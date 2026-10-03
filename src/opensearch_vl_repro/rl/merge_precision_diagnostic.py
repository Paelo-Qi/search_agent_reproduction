"""Diagnostic-only CPU FP32 LoRA merge -> BF16 save/reload, no production changes."""
from __future__ import annotations

import copy
import gc
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import weakref
from collections import Counter
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json
from opensearch_vl_repro.rl import policy_handoff_diagnostic as handoff
from opensearch_vl_repro.sft_tool_audit import sha256_file

VERSION = "merge-precision-forensic-v1"
META = dict(diagnostic_version=VERSION, diagnostic_only=True, evidence_only=True,
            formal_rl_initialization_allowed=False, not_for_rollout=True, not_for_training=True)
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
PAIRS = {
    "dynamic_peft_vs_production_merge": ("dynamic_peft_logprob", "production_merge_logprob"),
    "dynamic_peft_vs_fp32_merge_bf16": ("dynamic_peft_logprob", "fp32_merge_bf16_logprob"),
    "production_merge_vs_fp32_merge_bf16": ("production_merge_logprob", "fp32_merge_bf16_logprob"),
}


def module_key(name):
    suffix = name.removesuffix(".weight").rsplit(".", 1)[-1]
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if suffix not in TARGETS or match is None or any(x in name.lower() for x in ("visual", "vision", "merger", "projector")):
        return None
    return int(match[1]), suffix


def dtype_counts(model, *, lora=None):
    tensors, elements = Counter(), Counter()
    for name, value in model.named_parameters():
        if lora is not None and ("lora_" in name) != lora:
            continue
        tensors[str(value.dtype)] += 1
        elements[str(value.dtype)] += value.numel()
    return {"tensor_count_by_dtype": dict(tensors), "element_count_by_dtype": dict(elements)}


def lora_roster(model, *, expected_layers=36):
    result = []
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        key = module_key(name)
        if (key is None or set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"}
                or list(module.active_adapters) != ["default"] or module.merged
                or getattr(module, "fan_in_fan_out", False) or getattr(module, "lora_variant", {})
                or any(getattr(module, "use_dora", {}).values())
                or getattr(module.lora_B["default"], "bias", None) is not None):
            raise ValueError("diagnostic supports only original, unmerged vanilla language Linear LoRA")
        result.append((name, module, key))
    if len(result) != expected_layers * len(TARGETS) or {r[2] for r in result} != {
        (layer, suffix) for layer in range(expected_layers) for suffix in TARGETS}:
        raise ValueError("LoRA target roster does not match all decoder layers/seven suffixes")
    return result


def lora_dtype_audit(model, *, expected_layers=36):
    records = [dict(module_name=name, layer_index=layer, target_suffix=suffix,
        base_target_weight_dtype=str(module.get_base_layer().weight.dtype),
        lora_A_dtype=str(module.lora_A["default"].weight.dtype),
        lora_B_dtype=str(module.lora_B["default"].weight.dtype))
        for name, module, (layer, suffix) in lora_roster(model, expected_layers=expected_layers)]
    return {"per_module": records, "by_suffix": dtype_by_suffix(records)}


def dtype_by_suffix(records):
    fields = sorted({key for r in records for key in r if "dtype" in key})
    return {suffix: {field: dict(Counter(r[field] for r in records
        if r["target_suffix"] == suffix and field in r)) for field in fields}
        for suffix in sorted({r["target_suffix"] for r in records})}


def quantization_stats(weight_bf16, delta_fp32, *, chunk_size=1048576):
    """Full-element statistics in bounded chunks; never serialize weight tensors.

    'despite nonzero delta' uses NONZERO delta elements as its denominator.
    The separate all-element unchanged fraction also includes zero deltas.
    """
    import torch
    if (weight_bf16.dtype != torch.bfloat16 or delta_fp32.dtype != torch.float32
            or weight_bf16.shape != delta_fp32.shape or not weight_bf16.numel() or chunk_size < 1):
        raise ValueError("quantization statistics require aligned BF16 base and FP32 delta")
    base, delta = weight_bf16.detach().flatten().cpu(), delta_fp32.detach().flatten().cpu()
    ds = es = 0.; dm = em = 0.; unchanged = nonzero = lost = 0
    with torch.no_grad():
        for start in range(0, base.numel(), chunk_size):
            w, d = base[start:start + chunk_size], delta[start:start + chunk_size]
            target = w.float() + d
            rounded = target.to(torch.bfloat16)
            error = rounded.float() - target
            if not bool(torch.isfinite(d).all() & torch.isfinite(target).all() & torch.isfinite(error).all()):
                raise ValueError("nonfinite merge weight/delta/quantization residual")
            da, ea = d.abs(), error.abs()
            ds += da.double().sum().item(); es += ea.double().sum().item()
            dm = max(dm, da.max().item()); em = max(em, ea.max().item())
            nz, same = d != 0, rounded == w
            nonzero += int(nz.sum()); unchanged += int(same.sum()); lost += int((nz & same).sum())
    n = base.numel()
    return {"weight_numel": n, "delta_abs_sum": ds, "quant_error_abs_sum": es,
            "delta_abs_mean": ds / n, "delta_abs_max": dm,
            "quant_error_abs_mean": es / n, "quant_error_abs_max": em,
            "quant_error_mean_relative_to_delta": es / ds if ds else None,
            "quant_error_max_relative_to_delta": em / dm if dm else None,
            "nonzero_delta_count": nonzero, "unchanged_weight_count": unchanged,
            "unchanged_despite_nonzero_delta_count": lost,
            "fraction_delta_rounded_to_zero_effectively": unchanged / n,
            "fraction_merged_weight_unchanged_despite_nonzero_delta": lost / nonzero if nonzero else None,
            "unchanged_despite_delta_fraction": lost / nonzero if nonzero else None}


def suffix_summary(records):
    result = {}
    for suffix in sorted({r["target_suffix"] for r in records}):
        selected = [r for r in records if r["target_suffix"] == suffix]
        n = sum(r["weight_numel"] for r in selected)
        ds, es = sum(r["delta_abs_sum"] for r in selected), sum(r["quant_error_abs_sum"] for r in selected)
        nz = sum(r["nonzero_delta_count"] for r in selected)
        lost = sum(r["unchanged_despite_nonzero_delta_count"] for r in selected)
        result[suffix] = {"module_count": len(selected), "element_count": n,
            "delta_mean_abs": ds / n, "delta_max_abs": max(r["delta_abs_max"] for r in selected),
            "quant_error_mean_abs": es / n, "quant_error_max_abs": max(r["quant_error_abs_max"] for r in selected),
            "quant_error_mean_relative_to_delta": es / ds if ds else None,
            "nonzero_delta_count": nz, "unchanged_fraction": lost / nz if nz else None}
    return result


def fp32_merge_lora_model(wrapped, *, expected_layers=36):
    """Use actual PEFT merge, audit FP32 delta at the REAL merge calls on CPU."""
    import torch
    wrapped.eval().requires_grad_(False)
    roster = lora_roster(wrapped, expected_layers=expected_layers)
    records, guards = [], []
    before = {"base_parameter_dtypes": dtype_counts(wrapped, lora=False),
              "lora_parameter_dtypes": dtype_counts(wrapped, lora=True)}
    calls = Counter()
    try:
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            for name, module, (layer, suffix) in roster:
                base = module.get_base_layer()
                if base.weight.dtype != torch.bfloat16 or base.weight.device.type != "cpu":
                    raise ValueError("B1 merge must start from the ORIGINAL BF16 base on CPU")
                entry = dict(module_name=name, layer_index=layer, target_suffix=suffix,
                    base_target_weight_dtype_before_promotion=str(base.weight.dtype),
                    lora_A_dtype_before_promotion=str(module.lora_A["default"].weight.dtype),
                    lora_B_dtype_before_promotion=str(module.lora_B["default"].weight.dtype))
                module.lora_A["default"].float(); module.lora_B["default"].float()
                delta = module.get_delta_weight("default")
                if delta.dtype != torch.float32 or delta.device.type != "cpu":
                    raise ValueError("actual LoRA delta is not CPU FP32")
                entry.update(quantization_stats(base.weight, delta))
                entry.update(computed_delta_dtype=str(delta.dtype), lora_A_dtype=str(module.lora_A["default"].weight.dtype),
                             lora_B_dtype=str(module.lora_B["default"].weight.dtype))
                del delta
                base.float()  # promote only the weights/biases participating in LoRA merge
                entry["base_target_weight_dtype_at_merge"] = str(base.weight.dtype)
                original = module.get_delta_weight
                previous_override = module.__dict__.get("get_delta_weight")
                def guarded(adapter, original=original, base=base, name=name):
                    value = original(adapter)
                    if base.weight.dtype != torch.float32 or value.dtype != torch.float32 or value.device.type != "cpu":
                        raise ValueError("PEFT real merge arithmetic silently downcast")
                    calls[name] += 1
                    return value
                guards.append((module, previous_override))
                module.get_delta_weight = guarded
                records.append(entry)
            merged = wrapped.merge_and_unload(safe_merge=True)
            if any(calls[r["module_name"]] != 1 for r in records):
                raise ValueError("each original LoRA target must execute one audited FP32 delta merge")
            for _, module, _ in roster:
                if module.get_base_layer().weight.dtype != torch.float32:
                    raise ValueError("PEFT merged target is not FP32 before explicit cast")
    finally:
        for module, previous in guards:
            if previous is None:
                del module.get_delta_weight
            else:
                module.get_delta_weight = previous
    return merged, {"per_module": records, "by_suffix": suffix_summary(records)}, {
        **before, "merge_device": "cpu", "merge_arithmetic_dtype": "torch.float32",
        "actual_fp32_delta_merge_call_count": sum(calls.values()),
        "merged_parameter_dtypes_before_cast": dtype_counts(merged),
        "target_dtype_counts_by_suffix": dtype_by_suffix(records),
        "target_dtype_counts": {key: dict(Counter(r[key] for r in records)) for key in (
            "base_target_weight_dtype_at_merge", "lora_A_dtype", "lora_B_dtype", "computed_delta_dtype")}}


def disk_space_required(base_path, adapter, destination):
    size = sum(p.stat().st_size for p in base_path.glob("model*.safetensors"))
    if not size:
        raise ValueError("offline base weights missing for disk estimate")
    # One BF16 checkpoint + at least 1GiB margin; 2x source weight bytes is
    # conservative for the same architecture. No FP32 model is serialized.
    required = 2 * size + sum(p.stat().st_size for p in adapter.glob("*.safetensors")) + 1024 ** 3
    if shutil.disk_usage(destination).free < required:
        raise OSError(f"insufficient diagnostic disk before model load/save: need {required} bytes")
    return required


def load_cpu_peft(ctx, base_path):
    from peft import PeftModel
    from opensearch_vl_repro.model import load_base_model
    config = copy.deepcopy(ctx["sft"]); config["model"]["name_or_path"] = str(base_path)
    base = load_base_model(config, for_training=False).cpu()
    return PeftModel.from_pretrained(base, str(ctx["adapter"]), is_trainable=False, local_files_only=True).eval().requires_grad_(False)


def saved_dtype_counts(directory):
    """Inspect serialized safetensors headers, not a reload's forced load dtype."""
    from safetensors import safe_open
    counts, elements, names = Counter(), Counter(), set()
    for file in sorted(directory.glob("model*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in names:
                    raise ValueError("duplicate serialized model tensor")
                names.add(name)
                view = stream.get_slice(name)
                dtype = view.get_dtype()
                if dtype.startswith("F") and dtype != "BF16":
                    raise ValueError("serialized diagnostic floating weights are not BF16")
                counts[dtype] += 1
                elements[dtype] += math.prod(view.get_shape())
    if not counts.get("BF16"):
        raise ValueError("serialized diagnostic BF16 weights missing")
    return {"tensor_count_by_dtype": dict(counts), "element_count_by_dtype": dict(elements)}


def diagnostic_fp32_merge(ctx, base_path, destination, device):
    """Staged BF16 checkpoint; fresh CPU reload/hash BEFORE GPU diagnostic use."""
    import torch
    from peft import PeftModel
    from opensearch_vl_repro.model import load_base_model
    from opensearch_vl_repro.rl.rollout_sync import require_plain_merged_model, validate_merged_files
    required = disk_space_required(base_path, ctx["adapter"], destination)
    target = destination / "tmp_fp32_merge"
    if target.exists():
        raise FileExistsError("diagnostic model overwrite forbidden")
    staging = Path(tempfile.mkdtemp(prefix=".fp32-merge-", dir=destination))
    wrapped = load_cpu_peft(ctx, base_path)
    merged, stats, audit = fp32_merge_lora_model(wrapped)
    require_plain_merged_model(merged, PeftModel)
    merged.to(dtype=torch.bfloat16)
    audit.update(cast_target_dtype="torch.bfloat16", saved_parameter_dtypes=dtype_counts(merged),
                 base_load_dtype="torch.bfloat16", forward_autocast_dtype="torch.bfloat16",
                 required_free_disk_bytes=required)
    if set(audit["saved_parameter_dtypes"]["tensor_count_by_dtype"]) != {"torch.bfloat16"}:
        raise ValueError("explicit BF16 cast did not cover all model parameters")
    merged.config._name_or_path = BASE_MODEL
    merged.save_pretrained(str(staging), safe_serialization=True, max_shard_size="2GB")
    # Copy immutable processor/tokenizer assets, never invoke a processor.
    for file in ctx["merged"].iterdir():
        if file.is_file() and file.name not in {"config.json", "merge_manifest.json", "model.safetensors.index.json"} and not file.name.endswith(".safetensors"):
            shutil.copyfile(file, staging / file.name)
    model_ref, peft_ref = weakref.ref(merged), weakref.ref(wrapped)
    del merged, wrapped
    gc.collect()
    if model_ref() is not None or peft_ref() is not None:
        raise RuntimeError("CPU merge model still alive before fresh diagnostic reload")
    files = validate_merged_files(staging)
    audit["serialized_tensor_dtypes"] = saved_dtype_counts(staging)
    config = copy.deepcopy(ctx["sft"]); config["model"]["name_or_path"] = str(staging)
    fresh = load_base_model(config, for_training=False).cpu().eval().requires_grad_(False)
    require_plain_merged_model(fresh, PeftModel)
    audit["reload_parameter_dtypes"] = dtype_counts(fresh)
    if set(audit["reload_parameter_dtypes"]["tensor_count_by_dtype"]) != {"torch.bfloat16"}:
        raise ValueError("fresh diagnostic reload is not all BF16")
    if fresh.config._attn_implementation != ctx["sft"]["model"]["attn_implementation"]:
        raise ValueError("fresh diagnostic attention differs from formal SFT")
    audit.update(cpu_merge_model_destroyed=True, fresh_reload_completed=True,
                 diagnostic_model_file_sha256=files, diagnostic_model_fingerprint=canonical_json_sha256(files))
    atomic_json(staging / "diagnostic_metadata.json", {**META, "merge_complete": True,
        "source_adapter_fingerprint": ctx["identity"]["source_sft_actor"]["source_sft_adapter_fingerprint"],
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "dtype_audit": audit})
    staging.rename(target)  # only a fully saved, reload-validated diagnostic model is published
    return fresh.to(device), target, stats, audit


def cleanup_models(destination, *, keep=False):
    """Delete ONLY this invocation's exact direct-child temporary checkpoints."""
    deleted = []
    for path in list(destination.glob(".fp32-merge-*")) + [destination / "tmp_fp32_merge"]:
        if not path.exists():
            continue
        if path.resolve().parent != destination.resolve() or path.is_symlink() or not path.is_dir():
            raise ValueError("unsafe diagnostic cleanup target")
        if not keep:
            shutil.rmtree(path)
            deleted.append(path.name)
    return deleted


def weight_compare_stats(production, diagnostic, *, expected_layers=36):
    """Read one target at a time from safe tensor files, no resident HF models."""
    import torch
    from safetensors import safe_open
    def index(directory):
        found = {}
        for file in sorted(directory.glob("model*.safetensors")):
            with safe_open(str(file), framework="pt", device="cpu") as stream:
                for name in stream.keys():
                    key = module_key(name) if name.endswith(".weight") else None
                    if key is not None:
                        if key in found:
                            raise ValueError("duplicate target weight in diagnostic checkpoint")
                        found[key] = (file, name)
        return found
    a, b = index(production), index(diagnostic)
    expected = {(layer, suffix) for layer in range(expected_layers) for suffix in TARGETS}
    if a.keys() != b.keys() or a.keys() != expected:
        raise ValueError("B0/B1 target weight roster mismatch")
    rows = []
    for key in sorted(expected):
        with safe_open(str(a[key][0]), framework="pt", device="cpu") as sa, safe_open(str(b[key][0]), framework="pt", device="cpu") as sb:
            wa, wb = sa.get_tensor(a[key][1]), sb.get_tensor(b[key][1])
            if wa.shape != wb.shape or wa.dtype != torch.bfloat16 or wb.dtype != torch.bfloat16:
                raise ValueError("B0/B1 target weights must be aligned BF16 tensors")
            total = maximum = 0.; count = 0
            for ca, cb in zip(wa.flatten().split(1048576), wb.flatten().split(1048576), strict=True):
                diff = (ca.float() - cb.float()).abs()
                if not bool(torch.isfinite(diff).all()):
                    raise ValueError("nonfinite B0/B1 weight difference")
                total += diff.double().sum().item(); maximum = max(maximum, diff.max().item())
                count += int(torch.count_nonzero(diff))
            rows.append(dict(layer_index=key[0], target_suffix=key[1], weight_numel=wa.numel(),
                abs_weight_diff_sum=total, mean_abs_weight_diff=total / wa.numel(), max_abs_weight_diff=maximum,
                nonzero_diff_count=count, nonzero_diff_fraction=count / wa.numel()))
            del wa, wb
    by_suffix = {}
    for suffix in sorted(TARGETS):
        selected = [r for r in rows if r["target_suffix"] == suffix]; n = sum(r["weight_numel"] for r in selected)
        by_suffix[suffix] = dict(element_count=n, mean_abs_weight_diff=sum(r["abs_weight_diff_sum"] for r in selected) / n,
            max_abs_weight_diff=max(r["max_abs_weight_diff"] for r in selected),
            nonzero_diff_fraction=sum(r["nonzero_diff_count"] for r in selected) / n)
    return {"scope": "all LoRA target weights, streaming; no full tensor export", "per_module": rows, "by_suffix": by_suffix}


def load_previous_forensic(ctx):
    directory = ctx["reports"] / "policy_handoff_diagnostic"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    tokens = [json.loads(line) for line in (directory / "token_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    if (summary.get("execution_succeeded") is not True or summary.get("diagnostic_only") is not True
            or summary.get("formal_rl_initialization_allowed") is not False or summary.get("source_artifacts_unchanged") is not True
            or summary.get("gate_run_id") != ctx["identity"]["run_id"]
            or summary.get("formal_sft_adapter_fingerprint") != ctx["group"]["identity"]["pre_update_policy_fingerprint"]
            or summary.get("merged_checkpoint_fingerprint") != ctx["group"]["merged_checkpoint_fingerprint"]
            or summary.get("collection_attempt") != ctx["group"]["identity"]["collection_attempt"]):
        raise ValueError("prior forensic summary lineage/execution mismatch")
    selected = [(r, pos, token) for r in ctx["rows"] for pos, (token, mask) in enumerate(zip(r["responses"], r["response_mask"], strict=True)) if mask == 1]
    if len(tokens) != len(selected) or summary.get("token_count") != len(selected):
        raise ValueError("prior forensic token count mismatch")
    for idx, (record, (row, pos, token)) in enumerate(zip(tokens, selected, strict=True)):
        if (record.get("global_trainable_index") != idx or record.get("rollout_index") != row["rollout_index"]
                or record.get("step_index") != row["step_index"] or record.get("response_token_position") != pos
                or record.get("token_id") != token or record.get("formal_rl_initialization_allowed") is not False
                or any(not math.isfinite(record[k]) for k in ("dynamic_peft_logprob", "merged_hf_logprob"))):
            raise ValueError("prior forensic token IDs/order/logprobs mismatch")
    metrics = handoff.pair_metrics([r["dynamic_peft_logprob"] for r in tokens], [r["merged_hf_logprob"] for r in tokens])
    if summary.get("dynamic_peft_vs_merged_hf") != metrics:
        raise ValueError("prior A/B metrics do not match prior token artifact")
    return tokens, metrics


def reduction_fraction(baseline, measured):
    if not math.isfinite(baseline) or not math.isfinite(measured) or baseline < 0 or measured < 0:
        raise ValueError("finite nonnegative reduction metrics required")
    return (baseline - measured) / baseline if baseline else None


def analyze_tokens(rows, prior, dynamic, production, improved, *, top_n):
    if top_n < 1 or len(rows) != len(dynamic) or len(rows) != len(production) or len(rows) != len(improved):
        raise ValueError("invalid precision diagnostic row count/top_n")
    tokens = []
    for row, av, bv, cv in zip(rows, dynamic, production, improved, strict=True):
        if any(len(values) != len(row["responses"]) for values in (av, bv, cv)):
            raise ValueError("precision diagnostic HF token count mismatch")
        for pos, mask in enumerate(row["response_mask"]):
            if mask != 1:
                continue
            old = prior[len(tokens)]
            value = {**META, **{key: old[key] for key in ("global_trainable_index", "rollout_index", "step_index",
                "response_token_position", "token_id", "token_repr", "decoded_token", "is_special_token")},
                "dynamic_peft_logprob": av[pos], "production_merge_logprob": bv[pos], "fp32_merge_bf16_logprob": cv[pos],
                "prior_dynamic_peft_logprob": old["dynamic_peft_logprob"], "prior_production_merge_logprob": old["merged_hf_logprob"]}
            if not all(math.isfinite(value[k]) for k in ("dynamic_peft_logprob", "production_merge_logprob", "fp32_merge_bf16_logprob")):
                raise ValueError("nonfinite merge precision sampled logprobs")
            for name, (a, b) in PAIRS.items():
                value[name + "_diff"] = value[a] - value[b]
            value["prior_dynamic_vs_production_diff"] = old["dynamic_peft_logprob"] - old["merged_hf_logprob"]
            value["original_outlier_abs_diff_reduction_fraction"] = reduction_fraction(
                abs(value["prior_dynamic_vs_production_diff"]), abs(value["dynamic_peft_vs_fp32_merge_bf16_diff"]))
            tokens.append(value)
    if len(tokens) != len(prior):
        raise ValueError("precision diagnostic trainable token count mismatch")
    comparisons = {name: handoff.pair_metrics([r[a] for r in tokens], [r[b] for r in tokens]) for name, (a, b) in PAIRS.items()}
    prod, better = comparisons["dynamic_peft_vs_production_merge"], comparisons["dynamic_peft_vs_fp32_merge_bf16"]
    improvements = {name: reduction_fraction(prod[key], better[key]) for name, key in (
        ("mean_abs_reduction_fraction", "mean_abs_logprob_diff"), ("max_abs_reduction_fraction", "max_abs_logprob_diff"),
        ("clip_fraction_reduction_fraction", "clip_fraction_0p8_1p28"))}
    return tokens, {"comparisons": comparisons, "improvements": improvements,
        "original_production_outliers": sorted(tokens, key=lambda r: (-abs(r["prior_dynamic_vs_production_diff"]), r["global_trainable_index"]))[:top_n],
        "repeat_forward_deltas": {
            "dynamic": handoff.pair_metrics([r["dynamic_peft_logprob"] for r in tokens], [r["prior_dynamic_peft_logprob"] for r in tokens]),
            "production": handoff.pair_metrics([r["production_merge_logprob"] for r in tokens], [r["prior_production_merge_logprob"] for r in tokens])}}


def run_diagnostic(args, root):
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output, reports = handoff.diagnostic_paths(root, args.run_id)
    destination = reports / "merge_precision_diagnostic"
    if not destination.resolve().is_relative_to((root / "reports/rl_gate_c").absolute()):
        raise ValueError("diagnostic report path escapes permitted tree")
    if destination.exists():
        raise FileExistsError("existing merge precision diagnostic protected; overwrite forbidden")
    adapter = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    trees = [output, args.base_model_path, adapter, reports / "policy_handoff_diagnostic"]
    files = [reports / "gate_c_report.json", adapter.parent / "metadata.json", root / "configs/rl_main.yaml", root / "configs/sft_main_imageid_v3.yaml"]
    before = handoff.source_checksums(trees, files)
    destination.mkdir(parents=True, exist_ok=False)
    stage = "artifact_validation"
    try:
        ctx = handoff.validate_attempt(root, args.run_id, args.base_model_path)
        prior, old_metrics = load_previous_forensic(ctx)
        import torch
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("merge precision diagnostic requires one visible BF16 CUDA GPU, no distributed launch")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "verl")}
        if versions != {name: ctx["identity"]["software_versions"][name] for name in versions} or versions["verl"] != "0.6.1":
            raise ValueError("diagnostic software differs from historical forward environment")
        torch.cuda.set_device(0); device = torch.device("cuda", 0)
        results, dtypes, memory, destroyed = {}, {}, {}, {}
        # A and B0 retain the identical v4.5.2 loader/forward. B0 is NEVER regenerated.
        for kind in ("dynamic", "merged"):
            stage = kind + "_forward"
            torch.cuda.reset_peak_memory_stats(0)
            model = handoff.load_diagnostic_model(kind, ctx, args.base_model_path, device)
            ref = weakref.ref(model)
            try:
                dtypes[kind] = {"base": dtype_counts(model, lora=False), "lora": dtype_counts(model, lora=True)}
                if kind == "dynamic":
                    dtypes[kind]["target_modules"] = lora_dtype_audit(model)
                results[kind] = handoff.forward_rows(model, ctx["rows"], output / "group", device=device)
            finally:
                del model; gc.collect(); torch.cuda.empty_cache()
            destroyed[kind] = ref() is None
            if not destroyed[kind]: raise RuntimeError("model remains alive before next variant")
            memory[kind] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        stage = "diagnostic_fp32_merge_save_reload"
        torch.cuda.reset_peak_memory_stats(0)
        model, model_path, module_stats, merge_audit = diagnostic_fp32_merge(ctx, args.base_model_path, destination, device)
        ref = weakref.ref(model)
        try:
            stage = "fp32_merge_bf16_forward"
            results["improved"] = handoff.forward_rows(model, ctx["rows"], output / "group", device=device)
        finally:
            del model; gc.collect(); torch.cuda.empty_cache()
        destroyed["fp32_merge_bf16"] = ref() is None
        if not destroyed["fp32_merge_bf16"]: raise RuntimeError("B1 remains alive after diagnostic forward")
        memory["fp32_merge_bf16"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        stage = "weight_and_token_analysis"
        weights = weight_compare_stats(ctx["merged"], model_path)
        tokens, analysis = analyze_tokens(ctx["rows"], prior, results["dynamic"], results["merged"], results["improved"], top_n=args.top_n)
        if len(tokens) != ctx["alignment"]["masked_token_count"]: raise ValueError("precision diagnostic token selection changed")
        deleted = cleanup_models(destination, keep=args.keep_diagnostic_model)
        after = handoff.source_checksums(trees, files); handoff.assert_sources_unchanged(before, after)
        summary = {**META, "execution_succeeded": True, "gate_run_id": args.run_id, "token_count": len(tokens),
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "software_versions": versions,
            "formal_sft_adapter_fingerprint": ctx["group"]["identity"]["pre_update_policy_fingerprint"],
            "production_merged_checkpoint_fingerprint": ctx["group"]["merged_checkpoint_fingerprint"],
            "collection_attempt": ctx["group"]["identity"]["collection_attempt"],
            "variants": {"dynamic_peft": "original BF16 base + dynamic formal checkpoint-3k LoRA",
                "production_merge": str(ctx["merged"]), "fp32_merge_then_bf16": str(model_path), "optional_fp32_merge_fp32": None},
            "temperature": .7, "model_dtypes": dtypes, "merge_dtypes": merge_audit, "models_destroyed": destroyed,
            "peak_gpu_memory": memory, "module_merge_stats_summary": module_stats["by_suffix"],
            "prior_v452_dynamic_vs_production_metrics": old_metrics, **analysis,
            "diagnostic_model_kept": args.keep_diagnostic_model, "deleted_temporary_model_directories": deleted,
            "source_artifacts_unchanged": True, "interpretation": "Measurements only; no production/Gate change or automatic root-cause declaration."}
        stage = "report_publication"
        atomic_json(destination / "module_merge_stats.json", {**META, **module_stats})
        atomic_json(destination / "weight_compare_stats.json", {**META, **weights})
        atomic_json(destination / "source_checksums.json", {**META, "before": before, "after": after, "source_artifacts_unchanged": True})
        with (destination / "token_diagnostics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for token in tokens: stream.write(json.dumps(token, ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        atomic_json(destination / "summary.json", summary)
        print(f"TOKENS: {len(tokens)}")
        for name, values in analysis["comparisons"].items():
            print(f"{name}: mean_abs={values['mean_abs_logprob_diff']:.9g} max_abs={values['max_abs_logprob_diff']:.9g} clip={values['clip_fraction_0p8_1p28']:.9g}")
        print("IMPROVEMENT: " + json.dumps(analysis["improvements"]))
        print(f"REPORT: {destination / 'summary.json'}")
        if deleted: print("Deleted diagnostic-only temporary model directories: " + ", ".join(deleted))
        return 0
    except BaseException as exc:
        cleanup_error = None
        try: deleted = cleanup_models(destination, keep=args.keep_diagnostic_model)
        except BaseException as error: deleted = []; cleanup_error = str(error)
        try:
            after = handoff.source_checksums(trees, files); handoff.assert_sources_unchanged(before, after); unchanged = True
        except BaseException as error:
            unchanged = False; exc = RuntimeError(f"{exc}; source integrity failure: {error}")
        atomic_json(destination / "summary.json", {**META, "execution_succeeded": False, "gate_run_id": args.run_id,
            "stage": stage, "error": str(exc), "source_artifacts_unchanged": unchanged,
            "deleted_temporary_model_directories": deleted, "cleanup_error": cleanup_error})
        raise exc
