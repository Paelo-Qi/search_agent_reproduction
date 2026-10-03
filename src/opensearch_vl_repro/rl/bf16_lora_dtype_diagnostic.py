"""Runtime-only LoRA A/B dtype experiment; never a Gate or training path."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import weakref
from collections import Counter

from opensearch_vl_repro.rl import bf16_sdpa_diagnostic as backend
from opensearch_vl_repro.rl import merge_precision_diagnostic as precision
from opensearch_vl_repro.rl import policy_handoff_diagnostic as handoff
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json

VERSION = "bf16-lora-dtype-forensic-v1"
META = {**backend.META, "diagnostic_version": VERSION}
VARIANTS = dict(A3="dynamic_bf16_base_default_lora_sdpa",
                A4="dynamic_bf16_base_bf16_lora_sdpa", B3="merged_bf16_sdpa")
PAIRS = {"A3_vs_B3": ("A3", "B3"), "A4_vs_B3": ("A4", "B3"), "A3_vs_A4": ("A3", "A4")}
HISTORY_DIRS = (*backend.HISTORY_DIRS, "bf16_sdpa_backend_diagnostic")


class ExperimentNotInformative(ValueError):
    """Already-BF16 LoRA supplies no controlled dtype contrast."""


def tensor_fingerprint(value):
    """Hash shape/dtype/raw values in bounded CPU chunks; never export tensors."""
    import torch
    digest = hashlib.sha256(json.dumps([list(value.shape), str(value.dtype)]).encode())
    for chunk in value.detach().reshape(-1).split(1048576):
        cpu = chunk.to(device="cpu").contiguous()
        if cpu.is_floating_point() and not bool(torch.isfinite(cpu).all()):
            raise ValueError("nonfinite parameter values in dtype experiment")
        digest.update(memoryview(cpu.view(torch.uint8).numpy()))
    return digest.hexdigest()


def non_lora_fingerprint(model):
    """Exact non-LoRA parameter identity, including vision/projector weights."""
    digest, count = hashlib.sha256(), 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name:
            digest.update(json.dumps([name, tensor_fingerprint(parameter)]).encode())
            count += 1
    if not count:
        raise ValueError("non-LoRA parameters missing")
    return dict(parameter_count=count, sha256=digest.hexdigest())


def lora_state(model, *, expected_layers=36):
    """Inspect actual dynamic PEFT tensors and unchanged execution semantics."""
    from peft import PeftModel
    if not isinstance(model, PeftModel):
        raise ValueError("dynamic dtype experiment requires an actual PeftModel")
    backend.validate_model(model, backend="sdpa", dynamic=True)
    roster = precision.lora_roster(model, expected_layers=expected_layers)
    parameters = dict(model.named_parameters())
    names_by_id = {id(p): name for name, p in parameters.items()}
    modules, tensors = [], []
    for name, module, (layer, suffix) in roster:
        dropout = module.lora_dropout["default"]
        if (model.training or module.training or dropout.training or module.disable_adapters
                or any(p.requires_grad for p in module.parameters())):
            raise ValueError("LoRA must be enabled, frozen and in inference/eval mode")
        modules.append(dict(module_name=name, layer_index=layer, target_suffix=suffix,
            active_adapters=list(module.active_adapters), rank=module.r["default"],
            alpha=module.lora_alpha["default"], scaling=module.scaling["default"],
            dropout_p=getattr(dropout, "p", 0.), dropout_training=dropout.training,
            module_training=module.training, merged=module.merged, disabled=module.disable_adapters))
        for letter in ("A", "B"):
            weight = getattr(module, "lora_" + letter)["default"].weight
            if not weight.is_floating_point():
                raise ValueError("floating LoRA weights required")
            tensors.append(dict(name=names_by_id[id(weight)], module_name=name, letter=letter,
                shape=list(weight.shape), dtype=str(weight.dtype), value_sha256=tensor_fingerprint(weight)))
    if (len({t["name"] for t in tensors}) != 2*len(roster)
            or {t["name"] for t in tensors} != {n for n in parameters if "lora_" in n}):
        raise ValueError("extra/missing/aliased LoRA tensors outside the A/B target roster")
    return dict(module_count=len(modules), tensor_count=len(tensors), per_module=modules,
                tensors=tensors, dtype_counts=dict(Counter(t["dtype"] for t in tensors)))


def cast_dynamic_lora_to_bf16(model, *, expected_layers=36):
    """Only change real A/B weight runtime dtype; check exact cast values."""
    import torch
    before = lora_state(model, expected_layers=expected_layers)
    if set(before["dtype_counts"]) == {"torch.bfloat16"}:
        raise ExperimentNotInformative("all A3 LoRA A/B weights already BF16; experiment_not_informative=true")
    base_before = non_lora_fingerprint(model)
    parameters, records = dict(model.named_parameters()), []
    with torch.no_grad():
        for source in before["tensors"]:
            parameter = parameters[source["name"]]
            cast = parameter.detach().to(dtype=torch.bfloat16)
            expected = tensor_fingerprint(cast)
            # Preserve Parameter identity, device, frozen state and module semantics.
            parameter.data = cast
            records.append({**source, "original_dtype": source["dtype"], "target_dtype": str(parameter.dtype),
                "source_value_sha256": source["value_sha256"], "expected_cast_value_sha256": expected,
                "actual_cast_value_sha256": tensor_fingerprint(parameter)})
    after = lora_state(model, expected_layers=expected_layers)
    base_after = non_lora_fingerprint(model)
    if (base_before != base_after or before["per_module"] != after["per_module"]
            or after["dtype_counts"] != {"torch.bfloat16": before["tensor_count"]}
            or any(r["actual_cast_value_sha256"] != r["expected_cast_value_sha256"] for r in records)
            or {r["name"]: r["actual_cast_value_sha256"] for r in records}
               != {r["name"]: r["value_sha256"] for r in after["tensors"]}):
        raise ValueError("A4 changed something other than the exact LoRA A/B BF16 cast")
    return dict(module_count=before["module_count"], tensor_count=before["tensor_count"],
        source_dtype_counts=before["dtype_counts"], target_dtype_counts=after["dtype_counts"],
        before=before, after=after, per_tensor=records,
        non_lora_before=base_before, non_lora_after=base_after, base_unchanged=True)


def verify_after_forward(model, state, base, *, require_bf16=False, expected_layers=36):
    current = lora_state(model, expected_layers=expected_layers)
    if require_bf16 and set(current["dtype_counts"]) != {"torch.bfloat16"}:
        raise ValueError("A4 LoRA dtype promoted away from BF16 during forward")
    if current != state or non_lora_fingerprint(model) != base:
        raise ValueError("dynamic parameters/LoRA execution semantics changed during forward")
    return current


def load_history(ctx):
    """Bind five historical directories, including complete v4.5.5 token data."""
    history = backend.load_history(ctx)
    prior = history["v452"]["tokens"]
    directory = ctx["reports"] / HISTORY_DIRS[-1]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    backend._validate_lineage(summary, ctx, version=backend.VERSION, count=len(prior))
    if summary.get("variants") != backend.VARIANTS:
        raise ValueError("v4.5.5 variant semantics mismatch")
    tokens = [json.loads(line) for line in (directory / "token_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(tokens) != len(prior):
        raise ValueError("v4.5.5 token count mismatch")
    for old, record in zip(prior, tokens, strict=True):
        if (any(record.get(k) != old[k] for k in backend.TOKEN_KEYS + backend.TEXT_KEYS)
                or any(record.get(k) != v for k, v in backend.META.items())
                or any(not isinstance(record.get(k + "_logprob"), (int, float))
                       or not math.isfinite(record[k + "_logprob"]) for k in backend.VARIANTS)):
            raise ValueError("v4.5.5 token IDs/order/metadata/logprob mismatch")
    for name, (a, b) in backend.PAIRS.items():
        metric = handoff.pair_metrics([r[a + "_logprob"] for r in tokens], [r[b + "_logprob"] for r in tokens])
        if summary["comparisons"].get(name) != metric:
            raise ValueError("v4.5.5 metrics disagree with per-token evidence")
    for kind in ("A3", "B3"):
        audit = summary.get("attention_backend_audits", {}).get(kind, {})
        if audit.get("model") != "sdpa" or any(r.get("backend") != "sdpa" for r in audit.get("attention_modules", [])):
            raise ValueError("v4.5.5 SDPA backend provenance mismatch")
        if summary.get("model_dtypes", {}).get(kind, {}).get("base_parameter_dtypes", {}).get("tensor_count_by_dtype", {}).keys() != {"torch.bfloat16"}:
            raise ValueError("v4.5.5 base dtype audit missing or not BF16")
    if not summary["model_dtypes"]["A3"].get("lora_parameter_dtypes") or not summary["model_dtypes"]["A3"].get("lora_targets"):
        raise ValueError("v4.5.5 default LoRA dtype evidence missing")
    return dict(summary=summary, tokens=tokens, prior_history_labels=list(history))


def direction_analysis(shift, original, residual):
    if len(residual) != len(original) or not all(math.isfinite(x) for x in residual):
        raise ValueError("finite aligned residuals required")
    corr = backend.direction_metrics(shift, original)["backend_shift_correlation"]
    toward = sum(abs(r) < abs(g) for r, g in zip(residual, original, strict=True))
    away = sum(abs(r) > abs(g) for r, g in zip(residual, original, strict=True))
    n = len(original)
    return dict(lora_dtype_shift_correlation_with_A3B3_gap=corr, direction_denominator=n,
                toward_B3_fraction=toward/n, away_from_B3_fraction=away/n, equal_fraction=(n-toward-away)/n)


def analyze_tokens(rows, prior, results, *, top_n):
    if top_n < 1 or set(results) != set(VARIANTS):
        raise ValueError("positive top_n and all three variants required")
    flat = {}
    for variant, values in results.items():
        if len(values) != len(rows) or any(len(v) != len(r["responses"]) for r, v in zip(rows, values, strict=True)):
            raise ValueError("variant response/row length mismatch")
        flat[variant] = [v[pos] for row, v in zip(rows, values, strict=True) for pos, mask in enumerate(row["response_mask"]) if mask == 1]
        if len(flat[variant]) != len(prior):
            raise ValueError("historical trainable token count changed")
    comparisons = {name: handoff.pair_metrics(flat[a], flat[b]) for name, (a, b) in PAIRS.items()}
    selected = [(row, pos) for row in rows for pos, mask in enumerate(row["response_mask"]) if mask == 1]
    tokens = []
    for idx, (old, (row, pos)) in enumerate(zip(prior, selected, strict=True)):
        if tuple(old[k] for k in backend.TOKEN_KEYS) != (idx, row["rollout_index"], row["step_index"], pos, row["responses"][pos]):
            raise ValueError("historical token IDs/order mismatch")
        record = {**META, **{k: old[k] for k in backend.TOKEN_KEYS + backend.TEXT_KEYS},
                  **{v + "_logprob": flat[v][idx] for v in VARIANTS}}
        for name, (a, b) in PAIRS.items():
            diff = flat[a][idx] - flat[b][idx]
            ratio = math.exp(diff)
            record.update({name + "_diff": diff, name + "_ratio": ratio,
                           name + "_clipped": ratio < .8 or ratio > 1.28, name + "_sign": (diff > 0) - (diff < 0)})
        old_diff = old["A3_logprob"] - old["B3_logprob"]
        old_ratio = math.exp(old_diff)
        record.update(prior_v455_A3_vs_B3_diff=old_diff,
                      original_outlier_abs_reduction=abs(old_diff)-abs(record["A4_vs_B3_diff"]),
                      original_outlier_reduction_fraction=precision.reduction_fraction(abs(old_diff), abs(record["A4_vs_B3_diff"])),
                      original_clipped_to_nonclipped=(old_ratio < .8 or old_ratio > 1.28) and not record["A4_vs_B3_clipped"])
        tokens.append(record)
    baseline, measured = comparisons["A3_vs_B3"], comparisons["A4_vs_B3"]
    improvement = {name: precision.reduction_fraction(baseline[key], measured[key]) for name, key in (
        ("mean_abs_reduction_fraction", "mean_abs_logprob_diff"), ("max_abs_reduction_fraction", "max_abs_logprob_diff"),
        ("clip_fraction_reduction_fraction", "clip_fraction_0p8_1p28"))}
    clips = {name: {t["global_trainable_index"] for t in tokens if t[name + "_clipped"]} for name in PAIRS}
    intersections = {" & ".join(names): len(set.intersection(*(clips[name] for name in names)))
                     for size in (2, 3) for names in itertools.combinations(PAIRS, size)}
    clips.update(A3_vs_B3_only=clips["A3_vs_B3"]-clips["A4_vs_B3"], A4_vs_B3_only=clips["A4_vs_B3"]-clips["A3_vs_B3"])
    return tokens, dict(comparisons=comparisons, lora_dtype_alignment_improvement=improvement,
        **direction_analysis([t["A3_vs_A4_diff"] for t in tokens], [t["A3_vs_B3_diff"] for t in tokens], [t["A4_vs_B3_diff"] for t in tokens]),
        clip_sets={k: sorted(v) for k, v in clips.items()}, clip_set_counts={k: len(v) for k, v in clips.items()},
        clip_set_intersections=intersections,
        original_A3_vs_B3_outliers=sorted(tokens, key=lambda t: (-abs(t["prior_v455_A3_vs_B3_diff"]), t["global_trainable_index"]))[:top_n],
        top_outliers={name: sorted(tokens, key=lambda t: (-abs(t[name + "_diff"]), t["global_trainable_index"]))[:top_n] for name in PAIRS})


def repeat_checks(tokens, history):
    old = history["tokens"]
    if len(tokens) != len(old):
        raise ValueError("repeat token count mismatch")
    baseline = history["summary"]["comparisons"]["A3_vs_B3"]
    current = handoff.pair_metrics([r["A3_logprob"] for r in tokens], [r["B3_logprob"] for r in tokens])
    limits = dict(mean_abs=max(1e-4, .1*baseline["mean_abs_logprob_diff"]), max_abs=max(1e-3, .1*baseline["max_abs_logprob_diff"]))
    repeats = {v: handoff.pair_metrics([r[v + "_logprob"] for r in tokens], [r[v + "_logprob"] for r in old]) for v in ("A3", "B3")}
    return dict(pair_metric_deltas={k: current[k]-baseline[k] for k in current}, per_variant_repeat_deltas=repeats,
                diagnostic_repeat_limits=limits, interpretation_allowed=all(
                    r["mean_abs_logprob_diff"] <= limits["mean_abs"] and r["max_abs_logprob_diff"] <= limits["max_abs"] for r in repeats.values()))


def run_diagnostic(args, root):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output, reports = handoff.diagnostic_paths(root, args.run_id)
    destination = reports / "bf16_lora_dtype_diagnostic"
    if not destination.resolve().is_relative_to((root / "reports/rl_gate_c").resolve()):
        raise ValueError("diagnostic destination escapes report tree")
    if destination.exists():
        raise FileExistsError("existing LoRA dtype diagnostic protected; overwrite forbidden")
    if not args.local_files_only or args.top_n < 1:
        raise ValueError("local-files-only and positive top_n required")
    adapter = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    trees = [output, args.base_model_path, adapter, root / "configs", *(reports / d for d in HISTORY_DIRS)]
    files = [adapter.parent / "metadata.json", reports / "gate_c_report.json"]
    before = handoff.source_checksums(trees, files)
    destination.mkdir(parents=True, exist_ok=False)
    stage, analysis, audits, lora_audits, peaks, destroyed = "artifact_validation", {}, {}, {}, {}, {}
    informative, cast_audit = None, None
    try:
        ctx = handoff.validate_attempt(root, args.run_id, args.base_model_path)
        history = load_history(ctx)
        import torch
        if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("requires one visible BF16 CUDA GPU, no distributed launch")
        versions = {k: importlib.metadata.version(k) for k in backend.PACKAGES}
        if versions != {k: ctx["identity"]["software_versions"][k] for k in backend.PACKAGES} or versions["verl"] != "0.6.1":
            raise ValueError("software differs from historical environment")
        torch.cuda.set_device(0)
        device, results = torch.device("cuda", 0), {}
        numeric_flags = backend.numeric_environment(torch)
        original_state, original_base = None, None
        for kind in VARIANTS:
            stage = kind + "_load"
            torch.cuda.reset_peak_memory_stats(0)
            model, ref = None, None
            audits[kind] = {}
            try:
                model = backend.load_bf16_sdpa("A3" if kind == "A4" else kind, ctx, args.base_model_path, device)
                ref = weakref.ref(model)
                if kind != "B3":
                    stage = kind + "_lora_inspection"
                    state, base = lora_state(model), non_lora_fingerprint(model)
                    if kind == "A3":
                        original_state, original_base = state, base
                        lora_audits[kind] = dict(state)
                        informative = set(state["dtype_counts"]) != {"torch.bfloat16"}
                        if not informative:
                            raise ExperimentNotInformative("A3 LoRA A/B already BF16; experiment_not_informative=true")
                        old_audit = history["summary"]["model_dtypes"]["A3"]
                        if (precision.dtype_counts(model, lora=True) != old_audit["lora_parameter_dtypes"]
                                or precision.lora_dtype_audit(model) != old_audit["lora_targets"]):
                            raise ValueError("A3 actual default LoRA dtype/targets differ from v4.5.5")
                    else:
                        if state != original_state or base != original_base:
                            raise ValueError("fresh A4 differs from A3 before cast")
                        stage = "A4_runtime_lora_cast"
                        cast_audit = cast_dynamic_lora_to_bf16(model)
                        state = cast_audit["after"]
                        lora_audits[kind] = {"before": cast_audit["before"], "after": state}
                stage = kind + "_forward"
                audits[kind]["base_parameter_dtypes"] = precision.dtype_counts(model, lora=False)
                audits[kind]["lora_parameter_dtypes"] = precision.dtype_counts(model, lora=True)
                if kind != "B3":
                    audits[kind]["lora_targets"] = precision.lora_dtype_audit(model)
                results[kind] = backend.forward_with_audit(model, ctx["rows"], output / "group", device=device, audit=audits[kind])
                backend.validate_model(model, backend="sdpa", dynamic=kind != "B3")
                if kind != "B3":
                    stage = kind + "_post_forward_validation"
                    verified = verify_after_forward(model, state, base, require_bf16=kind == "A4")
                    lora_audits[kind]["post_forward_dtype_counts"] = verified["dtype_counts"]
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                peaks[kind] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            destroyed[kind] = ref is None or ref() is None
            if not destroyed[kind]:
                raise RuntimeError(f"{kind} still resident before next variant")
        if backend.numeric_environment(torch) != numeric_flags:
            raise ValueError("numerical/backend flags changed during diagnostic")
        stage = "token_analysis"
        tokens, analysis = analyze_tokens(ctx["rows"], history["tokens"], results, top_n=args.top_n)
        if len(tokens) != ctx["alignment"]["masked_token_count"]:
            raise ValueError("original alignment token count changed")
        analysis["repeat_delta_vs_v455"] = repeat_checks(tokens, history)
        if not analysis["repeat_delta_vs_v455"]["interpretation_allowed"]:
            raise ValueError("A3/B3 repeat differs materially from v4.5.5; diagnostic interpretation refused")
        stage = "report_publication"
        after = handoff.source_checksums(trees, files)
        handoff.assert_sources_unchanged(before, after)
        summary = {**META, **analysis, "execution_succeeded": True, "gate_run_id": args.run_id,
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "temperature": .7,
            "collection_attempt": ctx["group"]["identity"]["collection_attempt"],
            "formal_sft_adapter_fingerprint": ctx["group"]["identity"]["pre_update_policy_fingerprint"],
            "production_merged_checkpoint_fingerprint": ctx["group"]["merged_checkpoint_fingerprint"],
            "token_count": len(tokens), "software_versions": versions, "variants": VARIANTS,
            "model_sources": {"A3": str(args.base_model_path), "A4": str(args.base_model_path), "B3": str(ctx["merged"]), "adapter": str(ctx["adapter"])},
            "lora_dtype_audit": lora_audits, "lora_cast_audit": cast_audit, "models_destroyed": destroyed,
            "peak_gpu_memory": peaks, "numeric_environment": numeric_flags, "model_dtypes": audits,
            "attention_backend_audits": {k: v["resolved_attention"] for k, v in audits.items()},
            "source_artifacts_unchanged": True, "experiment_informative": informative,
            "experiment_not_informative": False,
            "interpretation": "Measurements only; no automatic root-cause confirmation, production repair or Gate decision."}
        atomic_json(destination / "source_checksums.json", {**META, "before": before, "after": after})
        atomic_json(destination / "lora_cast_audit.json", {**META, **cast_audit})
        atomic_json(destination / "clip_sets.json", {**META, "sets": analysis["clip_sets"], "intersections": analysis["clip_set_intersections"]})
        with (destination / "token_diagnostics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for token in tokens:
                stream.write(json.dumps(token, ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(destination / "summary.json", summary)
        print(f"TOKENS: {len(tokens)}")
        for name, (a, b) in PAIRS.items():
            m = analysis["comparisons"][name]
            print(f"{name} ({VARIANTS[a]} vs {VARIANTS[b]}): mean_abs={m['mean_abs_logprob_diff']:.9g} max_abs={m['max_abs_logprob_diff']:.9g} clip={m['clip_fraction_0p8_1p28']:.9g}")
        improvement = analysis["lora_dtype_alignment_improvement"]
        print("LORA DTYPE ALIGNMENT IMPROVEMENT: "
              f"mean_abs_reduction={improvement['mean_abs_reduction_fraction']} "
              f"max_abs_reduction={improvement['max_abs_reduction_fraction']} clip_reduction={improvement['clip_fraction_reduction_fraction']}")
        print("corr(lora_dtype_shift, original_A3_B3_gap)=" + str(analysis["lora_dtype_shift_correlation_with_A3B3_gap"]))
        for name in ("toward_B3_fraction", "away_from_B3_fraction", "equal_fraction"):
            print(f"{name}={analysis[name]}")
        print("A3_LORA_DTYPES: " + json.dumps(lora_audits["A3"]["dtype_counts"]))
        print("A4_LORA_DTYPES: " + json.dumps(cast_audit["target_dtype_counts"]))
        print(f"REPORT: {destination / 'summary.json'}")
        return 0
    except BaseException as exc:
        after = None
        try:
            after = handoff.source_checksums(trees, files)
            handoff.assert_sources_unchanged(before, after)
            unchanged = True
        except BaseException as error:
            unchanged = False
            exc = RuntimeError(f"{exc}; source integrity failure: {error}")
        atomic_json(destination / "source_checksums.json", {**META, "before": before, "after": after, "source_artifacts_unchanged": unchanged})
        atomic_json(destination / "summary.json", {**META, "execution_succeeded": False, "gate_run_id": args.run_id,
            "stage": stage, "error": str(exc), "source_artifacts_unchanged": unchanged,
            "experiment_informative": informative, "experiment_not_informative": informative is False,
            "partial_analysis": analysis, "model_dtypes": audits, "lora_dtype_audit": lora_audits,
            "lora_cast_audit": cast_audit, "peak_gpu_memory": peaks, "models_destroyed": destroyed})
        raise exc
