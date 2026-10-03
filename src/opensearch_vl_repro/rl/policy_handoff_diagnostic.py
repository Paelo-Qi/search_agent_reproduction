"""Offline, read-only A/B/C forensic forward; NOT a Gate or training path.

Heavy model dependencies are lazy. Old source-code hashes are historical
provenance, never re-bound to the currently installed integration source.
"""
from __future__ import annotations

import copy
import gc
import importlib.metadata
import json
import math
import os
import re
import weakref
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json
from opensearch_vl_repro.rl.group import read_group
from opensearch_vl_repro.rl.training_batch import training_rows
from opensearch_vl_repro.sft_tool_audit import sha256_file

DIAGNOSTIC_VERSION = "policy-handoff-forensic-v451-v1"
PAIRS = {
    "dynamic_peft_vs_rollout": ("dynamic_peft_logprob", "vllm_old_logprob"),
    "dynamic_peft_vs_merged_hf": ("dynamic_peft_logprob", "merged_hf_logprob"),
    "merged_hf_vs_rollout": ("merged_hf_logprob", "vllm_old_logprob"),
}
TOKEN_PAIR_NAMES = {
    "dynamic_peft_vs_rollout": "dynamic_vs_rollout",
    "dynamic_peft_vs_merged_hf": "dynamic_vs_merged",
    "merged_hf_vs_rollout": "merged_vs_rollout",
}
METADATA = {"diagnostic_version": DIAGNOSTIC_VERSION, "diagnostic_only": True,
            "formal_rl_initialization_allowed": False}


def diagnostic_paths(root, run_id):
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", run_id) is None:
        raise ValueError("safe explicit Gate run-id required")
    return (root / "outputs/rl_gate_c" / run_id, root / "reports/rl_gate_c" / run_id)


def exact_merged_path(output, group):
    attempt = group["identity"]["collection_attempt"]
    if not isinstance(attempt, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", attempt) is None:
        raise ValueError("unsafe collection_attempt")
    path = output / ("merged-" + attempt)
    if not path.is_dir() or not path.resolve().is_relative_to(output.resolve()):
        raise FileNotFoundError("exact rollout merged checkpoint unavailable")
    return path


def diagnostic_rows_from_group(group):
    # Reuse the exact formal response/fatal mask. Advantages are irrelevant to
    # this forward-only diagnostic; no reward or estimator is recomputed.
    return training_rows(group, [0., 0.])


def source_checksums(trees, files):
    result = {}
    for tree in trees:
        if not tree.is_dir():
            raise FileNotFoundError(f"protected source directory missing: {tree}")
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                result[str(path.absolute())] = sha256_file(path)
    for path in files:
        result[str(path.absolute())] = sha256_file(path)
    return result


def assert_sources_unchanged(before, after):
    if before != after:
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        raise RuntimeError(f"source artifacts changed during diagnostic: {changed}")


def validate_attempt(root, run_id, base_path):
    """Validate old artifact internal consistency only; never bind a Gate run."""
    from opensearch_vl_repro.rl.config import load_rl_config
    from opensearch_vl_repro.rl.rollout_sync import validate_actor_adapter, validate_merged_files
    from opensearch_vl_repro.sft_train_plan import load_main_config

    output, reports = diagnostic_paths(root, run_id)
    identity = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((reports / "gate_c_report.json").read_text(encoding="utf-8"))
    if (report.get("stage") != "pre_update_policy_alignment" or report.get("passed") is not False
            or type(report.get("optimizer_step_count")) is not int or report["optimizer_step_count"] != 0
            or report.get("identity") != identity):
        raise ValueError("forensic mode requires pre-update failed attempt with exactly zero optimizer steps")
    if (output / "update_verified.json").exists():
        raise ValueError("updated/verified Gate attempt refused by forensic mode")
    marker = output / "gate_manifest.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("passed") is not False:
        raise ValueError("PASS or ambiguous Gate manifest refused by forensic mode")
    if (identity.get("run_id") != run_id or identity.get("gate_version") != "minimum-rl-integration-c-v1"
            or (identity.get("base_model"), identity.get("base_revision")) != (BASE_MODEL, BASE_REVISION)
            or identity.get("formal_rl_initialization_allowed") is not False
            or identity.get("identity_sha256") != canonical_json_sha256(
                {k: v for k, v in identity.items() if k != "identity_sha256"})):
        raise ValueError("historical run identity/lineage checksum mismatch")
    started = output / "update_started.json"
    if started.exists() and json.loads(started.read_text(encoding="utf-8")).get("identity") != identity:
        raise ValueError("update phase marker identity mismatch")
    gate = identity["gate_config"]
    if (gate.get("rollout_n") != 2 or gate.get("actor_world_size") != 2
            or gate.get("clip_ratio_low") != .2 or gate.get("clip_ratio_high") != .28
            or gate["vllm"]["temperature"] != .7 or gate["vllm"]["top_p"] != 1
            or gate["vllm"]["top_k"] != -1 or identity.get("logprobs_mode") != "processed_logprobs"):
        raise ValueError("historical Gate sampling/logprob semantics mismatch")
    group = read_group(output / "group")
    if group["identity"]["context"] != identity["identity_sha256"]:
        raise ValueError("group/run historical context mismatch")
    merged = exact_merged_path(output, group)
    merge = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
    mi = merge["identity"]
    actual_files = validate_merged_files(merged)
    actual_files.pop("merge_manifest.json", None)  # manifest is added AFTER original model file hashes
    if (merge.get("merge_complete") is not True or merge.get("no_active_peft") is not True
            or actual_files != mi["merged_file_sha256"]
            or mi["merged_checkpoint_fingerprint"] != canonical_json_sha256(
                {k: v for k, v in mi.items() if k != "merged_checkpoint_fingerprint"})
            or mi["merged_checkpoint_fingerprint"] != group["merged_checkpoint_fingerprint"]
            or (mi.get("base_model"), mi.get("base_revision")) != (BASE_MODEL, BASE_REVISION)):
        raise ValueError("exact merged checkpoint fingerprint/completeness mismatch")
    rl_path = root / "configs/rl_main.yaml"
    sft_path = root / "configs/sft_main_imageid_v3.yaml"
    if (sha256_file(rl_path) != identity["formal_config_sha256"]
            or sha256_file(sft_path) != identity["sft_config_sha256"]):
        raise ValueError("formal config differs from historical attempt")
    rl = load_rl_config(rl_path)
    sft = load_main_config(sft_path, base_eval_config=root / "configs/eval_base_300.yaml")
    adapter = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    if (root / rl["model"]["sft_adapter"]).resolve() != adapter.resolve():
        raise ValueError("only formal checkpoint-3k may initialize diagnostic A")
    actor = validate_actor_adapter(adapter=adapter, gate_manifest=None, rl_config=rl,
                                  sft_config=sft, source_sft_adapter=adapter)
    fingerprint = actor["source_sft_adapter_fingerprint"]
    if (actor != identity["source_sft_actor"] or merge.get("actor_provenance") != actor
            or group["identity"]["pre_update_policy_fingerprint"] != fingerprint
            or mi["actor_adapter_fingerprint"] != fingerprint
            or mi["source_sft_adapter_fingerprint"] != fingerprint):
        raise ValueError("formal checkpoint-3k source fingerprint/lineage mismatch")
    base_config = json.loads((base_path / "config.json").read_text(encoding="utf-8"))
    base_files = {p.name: sha256_file(p) for p in sorted(base_path.iterdir())
                  if p.is_file() and p.suffix in {".json", ".safetensors"}}
    if (base_files != identity["offline_base_file_sha256"] or not list(base_path.glob("model*.safetensors"))
            or base_config.get("model_type") != "qwen3_vl"
            or base_config.get("text_config", {}).get("num_hidden_layers") != 36
            or base_config.get("_commit_hash") not in {None, BASE_REVISION}):
        raise ValueError("pinned offline base files/revision differ from historical run")
    rows, counts = diagnostic_rows_from_group(group)
    alignment = json.loads((output / "pre_update_policy_alignment.json").read_text(encoding="utf-8"))
    from opensearch_vl_repro.rl.policy_alignment import STAT_FIELDS, alignment_artifact, alignment_checks
    if (alignment.get("identity") != identity or alignment.get("trajectory_group_id") != group["identity"]["trajectory_group_id"]
            or alignment.get("pre_update_policy_fingerprint") != fingerprint
            or alignment.get("masked_token_count") != counts["supervised_response_tokens"]
            or alignment.get("temperature") != .7 or alignment.get("rollout_temperature") != .7
            or alignment.get("passed") is not False or alignment.get("all_finite") is not True
            or alignment.get("optimizer_step_count") != 0
            or alignment.get("max_abs_logprob_diff_bound") != .1
            or report.get("pre_update_policy_alignment") != alignment
            or len(alignment.get("per_rank", [])) != 2
            or {r["rank"] for r in alignment["per_rank"]} != {0, 1}):
        raise ValueError("historical pre-update alignment/group/report evidence mismatch")
    for rank in alignment["per_rank"]:
        if (rank.get("masked_token_count") != counts["supervised_response_tokens"]
                or rank.get("all_finite") is not True or rank.get("temperature") != .7
                or rank.get("checks") != alignment_checks(rank)
                or rank.get("passed") is not all(alignment_checks(rank).values())):
            raise ValueError("historical rank alignment token/finite/temperature mismatch")
    if any(not isinstance(record.get(key), (int, float)) or isinstance(record[key], bool)
           or not math.isfinite(record[key]) for record in [alignment, *alignment["per_rank"]] for key in STAT_FIELDS):
        raise ValueError("historical finite-alignment statistics missing/nonfinite")
    # Recompute ONLY the stored statistics aggregation, not current policy,
    # identity binding or a Gate decision. The v4.5.1 pure helper is unchanged.
    if alignment != alignment_artifact(alignment["per_rank"], gate_version=identity["gate_version"],
        identity=identity, trajectory_group_id=group["identity"]["trajectory_group_id"], policy_fingerprint=fingerprint):
        raise ValueError("historical alignment aggregate differs from recorded rank evidence")
    return dict(output=output, reports=reports, identity=identity, group=group, rows=rows, counts=counts,
                alignment=alignment, merged=merged, merge=merge, sft=sft, adapter=adapter)


def load_row_inputs(directory, row, *, device):
    import torch

    path = (directory / row["multimodal_file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("unsafe saved multimodal path")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if ("input_ids" not in saved or saved["input_ids"].shape != (1, len(row["prompt_ids"]))
            or saved["input_ids"][0].tolist() != row["prompt_ids"]):
        raise ValueError("saved multimodal prompt IDs mismatch")
    if (len(row["responses"]) != len(row["old_log_probs"])
            or len(row["responses"]) != len(row["response_mask"])):
        raise ValueError("response/logprob/mask length mismatch")
    vision = {k: saved[k].to(device) for k in ("pixel_values", "image_grid_thw") if k in saved}
    if set(vision) != {"pixel_values", "image_grid_thw"} or not vision["pixel_values"].numel():
        raise ValueError("saved real multimodal tensors required")
    ids = torch.tensor([row["prompt_ids"] + row["responses"]], device=device, dtype=torch.long)
    return ids, torch.ones_like(ids), vision


def sampled_response_logprobs(logits, response_ids, *, temperature=.7, logprob_function=None):
    """Pinned verl non-rmpad slice: response t uses its PREVIOUS position."""
    import torch
    if temperature != .7:
        raise ValueError("diagnostic temperature must remain 0.7")
    if logprob_function is None:
        from verl.utils.torch_functional import logprobs_from_logits
        logprob_function = logprobs_from_logits
    length = response_ids.shape[-1]
    if logits.ndim != 3 or logits.shape[0] != 1 or length < 1 or logits.shape[1] <= length:
        raise ValueError("invalid full sequence/response logit shape")
    # As in verl: scale BEFORE slicing, preserve the actual logits dtype.
    logits.div_(temperature)
    for chunk in logits.split(256, dim=1):
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("nonfinite HF forward logits")
    response_logits = logits[:, -length - 1 : -1, :]
    logprobs = logprob_function(response_logits, response_ids, inplace_backward=False)
    if logprobs.shape != response_ids.shape or not bool(torch.isfinite(logprobs).all()):
        raise ValueError("nonfinite/misaligned HF sampled-token logprobs")
    return logprobs.detach().float().cpu().tolist()[0]


def forward_rows(model, rows, directory, *, device, logprob_function=None):
    import torch
    result = []
    model.eval()
    model.requires_grad_(False)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    rope_model = base.model
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for row in rows:
            ids, attention, vision = load_row_inputs(directory, row, device=device)
            positions, _ = rope_model.get_rope_index(input_ids=ids, image_grid_thw=vision["image_grid_thw"],
                                                     attention_mask=attention)
            output = model(input_ids=ids, attention_mask=attention, position_ids=positions,
                           **vision, use_cache=False)
            response_ids = torch.tensor([row["responses"]], dtype=torch.long, device=device)
            result.append(sampled_response_logprobs(output.logits, response_ids, logprob_function=logprob_function))
            del output, response_ids, ids, attention, positions, vision
    return result


def load_diagnostic_model(kind, ctx, base_path, device):
    import torch
    from peft import PeftModel
    from opensearch_vl_repro.model import load_base_model
    from opensearch_vl_repro.rl.rollout_sync import require_plain_merged_model

    config = copy.deepcopy(ctx["sft"])
    config["model"]["name_or_path"] = str(base_path if kind == "dynamic" else ctx["merged"])
    model = load_base_model(config, for_training=False).to(device)
    if kind == "dynamic":
        model = PeftModel.from_pretrained(model, str(ctx["adapter"]), is_trainable=False, local_files_only=True)
        if not any("lora_" in name for name, _ in model.named_parameters()):
            raise ValueError("dynamic HF model lacks actual LoRA parameters")
    else:
        require_plain_merged_model(model, PeftModel)
    model.eval().requires_grad_(False)
    if model.config._attn_implementation != ctx["sft"]["model"]["attn_implementation"]:
        raise ValueError("resolved diagnostic attention differs from formal SFT")
    if next(model.get_base_model().parameters() if kind == "dynamic" else model.parameters()).dtype != torch.bfloat16:
        raise ValueError("diagnostic base must be BF16")
    return model


def quantile(values, q):
    """Sorted linear interpolation, matching the standard inclusive quantile."""
    ordered = sorted(values)
    if not ordered or not 0 <= q <= 1:
        raise ValueError("nonempty valid quantile input required")
    pos = (len(ordered) - 1) * q
    left = int(pos)
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (pos - left)


def pair_metrics(first, second):
    if not first or len(first) != len(second) or any(not math.isfinite(x) for x in first + second):
        raise ValueError("finite aligned sampled-token logprobs required")
    diffs = [a - b for a, b in zip(first, second, strict=True)]
    absolute = [abs(d) for d in diffs]
    try:
        ratios = [math.exp(d) for d in diffs]
    except OverflowError as exc:
        raise ValueError("nonfinite diagnostic importance ratios") from exc
    if not all(math.isfinite(r) for r in ratios):
        raise ValueError("nonfinite diagnostic importance ratios")
    return {"token_count": len(diffs), "mean_abs_logprob_diff": sum(absolute) / len(diffs),
            "max_abs_logprob_diff": max(absolute), "mean_signed_logprob_diff": sum(diffs) / len(diffs),
            "mean_importance_ratio": sum(ratios) / len(diffs), "min_importance_ratio": min(ratios),
            "max_importance_ratio": max(ratios),
            "clip_fraction_0p8_1p28": sum(r < .8 or r > 1.28 for r in ratios) / len(ratios),
            **{name + "_abs": quantile(absolute, q) for name, q in (
                ("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("p99_5", .995), ("p100", 1.))},
            **{f"count_abs_diff_gt_{name}": sum(d > threshold for d in absolute) for name, threshold in (
                ("0p01", .01), ("0p05", .05), ("0p10", .1), ("0p20", .2))}}


def token_diagnostics(rows, dynamic, merged, tokenizer):
    if len(rows) != len(dynamic) or len(rows) != len(merged):
        raise ValueError("HF row count mismatch")
    records = []
    special = set(tokenizer.all_special_ids)
    for row, a_values, b_values in zip(rows, dynamic, merged, strict=True):
        if len(a_values) != len(row["responses"]) or len(b_values) != len(row["responses"]):
            raise ValueError("HF sampled-token count mismatch")
        for pos, (token, mask, a, b, c) in enumerate(zip(row["responses"], row["response_mask"],
                                                        a_values, b_values, row["old_log_probs"], strict=True)):
            if mask != 1:
                continue
            if not all(math.isfinite(v) for v in (a, b, c)):
                raise ValueError("nonfinite HF/rollout token logprob")
            value = {**METADATA, "global_trainable_index": len(records), "rollout_index": row["rollout_index"],
                "step_index": row["step_index"], "response_token_position": pos, "token_id": token,
                "token_repr": tokenizer.convert_ids_to_tokens(token),
                "decoded_token": tokenizer.decode([token], skip_special_tokens=False),
                "is_special_token": token in special, "vllm_old_logprob": c,
                "dynamic_peft_logprob": a, "merged_hf_logprob": b}
            for name, (first, second) in PAIRS.items():
                diff = value[first] - value[second]
                prefix = TOKEN_PAIR_NAMES[name]
                value[prefix + "_diff"] = diff
                try:
                    value[prefix + "_ratio"] = math.exp(diff)
                except OverflowError as exc:
                    raise ValueError("nonfinite token importance ratio") from exc
            records.append(value)
    if not records:
        raise ValueError("no trainable diagnostic tokens")
    return records


def summarize_tokens(tokens, *, top_n, original_alignment):
    if type(top_n) is not int or top_n < 1:
        raise ValueError("positive top_n required")
    metrics = {name: pair_metrics([r[a] for r in tokens], [r[b] for r in tokens]) for name, (a, b) in PAIRS.items()}
    per_row = []
    for rollout, step in sorted({(r["rollout_index"], r["step_index"]) for r in tokens}):
        selected = [r for r in tokens if (r["rollout_index"], r["step_index"]) == (rollout, step)]
        per_row.append({"rollout_index": rollout, "step_index": step,
            **{name: pair_metrics([r[a] for r in selected], [r[b] for r in selected]) for name, (a, b) in PAIRS.items()}})
    current = metrics["dynamic_peft_vs_rollout"]
    deltas = {"delta_" + name: current[field] - original_alignment[old_field] for name, field, old_field in (
        ("mean_abs", "mean_abs_logprob_diff", "mean_abs_logprob_diff"),
        ("max_abs", "max_abs_logprob_diff", "max_abs_logprob_diff"),
        ("mean_signed", "mean_signed_logprob_diff", "mean_signed_logprob_diff"),
        ("clip_fraction", "clip_fraction_0p8_1p28", "initial_clip_fraction"))}
    return {**metrics, "per_row": per_row, "fsdp_proxy_metric_deltas": deltas,
            "top_outliers": {name: sorted(tokens, key=lambda r: (-abs(r[TOKEN_PAIR_NAMES[name] + "_diff"]),
                                   r["global_trainable_index"]))[:top_n] for name in PAIRS}}


def publish_diagnostic(destination, summary, tokens, rows, checksums):
    """Fresh diagnostic directory only; no Gate report/manifest publication."""
    if not destination.is_dir() or any(destination.iterdir()):
        raise FileExistsError("diagnostic output must be fresh and empty; overwrite forbidden")
    atomic_json(destination / "source_checksums.json", {**METADATA, **checksums})
    atomic_json(destination / "row_diagnostics.json", {**METADATA, "rows": rows})
    # Exclusive open: never overwrite an earlier token investigation.
    with (destination / "token_diagnostics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for token in tokens:
            stream.write(json.dumps(token, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(destination / "summary.json", {**METADATA, **summary})  # successful summary LAST


def run_diagnostic(args, root):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output, reports = diagnostic_paths(root, args.run_id)
    destination = reports / "policy_handoff_diagnostic"
    if not destination.resolve().is_relative_to((root / "reports/rl_gate_c").absolute()):
        raise ValueError("diagnostic reports path escapes its permitted report tree")
    if destination.exists():
        raise FileExistsError("existing diagnostic is protected; archive it explicitly before another invocation")
    # Hash protected originals BEFORE validation/forward, including every exact
    # model and tensor file. Never create a lock/marker in outputs/... .
    adapter = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    trees = [output, args.base_model_path, adapter]
    files = [reports / "gate_c_report.json", adapter.parent / "metadata.json",
             root / "configs/rl_main.yaml", root / "configs/sft_main_imageid_v3.yaml"]
    before = source_checksums(trees, files)
    destination.mkdir(parents=True, exist_ok=False)
    stage, after = "artifact_validation", None
    try:
        ctx = validate_attempt(root, args.run_id, args.base_model_path)
        import torch
        if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("diagnostic requires exactly one visible CUDA BF16 GPU, no distributed launch")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "verl")}
        historical = ctx["identity"]["software_versions"]
        if versions != {name: historical[name] for name in versions} or versions["verl"] != "0.6.1":
            raise ValueError("diagnostic software differs from historical pinned forward environment")
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_model_path), local_files_only=True,
                                                  revision=BASE_REVISION, trust_remote_code=False)
        outputs, memory, destruction = {}, {}, {}
        for kind in ("dynamic", "merged"):
            stage = kind + "_hf_forward"
            torch.cuda.reset_peak_memory_stats(0)
            model = load_diagnostic_model(kind, ctx, args.base_model_path, device)
            reference = weakref.ref(model)
            try:
                outputs[kind] = forward_rows(model, ctx["rows"], output / "group", device=device)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            destruction[kind + "_model_destroyed"] = reference() is None
            if reference() is not None:
                raise RuntimeError(kind + " HF model not destroyed; refusing overlapping model residency")
            memory[kind] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        stage = "token_analysis"
        tokens = token_diagnostics(ctx["rows"], outputs["dynamic"], outputs["merged"], tokenizer)
        if len(tokens) != ctx["alignment"]["masked_token_count"]:
            raise ValueError("diagnostic token selection differs from Gate alignment")
        measured = summarize_tokens(tokens, top_n=args.top_n, original_alignment=ctx["alignment"])
        after = source_checksums(trees, files)
        assert_sources_unchanged(before, after)
        summary = {"execution_succeeded": True, "gate_run_id": args.run_id, "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION, "formal_sft_adapter_fingerprint": ctx["identity"]["source_sft_actor"]["source_sft_adapter_fingerprint"],
            "merged_checkpoint_fingerprint": ctx["group"]["merged_checkpoint_fingerprint"],
            "collection_attempt": ctx["group"]["identity"]["collection_attempt"], "token_count": len(tokens),
            "temperature": .7, "top_p": 1., "top_k": -1, "software_versions": versions,
            "original_gate_alignment": ctx["alignment"], "original_fsdp_alignment": ctx["alignment"],
            **measured, **destruction, "peak_cuda_memory": memory, "source_artifacts_unchanged": True,
            "evidence_pattern": "Measurements only; no automatic root-cause classification or Gate decision."}
        stage = "diagnostic_publication"
        publish_diagnostic(destination, summary, tokens, measured["per_row"],
                           {"before": before, "after": after, "source_artifacts_unchanged": True})
        print(f"TOKENS: {len(tokens)}")
        for name in PAIRS:
            m = measured[name]
            print(f"{name}: mean_abs={m['mean_abs_logprob_diff']:.9g} max_abs={m['max_abs_logprob_diff']:.9g} "
                  f"clip_fraction={m['clip_fraction_0p8_1p28']:.9g}")
        old = ctx["alignment"]
        print(f"Original FSDP vs rollout: mean_abs={old['mean_abs_logprob_diff']:.9g} "
              f"max_abs={old['max_abs_logprob_diff']:.9g} clip_fraction={old['initial_clip_fraction']:.9g}")
        print(f"REPORT: {destination / 'summary.json'}")
        return 0
    except BaseException as exc:
        # Diagnostic failure only; never touch the old Gate's failure report.
        try:
            after = source_checksums(trees, files)
            assert_sources_unchanged(before, after)
            unchanged = True
        except BaseException as mutation:
            unchanged = False
            exc = RuntimeError(f"{exc}; source integrity verification failed: {mutation}")
        atomic_json(destination / "summary.json", {**METADATA, "execution_succeeded": False,
            "gate_run_id": args.run_id, "stage": stage, "error": str(exc),
            "source_artifacts_unchanged": unchanged})
        print(f"Diagnostic FAIL at {stage}: {exc}", flush=True)
        raise exc
