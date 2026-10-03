"""Read-only, serial BF16 FA2/SDPA forensic experiment, never a Gate/update.

Inputs, masks, M-RoPE and logprob arithmetic belong to the historical forward
helper. This module changes only the fresh-load attention backend for A3/B3.
"""
from __future__ import annotations

import gc
import importlib.metadata
import itertools
import json
import math
import os
import weakref

from opensearch_vl_repro.rl import policy_handoff_diagnostic as handoff
from opensearch_vl_repro.rl import merge_precision_diagnostic as precision
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json

VERSION = "bf16-sdpa-backend-forensic-v1"
META = dict(diagnostic_version=VERSION, diagnostic_only=True,
            formal_rl_initialization_allowed=False, not_for_training=True, not_for_rollout=True)
VARIANTS = dict(A0="dynamic_bf16_fa2", B0="merged_bf16_fa2",
                A3="dynamic_bf16_sdpa", B3="merged_bf16_sdpa")
CORE_PAIRS = {"A0_vs_B0": ("A0", "B0"), "A3_vs_B3": ("A3", "B3"),
              "A0_vs_A3": ("A0", "A3"), "B0_vs_B3": ("B0", "B3")}
PAIRS = {**CORE_PAIRS, "A0_vs_B3": ("A0", "B3"), "A3_vs_B0": ("A3", "B0")}
TOKEN_KEYS = ("global_trainable_index", "rollout_index", "step_index",
              "response_token_position", "token_id")
TEXT_KEYS = ("token_repr", "decoded_token", "is_special_token")
HISTORY_DIRS = ("policy_handoff_diagnostic", "merge_precision_diagnostic",
                "fp32_merged_forward_diagnostic_fa2_primary", "fp32_merged_forward_diagnostic")
PACKAGES = ("torch", "transformers", "peft", "verl")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_lineage(summary, ctx, *, version, count):
    expected = {**META, "diagnostic_version": version, "execution_succeeded": True,
        "source_artifacts_unchanged": True, "gate_run_id": ctx["identity"]["run_id"],
        "collection_attempt": ctx["group"]["identity"]["collection_attempt"],
        "formal_sft_adapter_fingerprint": ctx["group"]["identity"]["pre_update_policy_fingerprint"],
        "production_merged_checkpoint_fingerprint": ctx["group"]["merged_checkpoint_fingerprint"],
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "temperature": .7,
        "software_versions": {k: ctx["identity"]["software_versions"][k] for k in PACKAGES},
        "token_count": count}
    # Older reports do not carry the new not_for_* flags; their three common
    # diagnostic/execution flags, lineage and numerical semantics are mandatory.
    for key in expected.keys() - {"not_for_training", "not_for_rollout"}:
        if summary.get(key) != expected[key]:
            raise ValueError(f"historical {version} {key} mismatch")


def load_history(ctx):
    """Bind full token order, not just summary counts, for all four archives."""
    prior, prior_metrics = precision.load_previous_forensic(ctx)
    old_summary, old_tokens = precision.load_previous_precision(ctx, prior)
    history = {"v452": {"tokens": prior, "metrics": prior_metrics},
               "v453": {"tokens": old_tokens, "summary": old_summary}}
    forensic_summary = _read(ctx["reports"] / HISTORY_DIRS[0] / "summary.json")
    for key, expected in (("base_model", BASE_MODEL), ("base_revision", BASE_REVISION),
                          ("temperature", .7), ("software_versions", old_summary["software_versions"])):
        if forensic_summary.get(key) != expected:
            raise ValueError(f"v4.5.2 {key} conflicts with historical identity")
    if forensic_summary.get("diagnostic_version") != handoff.DIAGNOSTIC_VERSION:
        raise ValueError("v4.5.2 diagnostic version mismatch")
    for label, directory in zip(("v454_primary", "v454_secondary"), HISTORY_DIRS[2:], strict=True):
        path = ctx["reports"] / directory
        summary = _read(path / "summary.json")
        _validate_lineage(summary, ctx, version=precision.FP32_META["diagnostic_version"], count=len(prior))
        tokens = [json.loads(line) for line in (path / "token_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(tokens) != len(prior):
            raise ValueError("v4.5.4 token count mismatch")
        for old, token in zip(prior, tokens, strict=True):
            if (any(token.get(k) != old[k] for k in TOKEN_KEYS)
                    or token.get("formal_rl_initialization_allowed") is not False
                    or token.get("diagnostic_only") is not True
                    or token.get("diagnostic_version") != precision.FP32_META["diagnostic_version"]):
                raise ValueError("v4.5.4 token IDs/order/diagnostic provenance mismatch")
        for name, (a, b) in {**precision.PAIRS, **precision.FP32_PAIRS}.items():
            values = [(r.get(a), r.get(b)) for r in tokens]
            available = all(a is not None and b is not None for a, b in values)
            if not available and name in precision.PAIRS:
                raise ValueError("v4.5.4 BF16 token evidence missing")
            if any(v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v))
                   for pair in values for v in pair):
                raise ValueError("nonfinite historical logprob")
            measured = handoff.pair_metrics([v[0] for v in values], [v[1] for v in values]) if available else None
            if summary["comparisons"].get(name) != measured:
                raise ValueError("v4.5.4 metrics disagree with token evidence")
        for alias, original in precision.BF16_PAIR_ALIASES.items():
            if summary["comparisons"].get(alias) != summary["comparisons"][original]:
                raise ValueError("v4.5.4 comparison alias mismatch")
        secondary = summary.get("secondary")
        if label == "v454_primary" and secondary is not None:
            raise ValueError("primary archive must not contain secondary execution")
        if label == "v454_secondary":
            if (not isinstance(secondary, dict) or secondary.get("attention_backend") != "sdpa"
                    or any(secondary.get(k) is not True for k in
                           ("secondary_backend_changed", "exploratory_only", "excluded_from_primary_metrics"))
                    or any(secondary.get("variants", {}).get(k, {}).get("supported") is not True for k in ("A2", "B2"))):
                raise ValueError("historical secondary SDPA evidence incomplete")
            metric = secondary.get("comparisons", {}).get("dynamic_fp32_vs_merged_fp32_sdpa")
            template = handoff.pair_metrics([0.] * len(prior), [0.] * len(prior))
            if (not isinstance(metric, dict) or metric.keys() != template.keys()
                    or metric.get("token_count") != len(prior)
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in metric.values())):
                raise ValueError("invalid secondary aggregate metrics")
            # v4.5.4 did not persist secondary per-token FP32 values: validate
            # its aggregate and provenance only, never claim independent reproduction.
        history[label] = {"tokens": tokens, "summary": summary}
    return history


def load_bf16_sdpa(kind, ctx, base_path, device):
    """Fresh local HF load; default PEFT inference dtype behavior is preserved."""
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from opensearch_vl_repro.rl.rollout_sync import require_plain_merged_model

    if kind not in ("A3", "B3"):
        raise ValueError("SDPA loader accepts only A3/B3")
    path = base_path if kind == "A3" else ctx["merged"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(path), revision=BASE_REVISION,
        dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False,
        low_cpu_mem_usage=True, local_files_only=True).to(device)
    if kind == "A3":
        model = PeftModel.from_pretrained(model, str(ctx["adapter"]), is_trainable=False, local_files_only=True)
        if not isinstance(model, PeftModel) or not any("lora_" in n for n, _ in model.named_parameters()):
            raise ValueError("A3 requires actual dynamic PEFT LoRA")
    else:
        require_plain_merged_model(model, PeftModel)
    model.eval().requires_grad_(False)
    validate_model(model, backend="sdpa", dynamic=kind == "A3")
    return model


def validate_model(model, *, backend, dynamic):
    import torch
    counts = precision.dtype_counts(model, lora=False)["tensor_count_by_dtype"]
    if not counts or counts.keys() != {str(torch.bfloat16)}:
        raise ValueError("non-LoRA model parameters must remain BF16")
    audit = precision.attention_backend_audit(model)
    if audit["model"] != backend or any(r["backend"] != backend for r in audit["attention_modules"]):
        raise ValueError("effective attention backend mismatch")
    if dynamic != any("lora_" in n for n, _ in model.named_parameters()):
        raise ValueError("dynamic/static parameter identity mismatch")
    return audit


def forward_with_audit(model, rows, directory, *, device, audit):
    """Observe logits/autocast; delegate the unmodified historical forward."""
    import torch
    audit["saved_vision_dtypes"] = []
    # Reading original inputs on CPU records the saved dtype; no image processor
    # or text tokenizer is called. The forward helper validates these inputs again.
    for row in rows:
        ids, attention, vision = handoff.load_row_inputs(directory, row, device=torch.device("cpu"))
        audit["saved_vision_dtypes"].append({k: str(v.dtype) for k, v in vision.items()})
        del ids, attention, vision
    audit["actual_forward_states"] = []
    def post(module, args, kwargs, result):
        state = dict(logits_dtype=str(result.logits.dtype),
                     autocast_enabled=torch.is_autocast_enabled(device.type),
                     autocast_dtype=str(torch.get_autocast_dtype(device.type)))
        audit["actual_forward_states"].append(state)
        if device.type == "cuda" and (not state["autocast_enabled"]
                or state["autocast_dtype"] != "torch.bfloat16" or state["logits_dtype"] != "torch.bfloat16"):
            raise ValueError("BF16 logits/CUDA autocast contract violated")
    handle = model.register_forward_hook(post, with_kwargs=True)
    try:
        return precision.bf16_forward_audit(model, rows, directory, device=device, audit=audit)
    finally:
        handle.remove()


def direction_metrics(first, second):
    if not first or len(first) != len(second) or not all(math.isfinite(v) for v in first + second):
        raise ValueError("finite aligned backend shifts required")
    n = len(first)
    ac, bc = sum(first) / n, sum(second) / n
    aa, bb = [v - ac for v in first], [v - bc for v in second]
    av, bv = sum(v*v for v in aa), sum(v*v for v in bb)
    corr = max(-1., min(1., sum(a*b for a, b in zip(aa, bb, strict=True)) / math.sqrt(av*bv))) if av and bv else None
    opposite = sum((a < 0 < b) or (b < 0 < a) for a, b in zip(first, second, strict=True))
    return dict(backend_shift_correlation=corr, backend_shift_opposite_sign_fraction=opposite/n,
                backend_shift_opposite_sign_count=opposite, backend_shift_denominator=n,
                backend_shift_zero_pair_count=sum(a == 0 or b == 0 for a, b in zip(first, second, strict=True)))


def analyze_tokens(rows, prior, results, *, top_n):
    if top_n < 1 or set(results) != set(VARIANTS):
        raise ValueError("positive top_n and all four variants required")
    flat = {}
    for variant, values in results.items():
        if len(values) != len(rows) or any(len(v) != len(r["responses"]) for r, v in zip(rows, values, strict=True)):
            raise ValueError("variant row/response length mismatch")
        flat[variant] = [v[p] for row, v in zip(rows, values, strict=True) for p, mask in enumerate(row["response_mask"]) if mask == 1]
        if len(flat[variant]) != len(prior):
            raise ValueError("original trainable token count changed")
    comparisons = {name: handoff.pair_metrics(flat[a], flat[b]) for name, (a, b) in PAIRS.items()}
    tokens = []
    selected = [(r, p) for r in rows for p, mask in enumerate(r["response_mask"]) if mask == 1]
    for idx, (old, (row, pos)) in enumerate(zip(prior, selected, strict=True)):
        identity = (idx, row["rollout_index"], row["step_index"], pos, row["responses"][pos])
        if tuple(old[k] for k in TOKEN_KEYS) != identity:
            raise ValueError("historical token IDs/order mismatch")
        token = {**META, **{k: old[k] for k in TOKEN_KEYS + TEXT_KEYS},
                 **{v + "_logprob": flat[v][idx] for v in VARIANTS}}
        for name, (a, b) in PAIRS.items():
            diff = flat[a][idx] - flat[b][idx]
            ratio = math.exp(diff)
            token.update({name + "_diff": diff, name + "_ratio": ratio,
                          name + "_clipped": ratio < .8 or ratio > 1.28,
                          name + "_sign": (diff > 0) - (diff < 0)})
        token["prior_A0_vs_B0_diff"] = old["dynamic_peft_logprob"] - old["merged_hf_logprob"]
        token["original_outlier_abs_diff_reduction_fraction"] = precision.reduction_fraction(
            abs(token["prior_A0_vs_B0_diff"]), abs(token["A3_vs_B3_diff"]))
        tokens.append(token)
    baseline, measured = comparisons["A0_vs_B0"], comparisons["A3_vs_B3"]
    improvement = {name: precision.reduction_fraction(baseline[key], measured[key]) for name, key in (
        ("mean_abs_reduction_fraction", "mean_abs_logprob_diff"), ("max_abs_reduction_fraction", "max_abs_logprob_diff"),
        ("clip_fraction_reduction_fraction", "clip_fraction_0p8_1p28"))}
    clips = {name: {r["global_trainable_index"] for r in tokens if r[name + "_clipped"]} for name in CORE_PAIRS}
    intersections = {" & ".join(names): len(set.intersection(*(clips[n] for n in names)))
                     for size in range(2, 5) for names in itertools.combinations(CORE_PAIRS, size)}
    outliers = {name: sorted(tokens, key=lambda r: (-abs(r[name + "_diff"]), r["global_trainable_index"]))[:top_n]
                for name in CORE_PAIRS}
    original = sorted(tokens, key=lambda r: (-abs(r["prior_A0_vs_B0_diff"]), r["global_trainable_index"]))[:top_n]
    return tokens, dict(comparisons=comparisons, backend_alignment_improvement=improvement,
        **direction_metrics([r["A0_vs_A3_diff"] for r in tokens], [r["B0_vs_B3_diff"] for r in tokens]),
        clip_sets={name: sorted(indices) for name, indices in clips.items()},
        clip_set_counts={name: len(indices) for name, indices in clips.items()}, clip_set_intersections=intersections,
        original_A0_vs_B0_outliers=original, top_outliers=outliers)


def repeat_checks(tokens, history):
    """Diagnostic repeat sanity, deliberately separate from every Gate threshold.

    Mean/max repeat drift must be <=10% of each historical A/B gap, with
    1e-4/1e-3 absolute floors. Report the limits and all deltas explicitly.
    """
    reports = {}
    for label, evidence in history.items():
        old = evidence["tokens"]
        bfield = "merged_hf_logprob" if label == "v452" else "production_merge_logprob"
        ab = handoff.pair_metrics([r["dynamic_peft_logprob"] for r in old], [r[bfield] for r in old])
        current = handoff.pair_metrics([r["A0_logprob"] for r in tokens], [r["B0_logprob"] for r in tokens])
        limits = dict(mean_abs=max(1e-4, .1*ab["mean_abs_logprob_diff"]), max_abs=max(1e-3, .1*ab["max_abs_logprob_diff"]))
        changes = {k: current[k] - ab[k] for k in current}
        repeats = {v: handoff.pair_metrics([r[v + "_logprob"] for r in tokens], [r[f] for r in old])
                   for v, f in (("A0", "dynamic_peft_logprob"), ("B0", bfield))}
        accepted = all(m["mean_abs_logprob_diff"] <= limits["mean_abs"] and m["max_abs_logprob_diff"] <= limits["max_abs"]
                       for m in repeats.values())
        reports[label] = dict(pair_metric_deltas=changes, per_variant_repeat_deltas=repeats,
                              diagnostic_repeat_limits=limits, interpretation_allowed=accepted)
    return reports


def numeric_environment(torch):
    """Observe only: no TF32 flag or SDPA kernel selection is changed."""
    return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                sdpa_flash_enabled=torch.backends.cuda.flash_sdp_enabled(),
                sdpa_math_enabled=torch.backends.cuda.math_sdp_enabled(),
                sdpa_mem_efficient_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
                sdpa_cudnn_enabled=torch.backends.cuda.cudnn_sdp_enabled())


def run_diagnostic(args, root):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output, reports = handoff.diagnostic_paths(root, args.run_id)
    destination = reports / "bf16_sdpa_backend_diagnostic"
    if not destination.resolve().is_relative_to((root / "reports/rl_gate_c").resolve()):
        raise ValueError("report destination escapes permitted tree")
    if destination.exists():
        raise FileExistsError("existing BF16 SDPA diagnostic protected; overwrite forbidden")
    if not args.local_files_only or args.top_n < 1:
        raise ValueError("local-files-only and positive top_n required")
    adapter = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    trees = [output, args.base_model_path, adapter, root / "configs", *(reports / d for d in HISTORY_DIRS)]
    files = [reports / "gate_c_report.json", adapter.parent / "metadata.json"]
    before = handoff.source_checksums(trees, files)
    destination.mkdir(parents=True, exist_ok=False)
    stage, analysis, audits, peaks, destroyed = "artifact_validation", {}, {}, {}, {}
    try:
        ctx = handoff.validate_attempt(root, args.run_id, args.base_model_path)
        history = load_history(ctx)
        if ctx["sft"]["model"]["attn_implementation"] != "flash_attention_2":
            raise ValueError("historical formal backend must be FA2")
        import torch
        if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("requires exactly one visible BF16 CUDA GPU, no distributed launch")
        versions = {k: importlib.metadata.version(k) for k in PACKAGES}
        if versions != {k: ctx["identity"]["software_versions"][k] for k in PACKAGES} or versions["verl"] != "0.6.1":
            raise ValueError("diagnostic software differs from historical environment")
        torch.cuda.set_device(0)
        device, results = torch.device("cuda", 0), {}
        numeric_flags = numeric_environment(torch)
        for kind in VARIANTS:
            stage = kind + "_load_forward"
            torch.cuda.reset_peak_memory_stats(0)
            model, ref = None, None
            audits[kind] = {}
            try:
                model = (handoff.load_diagnostic_model("dynamic" if kind == "A0" else "merged", ctx, args.base_model_path, device)
                         if kind in ("A0", "B0") else load_bf16_sdpa(kind, ctx, args.base_model_path, device))
                ref = weakref.ref(model)
                validate_model(model, backend="flash_attention_2" if kind in ("A0", "B0") else "sdpa", dynamic=kind in ("A0", "A3"))
                audits[kind]["base_parameter_dtypes"] = precision.dtype_counts(model, lora=False)
                audits[kind]["lora_parameter_dtypes"] = precision.dtype_counts(model, lora=True)
                if kind in ("A0", "A3"):
                    audits[kind]["lora_targets"] = precision.lora_dtype_audit(model)
                results[kind] = forward_with_audit(model, ctx["rows"], output / "group", device=device, audit=audits[kind])
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                peaks[kind] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(0), peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
            destroyed[kind] = ref is None or ref() is None
            if not destroyed[kind]:
                raise RuntimeError(f"{kind} model still resident before next variant")
        if (audits["A0"]["lora_parameter_dtypes"] != audits["A3"]["lora_parameter_dtypes"]
                or audits["A0"]["lora_targets"] != audits["A3"]["lora_targets"]):
            raise ValueError("A3 changed original PEFT LoRA dtype/target semantics")
        if numeric_environment(torch) != numeric_flags:
            raise ValueError("global numerical/backend flags changed during diagnostic")
        stage = "token_analysis"
        tokens, analysis = analyze_tokens(ctx["rows"], history["v452"]["tokens"], results, top_n=args.top_n)
        if len(tokens) != ctx["alignment"]["masked_token_count"]:
            raise ValueError("historical alignment token count mismatch")
        analysis["repeat_delta_vs_history"] = repeat_checks(tokens, history)
        if not all(r["interpretation_allowed"] for r in analysis["repeat_delta_vs_history"].values()):
            raise ValueError("A0/B0 repeat differs materially from historical baselines; diagnostic interpretation refused")
        stage = "report_publication"
        after = handoff.source_checksums(trees, files)
        handoff.assert_sources_unchanged(before, after)
        summary = {**META, **analysis, "execution_succeeded": True, "gate_run_id": args.run_id,
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "temperature": .7,
            "collection_attempt": ctx["group"]["identity"]["collection_attempt"],
            "formal_sft_adapter_fingerprint": ctx["group"]["identity"]["pre_update_policy_fingerprint"],
            "production_merged_checkpoint_fingerprint": ctx["group"]["merged_checkpoint_fingerprint"],
            "token_count": len(tokens), "software_versions": versions, "variants": VARIANTS,
            "model_sources": {"A0": str(args.base_model_path), "A3": str(args.base_model_path),
                              "B0": str(ctx["merged"]), "B3": str(ctx["merged"]), "adapter": str(ctx["adapter"])},
            "models_destroyed": destroyed, "peak_gpu_memory": peaks, "numeric_environment": numeric_flags,
            "model_dtypes": audits, "attention_backend_audits": {k: v["resolved_attention"] for k, v in audits.items()},
            "source_artifacts_unchanged": True,
            "historical_secondary_validation_scope": "aggregate/provenance only; secondary token values were not persisted",
            "interpretation": "Measurements only; backend-specific evidence, not root-cause confirmation or Gate PASS."}
        atomic_json(destination / "source_checksums.json", {**META, "before": before, "after": after})
        atomic_json(destination / "clip_sets.json", {**META, "sets": analysis["clip_sets"], "intersections": analysis["clip_set_intersections"]})
        with (destination / "token_diagnostics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for token in tokens:
                stream.write(json.dumps(token, ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Successful summary is the final artifact; no Gate PASS artifact exists.
        atomic_json(destination / "summary.json", summary)
        print(f"TOKENS: {len(tokens)}")
        for name in CORE_PAIRS:
            m = analysis["comparisons"][name]
            print(f"{name}: mean_abs={m['mean_abs_logprob_diff']:.9g} max_abs={m['max_abs_logprob_diff']:.9g} clip={m['clip_fraction_0p8_1p28']:.9g}")
        improvement = analysis["backend_alignment_improvement"]
        print("BACKEND ALIGNMENT IMPROVEMENT: "
              f"mean_abs_reduction={improvement['mean_abs_reduction_fraction']} "
              f"max_abs_reduction={improvement['max_abs_reduction_fraction']} "
              f"clip_reduction={improvement['clip_fraction_reduction_fraction']}")
        print("corr(dynamic_backend_shift, merged_backend_shift)=" + str(analysis["backend_shift_correlation"]))
        print("opposite_sign_fraction=" + str(analysis["backend_shift_opposite_sign_fraction"]))
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
        atomic_json(destination / "source_checksums.json", {**META, "before": before, "after": after,
                                                           "source_artifacts_unchanged": unchanged})
        atomic_json(destination / "summary.json", {**META, "execution_succeeded": False, "gate_run_id": args.run_id,
            "stage": stage, "error": str(exc), "source_artifacts_unchanged": unchanged,
            "partial_analysis": analysis, "model_dtypes": audits, "peak_gpu_memory": peaks, "models_destroyed": destroyed})
        raise exc
