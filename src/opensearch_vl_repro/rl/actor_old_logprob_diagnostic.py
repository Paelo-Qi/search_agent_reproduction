"""v4.5.7 two independent PRE-UPDATE real FSDP2 actor computations.

Diagnostic-only. No loss, backward, optimizer step, rollout, export or Gate.
All source artifacts are read-only; only a new report directory is written.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path

from opensearch_vl_repro.rl import bf16_lora_dtype_diagnostic as lora
from opensearch_vl_repro.rl import bf16_sdpa_diagnostic as backend
from opensearch_vl_repro.rl import policy_handoff_diagnostic as handoff
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json, local_tensor, load_gate_config
from opensearch_vl_repro.rl.policy_alignment import compare_policy_logprobs
from opensearch_vl_repro.rl.training_batch import build_dataproto
from opensearch_vl_repro.rl.verl_actor_gate import CollectiveStages, construct_actor
from opensearch_vl_repro.rl.verl_policy_update import configure_one_update
from opensearch_vl_repro.rl.verl_old_logprob_semantics import inspect_verl_old_logprob_semantics
from opensearch_vl_repro.sft_tool_audit import sha256_file

VERSION = "actor-old-logprob-recompute-feasibility-v1"
META = {**backend.META, "diagnostic_version": VERSION, "optimizer_step_count": 0}
HISTORY_DIRS = (*lora.HISTORY_DIRS, "bf16_lora_dtype_diagnostic")
VARIANTS = dict(R="saved rollout/vLLM processed_logprobs (formal FP32 carrier)",
                O="first independent unchanged FSDP2 actor compute_log_prob",
                C="second independent unchanged FSDP2 actor compute_log_prob")
PAIRS = {"rollout_vs_actor_recomputed_old": ("O", "R"),
         "actor_recomputed_old_vs_actor_current": ("C", "O"),
         "rollout_vs_actor_current": ("C", "R")}


def recursive_fingerprint(value):
    """Typed stable recursive SHA256, including tensors in object arrays."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            digest.update(b"tensor:" + lora.tensor_fingerprint(v).encode())
        elif isinstance(v, np.ndarray):
            digest.update(json.dumps(["array", str(v.dtype), list(v.shape)]).encode())
            for item in v.flat:
                visit(item)
        elif isinstance(v, dict):
            digest.update(b"dict:")
            for key in sorted(v):
                visit(key)
                visit(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(type(v).__name__.encode() + str(len(v)).encode())
            for item in v:
                visit(item)
        elif isinstance(v, np.generic):
            visit(v.item())
        elif v is None or type(v) in (str, bool, int, float):
            digest.update(json.dumps([type(v).__name__, v], allow_nan=False).encode())
        else:
            raise TypeError(f"unsupported fingerprint value {type(v)}")
    visit(value)
    return digest.hexdigest()


def input_fingerprint(data):
    fields = {k: recursive_fingerprint(data.batch[k]) for k in sorted(data.batch.keys())}
    non_tensors = recursive_fingerprint(data.non_tensor_batch)
    metadata = recursive_fingerprint(data.meta_info)
    return dict(tensor_fields=fields, non_tensor_batch=non_tensors, metadata=metadata,
                sha256=recursive_fingerprint([fields, non_tensors, metadata]))


def actor_fingerprint(model):
    """All actual local FSDP shards, not a weak sample or full-state export."""
    roster = []
    for name, parameter in sorted(model.named_parameters()):
        shard = local_tensor(parameter)
        roster.append(dict(name=name, global_shape=list(parameter.shape), local_shape=list(shard.shape),
            dtype=str(shard.dtype), requires_grad=parameter.requires_grad,
            placements=str(getattr(parameter, "placements", None)), sha256=lora.tensor_fingerprint(shard)))
    if not roster:
        raise ValueError("empty actor parameters")
    return dict(parameter_count=len(roster), sha256=recursive_fingerprint(roster), local_shards=roster)


def rng_fingerprint():
    import numpy as np
    import torch
    return recursive_fingerprint(dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))


def require_no_gradients(model):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("unexpected actor gradients in forward-only diagnostic")


def independent_compute(actor, data, label, *, expected_input, expected_parameters):
    """A fresh DataProto and an observed real model forward for EACH call."""
    import torch
    batch = copy.deepcopy(data)
    before = input_fingerprint(batch)
    if before != expected_input:
        raise ValueError("O/C input fingerprint differs from original DataProto")
    pre = actor_fingerprint(actor.actor_module)
    if pre != expected_parameters:
        raise ValueError("actor parameter mutation before independent compute")
    require_no_gradients(actor.actor_module)
    rng_before = rng_fingerprint()
    forwards = []
    def pre_hook(module, args):
        dropouts = [(name, m.training) for name, m in module.named_modules()
                    if isinstance(m, torch.nn.modules.dropout._DropoutNd)]
        active = torch.is_autocast_enabled("cuda") if torch.cuda.is_available() else torch.is_autocast_enabled("cpu")
        record = dict(model_training=module.training, gradients_enabled=torch.is_grad_enabled(),
            autocast_enabled=active, autocast_dtype=str(torch.get_autocast_dtype("cuda" if torch.cuda.is_available() else "cpu")),
            dropout_training=dropouts)
        forwards.append(record)
        if module.training or torch.is_grad_enabled() or any(training for _, training in dropouts):
            raise ValueError("compute_log_prob forward is not deterministic eval/no_grad/dropout-disabled")
        if any(p.device.type == "cuda" for p in module.parameters()) and (
                not active or record["autocast_dtype"] != "torch.bfloat16"):
            raise ValueError("actual CUDA actor forward did not use BF16 autocast")
    hook = actor.actor_module.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():  # EXACT outer context of formal alignment helper
            result, _ = actor.compute_log_prob(batch, calculate_entropy=False)
        if not forwards:
            raise ValueError("compute_log_prob did not execute an actual actor module forward")
        if result.requires_grad or result.shape != data.batch["responses"].shape:
            raise ValueError("logprob output shape/gradient contract mismatch")
        post_input, post = input_fingerprint(batch), actor_fingerprint(actor.actor_module)
        require_no_gradients(actor.actor_module)
        rng_after = rng_fingerprint()
        if post_input != before or input_fingerprint(data) != expected_input:
            raise ValueError("compute_log_prob mutated DataProto")
        if post != pre:
            raise ValueError("actor parameter mutation during compute_log_prob")
        if rng_after != rng_before:
            raise ValueError("RNG changed during deterministic logprob computation")
        if actor.actor_optimizer.state:
            raise ValueError("optimizer state unexpectedly populated without an allowed step")
        return result.detach().cpu(), dict(label=label, input_before=before, input_after=post_input,
            parameters_before=pre, parameters_after=post, rng_before=rng_before, rng_after=rng_after,
            forward_count=len(forwards), forward_states=forwards, independent_dataproto_clone=True)
    finally:
        hook.remove()


def audit_batch(data, rows, count):
    import torch
    if data.meta_info != {"temperature": .7, "micro_batch_size": 1, "use_dynamic_bsz": False}:
        raise ValueError("formal temperature/microbatch metadata mismatch")
    tensors = data.batch
    carrier = rollout_carrier(data)
    if set(tensors.keys()) != {"input_ids", "attention_mask", "position_ids", "responses", "response_mask", carrier, "advantages"}:
        raise ValueError("formal DataProto tensor fields mismatch")
    if tensors["position_ids"].shape != (len(rows), 3, tensors["input_ids"].shape[-1]):
        raise ValueError("real M-RoPE shape mismatch")
    mask = tensors["response_mask"]
    if mask.shape != tensors["responses"].shape or int(mask.sum().item()) != count:
        raise ValueError("historical response mask/count changed")
    for i, row in enumerate(rows):
        n = len(row["responses"])
        if (tensors["responses"][i, :n].tolist() != row["responses"]
                or mask[i].tolist() != row["response_mask"] + [0] * (mask.shape[1]-n)
                or not torch.equal(tensors[carrier][i, :n].cpu(), torch.tensor(row["old_log_probs"], dtype=torch.float32))):
            raise ValueError("R, token IDs or fatal response mask differ from saved rollout")
    multimodal = data.non_tensor_batch.get("multi_modal_inputs")
    if multimodal is None or len(multimodal) != len(rows) or any(
        set(v) != {"pixel_values", "image_grid_thw"} or not v["pixel_values"].numel() for v in multimodal):
        raise ValueError("multimodal inputs missing")
    return dict(token_identity_verified=True, multimodal_inputs_verified=True, temperature=.7,
                mrope_shape=list(tensors["position_ids"].shape), response_mask_reused=True)


def rollout_carrier(data):
    # Read-only historical compatibility. The modern builder names saved R
    # rollout_log_probs; historical fixtures/artifacts called R old_log_probs.
    return "rollout_log_probs" if "rollout_log_probs" in data.batch else "old_log_probs"


def feasibility(metrics, historical):
    # Diagnostic-specific scale check, NEVER the formal Gate max_abs<.1 rule.
    limit = min(1e-3, historical["max_abs_logprob_diff"] * .01)
    ok = (metrics["token_count"] == historical["masked_token_count"]
          and metrics["clip_fraction_0p8_1p28"] == 0
          and metrics["max_abs_logprob_diff"] <= limit)
    return dict(actor_recompute_numerically_feasible=ok, diagnostic_max_abs_limit=limit,
                rule="finite/count equal, clip=0, max_abs<=min(1e-3,0.01*historical_max); not a Gate threshold")


def analyze_tokens(rows, prior, results, mask, *, top_n):
    import torch
    if type(top_n) is not int or top_n < 1 or set(results) != set(VARIANTS):
        raise ValueError("positive top_n and three independent R/O/C results required")
    chosen = mask.cpu().bool()
    flat = {}
    for name, value in results.items():
        if value.shape != mask.shape or not bool(torch.isfinite(value[chosen]).all()):
            raise ValueError("finite aligned masked logprobs required")
        flat[name] = value[chosen].double().tolist()
    positions = [(r, p) for r in rows for p, m in enumerate(r["response_mask"]) if m == 1]
    if len(prior) != len(positions) or len(flat["R"]) != len(prior):
        raise ValueError("historical token count mismatch")
    tokens = []
    for index, (old, (row, pos)) in enumerate(zip(prior, positions, strict=True)):
        identity = (index, row["rollout_index"], row["step_index"], pos, row["responses"][pos])
        if tuple(old.get(k) for k in backend.TOKEN_KEYS) != identity:
            raise ValueError("historical token IDs/order mismatch")
        record = {**META, **{k: old[k] for k in backend.TOKEN_KEYS + backend.TEXT_KEYS},
                  **{v + "_logprob": flat[v][index] for v in VARIANTS}, "historical_R_minus_C_diff": None}
        for a, b in (("R", "O"), ("O", "C"), ("R", "C")):
            diff = flat[a][index] - flat[b][index]
            record[a + "_minus_" + b + "_diff"] = diff
            record[a + "_to_" + b + "_ratio"] = math.exp(-diff)
        tokens.append(record)
    comparisons = {name: handoff.pair_metrics(flat[current], flat[old]) for name, (current, old) in PAIRS.items()}
    oc = comparisons["actor_recomputed_old_vs_actor_current"]
    ratio = dict(formula="exp(C - O)", mean=oc["mean_importance_ratio"], min=oc["min_importance_ratio"],
                 max=oc["max_importance_ratio"], clip_fraction=oc["clip_fraction_0p8_1p28"])
    top = {a + "_vs_" + b: sorted(tokens, key=lambda t: (-abs(t[a + "_minus_" + b + "_diff"]), t["global_trainable_index"]))[:top_n]
           for a, b in (("R", "C"), ("R", "O"), ("O", "C"))}
    return tokens, dict(comparisons=comparisons, actor_side_initial_ratio=ratio, top_differences=top,
        comparison_direction={name: f"{current}-{old}; ratio=exp({current}-{old})" for name, (current, old) in PAIRS.items()},
        historical_R_C_top50=dict(available=False, reason="Original formal alignment stores aggregate/per-rank stats, not original actor per-token C; no HF substitution."))


def repeat_guard(current, historical):
    keys = ("mean_abs_logprob_diff", "max_abs_logprob_diff", "mean_signed_logprob_diff",
            "mean_importance_ratio", "min_importance_ratio", "max_importance_ratio", "initial_clip_fraction")
    checks, deltas, limits = {}, {}, {}
    for key in keys:
        deltas[key] = abs(current[key] - historical[key])
        # Repeat-only tolerances, not modified formal acceptance thresholds.
        limits[key] = max(1e-4, .1 * abs(historical[key])) if key != "initial_clip_fraction" else max(1/current["masked_token_count"], .1 * historical[key])
        checks[key] = deltas[key] <= limits[key]
    checks["masked_token_count"] = current["masked_token_count"] == historical["masked_token_count"]
    return dict(passed=all(checks.values()), checks=checks, absolute_deltas=deltas,
                diagnostic_repeat_limits=limits, original=historical, measured=current)


def load_history(ctx):
    history = lora.load_history(ctx)  # validates the first five forensic directories
    directory = ctx["reports"] / HISTORY_DIRS[-1]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    prior = history["tokens"]
    backend._validate_lineage(summary, ctx, version=lora.VERSION, count=len(prior))
    if summary.get("variants") != lora.VARIANTS or summary.get("experiment_informative") is not True:
        raise ValueError("v4.5.6 variant/experiment provenance mismatch")
    tokens = [json.loads(line) for line in (directory / "token_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(tokens) != len(prior):
        raise ValueError("v4.5.6 token count mismatch")
    for old, token in zip(prior, tokens, strict=True):
        if any(token.get(k) != old[k] for k in backend.TOKEN_KEYS + backend.TEXT_KEYS) or any(token.get(k) != v for k, v in lora.META.items()):
            raise ValueError("v4.5.6 token IDs/order/metadata mismatch")
    for name, (a, b) in lora.PAIRS.items():
        measured = handoff.pair_metrics([t[a + "_logprob"] for t in tokens], [t[b + "_logprob"] for t in tokens])
        if summary["comparisons"].get(name) != measured:
            raise ValueError("v4.5.6 per-token metrics mismatch")
    return tokens


def software_versions(identity):
    versions = {k: importlib.metadata.version(k) for k in backend.PACKAGES}
    if versions != {k: identity["software_versions"][k] for k in backend.PACKAGES} or versions["verl"] != "0.6.1":
        raise ValueError("software differs from attempt1 / verl!=0.6.1")
    return versions


def protected_sources(root, run_id, base_path):
    output, reports = handoff.diagnostic_paths(root, run_id)
    adapter = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    return ([output, base_path, adapter, root / "configs", *(reports / d for d in HISTORY_DIRS)],
            [adapter.parent / "metadata.json", reports / "gate_c_report.json"])


def reserve_reports(root, run_id, base_path):
    _, reports = handoff.diagnostic_paths(root, run_id)
    destination = reports / "actor_old_logprob_recompute_diagnostic"
    if not destination.resolve().is_relative_to((root / "reports/rl_gate_c").resolve()):
        raise ValueError("unsafe diagnostic destination")
    if destination.exists():
        raise FileExistsError("existing actor recompute diagnostic protected; overwrite forbidden")
    trees, files = protected_sources(root, run_id, base_path)
    before = handoff.source_checksums(trees, files)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination / "summary.json", {**META, "execution_succeeded": False,
        "gate_run_id": run_id, "stage": "initialized", "actor_recompute_numerically_feasible": False,
        "ppo_semantic_acceptability": "undetermined"})
    return destination, trees, files, before


def forbid_updates(actor):
    """Defense in depth on this fresh diagnostic actor ONLY; restored at exit."""
    originals = [(actor, "update_policy", actor.update_policy),
                 (actor.actor_optimizer, "step", actor.actor_optimizer.step)]
    def reject(*args, **kwargs):
        raise RuntimeError("optimizer/update forbidden in actor-old-logprob diagnostic")
    for owner, name, _ in originals:
        setattr(owner, name, reject)
    return originals


def print_summary(summary, destination):
    print("TOKENS: " + str(summary["token_count"]), flush=True)
    for name, metric in summary["comparisons"].items():
        print(name.upper() + ": " + json.dumps(metric), flush=True)
    print("INITIAL ACTOR-SIDE PPO RATIO: " + json.dumps(summary["actor_side_initial_ratio"]), flush=True)
    for key in ("actor_recompute_numerically_feasible", "historical_alignment_repeat", "ppo_semantic_acceptability", "optimizer_step_count"):
        print(key.upper() + ": " + json.dumps(summary[key]), flush=True)
    audit = summary["verl_old_logprob_semantics"]
    print("VERL_VERSION: " + audit["verl_version"], flush=True)
    print("USE_ROLLOUT_LOG_PROBS: " + json.dumps(dict(symbol_found=audit["use_rollout_log_probs_symbol_found"], runtime_gate_config=audit["runtime_gate_config"])), flush=True)
    for key in ("when_true", "when_false", "denominator_source"):
        print(key.upper() + ": " + str(audit.get(key, "undetermined")), flush=True)
    print("REPORT: " + str(destination / "summary.json"), flush=True)


def run_diagnostic(args, root):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    if not args.local_files_only or args.top_n < 1:
        raise ValueError("local-files-only and positive top_n required")
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from opensearch_vl_repro.model import load_processor
    if (int(os.environ.get("WORLD_SIZE", "1")) != 2 or not torch.cuda.is_available() or torch.cuda.device_count() != 2):
        raise RuntimeError("requires torchrun --nproc_per_node=2 and exactly two visible CUDA GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA required")
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    destination, before, trees, files = None, {}, [], []
    report, original_methods = {**META, "execution_succeeded": False, "gate_run_id": args.run_id,
        "actor_recompute_numerically_feasible": False, "ppo_semantic_acceptability": "undetermined"}, []
    installed_before = {}
    try:
        reservation = [None]
        if rank == 0:
            try:
                destination, trees, files, before = reserve_reports(root, args.run_id, args.base_model_path)
                reservation[0] = dict(destination=str(destination), error=None)
            except Exception as exc:
                reservation[0] = dict(error=f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(reservation, src=0)
        if reservation[0]["error"]:
            raise RuntimeError(reservation[0]["error"])
        destination = Path(reservation[0]["destination"])
        stages = CollectiveStages(torch, destination / "stages.json", dict(run_id=args.run_id, **META), local_rank,
                                  required_checks=(), label="actor recompute diagnostic")
        stage = "artifact_validation"
        def run(name, operation):
            nonlocal stage
            stage = name
            return stages.run(name, operation)
        ctx = run("artifact_validation", lambda: handoff.validate_attempt(root, args.run_id, args.base_model_path))
        prior = run("historical_diagnostics", lambda: load_history(ctx))
        versions = run("software_binding", lambda: software_versions(ctx["identity"]))
        seed = ctx["identity"].get("seed")
        if type(seed) is not int:
            raise ValueError("original Gate seed missing; cannot guess")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        runtime_sft = copy.deepcopy(ctx["sft"])
        runtime_sft["model"]["name_or_path"] = str(args.base_model_path.resolve())
        gate = ctx["identity"]["gate_config"]
        mesh = run("device_mesh", lambda: init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",)))
        processor = run("processor_pad_id_only", lambda: load_processor(runtime_sft, local_files_only=True))
        a22 = run("formal_actor_configuration", lambda: load_gate_config(root / "configs/rl_gate_a22.yaml"))
        actor, actor_audit = run("real_fsdp2_actor", lambda: construct_actor(config=runtime_sft, gate=a22, adapter=ctx["adapter"], mesh=mesh))
        def configure():
            configure_one_update(actor, len(ctx["rows"]), gate)
            if actor.config.strategy != "fsdp2" or actor.config.use_rollout_log_probs is not True:
                raise ValueError("formal FSDP2 runtime actor config mismatch")
            if (actor_audit["model_training"] is not True or actor_audit["language_model_training"] is not True
                or actor_audit["decoder_layers_training"] != 36 or actor_audit["decoder_layers_gradient_checkpointing"] != 36
                or actor_audit["effective_attention_implementation"] != "flash_attention_2"):
                raise ValueError("formal actor initialization contract mismatch")
        run("unchanged_formal_actor_semantics", configure)
        semantics = run("installed_verl_source_audit", lambda: inspect_verl_old_logprob_semantics(actor, version=versions["verl"]) if rank == 0 else None)
        if rank == 0:
            installed_before = semantics["source_hashes"]
            semantics["runtime_gate_denominator_origin"] = "Historical diagnostic reads saved rollout R (legacy row old_log_probs, modern rollout_log_probs carrier); true flag unchanged; NO update executed. O was not installed as denominator."
            report.update(verl_old_logprob_semantics=semantics,
                          ppo_semantic_acceptability=semantics["ppo_semantic_acceptability"])
            atomic_json(destination / "verl_source_audit.json", {**META, **semantics})
        original_methods = forbid_updates(actor)
        rows = ctx["rows"]
        # Keep saved advantages if available, WITHOUT reward/RLOO recomputation.
        def original_rows():
            path = ctx["output"] / "training_masks.json"
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                actual = saved["rows"]
                if len(actual) != len(rows) or any({k: v for k, v in a.items() if k != "advantage"} != {k: v for k, v in b.items() if k != "advantage"}
                    for a, b in zip(actual, rows, strict=True)):
                    raise ValueError("saved training rows differ from original group/mask")
                return actual
            return rows  # existing diagnostic helper's irrelevant zero placeholders
        rows = run("original_training_rows", original_rows)
        data = run("real_multimodal_dataproto", lambda: build_dataproto(rows, directory=ctx["output"] / "group",
            model=actor.actor_module, pad_id=processor.tokenizer.pad_token_id, device=torch.device("cuda", local_rank), temperature=.7))
        batch_audit = run("historical_token_vision_mask_audit", lambda: audit_batch(data, rows, len(prior)))
        inputs = run("input_fingerprint", lambda: input_fingerprint(data))
        parameters = run("pre_O_parameter_fingerprint", lambda: actor_fingerprint(actor.actor_module))
        O, audit_O = run("independent_O_compute", lambda: independent_compute(actor, data, "O", expected_input=inputs, expected_parameters=parameters))
        if rank == 0:
            report["partial_computation_audits"] = {"O": audit_O}
        C, audit_C = run("independent_C_compute", lambda: independent_compute(actor, data, "C", expected_input=inputs, expected_parameters=parameters))
        if rank == 0:
            report["partial_computation_audits"]["C"] = audit_C
        def same_execution():
            if audit_O["forward_states"] != audit_C["forward_states"]:
                raise ValueError("O/C forward mode/autocast/dropout semantics differ")
        run("O_C_execution_state_match", same_execution)
        mask, R = data.batch["response_mask"].cpu(), data.batch[rollout_carrier(data)].cpu()
        def oc_check():
            metric = handoff.pair_metrics(C[mask.bool()].double().tolist(), O[mask.bool()].double().tolist())
            decision = feasibility(metric, ctx["alignment"])
            if not decision["actor_recompute_numerically_feasible"]:
                if rank == 0:
                    report.update(decision, O_C_metrics=metric)
                raise ValueError("actor-side recompute candidate rejected: O/C non-determinism")
            return decision
        decision = run("O_C_numerical_feasibility", oc_check)
        tokens, analysis = run("R_O_C_token_analysis", lambda: analyze_tokens(rows, prior, dict(R=R, O=O, C=C), mask, top_n=args.top_n))
        measured = run("formal_R_C_projection", lambda: compare_policy_logprobs(C, R, mask,
            clip_ratio_low=gate["clip_ratio_low"], clip_ratio_high=gate["clip_ratio_high"], expected_masked_token_count=len(prior)))
        originals = {r["rank"]: r for r in ctx["alignment"]["per_rank"]}
        repeat = run("historical_R_C_repeat", lambda: repeat_guard(measured, originals[rank]))
        run("require_historical_repeat", lambda: None if repeat["passed"] else (_ for _ in ()).throw(ValueError("R/C differs significantly from original FSDP alignment; interpretation refused")))
        local = dict(rank=rank, actor_audit=actor_audit, computation_audits=dict(O=audit_O, C=audit_C),
            parameter_fingerprints=dict(pre_O=audit_O["parameters_before"], post_O=audit_O["parameters_after"], pre_C=audit_C["parameters_before"], post_C=audit_C["parameters_after"]),
            input_fingerprints=dict(O=audit_O["input_before"], C=audit_C["input_before"]), historical_repeat=repeat,
            comparisons=analysis["comparisons"], feasibility=decision)
        def gather():
            collected = [None] * world
            dist.all_gather_object(collected, local)
            return collected
        per_rank = run("gather_actor_evidence", gather)
        def publish():
            if rank != 0:
                return
            after = handoff.source_checksums(trees, files)
            handoff.assert_sources_unchanged(before, after)
            installed_after = {p: sha256_file(Path(p)) for p in installed_before}
            handoff.assert_sources_unchanged(installed_before, installed_after)
            report.update(**analysis, **decision, **batch_audit, execution_succeeded=True, actor_world_size=world,
                token_count=len(tokens), software_versions=versions, base_model=BASE_MODEL, base_revision=BASE_REVISION,
                actor_source_adapter=str(ctx["adapter"]), actor_source_fingerprint=ctx["group"]["identity"]["pre_update_policy_fingerprint"],
                rollout_logprob_source=str(ctx["output"] / "group"), variants=VARIANTS, per_rank=per_rank,
                parameter_fingerprints={str(r["rank"]): r["parameter_fingerprints"] for r in per_rank},
                input_fingerprints={str(r["rank"]): r["input_fingerprints"] for r in per_rank},
                historical_alignment_repeat=dict(passed=all(r["historical_repeat"]["passed"] for r in per_rank), per_rank=[r["historical_repeat"] for r in per_rank]),
                verl_old_logprob_semantics=semantics, ppo_semantic_acceptability=semantics["ppo_semantic_acceptability"],
                source_artifacts_unchanged=True, installed_verl_sources_unchanged=True, numeric_environment=backend.numeric_environment(torch),
                interpretation=("Numeric feasibility confirmed. " +
                    ("Candidate fix cannot be justified by the assumed verl false-flag semantics. "
                     if semantics["ppo_semantic_acceptability"] == "not_supported_by_verl_implementation" else
                     "Semantic repair decision pending. ") +
                    "No formal repair/initialization authorized. Trajectories were generated by static merged vLLM; O/C are dynamic training actor; no PPO denominator was changed."))
            atomic_json(destination / "source_checksums.json", {**META, "before": before, "after": after, "installed_verl_before": installed_before, "installed_verl_after": installed_after})
            atomic_json(destination / "token_diagnostics.json", {**META, "tokens": tokens})
            # Same JSONL convention as the six read-only histories, new dir only.
            with (destination / "token_diagnostics.jsonl").open("x", encoding="utf-8") as stream:
                for token in tokens:
                    stream.write(json.dumps(token, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Print before the final success artifact; publication failure stays false.
            print_summary(report, destination)
            atomic_json(destination / "summary.json", report)
        run("report_publication", publish)
        return 0
    except BaseException as exc:
        if rank == 0 and destination is not None:
            unchanged, error, after = False, str(exc), None
            try:
                after = handoff.source_checksums(trees, files)
                handoff.assert_sources_unchanged(before, after)
                handoff.assert_sources_unchanged(installed_before, {p: sha256_file(Path(p)) for p in installed_before})
                unchanged = True
            except BaseException as integrity:
                error += "; source integrity failure: " + str(integrity)
            atomic_json(destination / "source_checksums.json", {**META, "before": before, "after": after, "source_artifacts_unchanged": unchanged})
            report.update(execution_succeeded=False, stage=locals().get("stage", "reservation"), error=error,
                          source_artifacts_unchanged=unchanged)
            atomic_json(destination / "summary.json", report)
        raise
    finally:
        for owner, name, original in original_methods:
            setattr(owner, name, original)
        try:
            dist.destroy_process_group()  # no failure-path barrier
        except BaseException as exc:
            if rank == 0 and destination is not None:
                report.update(execution_succeeded=False, stage="distributed_cleanup", error=str(exc))
                atomic_json(destination / "summary.json", report)
            raise
