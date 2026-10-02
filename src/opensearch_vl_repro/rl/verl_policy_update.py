"""One REAL verl 0.6.1 FSDP2 PPO policy update. No custom policy/SFT loss."""
from __future__ import annotations

import gc
import json
import math
import os
import time
import weakref
from dataclasses import replace
from datetime import timedelta

from opensearch_vl_repro.rl.actor_gate import (
    atomic_json, gradient_checks, lora_snapshot, reload_matches, trainable_policy, update_checks,
)
from opensearch_vl_repro.rl.gate_c import (
    UPDATE_CHECKS, bind_run, failure_report, paths_for, prepare_context, record_stage,
)
from opensearch_vl_repro.rl.group import read_group, run_lock
from opensearch_vl_repro.rl.rloo import official_rloo
from opensearch_vl_repro.rl.training_batch import build_dataproto, mask_artifact, training_rows
from opensearch_vl_repro.rl.policy_alignment import (
    alignment_artifact, alignment_checks, compare_policy_logprobs, require_policy_alignment,
)
from opensearch_vl_repro.sft_tool_audit import sha256_file
from opensearch_vl_repro.eval_subset import canonical_json_sha256


def configure_one_update(actor, row_count, gate):
    if type(row_count) is not int or row_count < 1:
        raise ValueError("nonempty policy rows required")
    # Pinned actor mini-batch/epoch loops yield EXACTLY ONE optimizer call.
    actor.config = replace(actor.config, ppo_mini_batch_size=row_count,
                           ppo_micro_batch_size_per_gpu=1, ppo_epochs=1, shuffle=False,
                           clip_ratio_low=gate["clip_ratio_low"], clip_ratio_high=gate["clip_ratio_high"],
                           entropy_coeff=0., use_kl_loss=False, use_rollout_log_probs=True,
                           loss_agg_mode="seq-mean-token-mean")


def audited_policy_update(actor, data):
    """Inspect clipped live gradients at the REAL AdamW.step, never manufacture them."""
    count, gradient_audit = 0, {}
    optimizer = actor.actor_optimizer
    original_step = optimizer.step
    def step(*args, **kwargs):
        nonlocal count, gradient_audit
        if count != 0:
            raise RuntimeError("Gate C refuses a second optimizer step")
        gradient_audit = gradient_checks(actor.actor_module)
        frozen_clean = all(p.grad is None for p in actor.actor_module.parameters() if not p.requires_grad)
        gradient_audit["vision_projector_base_frozen"] = frozen_clean
        trainable_policy(actor.actor_module, optimizer)
        if not all(gradient_audit.values()):
            raise RuntimeError(f"RL gradient gate failed (possibly zero reward variance): {gradient_audit}")
        value = original_step(*args, **kwargs)
        count += 1
        return value
    optimizer.step = step
    try:
        metrics = actor.update_policy(data)  # official RL clipped policy loss + backward + optimizer
    finally:
        optimizer.step = original_step
    if count != 1:
        raise RuntimeError("real verl update_policy did not perform exactly one optimizer step")
    losses = metrics.get("actor/pg_loss", [])
    if not losses or not all(math.isfinite(float(loss)) for loss in losses):
        raise RuntimeError("real verl policy loss missing/nonfinite")
    return metrics, {**gradient_audit, "optimizer_step_count": count}


def audit_pre_update_policy(actor, data, gate, *, expected_masked_token_count):
    """Real verl inference on the SAME DataProto that update_policy will consume."""
    import torch

    with torch.no_grad():
        current_log_probs, _ = actor.compute_log_prob(data, calculate_entropy=False)
    audit = compare_policy_logprobs(current_log_probs, data.batch["old_log_probs"],
        data.batch["response_mask"], clip_ratio_low=gate["clip_ratio_low"],
        clip_ratio_high=gate["clip_ratio_high"], expected_masked_token_count=expected_masked_token_count)
    audit.update(logprobs_computed=True, temperature=data.meta_info.get("temperature"),
                 rollout_temperature=gate["vllm"]["temperature"])
    audit["checks"] = alignment_checks(audit)
    audit["passed"] = all(audit["checks"].values())
    return audit


def policy_update_after_alignment(actor, data, alignment):
    require_policy_alignment(alignment)
    return audited_policy_update(actor, data)


def update(args, root):
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from opensearch_vl_repro.model import load_processor
    from opensearch_vl_repro.inference.adapter import adapter_file_identity
    from opensearch_vl_repro.rl.verl_actor_gate import (
        CollectiveStages, checkpoint_manager, construct_actor, memory_stats, save_checkpoint,
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Gate C actor update requires real CUDA/BF16; CPU test is not PASS")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    output, reports = paths_for(root, args.run_id)
    ctx = prepare_context(args, root)
    actor = fresh = before = after = None
    checks = dict.fromkeys(UPDATE_CHECKS, False)
    alignment = None
    stage, started = "actor_initializing", time.monotonic()
    lock = None
    try:
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 2:
            raise ValueError("Gate C actor_world_size must equal 2")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(local_rank)
        stages = CollectiveStages(torch, reports / "gate_c_actor_stages.json", ctx["identity"], local_rank,
                                  required_checks=UPDATE_CHECKS, label="Gate C")
        def run(name, operation):
            nonlocal stage
            stage = name
            if rank == 0:
                record_stage(reports, ctx["identity"], name)
            return stages.run(name, operation)
        def reserve():
            nonlocal lock
            if rank == 0:
                lock = run_lock(root / "outputs/rl_gate_c" / (args.run_id + ".lock"))
                lock.__enter__()
                bind_run(output, reports, ctx["identity"])
        run("actor_run_binding", reserve)
        group = run("committed_group_verify", lambda: read_group(output / "group"))
        if group["identity"]["context"] != ctx["identity"]["identity_sha256"]:
            raise ValueError("group/pre-update policy context changed")
        if (output / "update_verified.json").exists():
            # The finalizer will recheck checksums. Never repeat a completed update.
            receipt = json.loads((output / "update_verified.json").read_text(encoding="utf-8"))
            if receipt["identity"] != ctx["identity"] or receipt["group_sha256"] != sha256_file(output / "group/group.json"):
                raise ValueError("completed update receipt differs from run/group")
            print("Verified update exists; no second optimizer step. Run finalize.", flush=True)
            return 0
        def begin_update():
            if rank == 0:
                if (output / "update_started.json").exists():
                    raise ValueError("actor update previously entered; ambiguous step/save cannot resume. Use NEW run-id from checkpoint-3k")
                atomic_json(output / "update_started.json", {"identity": ctx["identity"], "formal_rl_initialization_allowed": False})
        run("actor_updating", begin_update)
        rewards = [m["reward"]["total"] for m in sorted(group["members"], key=lambda m: m["rollout_index"])]
        fatal = [m["fatal"]["fatal"] for m in sorted(group["members"], key=lambda m: m["rollout_index"])]
        raw, final = run("real_verl_rloo", lambda: official_rloo(rewards, fatal, group_id=group["identity"]["trajectory_group_id"]))
        checks["real_verl_rloo"] = len(raw) == 2
        checks["fatal_clamp_after_rloo"] = all(
            b == (max(a, 0.) if f else a) for a, b, f in zip(raw, final, fatal, strict=True))
        checks["finite_advantages"] = all(math.isfinite(v) for v in raw + final)
        rows, token_audit = run("fatal_aware_training_rows", lambda: training_rows(group, final))
        checks["response_only_mask"] = token_audit["supervised_response_tokens"] == sum(sum(r["response_mask"]) for r in rows)
        run("token_mask_audit_artifact", lambda: atomic_json(output / "training_masks.json", mask_artifact(group, final)) if rank == 0 else None)
        mesh = run("device_mesh", lambda: init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",)))
        processor = run("processor", lambda: load_processor(ctx["runtime_sft"], local_files_only=True))
        actor, audit = run("original_sft_lora_fsdp2", lambda: construct_actor(
            config=ctx["runtime_sft"], gate=ctx["a22"], adapter=ctx["adapter"], mesh=mesh))
        # construct_actor validates the original PEFT/frozen parameters and real
        # verl FSDP2 wrapping (including every decoder) before it returns.
        checks["formal_sft_actor_init"] = ctx["adapter"] == root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
        checks["real_fsdp2"] = actor.config.strategy == "fsdp2"
        active = (audit["model_training"] is True and audit["language_model_training"] is True
                  and audit["decoder_layers_training"] == 36 and audit["decoder_layers_gradient_checkpointing"] == 36
                  and audit["effective_attention_implementation"] == "flash_attention_2")
        if not active:
            raise RuntimeError("FSDP2 actor train/checkpointing/FA2 contract failed")
        checks["training_checkpointing_active"] = active
        configure_one_update(actor, len(rows), ctx["gate"])
        data = run("real_multimodal_dataproto", lambda: build_dataproto(rows, directory=output / "group",
            model=actor.actor_module, pad_id=processor.tokenizer.pad_token_id, device=torch.device("cuda", local_rank), temperature=.7))
        from verl import DataProto
        checks["real_dataproto"] = isinstance(data, DataProto)
        before = run("pre_update_lora_snapshot", lambda: lora_snapshot(actor.actor_module))
        local_alignment = run("pre_update_policy_alignment", lambda: audit_pre_update_policy(
            actor, data, ctx["gate"], expected_masked_token_count=token_audit["supervised_response_tokens"]))
        local_alignment["rank"] = rank
        checks.update(local_alignment["checks"])
        def gather_alignment():
            rows = [None] * world
            dist.all_gather_object(rows, local_alignment)
            return alignment_artifact(rows, gate_version=ctx["gate"]["gate_version"], identity=ctx["identity"],
                trajectory_group_id=group["identity"]["trajectory_group_id"],
                policy_fingerprint=ctx["actor"]["source_sft_adapter_fingerprint"])
        alignment = run("pre_update_policy_alignment", gather_alignment)
        run("pre_update_policy_alignment", lambda: atomic_json(
            output / "pre_update_policy_alignment.json", alignment) if rank == 0 else None)
        # Collective check BEFORE entering the policy-loss stage: neither rank
        # may update if its peer failed. Same data object, no old-logprob write.
        run("pre_update_policy_alignment", lambda: require_policy_alignment(alignment))
        metrics, grad_audit = run("real_verl_update_policy", lambda: policy_update_after_alignment(actor, data, alignment))
        checks["real_verl_policy_loss"] = bool(metrics.get("actor/pg_loss"))
        checks["loss_finite"] = all(math.isfinite(float(v)) for v in metrics["actor/pg_loss"])
        checks["exactly_one_optimizer_step"] = grad_audit["optimizer_step_count"] == 1
        for key in ("lora_grad_finite", "nonzero_lora_grad", "vision_projector_base_frozen"):
            checks[key] = grad_audit[key] is True
        after = run("post_update_lora_snapshot", lambda: lora_snapshot(actor.actor_module))
        changed = update_checks(before, after)
        if not all(changed.values()):
            raise RuntimeError("one RL step did not change finite LoRA parameters")
        checks.update(changed)
        run("post_update_frozen_audit", lambda: trainable_policy(actor.actor_module, actor.actor_optimizer))
        checks["optimizer_only_lora"] = True  # exact parameter-identity audit returned without error
        saved = run("native_checkpoint_and_peft_export", lambda: save_checkpoint(actor, processor, output / "updated_actor"))
        checks["native_checkpoint_saved"] = all(
            (output / "updated_actor/distributed" / f"{prefix}_world_size_{world}_rank_{r}.pt").is_file()
            for r in range(world) for prefix in ("model", "optim", "extra_state"))
        checks["peft_exported"] = bool(saved.get("adapter_fingerprint"))
        def gate_only_metadata():
            if rank == 0:
                atomic_json(output / "updated_actor/gate_only_metadata.json", {
                    "formal_rl_initialization_allowed": False, "source_policy_fingerprint":
                    ctx["actor"]["source_sft_adapter_fingerprint"], "optimizer_step_count": 1,
                    "group_id": group["identity"]["trajectory_group_id"], "adapter_identity": saved})
        run("gate_only_checkpoint_metadata", gate_only_metadata)
        post_fingerprint = saved["adapter_fingerprint"]
        if post_fingerprint == ctx["actor"]["source_sft_adapter_fingerprint"]:
            raise RuntimeError("post-update adapter fingerprint unchanged")
        checks["post_policy_fingerprint_changed"] = post_fingerprint != ctx["actor"]["source_sft_adapter_fingerprint"]
        model_ref, optimizer_ref = weakref.ref(actor.actor_module), weakref.ref(actor.actor_optimizer)
        del actor, before
        actor = before = None
        gc.collect()
        torch.cuda.empty_cache()
        destroyed = model_ref() is None and optimizer_ref() is None
        if not destroyed:
            raise RuntimeError("original actor/optimizer still alive before fresh reload")
        checks["original_actor_destroyed"] = destroyed
        run("destroy_and_barrier", dist.barrier)
        fresh, _ = run("fresh_base_updated_peft_fsdp2", lambda: construct_actor(
            config=ctx["runtime_sft"], gate=ctx["a22"], adapter=output / "updated_actor/adapter", mesh=mesh))
        checks["fresh_actor_reloaded"] = fresh is not None
        if not reload_matches(after, lora_snapshot(fresh.actor_module)):
            raise RuntimeError("fresh PEFT reload parameter mismatch")
        run("native_model_optimizer_reload", lambda: checkpoint_manager(fresh, processor).load_checkpoint(
            str(output / "updated_actor/distributed"), del_local_after_load=False))
        native_ok = bool(fresh.actor_optimizer.state)
        matched = reload_matches(after, lora_snapshot(fresh.actor_module))
        checks["native_checkpoint_reloaded"] = native_ok
        checks["reload_param_match"] = matched
        def finite_forward():
            with torch.no_grad():
                logprobs, _ = fresh.compute_log_prob(data, calculate_entropy=False)
            if not bool(torch.isfinite(logprobs).all()):
                raise RuntimeError("fresh native checkpoint forward logprobs nonfinite")
            return {"finite": True, "shape": list(logprobs.shape)}
        proof = run("fresh_multimodal_forward", finite_forward)
        checks["fresh_forward_finite"] = proof["finite"] is True
        if not all(checks.values()):
            raise RuntimeError(f"incomplete Gate C update checks: {checks}")
        rank_report = {"rank": rank, "world_size": world, "checks": checks, "model_audit": audit,
                       "pre_update_policy_alignment": local_alignment,
                       "metrics": {k: [float(v) for v in vals] for k, vals in metrics.items()},
                       "gradient_audit": grad_audit, "optimizer_step_count": grad_audit["optimizer_step_count"],
                       "reload_proof": proof, **memory_stats(torch, local_rank), "elapsed_seconds": time.monotonic() - started}
        ranks = [None] * world
        dist.all_gather_object(ranks, rank_report)
        del fresh, after, data
        fresh = after = None
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        def publish_update():
            if rank == 0:
                files = {p.relative_to(output / "updated_actor").as_posix(): sha256_file(p)
                         for p in sorted((output / "updated_actor").rglob("*")) if p.is_file()}
                report = {"passed": False, "stage": "actor_reloaded", "identity": ctx["identity"],
                          "formal_rl_initialization_allowed": False, "group_sha256": sha256_file(output / "group/group.json"),
                          "checks": {k: all(row["checks"].get(k) is True for row in ranks) for k in UPDATE_CHECKS},
                          "per_rank": ranks, "raw_advantages": raw, "final_advantages": final,
                          "raw_returns": raw, "final_returns": final, "rewards": rewards,
                          "training_token_counts": token_audit, "pre_update_policy_fingerprint": ctx["actor"]["source_sft_adapter_fingerprint"],
                          "training_masks_sha256": sha256_file(output / "training_masks.json"),
                          "pre_update_policy_alignment_sha256": sha256_file(output / "pre_update_policy_alignment.json"),
                          "pre_update_policy_alignment": alignment,
                          "post_update_policy_fingerprint": post_fingerprint,
                          "checkpoint_file_sha256": files, "checkpoint_fingerprint": canonical_json_sha256(files),
                          "completed_stages": stages.log,
                          "scope": "one n=2 group; per-generation policy rows replicated on both ranks; one optimizer update"}
                atomic_json(output / "update_verified.json", report)
                record_stage(reports, ctx["identity"], "actor_reloaded", update_receipt_sha256=sha256_file(output / "update_verified.json"))
        run("verified_update_publication", publish_update)
        return 0
    except BaseException as exc:
        if not dist.is_initialized() or dist.get_rank() == 0:
            extra = {"checks": checks}
            if stage == "pre_update_policy_alignment":
                extra.update(optimizer_step_count=0, pre_update_policy_alignment=alignment)
                if alignment is not None:
                    extra["checks"] = {**checks, **alignment["checks"]}
            failure_report(output, reports, ctx["identity"], stage, exc, **extra)
        raise
    finally:
        actor = fresh = before = after = None
        gc.collect()
        if lock is not None:
            lock.__exit__(None, None, None)
        if dist.is_initialized():
            dist.destroy_process_group()
