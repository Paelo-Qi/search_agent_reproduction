"""One shared static merge; each isolated collector owns one entire n4 group."""
from __future__ import annotations

import json
import hashlib
import uuid
from pathlib import Path

from . import checkpoint as cp
from .formal_main import VERSION, CONTEXT_BUDGET_POLICY, prepare_context, recover_main, main_paths, member_seed
from .formal_collection import (formal_merge_identity, verify_formal_merge, reserve_collection,
                                commit_collection, collect_member, model_task, load_source_images)
from .run_state import checkpoint_policy


def validate_main_handoff(root, run, policy, source_adapter):
    """Current authority only; no Gate/Smoke adapter path promotion."""
    from opensearch_vl_repro.inference.adapter import adapter_file_identity
    state = recover_main(root, run, cleanup=False)
    if state["policy"] != policy:
        raise ValueError("Main handoff must use authoritative CURRENT policy")
    if policy["policy_iteration"] == 0:
        directory = Path(source_adapter).resolve()
        identity = adapter_file_identity(directory)
        if identity["adapter_fingerprint"] != run["semantics"]["source_sft"]["adapter_sha256"]:
            raise ValueError("original SFT adapter changed")
        role_files = identity["file_sha256"]
    else:
        parent = Path(root) / "checkpoints" / f"policy-{policy['policy_iteration']:06d}"
        checkpoint = cp.read_verified_checkpoint(parent)
        if checkpoint_policy(checkpoint) != policy or checkpoint["eligibility"]["kind"] != "main_checkpoint":
            raise ValueError("foreign/diagnostic Main adapter")
        directory = parent / "adapter"
        identity = adapter_file_identity(directory)
        role_files = {n.removeprefix("adapter/"): sha for n, sha in checkpoint["artifact_role_files"]["adapter"].items()}
        if cp.artifact_inventory(directory) != role_files:
            raise ValueError("Main adapter export inventory mismatch")
    binding = dict(version=VERSION, run_identity_sha256=run["run_identity_sha256"],
        policy_iteration=policy["policy_iteration"], effective_policy_fingerprint=policy["effective_policy_fingerprint"],
        parent_checkpoint_identity=policy["checkpoint_identity"], adapter_file_sha256=role_files,
        base_snapshot_sha256=run["semantics"]["base_model"]["offline_snapshot_sha256"])
    return directory, dict(actor_adapter_fingerprint=identity["adapter_fingerprint"], formal_binding=binding,
        base_model=run["semantics"]["base_model"]["name"], base_revision=run["semantics"]["base_model"]["revision"])


def merged_path(output, policy):
    return Path(output) / "merges" / f"policy-{policy['policy_iteration']:06d}"


def run_merge_worker(args, root):
    ctx = prepare_context(args, root)
    output, _ = main_paths(root, args.run_id)
    recovered = recover_main(output, ctx["run"])
    if not recovered["missing_prompts"]:
        raise ValueError("complete window must retire merge, not recreate it")
    require_single_gpu()
    from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry
    from .live_workflow import LiveRLWorkflowAdapter
    from .rollout_sync import merge_actor_adapter
    policy = recovered["policy"]
    directory, actor = validate_main_handoff(output, ctx["run"], policy, args.sft_adapter)
    merged = merged_path(output, policy)
    merged.parent.mkdir(exist_ok=True)
    if not merged.exists():
        registry = create_phase3_tool_registry(search_config=args.search_config, layout_config=args.layout_config,
                                               cache_dir=args.tool_cache_dir or output / "tool_cache")
        row = next(r for r in ctx["records"] if r["prompt_id"] == recovered["missing_prompts"][0])
        adapter = LiveRLWorkflowAdapter(registry)
        adapter.initialize_episode(**model_task(row, load_source_images(row, args.source_root)))
        merge_actor_adapter(base_snapshot=args.base_model_path, adapter=directory, actor=actor,
            sft_config=ctx["canonical"], output=merged, versions=ctx["versions"],
            validation_messages=adapter.build_next_messages(), tools=registry.declarations_for_model(),
            identity_builder=formal_merge_identity)
    verify_formal_merge(merged, actor["formal_binding"])  # FULL ONCE before sharing/reuse
    return 0  # supervisor also verifies all HF descendants gone


def require_single_gpu():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("Main merge/collection requires exactly one visible CUDA/BF16 GPU")
    torch.cuda.set_device(0)


def read_shared_merge(output, run, policy):
    """Metadata after exclusive merge-worker FULL verification + process teardown.

    Immutable shared bytes are fully checked again at the retirement boundary.
    This is not a generic merge reader nor permission to consume foreign bytes.
    """
    merged = merged_path(output, policy)
    value = json.loads((merged / "merge_manifest.json").read_text(encoding="utf-8"))
    ident = value["identity"]
    cp.check_seal(ident, "merged_checkpoint_fingerprint")
    if (ident.get("version") != VERSION or ident.get("run_identity_sha256") != run["run_identity_sha256"]
            or ident.get("policy_iteration") != policy["policy_iteration"]
            or ident.get("parent_checkpoint_identity") != policy["checkpoint_identity"]
            or ident.get("effective_policy_fingerprint") != policy["effective_policy_fingerprint"]
            or ident.get("base_snapshot_sha256") != run["semantics"]["base_model"]["offline_snapshot_sha256"]
            or any(value.get(k) is not True for k in ("merge_complete", "fresh_hf_forward_finite",
                 "no_active_peft", "merge_hf_destroyed", "reload_hf_destroyed"))):
        raise ValueError("Main shared merge identity/fresh HF verification mismatch")
    return merged, ident


def run_collection_worker(args, root):
    from .formal_smoke import redact_runtime_secrets
    from .live_workflow import ProviderInterruption
    ctx = prepare_context(args, root)
    output, reports = main_paths(root, args.run_id)
    backend, stage, identity = None, "recovery", None
    invocation = str(uuid.uuid4())
    prompt_tag = hashlib.sha256(args.prompt_id.encode()).hexdigest()[:16]
    try:
        # Read-only while peers publish: no attempt reconciliation/cleanup writes.
        state = recover_main(output, ctx["run"], reconcile=False, cleanup=False)
        if args.prompt_id not in state["missing_prompts"]:
            raise ValueError("collection worker owns exactly one missing current-window prompt")
        require_single_gpu()
        from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry
        from opensearch_vl_repro.evaluation.judge import DeepSeekJudge, load_judge_config
        from .rollout_gate import VLLMStaticBackend
        policy, run = state["policy"], ctx["run"]
        merged, merge = read_shared_merge(output, run, policy)
        row = next(r for r in ctx["records"] if r["prompt_id"] == args.prompt_id)
        registry = create_phase3_tool_registry(search_config=args.search_config, layout_config=args.layout_config,
            cache_dir=args.tool_cache_dir or output / "tool_cache")
        judge = DeepSeekJudge(load_judge_config(args.judge_config))
        rollout = run["semantics"]["rollout"]["config"]
        settings = dict(vllm={k: rollout[k] for k in ("max_model_len", "max_new_tokens", "tensor_parallel_size", "gpu_memory_utilization")},
                        agent=dict(max_turns=rollout["max_turns"]))
        stage = "vllm_init"
        backend = VLLMStaticBackend(checkpoint=merged, sft_config=ctx["canonical"], gate=settings,
                                   seed=member_seed(run, args.prompt_id, 0), capture_tokens=True,
                                   context_budget_policy=CONTEXT_BUDGET_POLICY)
        staging, identity = reserve_collection(output, run, policy, row)
        images, members = load_source_images(row, args.source_root), []
        stage = "collecting_and_rewarding"
        for index in range(4):
            member = collect_member(row=row, images=images, identity=identity, index=index,
                backend=backend, registry=registry, judge=judge, reward_cache=args.reward_cache_dir or output / "reward_cache",
                staging=staging, rollout=rollout, seed=member_seed(run, args.prompt_id, 0))
            member["member_seed"] = member_seed(run, args.prompt_id, index)
            cp.durable_json(staging / member["trajectory_file"], member)
            members.append(member)
        stage = "group_publication"
        commit_collection(output, staging, identity, members, merge_identity=merge)
        stage = "shutdown"
        backend.close()
        backend = None
        cp.durable_json(reports / f"collection-{prompt_tag}-{invocation}.json", dict(
            passed=False, status="group_committed", collection_identity=identity, worker_shutdown=True))
        return 0
    except BaseException as exc:
        cp.durable_json(reports / f"collection_failure-{prompt_tag}-{invocation}.json", redact_runtime_secrets(dict(
            passed=False, status="interrupted", prompt_id=args.prompt_id, stage=stage,
            error=f"{type(exc).__name__}: {exc}", collection_identity=identity,
            provider_interruption=dict(error_type=exc.error_type, reason=exc.reason) if isinstance(exc, ProviderInterruption) else None)))
        raise
    finally:
        if backend is not None:
            backend.close()
