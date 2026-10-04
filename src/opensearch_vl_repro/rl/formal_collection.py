"""Isolated real Formal collection: current verified policy -> atomic n=2 groups."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256 as digest
from . import checkpoint as cp
from .formal_smoke import (VERSION, prepare_context, recover_smoke,
                           require_smoke_run, smoke_paths)
from .group import formal_group_identity, publish_formal_group, validate_formal_group
from .rollout_inputs import load_source_images, model_task
from .run_state import checkpoint_policy


def validate_formal_handoff(root, run, policy, source_adapter, *, cpu_fixture=False):
    """Validate authority, not a directory spelling or Gate evidence booleans."""
    from opensearch_vl_repro.inference.adapter import adapter_file_identity
    require_smoke_run(run)
    state = recover_smoke(root, run, cpu_fixture=cpu_fixture)
    if state["policy"] != policy:
        raise ValueError("static handoff must use the authoritative CURRENT policy")
    if policy["policy_iteration"] == 0:
        directory = Path(source_adapter).resolve()
        # prepare_context already validates full original SFT lineage/metadata.
        identity = adapter_file_identity(directory)
        if identity["adapter_fingerprint"] != run["semantics"]["source_sft"]["adapter_sha256"]:
            raise ValueError("original SFT adapter changed")
        role_files = identity["file_sha256"]
    else:
        directory = Path(root) / "checkpoints" / f"policy-{policy['policy_iteration']:06d}"
        checkpoint = cp.read_verified_checkpoint(directory)
        if checkpoint["run"]["run_identity_sha256"] != run["run_identity_sha256"] or checkpoint_policy(checkpoint) != policy:
            raise ValueError("foreign/diagnostic/unverified Formal adapter")
        role_files = {name.removeprefix("adapter/"): sha for name, sha in checkpoint["artifact_role_files"]["adapter"].items()}
        if any(not name.startswith("adapter/") for name in checkpoint["artifact_role_files"]["adapter"]):
            raise ValueError("Formal adapter role must be a complete adapter export")
        directory = directory / "adapter"
        identity = adapter_file_identity(directory)
        if cp.artifact_inventory(directory) != role_files:
            raise ValueError("Formal adapter role/inventory mismatch")
    binding = dict(version=VERSION, run_identity_sha256=run["run_identity_sha256"],
        policy_iteration=policy["policy_iteration"], effective_policy_fingerprint=policy["effective_policy_fingerprint"],
        parent_checkpoint_identity=policy["checkpoint_identity"], adapter_file_sha256=role_files,
        base_snapshot_sha256=run["semantics"]["base_model"]["offline_snapshot_sha256"])
    actor = dict(actor_adapter_fingerprint=identity["adapter_fingerprint"], formal_binding=binding,
        base_model=run["semantics"]["base_model"]["name"], base_revision=run["semantics"]["base_model"]["revision"])
    return directory, actor


def formal_merge_identity(*, actor, versions, file_hashes):
    value = dict(**actor["formal_binding"], merge_method="peft.merge_and_unload",
                 merged_file_sha256=file_hashes, software_versions=versions)
    return {**value, "merged_checkpoint_fingerprint": digest(value)}


def verify_formal_merge(directory, binding):
    directory = Path(directory)
    value = json.loads((directory / "merge_manifest.json").read_text(encoding="utf-8"))
    identity = value["identity"]
    cp.check_seal(identity, "merged_checkpoint_fingerprint")
    if any(identity.get(k) != v for k, v in binding.items()):
        raise ValueError("stale/foreign static merged policy")
    if any(value.get(k) is not True for k in (
            "merge_complete", "fresh_hf_forward_finite", "no_active_peft", "merge_hf_destroyed", "reload_hf_destroyed")):
        raise ValueError("static merge/fresh HF verification incomplete")
    cp.verify_artifacts(directory, identity["merged_file_sha256"], exclude=("merge_manifest.json",))
    return value


def reserve_collection(root, run, policy, row):
    """Private forensic attempt is never a reusable half-group; always fresh UUID."""
    prompt = row["prompt_id"]
    previous = 0
    for path in (Path(root) / "groups").glob(".collect-*/attempt.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["prompt_id"] == prompt and value["pre_update_policy_fingerprint"] == policy["effective_policy_fingerprint"]:
            previous = max(previous, value["collection_attempt_index"] + 1)
    identity = formal_group_identity(run, policy, prompt_id=prompt,
        source_identity=source_identity_for_row(run, row), attempt_id=str(uuid.uuid4()), attempt_index=previous)
    staging = Path(root) / "groups" / (".collect-" + identity["collection_attempt"])
    staging.mkdir(exist_ok=False)
    cp.durable_json(staging / "attempt.json", identity)
    return staging, identity


def source_identity_for_row(run, row):
    from .formal_smoke import source_identity
    actual = source_identity(row)
    if actual != cp.source_identity_for_prompt(run, row["prompt_id"]):
        raise ValueError("collection source record changed")
    return actual


def commit_collection(root, staging, identity, members, *, merge_identity, cpu_fixture=False):
    draft = dict(identity=identity, members=members, rollout_executed=True, static_merge=merge_identity)
    validate_formal_group(draft, committed=False)
    for member in members:
        cutoff = member.get("fatal_step")
        if member["fatal"] and (type(cutoff) is not int or not 0 <= cutoff < len(member["steps"])):
            raise ValueError("Formal fatal member requires an explicit actual generation cutoff")
    return publish_formal_group(staging, Path(root) / "groups" / identity["trajectory_group_id"], draft,
                                cpu_fixture=cpu_fixture)


def collect_member(*, row, images, identity, index, backend, registry, judge, reward_cache, staging, rollout, seed):
    """Actual rLLM owns the loop; project code persists/rewards its captured Steps."""
    import asyncio
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from vllm import SamplingParams
    from .formal_smoke import redact_runtime_secrets as redact_secrets
    from .live_workflow import LiveRLWorkflowAdapter
    from .workflow_adapter import build_rllm_workflow
    from .workflow_types import RLInfrastructureError
    from .reward_judges import live_rewards, bind_trajectory_reward
    adapter = LiveRLWorkflowAdapter(registry)
    task = model_task(row, images)  # Reference is deliberately not copied here.
    backend.sampling = SamplingParams(temperature=rollout["temperature"], top_p=rollout["top_p"], top_k=rollout["top_k"],
        logprobs=1, max_tokens=rollout["max_new_tokens"], seed=seed + index)
    begin = len(backend.training_inputs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        workflow, provenance = build_rllm_workflow(adapter=adapter, backend=backend, executor=executor,
                                                   max_turns=rollout["max_turns"], capture_tokens=True)
        episode = asyncio.run(workflow.run_with_termination_handling(task=task, uid=f"{identity['trajectory_group_id']}:{index}"))
    if adapter.infrastructure_failure is not None:
        # rLLM catches environment exceptions. The original provider error still
        # aborts the ENTIRE group/window; never convert it to reward=0.
        cp.durable_json(staging / f"provider-partial-{index}.json", redact_secrets(dict(
            complete=False, trajectory=adapter.finalize_episode(termination="error").to_dict(),
            steps=[s.to_dict() for t in episode.trajectories for s in t.steps],
            error_type=adapter.infrastructure_failure.error_type)))
        raise adapter.infrastructure_failure
    termination = episode.termination_reason.value if episode.termination_reason else "unknown"
    if termination not in {"env_done", "max_turns_exceeded", "max_response_length_exceeded"}:
        raise RLInfrastructureError(f"rLLM failed: {termination}: {episode.info}")
    trajectory = adapter.finalize_episode(termination=termination)
    if trajectory.status == "fatal":
        trajectory.status = "max_agent_turns_exceeded"  # Existing shared format scorer view.
    if termination == "max_response_length_exceeded":
        trajectory.status, trajectory.final_answer = "max_agent_turns_exceeded", None
    if len(episode.trajectories) != 1:
        raise ValueError("one actual rLLM trajectory per Formal member required")
    actual = episode.trajectories[0]
    steps = [step.to_dict() for step in actual.steps]
    inputs = backend.training_inputs[begin:]
    if not steps or len(steps) != len(inputs) or len(steps) != len(adapter.state().assistant_outputs):
        raise ValueError("actual rLLM steps/processor captures misaligned")
    for step_index, (step, tensors) in enumerate(zip(steps, inputs, strict=True)):
        if tensors["input_ids"][0].tolist() != step["prompt_ids"]:
            raise ValueError("actual rLLM token/processor binding mismatch")
        step["multimodal_file"] = f"multimodal-{index}-{step_index}.pt"
        torch.save(tensors, staging / step["multimodal_file"])
    fatal = adapter.fatal_metadata(termination=termination, step_count=len(steps))
    member = dict(identity=identity, rollout_index=index, member_id=f"{identity['trajectory_group_id']}:{index}",
        complete=False, fatal=fatal["fatal"], fatal_step=fatal["fatal_step"], fatal_metadata=fatal,
        steps=steps, trajectory=trajectory.to_dict(), trajectory_file=f"trajectory-{index}.json",
        termination=termination, rllm_provenance=provenance, rllm_episode_identity=episode.id, rollout_executed=True)
    cp.durable_json(staging / member["trajectory_file"], redact_secrets(member))
    member["reward"] = live_rewards(client=judge, directory=reward_cache, row=row, trajectory=trajectory, fatal=fatal)
    bind_trajectory_reward(actual, member["reward"]["total"])
    for stored, actual_step in zip(steps, actual.steps, strict=True):
        stored.update(reward=actual_step.reward, mc_return=actual_step.mc_return, info=actual_step.info)
    member["complete"] = True
    cp.durable_json(staging / member["trajectory_file"], redact_secrets(member))
    # A window's CPU carrier is saved on disk. Do not retain all previous prompt
    # vision tensors in the long-lived vLLM backend's capture buffer.
    del backend.training_inputs[begin:]
    return redact_secrets(member)


def run_collection(args, root):
    from .formal_smoke import redact_runtime_secrets as redact_secrets
    from .live_workflow import ProviderInterruption, LiveRLWorkflowAdapter
    root = Path(root).resolve()
    ctx = prepare_context(args, root)
    run = ctx["run"]
    output, reports = smoke_paths(root, args.run_id)
    backend, stage, identity = None, "recovery", None
    started = time.monotonic()
    try:
        recovered = recover_smoke(output, run)
        if not recovered["missing_prompts"]:
            return 0
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError("Formal collection requires exactly one visible CUDA/BF16 GPU")
        torch.cuda.set_device(0)
        from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry
        from opensearch_vl_repro.evaluation.judge import DeepSeekJudge, load_judge_config
        from .rollout_gate import VLLMStaticBackend
        from .rollout_sync import merge_actor_adapter
        policy = recovered["policy"]
        rows = {r["prompt_id"]: r for r in ctx["records"]}
        registry = create_phase3_tool_registry(search_config=args.search_config, layout_config=args.layout_config,
            cache_dir=args.tool_cache_dir or output / "tool_cache")
        judge = DeepSeekJudge(load_judge_config(args.judge_config))
        reward_cache = args.reward_cache_dir or output / "reward_cache"
        directory, actor = validate_formal_handoff(output, run, policy, args.sft_adapter)
        merge_root = output / "merges"
        merge_root.mkdir(exist_ok=True)
        merged = merge_root / f"policy-{policy['policy_iteration']:06d}"
        first = rows[recovered["missing_prompts"][0]]
        adapter = LiveRLWorkflowAdapter(registry)
        adapter.initialize_episode(**model_task(first, load_source_images(first, args.source_root)))
        stage = "static_merge"
        if not merged.exists():
            merge_actor_adapter(base_snapshot=args.base_model_path, adapter=directory, actor=actor,
                sft_config=ctx["canonical"], output=merged, versions=ctx["versions"],
                validation_messages=adapter.build_next_messages(), tools=registry.declarations_for_model(),
                identity_builder=formal_merge_identity)
        merge = verify_formal_merge(merged, actor["formal_binding"])
        rollout = run["semantics"]["rollout"]["config"]
        settings = dict(vllm={k: rollout[k] for k in ("max_model_len", "max_new_tokens", "tensor_parallel_size", "gpu_memory_utilization")},
                        agent=dict(max_turns=rollout["max_turns"]))
        stage = "vllm_init"
        backend = VLLMStaticBackend(checkpoint=merged, sft_config=ctx["canonical"], gate=settings,
                                   seed=run["semantics"]["initial_seed"], capture_tokens=True)
        for prompt in recovered["missing_prompts"]:
            row = rows[prompt]
            images = load_source_images(row, args.source_root)
            staging, identity = reserve_collection(output, run, policy, row)
            stage = "collecting_and_rewarding"
            members = []
            for index in range(2):
                members.append(collect_member(row=row, images=images, identity=identity, index=index,
                    backend=backend, registry=registry, judge=judge, reward_cache=reward_cache,
                    staging=staging, rollout=rollout,
                    seed=run["semantics"]["initial_seed"] + 2 * run["prompt_ids"].index(prompt)))
            stage = "group_publication"
            group = commit_collection(output, staging, identity, members, merge_identity=merge["identity"])
            print(f"[Smoke20] prompt {run['prompt_ids'].index(prompt) + 1}/20 group committed {group['identity']['trajectory_group_id']}", flush=True)
        stage = "vllm_shutdown"
        backend.close()
        backend = None
        cp.durable_json(reports / "collection.json", dict(passed=False, status="collection_complete", scope="runtime",
            run_identity_sha256=run["run_identity_sha256"], policy=policy, merge=merge["identity"],
            worker_shutdown=True, elapsed_seconds=time.monotonic() - started))
        return 0
    except BaseException as exc:
        failure = redact_secrets(dict(passed=False, status="interrupted", stage=stage,
            error=f"{type(exc).__name__}: {exc}", collection_identity=identity,
            provider_interruption=dict(error_type=exc.error_type, reason=exc.reason) if isinstance(exc, ProviderInterruption) else None))
        cp.durable_json(reports / "collection_failure.json", failure)
        raise
    finally:
        if backend is not None:
            backend.close()  # Failure propagates; coordinator also kills/reaps process group.
