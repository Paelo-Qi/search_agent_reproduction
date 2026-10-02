"""Gate C lifecycle: collect+reward -> FSDP2 policy update -> verify/publish.

Import safe: no GPU, rLLM, verl, vLLM or API invocation at module import.
Only these isolated Gate outputs may be mutated; no smoke20/main400 trainer.
"""
from __future__ import annotations

import asyncio
import copy
import importlib.metadata
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

from opensearch_vl_repro.agent.reliability import image_sha256, redact_secrets
from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json, load_smoke_records
from opensearch_vl_repro.rl.group import group_identity, publish_group, read_group, run_lock
from opensearch_vl_repro.rl.live_workflow import LiveRLWorkflowAdapter, ProviderInterruption
from opensearch_vl_repro.rl.rollout_gate import (
    GATE_B_CHECKS, VLLMStaticBackend, validate_rollout_versions,
)
from opensearch_vl_repro.rl.rollout_sync import merge_actor_adapter, validate_actor_adapter
from opensearch_vl_repro.rl.workflow_adapter import build_rllm_workflow
from opensearch_vl_repro.rl.data import safe_image_relpath, question_sha256
from opensearch_vl_repro.rl.workflow_types import RLInfrastructureError
from opensearch_vl_repro.sft_tool_audit import sha256_file

GATE_C_VERSION = "minimum-rl-integration-c-v1"
COLLECT_CHECKS = (
    "software_versions", "formal_sft_lineage", "gate_b_prerequisite", "real_smoke_prompt",
    "no_reference_in_model_task", "runtime_group_identity", "exactly_two_complete_rollouts",
    "same_pre_update_policy", "actual_rllm_tokens", "real_multimodal_inputs",
    "format_reward", "live_accuracy_judge", "live_query_judge", "exact_reward_formula",
    "provider_fail_closed", "vllm_shutdown", "atomic_group_commit",
)
UPDATE_CHECKS = (
    "real_verl_rloo", "fatal_clamp_after_rloo", "finite_advantages", "real_dataproto",
    "response_only_mask", "formal_sft_actor_init", "real_fsdp2", "training_checkpointing_active",
    "real_verl_policy_loss", "loss_finite", "lora_grad_finite", "nonzero_lora_grad",
    "vision_projector_base_frozen", "optimizer_only_lora", "exactly_one_optimizer_step",
    "parameters_finite", "lora_param_changed", "post_policy_fingerprint_changed",
    "native_checkpoint_saved", "peft_exported", "original_actor_destroyed", "fresh_actor_reloaded",
    "native_checkpoint_reloaded", "reload_param_match", "fresh_forward_finite",
)
ALL_CHECKS = COLLECT_CHECKS + UPDATE_CHECKS + ("artifacts_verified", "gate_only_output")


def load_gate_c_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"gate_version": GATE_C_VERSION, "rollout_n": 2, "actor_world_size": 2,
                "optimizer_steps": 1, "learning_rate": 1e-6, "weight_decay": 0.,
                "clip_ratio_low": .2, "clip_ratio_high": .28, "entropy_coeff": 0.}
    if not isinstance(value, dict) or set(value) != set(required) | {"vllm", "agent"}:
        raise ValueError("unknown/missing Gate C settings")
    if any(value[k] != v or isinstance(value[k], bool) for k, v in required.items()):
        raise ValueError("Gate C frozen one-update contract mismatch")
    if value["agent"] != {"max_turns": 16}:
        raise ValueError("Gate C requires 16 max turns")
    inference = value["vllm"]
    expected = {"tensor_parallel_size": 1, "max_model_len": 8192, "max_new_tokens": 512,
                "temperature": .7, "top_p": 1., "top_k": -1, "gpu_memory_utilization": .6}
    if inference != expected:
        raise ValueError("Gate C bounded rollout settings mismatch")
    return value


def validate_gate_b(path, source_fingerprint):
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("passed") is not True or value.get("gate_version") != "actor-rollout-roundtrip-b-v1"
            or value.get("formal_rl_initialization_allowed") is not False
            or (value.get("base_model"), value.get("base_revision")) != (BASE_MODEL, BASE_REVISION)
            or value.get("source_sft_adapter_fingerprint") != source_fingerprint
            or value.get("runtime_image_protocol_version") != RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
            or any(value.get("checks", {}).get(k) is not True for k in GATE_B_CHECKS)):
        raise ValueError("Gate B prerequisite/source lineage PASS missing or inconsistent")
    return {"manifest_sha256": sha256_file(path), "merged_checkpoint_fingerprint": value["merged_checkpoint_fingerprint"],
            "source_sft_adapter_fingerprint": source_fingerprint, "evidence_only_not_rl_initialization": True}


def model_task(row, images):
    if "trajectory_group_id" in row or row.get("prompt_id") != row.get("source_sample_id"):
        raise ValueError("prepared data must contain source identity, not runtime group identity")
    # Explicit allowlist, never a copy of the reward/source record.
    return {"sample_id": row["source_sample_id"], "question": row["question"], "images": images}


def load_source_images(row, root):
    root = root.resolve()
    if row.get("question_hash") != question_sha256(row["question"]):
        raise ValueError("frozen question identity mismatch")
    if len(row["image_relpaths"]) != len(row["image_hashes"]) or not row["image_relpaths"]:
        raise ValueError("source image identities missing")
    images = []
    for name, digest in zip(row["image_relpaths"], row["image_hashes"], strict=True):
        path = (root / safe_image_relpath(name)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError("source image missing/escaping root")
        with Image.open(path) as loaded:
            image = loaded.convert("RGB").copy()
        if image_sha256(image) != digest:
            raise ValueError("source image fingerprint mismatch")
        images.append(image)
    return images


def prepare_context(args, root):
    from opensearch_vl_repro.rl.config import load_rl_config
    from opensearch_vl_repro.sft_train_plan import load_main_config
    from opensearch_vl_repro.rl.actor_gate import load_gate_config, validate_software
    gate = load_gate_c_config(args.gate_config)
    rl = load_rl_config(args.config)
    sft = load_main_config(root / rl["model"]["sft_config"], base_eval_config=root / "configs/eval_base_300.yaml")
    formal_path = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    if (root / rl["model"]["sft_adapter"]).resolve() != formal_path.resolve():
        raise ValueError("Gate C RL init must be original formal checkpoint-3k, NOT Gate A/B/C weights")
    actor = validate_actor_adapter(adapter=formal_path, gate_manifest=None, rl_config=rl,
                                  sft_config=sft, source_sft_adapter=formal_path)
    prerequisite = validate_gate_b(args.gate_b_manifest.resolve(), actor["source_sft_adapter_fingerprint"])
    rl["data"]["quality_audit_dir"] = str(root / rl["data"]["quality_audit_dir"])
    records, manifest = load_smoke_records(args.data.resolve(), rl)
    if not 0 <= args.sample_index < len(records):
        raise ValueError("sample index outside quality-clean smoke records")
    row = records[args.sample_index]
    images = load_source_images(row, args.source_root)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "verl", "vllm", "rllm")}
    a22 = load_gate_config(root / "configs/rl_gate_a22.yaml")
    validate_software(versions, a22)
    validate_rollout_versions(versions)
    if not args.base_model_path.is_dir():
        raise FileNotFoundError("pinned offline base snapshot required")
    # Family/shape check. Revision identity comes from formal lineage; bind the
    # actual supplied offline snapshot files as well (do not trust its directory name).
    base_config = json.loads((args.base_model_path / "config.json").read_text(encoding="utf-8"))
    if base_config.get("model_type") != "qwen3_vl" or base_config.get("text_config", {}).get("num_hidden_layers") != 36:
        raise ValueError("offline snapshot is not Qwen3-VL-4B")
    if base_config.get("_commit_hash") not in {None, BASE_REVISION}:
        raise ValueError("offline snapshot declares a different pinned revision")
    weights = sorted(args.base_model_path.glob("model*.safetensors"))
    if not weights:
        raise ValueError("pinned offline base snapshot weights missing")
    base_files = {path.name: sha256_file(path) for path in
                  sorted(args.base_model_path.iterdir()) if path.is_file() and path.suffix in {".json", ".safetensors"}}
    from opensearch_vl_repro.evaluation.judge import load_judge_config
    from dataclasses import asdict
    judge = load_judge_config(args.judge_config)
    identity = {"run_id": args.run_id, "gate_version": GATE_C_VERSION, "gate_config": gate,
                "formal_rl_initialization_allowed": False,
                "formal_config_sha256": sha256_file(args.config), "sft_config_sha256": sha256_file(root / rl["model"]["sft_config"]),
                "source_sft_actor": actor, "gate_b_prerequisite": prerequisite,
                "sample_id": row["source_sample_id"], "prompt_id": row["prompt_id"],
                "source_data_manifest_sha256": manifest["manifest_sha256"], "software_versions": versions,
                "judge_config": asdict(judge), "search_config_sha256": sha256_file(args.search_config),
                "layout_config_sha256": sha256_file(args.layout_config), "seed": args.seed,
                "integration_source_sha256": {name: sha256_file(root / "src/opensearch_vl_repro/rl" / name)
                    for name in ("gate_c.py", "group.py", "live_workflow.py", "reward_judges.py", "rloo.py",
                                 "training_batch.py", "verl_policy_update.py", "workflow_adapter.py", "rollout_gate.py", "rollout_sync.py")},
                "logprobs_mode": "processed_logprobs",
                "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                "runtime_protocol": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION}
    identity["offline_base_file_sha256"] = base_files
    identity["identity_sha256"] = canonical_json_sha256(identity)
    runtime_sft = copy.deepcopy(sft)
    runtime_sft["model"]["name_or_path"] = str(args.base_model_path.resolve())
    return dict(gate=gate, rl=rl, sft=sft, runtime_sft=runtime_sft, actor=actor, a22=a22,
                row=row, images=images, versions=versions, judge=judge, identity=identity, adapter=formal_path)


def paths_for(root, run_id):
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", run_id) is None:
        raise ValueError("safe explicit Gate C run-id required")
    return root / "outputs/rl_gate_c" / run_id, root / "reports/rl_gate_c" / run_id


def bind_run(output, reports, identity):
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("same run-id identity/config/policy/data changed; use a new run-id")
    else:
        if any(output.iterdir()):
            raise ValueError("unbound nonempty output refused")
        atomic_json(path, identity)


def record_stage(reports, identity, stage, **values):
    value = {"passed": False, "status": "running", "stage": stage, "identity": identity,
             "gate_version": GATE_C_VERSION, "formal_rl_initialization_allowed": False, **values}
    atomic_json(reports / "gate_c_report.json", redact_secrets(value))
    print(f"[Gate C] {stage}", flush=True)
    return value


def failure_report(output, reports, identity, stage, exc, **audit):
    (output / "gate_manifest.json").unlink(missing_ok=True)  # revoke before any other I/O
    interrupted = isinstance(exc, (ProviderInterruption, KeyboardInterrupt))
    value = {"passed": False, "stage": stage, "status": "interrupted" if interrupted else "failed",
             "gate_version": GATE_C_VERSION, "identity": identity, "formal_rl_initialization_allowed": False,
             "interrupt_reason": getattr(exc, "reason", "manual_interrupt" if interrupted else None),
             "error": {"type": type(exc).__name__, "message": str(exc)}, **audit}
    value = redact_secrets(value)
    # Append-only failure history; never erase earlier provider audit.
    atomic_json(reports / "failures" / (uuid.uuid4().hex + ".json"), value)
    atomic_json(reports / "gate_c_report.json", value)


def collect(args, root):
    output, reports = paths_for(root, args.run_id)
    with run_lock(root / "outputs/rl_gate_c" / (args.run_id + ".lock")):
        ctx = prepare_context(args, root)
        bind_run(output, reports, ctx["identity"])
        committed = output / "group"
        if committed.exists():
            value = read_group(committed)
            if value["identity"]["context"] != ctx["identity"]["identity_sha256"]:
                raise ValueError("committed group belongs to another run context")
            print("Complete committed group reused; no rollout recollection", flush=True)
            return 0
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("collect requires exactly one real visible CUDA/BF16 GPU")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        attempt = uuid.uuid4().hex
        group_id = group_identity(prompt_id=ctx["row"]["prompt_id"],
                                 policy_fingerprint=ctx["actor"]["source_sft_adapter_fingerprint"],
                                 rollout_fingerprint=canonical_json_sha256(ctx["gate"]), attempt=attempt,
                                 context=ctx["identity"]["identity_sha256"])
        staging = output / (".group-" + attempt)
        staging.mkdir(exist_ok=False)
        members, backend, adapter, stage = [], None, None, "initialized"
        checks = dict.fromkeys(COLLECT_CHECKS, False)
        started = time.monotonic()
        def enter(name):
            nonlocal stage
            stage = name
            record_stage(reports, ctx["identity"], name, trajectory_group=group_id)
        try:
            from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry
            from opensearch_vl_repro.evaluation.judge import DeepSeekJudge
            from opensearch_vl_repro.rl.reward_judges import bind_trajectory_reward, live_rewards
            registry = create_phase3_tool_registry(search_config=args.search_config, layout_config=args.layout_config,
                                                  cache_dir=args.tool_cache_dir or root / ctx["rl"]["paths"]["tool_cache_dir"])
            adapter = LiveRLWorkflowAdapter(registry)
            task = model_task(ctx["row"], ctx["images"])
            adapter.initialize_episode(**task)
            enter("static_merge_original_checkpoint3k")
            # Every collection attempt has a new private merge, never uses Gate B weights.
            merge = merge_actor_adapter(base_snapshot=args.base_model_path.resolve(), adapter=ctx["adapter"],
                actor=ctx["actor"], sft_config=ctx["sft"], output=output / ("merged-" + attempt), versions=ctx["versions"],
                validation_messages=adapter.build_next_messages(), tools=registry.declarations_for_model(), on_stage=enter)
            enter("rollout_collecting")
            backend = VLLMStaticBackend(checkpoint=output / ("merged-" + attempt), sft_config=ctx["sft"],
                                        gate=ctx["gate"], seed=args.seed, capture_tokens=True)
            from vllm import SamplingParams
            for index in range(2):
                backend.sampling = SamplingParams(temperature=.7, top_p=1., top_k=-1, logprobs=1,
                                                   max_tokens=ctx["gate"]["vllm"]["max_new_tokens"], seed=args.seed + index)
                begin = len(backend.training_inputs)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    workflow, provenance = build_rllm_workflow(adapter=adapter, backend=backend, executor=executor,
                                                              max_turns=16, capture_tokens=True)
                    episode = asyncio.run(workflow.run_with_termination_handling(task=task, uid=f"{group_id['trajectory_group_id']}:{index}"))
                termination = episode.termination_reason.value if episode.termination_reason else "unknown"
                # Upstream converts exceptions into Episode.ERROR: promote original provider exception.
                if adapter.infrastructure_failure is not None:
                    raise adapter.infrastructure_failure
                if termination not in {"env_done", "max_turns_exceeded", "max_response_length_exceeded"}:
                    raise RLInfrastructureError(f"rLLM failed: {termination}: {episode.info}")
                trajectory = adapter.finalize_episode(termination=termination)
                if trajectory.status == "fatal":
                    trajectory.status = "max_agent_turns_exceeded"  # shared format scorer legacy view, not live cutoff
                if termination == "max_response_length_exceeded":
                    trajectory.status = "max_agent_turns_exceeded"
                    trajectory.final_answer = None  # length-limited text is not a completed terminal answer
                if len(episode.trajectories) != 1:
                    raise ValueError("one actual rLLM trajectory per group member required")
                steps = [step.to_dict() for step in episode.trajectories[0].steps]
                actual_inputs = backend.training_inputs[begin:]
                if len(steps) != len(actual_inputs) or len(steps) != len(adapter.state().assistant_outputs):
                    raise ValueError("actual rLLM steps/processor captures misaligned")
                for step_index, (step, inputs) in enumerate(zip(steps, actual_inputs, strict=True)):
                    if inputs["input_ids"][0].tolist() != step["prompt_ids"]:
                        raise ValueError("rLLM actual prompt/vision binding mismatch")
                    name = f"multimodal-{index}-{step_index}.pt"
                    torch.save(inputs, staging / name)
                    step["multimodal_file"] = name
                fatal = adapter.fatal_metadata(termination=termination, step_count=len(steps))
                member = {"rollout_index": index, "identity": group_id, "complete": False,
                          "rllm_episode_identity": episode.id, "rllm_provenance": provenance,
                          "termination": termination, "trajectory": trajectory.to_dict(), "steps": steps,
                          "fatal": fatal, "generated_token_count": sum(len(s["response_ids"]) for s in steps)}
                # Persist raw complete tokens for audit; NOT a committed usable half-group.
                atomic_json(staging / f"rollout-{index}.json", redact_secrets(member))
                enter("rewarding")
                member["reward"] = live_rewards(client=DeepSeekJudge(ctx["judge"]),
                    directory=root / "outputs/rl_gate_c/reward_cache", row=ctx["row"], trajectory=trajectory, fatal=fatal)
                real_trajectory = episode.trajectories[0]
                bind_trajectory_reward(real_trajectory, member["reward"]["total"])
                member["rllm_trajectory_reward"] = real_trajectory.reward
                for actual_step, stored in zip(real_trajectory.steps, steps, strict=True):
                    stored.update(reward=actual_step.reward, mc_return=actual_step.mc_return, info=actual_step.info)
                member["complete"] = True
                members.append(member)
                enter("rollout_collecting" if index == 0 else "group_rewarded")
            enter("vllm_shutdown")
            backend.close()
            backend = None
            checks.update(dict.fromkeys(COLLECT_CHECKS, True))
            enter("group_atomic_publication")
            value = publish_group(staging, committed, {"identity": group_id, "members": redact_secrets(members),
                "checks": checks, "merged_checkpoint_fingerprint": merge["identity"]["merged_checkpoint_fingerprint"],
                "collect_elapsed_seconds": time.monotonic() - started,
                "peak_cuda_memory": {"allocated": torch.cuda.max_memory_allocated(0), "reserved": torch.cuda.max_memory_reserved(0)}})
            record_stage(reports, ctx["identity"], "training_batch_ready", group_sha256=sha256_file(committed / "group.json"),
                         trajectory_group=value["identity"], checks=checks)
            return 0
        except BaseException as exc:
            failure_report(output, reports, ctx["identity"], stage, exc, group_identity=group_id, complete_members=members,
                           partial_trajectory=adapter.finalize_episode(termination="error").to_dict() if adapter else None)
            raise
        finally:
            if backend is not None:
                try:
                    backend.close()
                except BaseException as exc:
                    failure_report(output, reports, ctx["identity"], "vllm_shutdown", exc)
                    raise


def final_passed(checks):
    return all(checks.get(k) is True for k in ALL_CHECKS)


def publish_pass(output, report_path, report, writer=atomic_json):
    if not final_passed(report.get("checks", {})) or report.get("formal_rl_initialization_allowed") is not False:
        raise ValueError("Gate C complete literal checks required")
    marker = output / "gate_manifest.json"
    marker.unlink(missing_ok=True)
    try:
        writer(report_path, {**report, "passed": True, "stage": "complete", "status": "completed"})
        writer(marker, {**report, "passed": True, "stage": "complete", "status": "completed"})  # LAST durable PASS
    except BaseException:
        marker.unlink(missing_ok=True)
        raise


def finalize(args, root):
    output, reports = paths_for(root, args.run_id)
    with run_lock(root / "outputs/rl_gate_c" / (args.run_id + ".lock")):
        ctx = prepare_context(args, root)
        bind_run(output, reports, ctx["identity"])
        try:
            group = read_group(output / "group")
            update = json.loads((output / "update_verified.json").read_text(encoding="utf-8"))
            if (group["identity"]["context"] != ctx["identity"]["identity_sha256"]
                    or update["identity"] != ctx["identity"]
                    or update["group_sha256"] != sha256_file(output / "group/group.json")
                    or update["training_masks_sha256"] != sha256_file(output / "training_masks.json")
                    or len(update["per_rank"]) != 2 or {r["rank"] for r in update["per_rank"]} != {0, 1}):
                raise ValueError("Gate update/group/rank identity mismatch")
            for name, digest in update["checkpoint_file_sha256"].items():
                path = (output / "updated_actor" / name).resolve()
                if not path.is_relative_to((output / "updated_actor").resolve()) or sha256_file(path) != digest:
                    raise ValueError("post-update checkpoint changed")
            checks = {**group["checks"], **update["checks"], "artifacts_verified": True, "gate_only_output": True}
            if any(any(row["checks"].get(k) is not True for k in UPDATE_CHECKS) for row in update["per_rank"]):
                raise ValueError("not all actor ranks passed")
            report = {**update, "gate_version": GATE_C_VERSION, "checks": checks, "formal_rl_initialization_allowed": False,
                      "software_versions": ctx["identity"]["software_versions"],
                      "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
                      "source_sft_adapter_fingerprint": ctx["actor"]["source_sft_adapter_fingerprint"],
                      "gate_b_prerequisite_identity": ctx["identity"]["gate_b_prerequisite"],
                      "sample_id": ctx["identity"]["sample_id"], "prompt_id": ctx["identity"]["prompt_id"],
                      "trajectory_group_id": group["identity"]["trajectory_group_id"],
                      "collection_attempt": group["identity"]["collection_attempt"], "rollout_n": 2,
                      "actor_world_size": 2, "optimizer_step_count": 1,
                      "loss": [r["metrics"]["actor/pg_loss"] for r in update["per_rank"]],
                      "grad_stats": [r["gradient_audit"] for r in update["per_rank"]],
                      "fatal_summary": [m["fatal"] for m in group["members"]],
                      "trajectory_group": group["identity"], "trajectory_summaries": [{k: m[k] for k in
                        ("rollout_index", "rllm_episode_identity", "termination", "fatal", "generated_token_count", "reward")} for m in group["members"]],
                      "collect_peak_cuda_memory": group["peak_cuda_memory"],
                      "collect_elapsed_seconds": group["collect_elapsed_seconds"]}
            publish_pass(output, reports / "gate_c_report.json", redact_secrets(report))
            print(f"Gate C PASS: {reports / 'gate_c_report.json'}", flush=True)
            return 0
        except BaseException as exc:
            failure_report(output, reports, ctx["identity"], "finalize", exc)
            raise
