"""Fixed two-window diagnostic wrapper around S1/S2 production primitives.

No rollout, scheduler, retry, reward provider or trainer coordinator. Heavy
dependencies are lazy. CPU orchestration fixtures can never publish GPU PASS.
"""
from __future__ import annotations

import copy
import gc
import importlib.metadata
import json
import math
import os
import re
import subprocess
import uuid
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from .actor_gate import BASE_MODEL, BASE_REVISION, atomic_json, validate_output_paths

VERSION = "formal-s2-gpu-validation-v1"
SNAPSHOT_SHA256 = "5c655eb7bd80fb959428f2194acb217424d0e658dfd54ff1eb27f30bcedc236b"
WORLD_SIZE = 2
WINDOW_COUNT = 2
GROUPS_PER_WINDOW = 2
ROLLOUT_N = 2
MEMORY_PHASES = ("actor_load", "before_O", "after_OC", "after_update", "after_save", "after_fresh_reload")
STATE_FIELDS = ("parameter_sha256", "optimizer_state_sha256", "native_rng_sha256", "global_optimizer_step")


def require_launcher(environ):
    world, rank, local = (int(environ.get(k, "-1")) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if world != WORLD_SIZE or rank not in (0, 1) or local not in (0, 1):
        raise ValueError("S2 validation requires torchrun WORLD_SIZE == 2 and local ranks 0/1")
    return rank, local


def validation_paths(root, run_id, protected):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("safe, nonempty diagnostic run-id required")
    root = Path(root).resolve()
    output = root / "outputs/rl_formal_s2_validation" / run_id
    reports = root / "reports/rl_formal_s2_validation" / run_id
    # No redirection through symlink/junction ancestors, including output roots.
    for target in (output, reports):
        for ancestor in (target, *target.parents):
            if ancestor == root:
                break
            if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
                raise ValueError("diagnostic output ancestors may not redirect writes")
        if not target.resolve().is_relative_to(root):
            raise ValueError("diagnostic output escapes workspace")
    validate_output_paths(output, reports, list(protected) + [root / "data", root / "configs",
        root / "outputs/rl_gate_c", root / "outputs/rl_smoke", root / "outputs/rl_main"])
    return output, reports


def require_snapshot_sha(files):
    actual = canonical_json_sha256(files)
    if actual != SNAPSHOT_SHA256:
        raise ValueError(f"offline snapshot SHA mismatch: expected {SNAPSHOT_SHA256}, got {actual}")
    return actual


def source_identity(row):
    return canonical_json_sha256({k: row[k] for k in (
        "source_sample_id", "prompt_id", "question_hash", "image_hashes", "image_relpaths")})


def require_continuation(saved, receipt, *, optimizer_nonempty, moment_step):
    if not optimizer_nonempty or moment_step != 1:
        raise ValueError("checkpoint1 continuation requires nonempty AdamW moments at step 1")
    if saved.get("global_optimizer_step") != 1 or receipt.get("global_optimizer_step") != 1:
        raise ValueError("checkpoint1 continuation requires global step 1")
    if any(saved.get(k) is None or saved[k] != receipt.get(k) for k in STATE_FIELDS):
        raise ValueError("checkpoint1 native model/optimizer/RNG continuation mismatch")
    if any(receipt.get(k) is not True for k in (
        "native_identity_match", "optimizer_identity_match", "rng_identity_match", "global_step_match")):
        raise ValueError("actual S2 continuation receipt required")
    return dict(saved={k: saved[k] for k in STATE_FIELDS},
                reloaded={k: receipt[k] for k in STATE_FIELDS},
                optimizer_nonempty=True, moment_step=1, passed=True)


def require_checkpoint_result(checkpoint, step):
    if (checkpoint.get("policy_iteration") != step or checkpoint.get("global_optimizer_step") != step
            or checkpoint.get("eligibility", {}).get("eligible_for_main_init") is not False):
        raise ValueError("diagnostic checkpoint iteration/step/eligibility mismatch")


def publish_result(output, reports, report):
    """Report first; final PASS manifest last. Revoke manifest on publication error."""
    report_path, manifest_path = reports / "report.json", output / "manifest.json"
    if report.get("passed") is True and (report.get("scope") != "runtime"
            or not report.get("checks") or not all(v is True for v in report["checks"].values())):
        raise ValueError("CPU/incomplete evidence cannot publish GPU PASS")
    try:
        atomic_json(report_path, report)
        atomic_json(manifest_path, report)
    except BaseException:
        # Removing only this wrapper's PASS marker is safer than leaving a stale
        # PASS when an injected write fails *after* atomic replace. Keep report,
        # native checkpoints and attempts for forensics, never auto-resume.
        manifest_path.unlink(missing_ok=True)
        raise


def failure_result(output, reports, *, stage, error, evidence=None):
    failed = dict(version=VERSION, passed=False, stage=stage, error=str(error),
        diagnostic_mode=True, rollout_executed=False, resume_allowed=False,
        requires_new_run_id=True, eligible_for_main_init=False, evidence=evidence or {})
    # Revoke first even if writing the false report subsequently fails.
    (output / "manifest.json").unlink(missing_ok=True)
    atomic_json(reports / "report.json", failed)
    atomic_json(output / "manifest.json", failed)
    return failed


def tensor_devices(value):
    """JSON-safe recursive evidence; never keep live tensor/actor references."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        return dict(device=str(value.device), shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {str(k): tensor_devices(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [tensor_devices(v) for v in value]
    return None


def diagnostic_group(root, run, policy, row, processor, source_root, *, continuation=False, cpu_fixture=False):
    """Real source PIL/processor inputs + explicit deterministic diagnostic output.

    The legacy token_origin wire field is required by unchanged group schema;
    actual_token_origin/diagnostic_fixture are the authoritative diagnostic
    provenance. It must NOT be read as evidence of a vLLM rollout.
    """
    import torch
    from PIL import Image
    from .actor_gate import temporary_sft_record
    from .group import formal_group_identity, publish_formal_group
    from .rollout_sync import prepare_qwen_vl_processor_inputs
    record = temporary_sft_record(row, source_root)
    images = []
    for filename in record["images"]:
        with Image.open(filename) as image:
            images.append(image.convert("RGB").copy())
    messages = [dict(role="user", content=[
        *[dict(type="image", image=image) for image in images],
        dict(type="text", text=row["question"])])]
    tokenized = []
    for texts in (messages, messages + [dict(role="assistant", content="Yes."),
            dict(role="user", content="Diagnostic follow-up: reply briefly.")]):
        _, _, inputs = prepare_qwen_vl_processor_inputs(processor, texts, [])
        inputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if not {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"} <= inputs.keys():
            raise ValueError("real processor multimodal fields missing")
        if inputs["input_ids"].shape[0] != 1 or not inputs["pixel_values"].numel():
            raise ValueError("nonempty single-prompt processor vision required")
        tokenized.append(inputs)
        if not continuation:
            break
    prompt_id = row["prompt_id"]
    attempt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run['run_identity_sha256']}:{policy['policy_iteration']}:{prompt_id}"))
    identity = formal_group_identity(run, policy, prompt_id=prompt_id, source_identity=source_identity(row),
                                    attempt_id=attempt_id, attempt_index=0)
    gid = identity["trajectory_group_id"]
    staging = root / "groups" / f".diagnostic-{gid}"
    staging.mkdir(parents=True, exist_ok=False)
    members = []
    for rollout in range(ROLLOUT_N):
        steps = []
        for step_index, inputs in enumerate(tokenized if rollout == 0 else tokenized[:1]):
            completion = "Proceed." if step_index else ("Yes." if rollout == 0 else "No.")
            ids = processor.tokenizer.encode(completion, add_special_tokens=False)
            eos = processor.tokenizer.eos_token_id
            if type(eos) is not int or not ids:
                raise ValueError("diagnostic completion needs actual nonempty tokenizer IDs and EOS")
            response, prompt = list(ids) + [eos], inputs["input_ids"][0].tolist()
            mm_file = f"multimodal-{rollout}-{step_index}.pt"
            torch.save(inputs, staging / mm_file)
            logprobs = [-1.] * len(response)
            steps.append(dict(prompt_ids=prompt, response_ids=response, logprobs=logprobs,
                model_output=dict(prompt_ids=prompt, completion_ids=response, logprobs=logprobs),
                info=dict(token_origin="vllm.RequestOutput", logprobs_mode="processed_logprobs",
                    actual_token_origin="deterministic_processor_encoded_completion",
                    diagnostic_fixture=True, rollout_executed=False), multimodal_file=mm_file))
        accuracy = 1. if rollout == 0 else 0.
        member = dict(identity=identity, rollout_index=rollout, member_id=f"{gid}:{rollout}",
            complete=True, fatal=False, steps=steps, trajectory_file=f"trajectory-{rollout}.json",
            reward=dict(format=1., accuracy=accuracy, query=.5, total=.8 * accuracy + .1),
            reward_origin="deterministic_diagnostic_fixture", diagnostic_fixture=True, rollout_executed=False)
        atomic_json(staging / member["trajectory_file"], member)
        members.append(member)
    return publish_formal_group(staging, root / "groups" / gid,
        dict(identity=identity, members=members, diagnostic_mode=True), cpu_fixture=cpu_fixture)


@contextmanager
def observe_residency(actor, data, evidence, sample_memory):
    """Existing S2 materializer invokes this observer AFTER active-only CUDA transfer."""
    import numpy as np
    import torch
    from .training_batch import assert_cpu_multimodal
    assert_cpu_multimodal(data.non_tensor_batch)
    evidence["before"] = tensor_devices(data.non_tensor_batch)
    forwards, computes = actor._forward_micro_batch, actor.compute_log_prob
    evidence["active_microbatches"] = []
    calls = 0
    def forward(*args, **kwargs):
        micro = kwargs.get("micro_batch", args[0] if args else None)
        modalities = micro["multi_modal_inputs"]
        devices = tensor_devices(modalities)
        # At this point materialize_actor_microbatches has moved ONLY this row.
        def tensors(value):
            if isinstance(value, torch.Tensor):
                return [value]
            if isinstance(value, dict):
                return [t for v in value.values() for t in tensors(v)]
            if isinstance(value, (list, tuple, np.ndarray)):
                return [t for v in value for t in tensors(v)]
            return []
        entries = [modalities] if isinstance(modalities, dict) else list(modalities)
        if len(entries) != 1 or not isinstance(entries[0], dict) or not {"pixel_values", "image_grid_thw"} <= entries[0].keys():
            raise ValueError("active microbatch vision/grid missing")
        if any(t.device.type != "cuda" for t in tensors(modalities)) or micro["responses"].shape[0] != 1:
            raise ValueError("active microbatch must be single-row CUDA multimodal")
        evidence["active_microbatches"].append(devices)
        result = forwards(*args, **kwargs)
        assert_cpu_multimodal(data.non_tensor_batch)
        return result
    def compute(*args, **kwargs):
        nonlocal calls
        result = computes(*args, **kwargs)
        calls += 1
        if calls == 2:
            sample_memory("after_OC")
        return result
    actor._forward_micro_batch, actor.compute_log_prob = forward, compute
    try:
        yield
        if calls != 2 or not evidence["active_microbatches"]:
            raise ValueError("two real independent O/C and CUDA micro-forwards required")
        assert_cpu_multimodal(data.non_tensor_batch)
        evidence["after"] = tensor_devices(data.non_tensor_batch)
        evidence["passed"] = evidence["before"] == evidence["after"]
        if not evidence["passed"]:
            raise ValueError("original CPU multimodal carrier changed")
    finally:
        actor._forward_micro_batch, actor.compute_log_prob = forwards, computes


def full_lora_fingerprint(actor):
    """All-gather each LoRA DTensor, hash FULL tensors, not unequal local shards."""
    import torch
    from .old_logprob import fingerprint
    hashes = {}
    for name, parameter in sorted(actor.actor_module.named_parameters()):
        if not parameter.requires_grad:
            continue
        if "lora_" not in name:
            raise ValueError("only LoRA may be trainable")
        tensor = parameter.detach()
        tensor = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("nonfinite post-step LoRA tensor")
        hashes[name] = fingerprint(tensor.cpu())
    if not hashes:
        raise ValueError("no trainable LoRA tensors")
    return canonical_json_sha256(hashes)


def actor_contract(actor, canonical):
    from verl.utils.fsdp_utils import fsdp_version
    from .actor_gate import trainable_policy
    from .rl_actor_semantics import require_rl_lora_dropout_runtime
    model = actor.actor_module
    stacks = [m for n, m in model.named_modules() if n.endswith("language_model") and hasattr(m, "layers")]
    if len(stacks) != 1:
        raise ValueError("one real language decoder stack required")
    stack = stacks[0]
    dtypes = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
    training = sum(bool(layer.training) for layer in stack.layers)
    checkpointing = sum(bool(layer.gradient_checkpointing) for layer in stack.layers)
    if (not model.training or not stack.training or len(stack.layers) != 36 or training != 36 or checkpointing != 36
            or any(fsdp_version(layer) != 2 for layer in stack.layers)
            or model.config._attn_implementation != "flash_attention_2"
            or stack.config._attn_implementation != "flash_attention_2"
            or actor.config.strategy != "fsdp2" or actor.config.fsdp_config.dtype != "bfloat16"
            or not dtypes.get("torch.bfloat16")):
        raise ValueError("FSDP2/FA2/36-layer train/checkpoint contract failed")
    return dict(passed=True, fsdp2=True, bf16=True, parameter_dtypes=dtypes,
        model_training=bool(model.training), language_model_training=bool(stack.training),
        decoder_layers=36, decoder_layers_training=training,
        decoder_layers_gradient_checkpointing=checkpointing, effective_attention_implementation="flash_attention_2",
        parameter_audit=trainable_policy(model, actor.actor_optimizer),
        dropout=require_rl_lora_dropout_runtime(model, canonical))


def strict_oc_summary(alignment, *, world_size=2):
    from .policy_alignment import formal_alignment_artifact
    try:
        derived = formal_alignment_artifact(alignment["per_rank"], world_size=world_size,
                                             window_sha256=alignment["window_sha256"])
    except (ValueError, KeyError, TypeError):
        return False
    return alignment == derived and derived["passed"] is True


def aggregate_report(run, ranks, *, scope):
    """Require complete evidence; JSON fixture claims never authorize GPU PASS."""
    if len(ranks) != 2 or {r.get("rank") for r in ranks} != {0, 1}:
        raise ValueError("complete evidence from exactly ranks 0 and 1 required")
    ranks = sorted(ranks, key=lambda r: r["rank"])
    if any(len(r.get("windows", [])) != 2 for r in ranks):
        raise ValueError("exactly two windows per rank required")
    checks = dict(world_size_2=run["semantics"]["world_size"] == 2,
        diagnostic_identity=run["semantics"].get("diagnostic_version") == VERSION,
        runtime_scope=scope == "runtime" and all(r.get("scope") == "runtime" for r in ranks),
        no_rollout_or_providers=all(r.get("rollout_executed") is False for r in ranks))
    for iteration in range(2):
        rows = [r["windows"][iteration] for r in ranks]
        for rank, row in enumerate(rows):
            require_checkpoint_result(row["checkpoint"], iteration + 1)
            if row["update"]["before_step"] != iteration or row["update"]["after_step"] != iteration + 1:
                raise ValueError("PASS requires optimizer step sequence [1, 2]")
            plan = row["rank_plan"]
            expected_count = 4 if iteration == 0 else 5
            factor = 1 if iteration == 0 else 2
            if (plan["logical_row_count"] != expected_count or plan["physical_row_count"] != expected_count * factor
                    or plan["replication_factor"] != factor or plan["local_row_count"] != expected_count * factor // 2
                    or any(len(a) != plan["local_row_count"] for a in plan["rank_assignment"])):
                raise ValueError("uniform fixed diagnostic rank plan mismatch")
            from .training_window import deterministic_rank_plan
            ids = list(row["deterministic_plan"]["multiplicity"])
            expected_plan = deterministic_rank_plan(ids, 2)
            if row["deterministic_plan"] != expected_plan or plan["rank"] != rank:
                raise ValueError("non-deterministic rank plan/rank")
            if plan["rank_assignment"] != expected_plan["rank_assignments"]:
                raise ValueError("nonuniform rank assignment")
            saved = row["rng_optimizer_save"]
            reloaded = [r for r in row["reload"]["per_rank"] if r["rank"] == plan["rank"]]
            if len(reloaded) != 1 or any(saved.get(k) is None or saved[k] != reloaded[0]["state"].get(k) for k in STATE_FIELDS):
                raise ValueError("saved native model/optimizer/RNG differs from fresh reload")
            losses = row["update"]["metrics"].get("actor/pg_loss", [])
            if not losses or not all(math.isfinite(float(loss)) for loss in losses):
                raise ValueError("finite real PPO loss evidence required")
        if rows[0]["full_lora_sha256"] != rows[1]["full_lora_sha256"]:
            raise ValueError("post-step FULL LoRA fingerprints differ across ranks")
        if rows[0]["checkpoint"]["checkpoint_manifest_sha256"] != rows[1]["checkpoint"]["checkpoint_manifest_sha256"]:
            raise ValueError("ranks disagree on verified checkpoint")
        checks[f"window_{iteration}_passed"] = all(
            row["actor_contract"].get("passed") is True
            and all(row["actor_contract"].get(k) is True for k in (
                "fsdp2", "bf16", "model_training", "language_model_training"))
            and all(row["actor_contract"].get(k) == 36 for k in (
                "decoder_layers", "decoder_layers_training", "decoder_layers_gradient_checkpointing"))
            and row["actor_contract"].get("effective_attention_implementation") == "flash_attention_2"
            and row["actor_contract"]["dropout"].get("source_adapter_lora_dropout") == .05
            and row["actor_contract"]["dropout"].get("runtime_effective_lora_dropout") == 0.
            and strict_oc_summary(row["update"]["alignment"])
            and row["update"]["rollout_actor_handoff"].get("ratio_finite") is True
            and all(row["update"]["update_audit"].get(k) is True for k in (
                "lora_grad_finite", "nonzero_lora_grad", "vision_projector_base_frozen"))
            and row["update"]["update_audit"].get("optimizer_step_count") == 1
            and row["residency"].get("passed") is True and bool(row["residency"].get("active_microbatches"))
            and {m["phase"] for m in row["memory"]} >= set(MEMORY_PHASES)
            and all(row["reload"].get(k) is True for k in (
                "original_actor_destroyed", "fresh_multimodal_forward_finite", "adapter_reloaded",
                "native_reloaded", "optimizer_reloaded", "rng_reloaded", "execution_contract_verified"))
            for row in rows)
    checks["optimizer_rng_continuation"] = all(r["windows"][1]["continuation"].get("passed") is True for r in ranks)
    return dict(version=VERSION, scope=scope, diagnostic_mode=True, rollout_executed=False, run_identity=run,
        world_size=2, windows_completed=2, optimizer_steps=[1, 2], per_rank=ranks, checks=checks,
        passed=all(checks.values()), eligible_for_main_init=False, resume_allowed=False,
        requires_new_run_id_on_failure=True,
        legacy_token_origin_note="vllm.RequestOutput is schema compatibility ONLY; no rollout executed")


def execute_two_windows(session, *, cpu_fixture=False):
    """Fixed validation sequence, not a general/resumable trainer FSM.

    Mock sessions are accepted ONLY for non-PASS CPU orchestration tests.
    """
    if not cpu_fixture and type(session) is not RuntimeSession:
        raise ValueError("GPU validation requires the production runtime session")
    session.load_initial()
    for iteration in (0, 1):
        if iteration == 1:
            session.load_continuation()
        session.validate_window(iteration)
    return session.finish(scope="cpu_fixture" if cpu_fixture else "runtime")


class RuntimeSession:
    """Lifetime owner: no old actor references may survive staging fresh reload."""
    def __init__(self, *, torch, args, root, output, reports, mesh, rank, device):
        self.torch, self.args, self.root = torch, args, root
        self.output, self.reports, self.mesh, self.rank, self.device = output, reports, mesh, rank, device
        self.stage, self.windows, self.loaded = "inputs", [], None
        self.rank_report = dict(rank=rank, scope="runtime", rollout_executed=False,
            gpu=torch.cuda.get_device_name(device), windows=self.windows)

    def collect(self, value):
        rows = [None, None]
        self.torch.distributed.all_gather_object(rows, value)
        return rows

    def boundary(self, stage, operation):
        self.stage = stage
        # Rank-local false progress is durable even when NCCL/process death
        # prevents aggregation. No cleanup collectives on exception paths.
        error, value = None, None
        try:
            atomic_json(self.reports / f"rank-{self.rank}.json", dict(passed=False, stage=stage,
                rank=self.rank, windows=self.windows, resume_allowed=False, diagnostic_mode=True))
            value = operation()
            self.torch.cuda.synchronize(self.device)
        except Exception as exc:
            error = dict(rank=self.rank, stage=stage, error=f"{type(exc).__name__}: {exc}")
        errors = self.collect(error)
        if any(e is not None for e in errors):
            raise RuntimeError(f"S2 validation failed at {stage}: {errors}")
        return value

    def memory(self, phase, rows):
        from .verl_actor_gate import memory_stats
        self.torch.cuda.synchronize(self.device)
        value = dict(phase=phase, rank=self.rank, **memory_stats(self.torch, self.device))
        rows.append(value)
        print("[S2 VALIDATION MEMORY] " + json.dumps(value), flush=True)

    def prepare(self):
        from .config import load_rl_config
        from .actor_gate import load_smoke_records
        from .checkpoint import build_rl_lineage, build_training_run_identity
        from .offline_snapshot import offline_snapshot_files
        from .rl_actor_semantics import execution_contract
        from opensearch_vl_repro.sft_train_plan import load_main_config
        from opensearch_vl_repro.sft_tool_audit import sha256_file
        from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
        from opensearch_vl_repro.model import load_processor
        config = load_rl_config(self.args.config)
        config["data"]["quality_audit_dir"] = str(self.root / config["data"]["quality_audit_dir"])
        records, manifest = load_smoke_records(self.args.data, config)
        start = self.args.prompt_start
        if type(start) is not int or start < 0 or start + 4 > len(records):
            raise ValueError("prompt-start must select four distinct frozen smoke prompts")
        self.records = records[start:start + 4]
        canonical = load_main_config(self.root / config["model"]["sft_config"],
                                     base_eval_config=self.root / "configs/eval_base_300.yaml")
        self.canonical, self.runtime = canonical, copy.deepcopy(canonical)
        self.runtime["model"]["name_or_path"] = str(self.args.base_model_path.resolve())
        files = offline_snapshot_files(self.args.base_model_path, revision=BASE_REVISION, strict=True)
        snapshot_sha = require_snapshot_sha(files)
        if self.args.sft_adapter.resolve() != (self.root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter").resolve():
            raise ValueError("validation must use original formal checkpoint-3k adapter, not a Gate artifact")
        lineage = build_rl_lineage(config=config, sft_config=canonical, adapter_path=self.args.sft_adapter,
                                    run_id=self.args.run_id)
        sources = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        sources["validate_rl_formal_s2.py"] = sha256_file(self.root / "scripts/validate_rl_formal_s2.py")
        semantics = dict(dataset=dict(sha256=manifest["samples_sha256"], split="diagnostic_smoke_source",
                manifest_sha256=manifest["manifest_sha256"], config_sha256=sha256_file(self.args.config)),
            base_model=dict(name=BASE_MODEL, revision=BASE_REVISION, offline_snapshot_sha256=snapshot_sha),
            source_sft=dict(adapter_sha256=lineage.sft_adapter_fingerprint,
                metadata_sha256=lineage.sft_checkpoint_metadata_fingerprint, stage=lineage.sft_stage,
                lineage=list(lineage.sft_lineage)), execution_contract=execution_contract(canonical),
            rollout=dict(behavior_version=VERSION, config=dict(temperature=.7,
                diagnostic_fixture=True, rollout_executed=False, completion_recipe="yes-no-proceed-eos-v1",
                rollout_logprob=-1., continuation_extra_row_window=1)), rollout_n=2,
            groups_per_window=2, require_complete_windows=True, weighting="per_generation_row_mean_v1",
            optimizer=dict(name="AdamW", learning_rate=1e-6, weight_decay=0.),
            ppo=dict(epochs=1, microbatch=1, clip_ratio_low=.2, clip_ratio_high=.28,
                entropy=0., loss_mode="vanilla"), world_size=2,
            reward=dict(version="deterministic-diagnostic-v1", semantics="format*(.8*accuracy+.2*query)",
                fixture_accuracy=[1., 0.], fixture_query=.5, fixture_format=1.),
            tool_protocol_version=RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
            image_protocol_version="runtime-image-id-grounding-v3", integration_source_hashes=sources,
            diagnostic_version=VERSION, initial_seed=config["data"]["seed"])
        self.run = build_training_run_identity(self.args.run_id, semantics=semantics,
            prompt_ids=[r["prompt_id"] for r in self.records],
            prompt_sources=[dict(prompt_id=r["prompt_id"], source_identity=source_identity(r)) for r in self.records],
            locators=dict(base_snapshot=str(self.args.base_model_path.resolve()), source_root=str(self.args.source_root.resolve())))
        self.processor = load_processor(self.runtime, local_files_only=True)
        self.rank_report["software"] = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "verl", "flash-attn")}
        for name, pinned in dict(transformers="4.57.1", peft="0.21.1", verl="0.6.1").items():
            if self.rank_report["software"][name] != pinned:
                raise ValueError(f"validation requires {name}=={pinned}")
        self.rank_report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()

    def load_initial(self):
        from .formal_policy_update import load_formal_actor
        from .checkpoint import initialize_formal_run
        self.boundary("inputs", self.prepare)
        if len(set(self.collect(self.run["run_identity_sha256"]))) != 1:
            raise ValueError("rank run identities differ")
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.loaded = self.boundary("initial_actor_load", lambda: load_formal_actor(self.run,
            canonical_config=self.canonical, runtime_config=self.runtime, source_adapter=self.args.sft_adapter,
            processor=self.processor, mesh=self.mesh, initial_seed=self.run["semantics"]["initial_seed"]))
        self.boundary("initialize_s1", lambda: initialize_formal_run(self.output, self.run, self.loaded.policy)
                      if self.rank == 0 else None)

    def load_continuation(self):
        from .formal_policy_update import load_formal_actor, optimizer_step_counter
        from .run_state import checkpoint_policy
        from .checkpoint import read_verified_checkpoint
        directory = self.output / "checkpoints/policy-000001"
        manifest = self.boundary("read_checkpoint1", lambda: read_verified_checkpoint(directory))
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.loaded = self.boundary("native_continuation_load", lambda: load_formal_actor(self.run,
            canonical_config=self.canonical, runtime_config=self.runtime, source_adapter=self.args.sft_adapter,
            processor=self.processor, mesh=self.mesh, policy=checkpoint_policy(manifest), checkpoint_directory=directory))
        def check():
            saved = json.loads((directory / f"runtime_state_rank_{self.rank}.json").read_text(encoding="utf-8"))
            actor = self.loaded.actor
            return require_continuation(saved, self.loaded.reload_receipt.artifact,
                optimizer_nonempty=bool(actor.actor_optimizer.state), moment_step=optimizer_step_counter(actor, 1))
        self.continuation = self.boundary("continuation_identity", check)

    def validate_window(self, iteration):
        from .checkpoint import build_checkpoint_manifest, commit_verified_checkpoint, read_verified_checkpoint
        from .formal_policy_update import update_formal_window, save_formal_staging, fresh_reload_staging
        from .group import read_formal_group
        from .training_window import build_training_window, deterministic_rank_plan
        from .training_batch import formal_training_rows, build_rank_local_dataproto
        from .rloo import assemble_window_rloo
        from .run_state import new_update_attempt
        policy = self.loaded.policy
        memory, residency = [], {}
        self.memory("actor_load", memory)
        contract = self.boundary(f"window{iteration}_actor_contract", lambda: actor_contract(self.loaded.actor, self.canonical))
        rows = self.records[iteration * 2:iteration * 2 + 2]
        def publish():
            if self.rank == 0:
                return [diagnostic_group(self.output, self.run, policy, row, self.processor, self.args.source_root,
                    continuation=iteration == 1 and index == 0) for index, row in enumerate(rows)]
        groups = self.boundary(f"window{iteration}_groups", publish)
        groups = self.collect(groups)[0]
        directories = {g["identity"]["trajectory_group_id"]: self.output / "groups" / g["identity"]["trajectory_group_id"] for g in groups}
        groups = self.boundary(f"window{iteration}_read_groups", lambda: [read_formal_group(directories[g["identity"]["trajectory_group_id"]]) for g in groups])
        window = build_training_window(self.run, policy, groups, window_id=f"diagnostic-{iteration}")
        reward = self.boundary(f"window{iteration}_official_rloo", lambda: assemble_window_rloo(window, self.run, policy, groups))
        logical, _ = formal_training_rows(window, self.run, policy, groups, reward)
        plan = deterministic_rank_plan([r["logical_row_id"] for r in logical], 2)
        if plan["logical_count"] != (4 if iteration == 0 else 5):
            raise ValueError("fixed diagnostic row count changed")
        data, receipt = self.boundary(f"window{iteration}_batch", lambda: build_rank_local_dataproto(
            window, self.run, policy, groups, reward, group_directories=directories, rank=self.rank,
            model=self.loaded.actor.actor_module, pad_id=self.processor.tokenizer.pad_token_id, temperature=.7))
        before_sha = self.boundary(f"window{iteration}_pre_lora", lambda: full_lora_fingerprint(self.loaded.actor))
        attempt = self.collect(new_update_attempt(window) if self.rank == 0 else None)[0]
        self.memory("before_O", memory)
        def update():
            with observe_residency(self.loaded.actor, data, residency, lambda phase: self.memory(phase, memory)):
                return update_formal_window(self.loaded, data, receipt, root=self.output, run=self.run,
                    groups=groups, window=window, reward_window=reward, attempt=attempt)
        updated = self.boundary(f"window{iteration}_OC_and_update", update)
        self.memory("after_update", memory)
        full_sha = self.boundary(f"window{iteration}_post_lora", lambda: full_lora_fingerprint(self.loaded.actor))
        if len(set(self.collect(full_sha))) != 1 or full_sha == before_sha:
            raise ValueError("full LoRA weights did not change consistently across ranks")
        staging = self.output / "checkpoints" / f".validation-{iteration + 1}"
        description = self.boundary(f"window{iteration}_save", lambda: save_formal_staging(self.loaded,
            self.processor, staging, window=window, update_evidence=updated))
        self.memory("after_save", memory)
        saved = self.boundary(f"window{iteration}_saved_state", lambda: json.loads(
            (staging / f"runtime_state_rank_{self.rank}.json").read_text(encoding="utf-8")))
        fresh, reload = self.boundary(f"window{iteration}_fresh_reload", lambda: fresh_reload_staging(self.loaded,
            data, staging, description, canonical_config=self.canonical, runtime_config=self.runtime,
            processor=self.processor, mesh=self.mesh))
        self.memory("after_fresh_reload", memory)
        # Only receipt/data dictionaries remain: next iteration MUST construct
        # again from immutable checkpoint1, not reuse the verification actor.
        del fresh
        gc.collect()
        self.torch.cuda.empty_cache()
        manifest = self.boundary(f"window{iteration}_checkpoint_manifest", lambda: build_checkpoint_manifest(self.run, policy, groups, window, updated["attempt"], reward,
            artifact_role_files=description["artifact_role_files"], file_sha256=description["file_sha256"],
            kind="smoke_continuation", reload_evidence=reload))
        self.boundary(f"window{iteration}_commit", lambda: commit_verified_checkpoint(self.output, staging, manifest)
                      if self.rank == 0 else None)
        checkpoint = self.boundary(f"window{iteration}_verify_commit", lambda: read_verified_checkpoint(
            self.output / "checkpoints" / f"policy-{iteration + 1:06d}"))
        require_checkpoint_result(checkpoint, iteration + 1)
        evidence = dict(iteration=iteration, rank_plan=receipt.artifact, deterministic_plan=plan,
            rloo=reward, update=updated, actor_contract=contract, full_lora_sha256=full_sha,
            previous_full_lora_sha256=before_sha, residency=residency, memory=memory,
            reload=reload, rng_optimizer_save=saved,
            reload_receipt_before_update=self.loaded.reload_receipt.artifact,
            continuation=self.continuation if iteration else None, checkpoint=checkpoint)
        self.windows.append(evidence)
        atomic_json(self.reports / f"rank-{self.rank}.json", dict(passed=False, rank=self.rank, windows=self.windows,
            stage=f"window{iteration}_verified", diagnostic_mode=True, resume_allowed=False))
        self.loaded = None
        del data
        gc.collect()
        self.torch.cuda.empty_cache()

    def finish(self, *, scope):
        ranks = self.collect(self.rank_report)
        report = aggregate_report(self.run, ranks, scope=scope)
        if not report["passed"]:
            raise ValueError(f"validation checks failed: {report['checks']}")
        return report


def run_validation(args, root):
    rank, device = require_launcher(os.environ)  # BEFORE any CUDA/model construction.
    root = Path(root).resolve()
    output, reports = validation_paths(root, args.run_id, [args.sft_adapter.parent, args.base_model_path, args.source_root])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ValueError("exactly two visible CUDA GPUs required")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 GPU support required")
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    session, reserved = None, False
    try:
        # Rendezvous before reserving; both ranks checked the SAME absent paths.
        dist.barrier()
        error = None
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=False)
                reserved = True
                reports.mkdir(parents=True, exist_ok=False)
                failure_result(output, reports, stage="started", error="validation incomplete")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = [None, None]
        dist.all_gather_object(errors, error)
        if any(e is not None for e in errors):
            raise RuntimeError(f"validation reservation failed: {errors}")
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
        session = RuntimeSession(torch=torch, args=args, root=root, output=output, reports=reports,
                                 mesh=mesh, rank=rank, device=device)
        report = execute_two_windows(session)
        dist.barrier()
        dist.destroy_process_group()
        # No actor/collective work follows PASS publication.
        if rank == 0:
            publish_result(output, reports, report)
            print("FORMAL S2 DIAGNOSTIC GPU VALIDATION PASSED (not Smoke20/main RL)", flush=True)
        return 0
    except BaseException as exc:
        stage = session.stage if session else "initialization"
        print(f"[S2 VALIDATION FAIL] rank={rank} stage={stage}: {type(exc).__name__}: {exc}; use a NEW run-id", flush=True)
        if rank == 0 and reserved:
            failure_result(output, reports, stage=stage, error=f"{type(exc).__name__}: {exc}",
                           evidence=dict(windows=session.windows if session else []))
        # No barrier/teardown collective here: torchrun terminates failed peers.
        raise
