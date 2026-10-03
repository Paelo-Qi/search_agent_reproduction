"""Real verl 0.6.1 FSDP2 infrastructure, isolated from rollout/RL policy loss.

The actor owns model/optimizer; the temporary supervised step does NOT exercise
DataParallelPPOActor.update_policy, PPO/RLOO loss, Ray workers or rollout sync.
Heavy dependencies are imported only by the GPU entry point.
"""

from __future__ import annotations

import copy
import gc
import importlib.metadata
import json
import os
import platform
import random
import time
import weakref
from datetime import timedelta
from pathlib import Path
from typing import Any

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl.actor_gate import (
    REQUIRED_CHECKS, aggregate_reports, atomic_json, finite_loss, gate_identity,
    gradient_checks, load_gate_config, load_smoke_records, lora_snapshot,
    reload_matches, select_rank_sample, temporary_sft_record, trainable_policy,
    update_checks, validate_output_paths, validate_software,
)


def memory_stats(torch: Any, device: int) -> dict[str, int]:
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


class CollectiveStages:
    """Publish false first; aggregate ordinary exceptions at stage boundaries.

    Fatal process/NCCL failure may prevent aggregation, but leaves a false report
    at the last entered stage. torchrun + the bounded NCCL timeout propagate it.
    There is deliberately no barrier in the failure cleanup path.
    """

    def __init__(self, torch: Any, report: Path, identity: dict[str, Any], device: int,
                 required_checks: tuple[str, ...] = REQUIRED_CHECKS, label: str = "Gate A2.2"):
        self.torch, self.dist, self.report = torch, torch.distributed, report
        self.rank, self.world, self.device = self.dist.get_rank(), self.dist.get_world_size(), device
        self.identity = identity
        self.required_checks, self.label = required_checks, label
        self.log: list[dict[str, Any]] = []

    def run(self, stage: str, operation: Any) -> Any:
        if self.rank == 0:
            atomic_json(self.report, {"passed": False, "stage": stage, "identity": self.identity,
                                      "world_size": self.world, "fsdp_mode": "fsdp2",
                                      "checks": dict.fromkeys(self.required_checks, False),
                                      "completed_stages": self.log})
        error, result = None, None
        try:
            result = operation()
            self.torch.cuda.synchronize(self.device)
        except Exception as exc:
            error = {"rank": self.rank, "stage": stage,
                     "error_type": type(exc).__name__, "message": str(exc)}
            print("[GATE FAIL] " + json.dumps(error), flush=True)
        payload = {"rank": self.rank, "stage": stage, "error": error,
                   **memory_stats(self.torch, self.device)}
        print("[GATE MEMORY] " + json.dumps(payload), flush=True)
        rows: list[Any] = [None] * self.world
        self.dist.all_gather_object(rows, payload)
        self.log.append({"stage": stage, "per_rank": rows})
        failures = [row["error"] for row in rows if row["error"] is not None]
        if failures:
            if self.rank == 0:
                atomic_json(self.report, {"passed": False, "stage": stage, "identity": self.identity,
                                          "world_size": self.world, "fsdp_mode": "fsdp2",
                                          "checks": dict.fromkeys(self.required_checks, False),
                                          "errors": failures, "completed_stages": self.log})
            raise RuntimeError(f"{self.label} failed at {stage}: {failures}")
        return result


def require_checks(checks: dict[str, bool], names: tuple[str, ...]) -> None:
    failed = [name for name in names if checks.get(name) is not True]
    if failed:
        raise RuntimeError(f"Gate A2.2 failed checks: {failed}")


def construct_actor(*, config: dict[str, Any], gate: dict[str, Any], adapter: Path,
                    mesh: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from peft import PeftModel
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from verl.utils.fsdp_utils import (apply_fsdp2, fsdp2_load_full_state_dict,
                                       fsdp_version, get_shard_placement_fn)
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.config.actor import FSDPActorConfig
    from verl.workers.config.engine import FSDPEngineConfig
    from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer

    from opensearch_vl_repro.model import freeze_vision_components, load_base_model
    from opensearch_vl_repro.rl.actor_gate import BASE_MODEL
    from opensearch_vl_repro.sft_long_training import activate_sft_training_mode

    # Same load/freeze/checkpointing path as SFT, continuing real weights, not
    # creating a new LoRA. PEFT keeps trainable adapters FP32, MP forward BF16.
    model = load_base_model(config, for_training=True)
    freeze_vision_components(model)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True, local_files_only=True)
    model.peft_config["default"].base_model_name_or_path = BASE_MODEL
    freeze_vision_components(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    activate_sft_training_mode(model)
    audit = trainable_policy(model)
    if model.config._attn_implementation != "flash_attention_2":
        raise RuntimeError("resolved model attention is not flash_attention_2")
    audit["parameter_dtypes"] = {}
    for p in model.parameters():
        name = str(p.dtype)
        audit["parameter_dtypes"][name] = audit["parameter_dtypes"].get(name, 0) + p.numel()

    # This is the actual v0.6.1 ActorRolloutRefWorker FSDP2 branch, without
    # constructing its Ray/rollout/trainer stack or applying PPO loss patches.
    engine = FSDPEngineConfig(strategy="fsdp2", dtype="bfloat16", model_dtype="bf16",
                              fsdp_size=-1, reshard_after_forward=True)
    full_state = model.state_dict() if torch.distributed.get_rank() == 0 else {}
    wrap_names = list(model._no_split_modules)
    audit["wrap_layer_classes"] = wrap_names
    apply_fsdp2(model, {"mesh": mesh,
                       "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                         reduce_dtype=torch.float32,
                                                         cast_forward_inputs=True),
                       "offload_policy": None, "reshard_after_forward": True,
                       "shard_placement_fn": get_shard_placement_fn(fsdp_size=mesh.shape[-1])}, engine)
    fsdp2_load_full_state_dict(model, full_state, mesh, None)
    del full_state
    if fsdp_version(model) != 2:
        raise RuntimeError("verl did not produce a real FSDP2 model")
    activate_sft_training_mode(model)
    stacks = [m for n, m in model.named_modules() if n.endswith("language_model") and hasattr(m, "layers")]
    if len(stacks) != 1 or any(fsdp_version(layer) != 2 for layer in stacks[0].layers):
        raise RuntimeError("all 36 language decoder layers must be wrapped by verl FSDP2")
    if stacks[0].config._attn_implementation != "flash_attention_2":
        raise RuntimeError("resolved language-model attention is not flash_attention_2")
    opt_config = FSDPOptimizerConfig(lr=gate["optimizer"]["learning_rate"],
                                    weight_decay=gate["optimizer"]["weight_decay"],
                                    optimizer="AdamW", optimizer_impl="torch.optim")
    optimizer = build_optimizer([p for p in model.parameters() if p.requires_grad], opt_config)
    trainable_policy(model, optimizer)
    actor_config = FSDPActorConfig(strategy="fsdp2", fsdp_config=engine, optim=opt_config,
                                   ppo_micro_batch_size_per_gpu=1, use_dynamic_bsz=False,
                                   use_torch_compile=False, use_remove_padding=False,
                                   use_fused_kernels=False, freeze_vision_tower=True)
    actor = DataParallelPPOActor(actor_config, actor_module=model, actor_optimizer=optimizer)
    audit.update(model_training=model.training, language_model_training=stacks[0].training,
                 decoder_layers_training=sum(layer.training for layer in stacks[0].layers),
                 decoder_layers_gradient_checkpointing=sum(layer.gradient_checkpointing for layer in stacks[0].layers),
                 effective_attention_implementation=model.config._attn_implementation,
                 use_cache=model.config.use_cache)
    return actor, audit


def construct_rl_actor(*, config: dict[str, Any], gate: dict[str, Any], adapter: Path,
                       mesh: Any) -> tuple[Any, dict[str, Any]]:
    """Explicit formal RL semantics; raw Gate A/forensic constructor stays intact."""
    from opensearch_vl_repro.rl.rl_actor_semantics import configure_rl_lora_dropout_runtime
    actor, audit = construct_actor(config=config, gate=gate, adapter=adapter, mesh=mesh)
    audit["rl_lora_dropout_runtime"] = configure_rl_lora_dropout_runtime(actor.actor_module, config)
    return actor, audit


def checkpoint_manager(actor: Any, processor: Any) -> Any:
    from omegaconf import OmegaConf
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    return FSDPCheckpointManager(model=actor.actor_module, optimizer=actor.actor_optimizer,
                                lr_scheduler=None, processing_class=processor,
                                checkpoint_config=OmegaConf.create({"save_contents": ["model", "optimizer", "extra"],
                                                                    "load_contents": ["model", "optimizer", "extra"]}))


def save_checkpoint(actor: Any, processor: Any, output: Path, *, global_step: int = 1) -> dict[str, Any]:
    import torch
    from verl.utils.fsdp_utils import get_fsdp_full_state_dict
    from opensearch_vl_repro.inference.adapter import adapter_file_identity

    if type(global_step) is not int or global_step < 1:
        raise ValueError("positive native checkpoint optimizer step required")
    manager = checkpoint_manager(actor, processor)
    manager.save_checkpoint(str(output / "distributed"), global_step=global_step)
    # All ranks participate in the official full-state gather; rank0 exports
    # only the PEFT adapter. No manual DTensor concatenate/sharding format.
    full_state = get_fsdp_full_state_dict(actor.actor_module, offload_to_cpu=True, rank0_only=True)
    result = None
    if torch.distributed.get_rank() == 0:
        actor.actor_module.save_pretrained(str(output / "adapter"), state_dict=full_state,
                                           safe_serialization=True, save_embedding_layers=False)
        result = adapter_file_identity(output / "adapter")
    del full_state, manager
    items = [result]
    torch.distributed.broadcast_object_list(items, src=0)
    for rank in range(torch.distributed.get_world_size()):
        for prefix in ("model", "optim", "extra_state"):
            path = output / "distributed" / f"{prefix}_world_size_{torch.distributed.get_world_size()}_rank_{rank}.pt"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"verl native checkpoint artifact missing: {path.name}")
    return items[0]


def run_gate(args: Any, root: Path) -> int:
    # Enforce offline before importing any HF/verl dependency. No opt-in download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    from opensearch_vl_repro.rl.checkpoint import build_rl_lineage, validate_sft_overlap_scope
    from opensearch_vl_repro.rl.config import load_rl_config
    from opensearch_vl_repro.sft_train_plan import load_main_config

    gate = load_gate_config(args.gate_config)
    config = load_rl_config(args.config)
    sft = load_main_config(root / config["model"]["sft_config"], base_eval_config=root / "configs/eval_base_300.yaml")
    if ((args.base_model, args.base_revision) != (gate["model"], gate["revision"])
            or (sft["model"]["name_or_path"], sft["model"]["revision"]) != (gate["model"], gate["revision"])
            or sft["model"]["attn_implementation"] != "flash_attention_2"):
        raise ValueError("gate/base/SFT identity or FA2 mismatch")
    adapter = (args.sft_adapter or root / config["model"]["sft_adapter"]).resolve()
    lineage = build_rl_lineage(config=config, sft_config=sft, adapter_path=adapter, run_id="gate-a22")
    validate_sft_overlap_scope(lineage, ["main_a_1k", "main_b_2k"])
    if adapter.parent.name != "checkpoint-3k":
        raise ValueError("input must be the completed checkpoint-3k SFT adapter")
    config["data"]["quality_audit_dir"] = str(root / config["data"]["quality_audit_dir"])
    records, data_manifest = load_smoke_records(args.data, config)
    output, reports = args.output_dir.resolve(), args.report_dir.resolve()
    protected = [adapter.parent, args.data.parent, args.source_root,
                 Path(config["data"]["quality_audit_dir"]),
                 *([args.base_model_path] if args.base_model_path else [])]
    runtime_sft = copy.deepcopy(sft)
    if args.base_model_path:
        if not args.base_model_path.is_dir():
            raise FileNotFoundError("offline base snapshot directory missing")
        runtime_sft["model"]["name_or_path"] = str(args.base_model_path.resolve())

    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from opensearch_vl_repro.data import OpenSearchVLCollator
    from opensearch_vl_repro.inference.adapter import adapter_file_identity
    from opensearch_vl_repro.model import load_processor, move_batch

    if not torch.cuda.is_available():
        raise RuntimeError("Gate A2.2 requires real CUDA; CPU tests are not a GPU PASS")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    started = time.monotonic()
    try:
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
        rank, world = dist.get_rank(), dist.get_world_size()
        row = select_rank_sample(records, rank=rank, world_size=world, max_samples=args.max_samples)
        versions = {}
        for name in ("torch", "transformers", "peft", "verl"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "missing"  # Fail in the collectively reported software stage.
        versions["python"] = platform.python_version()
        identity = gate_identity(gate=gate, sft=sft, lineage=lineage.to_dict(),
                                 data_manifest=data_manifest, sample_ids=[r["source_sample_id"] for r in records[:world]],
                                 world_size=world, versions=versions, seed=args.seed)
        reservation_error = None
        if rank == 0:
            try:
                validate_output_paths(output, reports, protected)
                output.mkdir(parents=True, exist_ok=False)
                reports.mkdir(parents=True, exist_ok=False)
                atomic_json(reports / "gate_a22_report.json", {"passed": False, "stage": "initialized",
                                                              "world_size": world, "fsdp_mode": "fsdp2",
                                                              "checks": dict.fromkeys(REQUIRED_CHECKS, False),
                                                              "identity": identity})
            except Exception as exc:
                reservation_error = f"{type(exc).__name__}: {exc}"
        reservation = [reservation_error]
        dist.broadcast_object_list(reservation, src=0)
        if reservation[0]:
            raise RuntimeError(reservation[0])
        stages = CollectiveStages(torch, reports / "gate_a22_report.json", identity, local_rank)
        checks = {name: False for name in REQUIRED_CHECKS}
        checks.update(distributed_initialized=dist.is_initialized(),
                      expected_world_size=world == int(os.environ["WORLD_SIZE"]))
        stages.run("software_and_bf16", lambda: (validate_software(versions, gate),
                   require_checks({"bf16": torch.cuda.is_bf16_supported()}, ("bf16",))))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(local_rank)
        initial_memory = memory_stats(torch, local_rank)
        mesh = stages.run("device_mesh", lambda: init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",)))
        processor = stages.run("processor", lambda: load_processor(runtime_sft, local_files_only=True))
        actor, audit = stages.run("base_sft_load_fsdp2_optimizer", lambda: construct_actor(
            config=runtime_sft, gate=gate, adapter=adapter, mesh=mesh))
        for name in ("base_loaded", "sft_lora_loaded", "vision_frozen", "projector_frozen", "base_frozen",
                     "trainable_lora_present", "optimizer_only_lora", "training_checkpointing_active", "fsdp2_wrapped"):
            checks[name] = True
        print("[GATE MODEL] " + json.dumps({"rank": rank,
              **{key: value for key, value in audit.items() if key != "trainable_parameter_names"},
              "trainable_name_examples": audit["trainable_parameter_names"][:5]}), flush=True)

        def collate() -> Any:
            sample = temporary_sft_record(row, args.source_root)
            batch = OpenSearchVLCollator(processor, args.data, max_length=sft["data"]["max_length"])([sample])
            if (not {"input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"} <= batch.keys()
                    or batch["pixel_values"].numel() == 0 or batch["input_ids"].shape != batch["labels"].shape
                    or batch["input_ids"].shape[0] != 1 or not bool((batch["labels"] != -100).any())):
                raise RuntimeError("real multimodal input/assistant labels missing")
            print("[GATE BATCH] " + json.dumps({"rank": rank, "sample_id": row["source_sample_id"],
                  "actual_token_length": int(batch["attention_mask"].sum()),
                  "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in batch.items()}}), flush=True)
            return move_batch(batch, local_rank)

        batch = stages.run("real_multimodal_batch", collate)
        checks["real_multimodal_batch"] = True
        before = stages.run("pre_step_lora_snapshot", lambda: lora_snapshot(actor.actor_module))

        def forward() -> Any:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = actor.actor_module(**batch).loss
            checks["loss_finite"] = finite_loss(float(loss.detach()))
            require_checks(checks, ("loss_finite",))
            return loss

        loss = stages.run("forward", forward)
        loss_before = float(loss.detach())
        stages.run("backward", loss.backward)
        checks["backward_completed"] = True
        checks.update(stages.run("gradient_audit", lambda: gradient_checks(actor.actor_module)))
        stages.run("gradient_gate", lambda: require_checks(checks, ("lora_grad_finite", "nonzero_lora_grad")))
        stages.run("optimizer_step", actor.actor_optimizer.step)
        checks["optimizer_step_completed"] = True
        after = stages.run("updated_lora_snapshot", lambda: lora_snapshot(actor.actor_module))
        checks.update(update_checks(before, after))
        changed_names = [name for name in before if not torch.equal(before[name], after[name])]
        stages.run("update_gate", lambda: require_checks(checks, ("parameters_finite", "lora_param_changed")))
        stages.run("zero_grad_and_frozen_audit", lambda: (actor.actor_optimizer.zero_grad(set_to_none=True),
                                                         trainable_policy(actor.actor_module, actor.actor_optimizer)))
        saved_identity = stages.run("save_verl_checkpoint_and_adapter", lambda: save_checkpoint(actor, processor, output))
        checks.update(checkpoint_saved=True, optimizer_state_saved=True)

        model_ref, optimizer_ref = weakref.ref(actor.actor_module), weakref.ref(actor.actor_optimizer)
        del loss, actor, before
        gc.collect()
        torch.cuda.empty_cache()
        checks["original_model_destroyed"] = model_ref() is None and optimizer_ref() is None
        stages.run("destroy_and_barrier", lambda: (require_checks(checks, ("original_model_destroyed",)), dist.barrier()))

        fresh, reload_audit = stages.run("fresh_base_saved_adapter_fsdp2", lambda: construct_actor(
            config=runtime_sft, gate=gate, adapter=output / "adapter", mesh=mesh))
        checks["fresh_adapter_reloaded"] = True
        checks["adapter_fingerprint_match"] = (adapter_file_identity(output / "adapter") == saved_identity)
        checks["reload_param_match"] = stages.run("fresh_adapter_parameter_match", lambda: reload_matches(
            after, lora_snapshot(fresh.actor_module)))
        stages.run("reload_gate", lambda: require_checks(checks, ("adapter_fingerprint_match", "reload_param_match")))
        stages.run("native_checkpoint_reload", lambda: checkpoint_manager(fresh, processor).load_checkpoint(
            str(output / "distributed"), del_local_after_load=False))
        checks["native_checkpoint_reloaded"] = bool(fresh.actor_optimizer.state)
        checks["reload_param_match"] = reload_matches(after, lora_snapshot(fresh.actor_module))

        def validation_forward() -> float:
            fresh.actor_module.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                value = float(fresh.actor_module(**batch).loss)
            checks["fresh_forward_finite"] = finite_loss(value)
            require_checks(checks, REQUIRED_CHECKS)
            return value

        loss_after = stages.run("fresh_forward", validation_forward)
        rank_report = {"rank": rank, "local_rank": local_rank, "world_size": world,
                       "gpu_name": torch.cuda.get_device_name(local_rank), "bf16": True,
                       "sample_id": row["source_sample_id"], "prompt_id": row["prompt_id"],
                       "actual_token_length": int(batch["attention_mask"].sum()),
                       "tensor_summary": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in batch.items()},
                       "loss_before": loss_before, "loss_after_reload": loss_after,
                       "nonzero_lora_grad": checks["nonzero_lora_grad"],
                       "lora_param_changed": checks["lora_param_changed"], "reload_param_match": checks["reload_param_match"],
                       "changed_lora_tensor_count": len(changed_names), "changed_lora_tensor_examples": changed_names[:5],
                       "total_parameters": audit["total_parameters"], "trainable_parameters": audit["trainable_parameters"],
                       "initial_memory": initial_memory, **memory_stats(torch, local_rank),
                       "checks": checks, "elapsed_seconds": time.monotonic() - started}
        ranks: list[Any] = [None] * world
        dist.all_gather_object(ranks, rank_report)
        summary = aggregate_reports(ranks, world)
        if not summary["passed"]:
            raise RuntimeError("not all ranks satisfied the complete Gate PASS contract")
        dist.barrier()
        if rank == 0:
            artifact_files = {p.relative_to(output).as_posix(): p for p in output.rglob("*") if p.is_file()}
            from opensearch_vl_repro.sft_tool_audit import sha256_file
            summary.update(identity=identity, software_versions=versions, model=gate["model"], revision=gate["revision"],
                           input_adapter_fingerprint=lineage.sft_adapter_fingerprint,
                           output_adapter_fingerprint=saved_identity["adapter_fingerprint"],
                           checkpoint_file_sha256={name: sha256_file(p) for name, p in sorted(artifact_files.items())},
                           sample_ids=identity["sample_ids"], completed_stages=stages.log,
                           loss_before=[r["loss_before"] for r in ranks],
                           loss_after_reload=[r["loss_after_reload"] for r in ranks],
                           trainable_parameter_count=audit["trainable_parameters"],
                           total_parameter_count=audit["total_parameters"],
                           nonzero_lora_grad=summary["checks"]["nonzero_lora_grad"],
                           lora_param_changed=summary["checks"]["lora_param_changed"],
                           reload_param_match=summary["checks"]["reload_param_match"],
                           scope="verl FSDP2 actor infrastructure; temporary supervised step, not update_policy/RL",
                           runtime_locators={"base_snapshot": str(args.base_model_path) if args.base_model_path else None,
                                             "source_root": str(args.source_root), "input_adapter": str(adapter)})
            summary["checkpoint_fingerprint"] = canonical_json_sha256(summary["checkpoint_file_sha256"])
            atomic_json(output / "gate_manifest.json", summary)
            atomic_json(stages.report, summary)
            print("Gate A2.2 PASS " + json.dumps({"world_size": world, "fsdp_mode": "fsdp2",
                   "sample_ids": identity["sample_ids"], "report": str(stages.report)}), flush=True)
        del fresh, after, batch
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
