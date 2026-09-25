"""Exact-shard BF16/DDP LoRA training with complete, resumable state.

This is intentionally separate from the Phase 0 evidence-only Trainer. Checkpoints
are written only immediately after optimizer steps, so no partial gradient state is
needed to resume an interrupted microbatch accumulation.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .data import OpenSearchVLCollator, load_json_records
from .model import (add_lora, freeze_vision_components, load_base_model,
                    load_processor, move_batch, parameter_audit, select_probe_parameter)
from .reporting import environment_report, write_json
from .sft_main_data import SHARD_SIZES, load_sft_manifest
from .sft_tool_audit import sha256_file
from .sft_train_plan import (STAGE_NAMES, STAGE_ORDER, StagePlan, cosine_factor,
                             load_main_config, plan_stage, validate_resume_metadata)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_FILES = ("adapter/adapter_config.json", "optimizer.pt", "scheduler.pt",
                    "rng.pt", "trainer_state.json", "metadata.json")


def check_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    missing = [name for name in CHECKPOINT_FILES if not (path / name).is_file()]
    if missing or not list((path / "adapter").glob("*.safetensors")):
        raise ValueError(f"incomplete SFT checkpoint {path}: missing {missing or 'adapter weights'}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("checkpoint_complete") is not True:
        raise ValueError(f"SFT checkpoint is not marked complete: {path}")
    checksums = metadata.get("file_sha256")
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("SFT checkpoint has no artifact checksums")
    for name, expected in checksums.items():
        if sha256_file(path / name) != expected:
            raise ValueError(f"SFT checkpoint file checksum mismatch: {name}")
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    if metadata.get("global_step") != state.get("global_step"):
        raise ValueError("checkpoint metadata/trainer state global-step mismatch")
    return metadata


def checkpoint_metadata(plan: StagePlan, state: dict[str, Any], *, config: dict[str, Any],
                        pool_sha256: str, shard_sha256: str, resumed_from: str | None,
                        complete_stage: bool) -> dict[str, Any]:
    return {
        "checkpoint_complete": True, "model": config["model"]["name_or_path"],
        "model_revision": config["model"]["revision"], "stage": plan.stage,
        "lineage": list(plan.lineage if complete_stage else plan.lineage[:-1]),
        "stage_complete": complete_stage, "pool_manifest_sha256": pool_sha256,
        "shard_sha256": shard_sha256, "shard_samples": plan.shard_samples,
        "world_size": plan.world_size, "micro_batch": plan.micro_batch,
        "gradient_accumulation": plan.gradient_accumulation,
        "effective_global_batch": plan.effective_global_batch,
        "scheduler_phase": plan.scheduler_phase,
        "scheduler_type": plan.scheduler_type,
        "phase_total_steps": plan.phase_total_steps,
        "phase_step": state["phase_step"], "warmup_steps": plan.warmup_steps,
        "peak_lr": plan.peak_lr, "current_lr": state["current_lr"],
        "global_step": state["global_step"],
        "stage_optimizer_steps": state["global_step"] - plan.global_start_step,
        "epoch": state["epoch"], "microbatch_offset": state["microbatch_offset"],
        "cumulative_samples_seen": state["cumulative_samples_seen"],
        "stage_samples_seen": state["stage_samples_seen"],
        "resumed_from": resumed_from,
        "leakage_acknowledged": bool(state.get("leakage_acknowledged", False)),
    }


def _rng_state(torch: Any, np: Any, device: int) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device)}


def _restore_rng(value: dict[str, Any], torch: Any, np: Any, device: int) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state(value["torch_cuda"], device)


def _save_checkpoint(path: Path, *, model: Any, processor: Any, optimizer: Any,
                     scheduler: Any, state: dict[str, Any], plan: StagePlan,
                     config: dict[str, Any], pool_sha256: str, shard_sha256: str,
                     resumed_from: str | None, complete_stage: bool,
                     torch: Any, np: Any, dist: Any, rank: int, device: int) -> None:
    if scheduler.last_epoch != state["phase_step"]:
        raise RuntimeError("scheduler step diverged from persisted phase step")
    rng = [None] * plan.world_size
    dist.all_gather_object(rng, _rng_state(torch, np, device))
    failure = [None]
    if rank == 0:
        try:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite SFT checkpoint: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
            unwrapped = model.module
            unwrapped.save_pretrained(temporary / "adapter", safe_serialization=True)
            processor.save_pretrained(temporary / "adapter")
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
            torch.save(rng, temporary / "rng.pt")
            write_json(temporary / "trainer_state.json", state)
            metadata = checkpoint_metadata(
                plan, state, config=config, pool_sha256=pool_sha256,
                shard_sha256=shard_sha256, resumed_from=resumed_from,
                complete_stage=complete_stage)
            metadata["trainable_parameter_names"] = [
                name for name, parameter in unwrapped.named_parameters() if parameter.requires_grad
            ]
            metadata["file_sha256"] = {
                item.relative_to(temporary).as_posix(): sha256_file(item)
                for item in sorted(temporary.rglob("*")) if item.is_file()
            }
            write_json(temporary / "metadata.json", metadata)
            os.replace(temporary, path)
        except Exception as exc:
            # Leave a partially written temp directory visible for diagnosis;
            # it is never considered a valid checkpoint.
            failure[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"SFT checkpoint save failed: {failure[0]}")
    dist.barrier()


def run_sft_stage(config_path: str | Path, *, stage: str,
                  resume_from: str | Path | None = None, micro_batch: int | None = None,
                  gradient_accumulation: int | None = None,
                  phase_2_peak_lr: float | None = None,
                  stop_after_steps: int | None = None, run_tag: str | None = None,
                  acknowledge_leakage: bool = False) -> dict[str, Any] | None:
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.utils.tensorboard import SummaryWriter

    config = load_main_config(config_path, base_eval_config=PROJECT_ROOT / "configs/eval_base_300.yaml")
    if int(config["training"].get("dataloader_num_workers", 0)) != 0:
        raise ValueError("exact mid-shard RNG resume currently requires dataloader_num_workers=0")
    plan = plan_stage(config, stage, micro_batch=micro_batch,
                      gradient_accumulation=gradient_accumulation,
                      phase_2_peak_lr=phase_2_peak_lr)
    if not torch.cuda.is_available():
        raise RuntimeError("4B SFT requires CUDA with BF16 support")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("4B SFT requires torchrun on exactly two visible CUDA GPUs")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"GPU {local_rank} does not support BF16")
    dist.init_process_group("nccl")
    if run_tag is not None and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_tag) is None:
        raise ValueError("run_tag must be a safe short identifier")
    seed = int(config["project"]["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("tf32", True))
    if stage == "smoke":
        data_path = (PROJECT_ROOT / config["data"]["path"]).resolve()
        records = load_json_records(data_path)
        if len(records) != plan.shard_samples:
            raise ValueError("4B smoke dataset must contain exactly 100 official trajectories")
        pool_sha = shard_sha = sha256_file(data_path)
    else:
        pool_dir = (PROJECT_ROOT / config["data"]["pool_dir"]).resolve()
        manifest_path = pool_dir / "manifest.json"
        manifest = load_sft_manifest(manifest_path)
        pool_sha = sha256_file(manifest_path)
        data_path = pool_dir / manifest["shards"][stage]["path"]
        shard_sha = manifest["shards"][stage]["sha256"]
        records = load_json_records(data_path)
        if len(records) != plan.shard_samples:
            raise ValueError("SFT shard count changed")
        preflight_path = PROJECT_ROOT / "reports/sft_preflight/summary.json"
        if not preflight_path.is_file():
            raise RuntimeError("full SFT preflight audit must run before formal training")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (not preflight.get("passed") or not preflight.get("sequence_complete")
                or not preflight.get("tool_contract_passed") or not preflight.get("leakage_complete")):
            raise RuntimeError("SFT preflight audit has not passed")
        if preflight.get("pool_manifest_sha256") != pool_sha:
            raise RuntimeError("SFT preflight belongs to a different pool manifest")
        if preflight.get("leakage_review_required") and not acknowledge_leakage:
            raise RuntimeError("review leakage report and explicitly acknowledge it")
    if any(not (data_path.parent / image).is_file()
           for record in records for image in record["images"]):
        raise FileNotFoundError("selected SFT images are missing; materialize official ZIP images")

    root = (PROJECT_ROOT / config["project"]["output_dir"]).resolve()
    report_root = (PROJECT_ROOT / config["project"]["report_dir"]).resolve()
    if stage == "smoke":
        tag = run_tag or f"micro{plan.micro_batch}_accum{plan.gradient_accumulation}"
        root, report_root = root / tag, report_root / tag
        final_path = root / "checkpoint-20"
    else:
        final_path = root / STAGE_NAMES[stage]
    if final_path.exists():
        raise FileExistsError(f"stage output already exists: {final_path}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    resume_path = Path(resume_from).expanduser().resolve() if resume_from else None
    resume_mode = None
    if resume_path is not None:
        metadata = check_checkpoint(resume_path)
        resume_mode = validate_resume_metadata(
            plan, metadata, model_name=config["model"]["name_or_path"],
            revision=config["model"]["revision"], pool_sha256=pool_sha,
        )
        if resume_mode == "same_stage" and metadata["shard_sha256"] != shard_sha:
            raise ValueError("resumed stage shard checksum changed")
    elif stage not in ("main_a_1k", "smoke"):
        raise ValueError("continuation stage requires --resume-from complete prior checkpoint")

    processor = load_processor(config)  # Always pinned Base processor, never an arbitrary adapter copy.
    base = load_base_model(config, for_training=True)
    frozen_names = freeze_vision_components(base)
    if resume_path is None:
        model, resolved_targets = add_lora(base, config)
    else:
        model = PeftModel.from_pretrained(base, resume_path / "adapter", is_trainable=True)
        resolved_targets = list(model.peft_config["default"].target_modules)
    if config["training"].get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model = model.to(f"cuda:{local_rank}")
    audit = parameter_audit(model)
    if resume_path is not None and metadata.get("trainable_parameter_names") != audit["trainable_parameter_names"]:
        raise ValueError("checkpoint LoRA parameter order changed; optimizer resume is unsafe")
    probe_name, probe = select_probe_parameter(model)
    probe_before = probe.detach().float().cpu().clone()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank],
                                      output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=plan.peak_lr, weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    if resume_path is not None:
        optimizer.load_state_dict(torch.load(resume_path / "optimizer.pt", map_location="cpu",
                                              weights_only=False))
    if resume_mode == "next_stage" and plan.scheduler_phase == "phase_2":
        for group in optimizer.param_groups:
            group["lr"] = plan.peak_lr
            group["initial_lr"] = plan.peak_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: cosine_factor(
            step, plan.warmup_steps, plan.phase_total_steps),
    )
    if resume_path is not None and not (resume_mode == "next_stage" and plan.scheduler_phase == "phase_2"):
        scheduler.load_state_dict(torch.load(resume_path / "scheduler.pt", map_location="cpu",
                                               weights_only=False))
        for group in optimizer.param_groups:
            group["lr"] = metadata["current_lr"]
    if resume_path is not None:
        state = json.loads((resume_path / "trainer_state.json").read_text(encoding="utf-8"))
        saved_rng = torch.load(resume_path / "rng.pt", map_location="cpu", weights_only=False)[rank]
        if resume_mode == "next_stage":
            state.update(epoch=0, microbatch_offset=0, stage_samples_seen=0)
            if plan.scheduler_phase == "phase_2":
                state["phase_step"] = 0
    else:
        state = {"global_step": 0, "phase_step": 0, "epoch": 0,
                 "microbatch_offset": 0, "cumulative_samples_seen": 0,
                 "stage_samples_seen": 0, "current_lr": optimizer.param_groups[0]["lr"]}
        saved_rng = None
    if state["global_step"] != plan.global_start_step and resume_mode != "same_stage":
        raise ValueError("checkpoint global step does not match planned stage boundary")
    state["leakage_acknowledged"] = bool(acknowledge_leakage)
    if scheduler.last_epoch != state["phase_step"]:
        raise ValueError("checkpoint scheduler state does not match phase step")
    collator = OpenSearchVLCollator(processor, data_path, int(config["data"]["max_length"]))
    sampler = DistributedSampler(records, num_replicas=plan.world_size, rank=rank,
                                 shuffle=True, seed=seed + 100 * (STAGE_ORDER.index(stage)
                                                                   if stage != "smoke" else 0))
    writer = SummaryWriter(str(root / "tensorboard")) if rank == 0 else None
    optimizer.zero_grad(set_to_none=True)
    load_peak = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(local_rank),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(local_rank)}
    torch.cuda.reset_peak_memory_stats(local_rank)
    started = time.perf_counter()
    initial_step = state["global_step"]
    initial_stage_samples = state["stage_samples_seen"]
    target = plan.global_target_step
    if stop_after_steps is not None:
        if stage != "smoke" or not initial_step < stop_after_steps <= target:
            raise ValueError("--stop-after-steps is only for an in-progress 4B smoke")
        target = stop_after_steps
    step_losses, step_lrs, step_times = [], [], []
    accumulated_loss = 0.0
    micro_in_step = 0
    device = torch.device(f"cuda:{local_rank}")
    step_started = time.perf_counter()
    while state["global_step"] < target:
        sampler.set_epoch(state["epoch"])
        generator = torch.Generator().manual_seed(seed + state["epoch"])
        loader = DataLoader(records, batch_size=plan.micro_batch, sampler=sampler,
                            collate_fn=collator, num_workers=0, generator=generator)
        iterator = iter(loader)
        offset = state["microbatch_offset"]
        for _ in range(offset):
            next(iterator)
        if saved_rng is not None:
            _restore_rng(saved_rng, torch, np, local_rank)
            saved_rng = None
        for batch in iterator:
            batch = move_batch(batch, device)
            next_micro = micro_in_step + 1
            sync_context = (nullcontext() if next_micro == plan.gradient_accumulation
                            else wrapped.no_sync())
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = wrapped(**batch).loss
                finite = torch.tensor(bool(torch.isfinite(loss.detach()).all()), device=device)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite.item()):
                    raise FloatingPointError(f"non-finite SFT loss at step {state['global_step']}")
                (loss / plan.gradient_accumulation).backward()
            accumulated_loss += float(loss.detach().float().item())
            micro_in_step = next_micro
            batch_samples = int(batch["input_ids"].shape[0]) * plan.world_size
            state["cumulative_samples_seen"] += batch_samples
            state["stage_samples_seen"] += batch_samples
            state["microbatch_offset"] += 1
            if micro_in_step < plan.gradient_accumulation:
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step_time = time.perf_counter() - step_started
            state["global_step"] += 1
            state["phase_step"] += 1
            state["current_lr"] = optimizer.param_groups[0]["lr"]
            local_loss = torch.tensor(accumulated_loss / plan.gradient_accumulation,
                                      device=device, dtype=torch.float32)
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            step_loss = float(local_loss.item() / plan.world_size)
            if not math.isfinite(step_loss):
                raise FloatingPointError("non-finite aggregated SFT loss")
            step_losses.append(step_loss)
            step_lrs.append(state["current_lr"])
            step_times.append(step_time)
            if writer is not None and state["global_step"] % int(config["training"].get("logging_steps", 1)) == 0:
                writer.add_scalar("train/loss", step_loss, state["global_step"])
                writer.add_scalar("train/learning_rate", state["current_lr"], state["global_step"])
                writer.add_scalar("train/step_time_seconds", step_time, state["global_step"])
            accumulated_loss = 0.0
            micro_in_step = 0
            save_every = int(config["training"].get("save_every_steps", 0))
            if save_every and state["global_step"] % save_every == 0 and state["global_step"] < target:
                _save_checkpoint(root / "periodic" / stage / f"step-{state['global_step']}",
                                 model=wrapped, processor=processor, optimizer=optimizer,
                                 scheduler=scheduler, state=state, plan=plan, config=config,
                                 pool_sha256=pool_sha, shard_sha256=shard_sha,
                                 resumed_from=str(resume_path) if resume_path else None,
                                 complete_stage=False, torch=torch, np=np, dist=dist,
                                 rank=rank, device=local_rank)
            step_started = time.perf_counter()
            if state["global_step"] >= target:
                if state["microbatch_offset"] == len(loader):
                    state["epoch"] += 1
                    state["microbatch_offset"] = 0
                break
        else:
            state["epoch"] += 1
            state["microbatch_offset"] = 0
            if stage != "smoke" and state["epoch"] > plan.epochs:
                raise RuntimeError("stage exhausted its exact shard epochs before planned steps")
            continue
        if state["global_step"] >= target:
            break
    if micro_in_step:
        raise RuntimeError("stage stopped with an unsaved partial accumulation")
    delta = float((probe.detach().float().cpu() - probe_before).abs().max().item())
    if delta == 0:
        raise RuntimeError(f"LoRA parameter did not change: {probe_name}")
    runtime = time.perf_counter() - started
    peak = {"rank": rank,
            "load_peak_allocated_bytes": load_peak["peak_allocated_bytes"],
            "load_peak_reserved_bytes": load_peak["peak_reserved_bytes"],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(local_rank),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(local_rank)}
    all_peaks = [None] * plan.world_size
    dist.all_gather_object(all_peaks, peak)
    complete_stage = state["global_step"] == plan.global_target_step
    final_checkpoint = final_path if complete_stage else root / f"checkpoint-step-{state['global_step']}"
    _save_checkpoint(final_checkpoint, model=wrapped, processor=processor,
                     optimizer=optimizer, scheduler=scheduler, state=state, plan=plan,
                     config=config, pool_sha256=pool_sha, shard_sha256=shard_sha,
                     resumed_from=str(resume_path) if resume_path else None,
                     complete_stage=complete_stage, torch=torch, np=np, dist=dist,
                     rank=rank, device=local_rank)
    if writer is not None:
        writer.flush()
        writer.close()
    report = None
    if rank == 0:
        report = {
            "passed": complete_stage, "stage": stage, "checkpoint": str(final_checkpoint),
            "model": config["model"]["name_or_path"],
            "model_revision": config["model"]["revision"],
            "dataset_path": str(data_path), "shard_sha256": shard_sha,
            "pool_manifest_sha256": pool_sha, "lineage": list(plan.lineage),
            "world_size": plan.world_size, "micro_batch": plan.micro_batch,
            "gradient_accumulation": plan.gradient_accumulation,
            "effective_global_batch": plan.effective_global_batch,
            "epochs": plan.epochs, "global_step": state["global_step"],
            "optimizer_steps_this_invocation": state["global_step"] - initial_step,
            "scheduler_phase": plan.scheduler_phase,
            "scheduler_total_steps": plan.phase_total_steps,
            "phase_step": state["phase_step"], "warmup_steps": plan.warmup_steps,
            "scheduler_type": plan.scheduler_type, "peak_lr": plan.peak_lr,
            "current_lr": state["current_lr"], "losses": step_losses, "lrs": step_lrs,
            "step_times_seconds": step_times,
            "all_losses_finite": True, "runtime_seconds": runtime,
            "samples_per_second": ((state["stage_samples_seen"] - initial_stage_samples) / runtime
                                   if runtime else None),
            "seconds_per_optimizer_step": (runtime / len(step_losses) if step_losses else None),
            "mean_step_time_seconds": (sum(step_times) / len(step_times) if step_times else None),
            "cumulative_samples_seen": state["cumulative_samples_seen"],
            "stage_samples_seen": state["stage_samples_seen"],
            "peak_vram_by_gpu": all_peaks,
            "peak_allocated_bytes_overall": max(
                max(row["peak_allocated_bytes"], row["load_peak_allocated_bytes"])
                for row in all_peaks),
            "peak_reserved_bytes_overall": max(
                max(row["peak_reserved_bytes"], row["load_peak_reserved_bytes"])
                for row in all_peaks),
            "parameter_audit": audit, "frozen_vision_tensor_count": len(frozen_names),
            "vision_tower_frozen": any("visual" in name.lower() or "vision" in name.lower()
                                       for name in frozen_names),
            "multimodal_projector_frozen": any("projector" in name.lower() or "merger" in name.lower()
                                               for name in frozen_names),
            "resolved_lora_target_count": len(resolved_targets),
            "lora_probe": probe_name, "lora_probe_max_abs_delta": delta,
            "resumed_from": str(resume_path) if resume_path else None,
            "leakage_acknowledged": bool(acknowledge_leakage),
            "environment": environment_report(),
        }
        write_json(report_root / f"{stage}_step{state['global_step']}.json", report)
        print(json.dumps({key: value for key, value in report.items()
                          if key not in {"losses", "lrs", "parameter_audit", "environment"}},
                         ensure_ascii=False, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return report
