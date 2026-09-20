from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .config import load_config
from .data import OpenSearchVLCollator, load_json_records
from .model import (
    add_lora,
    freeze_vision_components,
    load_base_model,
    load_processor,
    parameter_audit,
    select_probe_parameter,
)
from .reporting import environment_report, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_sft_training(
    config_path: str | Path,
    *,
    expected_world_size: int,
    minimum_optimizer_steps: int,
    report_filename: str,
    gate_name: str,
) -> dict[str, Any] | None:
    """Run an evidence-producing BF16 LoRA smoke test."""

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    class EvidenceTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError(
                    f"non-finite training loss at step {self.state.global_step}"
                )
            return (loss, outputs) if return_outputs else loss

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                logs = dict(logs)
                logs["gpu_allocated_gib"] = torch.cuda.memory_allocated(device) / 2**30
                logs["gpu_reserved_gib"] = torch.cuda.memory_reserved(device) / 2**30
                logs["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            super().log(logs, start_time=start_time)

    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for {gate_name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 support is required for {gate_name}")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size:
        raise RuntimeError(
            f"{gate_name} requires WORLD_SIZE={expected_world_size}; got {world_size}"
        )
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count != expected_world_size:
        raise RuntimeError(
            f"{gate_name} requires exactly {expected_world_size} visible CUDA GPU(s); "
            f"got {visible_gpu_count}. Set CUDA_VISIBLE_DEVICES explicitly."
        )

    seed = int(config["project"]["seed"])
    set_seed(seed)
    dataset_path = (PROJECT_ROOT / config["data"]["path"]).resolve()
    records = load_json_records(dataset_path)
    expected_samples = int(config["data"]["expected_samples"])
    if len(records) != expected_samples:
        raise RuntimeError(
            f"{gate_name} expected {expected_samples} samples, found {len(records)}"
        )

    processor = load_processor(config)
    base_model = load_base_model(config, for_training=True)
    frozen_names = freeze_vision_components(base_model)
    model, resolved_lora_targets = add_lora(base_model, config)
    if bool(config["training"].get("gradient_checkpointing", True)):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    audit = parameter_audit(model)
    probe_name, probe_parameter = select_probe_parameter(model)
    probe_before = probe_parameter.detach().float().cpu().clone()

    data_collator = OpenSearchVLCollator(
        processor=processor,
        dataset_path=dataset_path,
        max_length=int(config["data"]["max_length"]),
    )
    output_dir = (PROJECT_ROOT / config["project"]["output_dir"]).resolve()
    report_dir = (PROJECT_ROOT / config["project"]["report_dir"]).resolve()
    train_cfg = config["training"]
    # Transformers 5 removed TrainingArguments.logging_dir. Its documented
    # replacement preserves our existing TensorBoard destination.
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_dir / "tensorboard")
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_steps=int(train_cfg["max_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        # In Transformers 5, a float warmup_steps in [0, 1) is a ratio.
        warmup_steps=float(train_cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        # The adapter is saved once at the end, after all evidence checks pass.
        save_strategy="no",
        bf16=bool(train_cfg.get("bf16", True)),
        tf32=bool(train_cfg.get("tf32", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
        seed=seed,
        data_seed=seed,
        logging_nan_inf_filter=False,
    )
    trainer = EvidenceTrainer(
        model=model,
        args=arguments,
        train_dataset=records,
        data_collator=data_collator,
        processing_class=processor,
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()

    probe_after = probe_parameter.detach().float().cpu()
    probe_max_abs_delta = float((probe_after - probe_before).abs().max().item())
    if probe_max_abs_delta == 0.0:
        raise RuntimeError(f"LoRA probe parameter did not change: {probe_name}")

    losses = [
        float(entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry and entry.get("epoch") is not None
    ]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise FloatingPointError(f"invalid logged losses: {losses}")
    if trainer.state.global_step < minimum_optimizer_steps:
        raise RuntimeError(
            f"{gate_name} completed only {trainer.state.global_step} optimizer steps; "
            f"minimum is {minimum_optimizer_steps}"
        )

    peak_allocated = torch.tensor(
        torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.long
    )
    peak_reserved = torch.tensor(
        torch.cuda.max_memory_reserved(), device="cuda", dtype=torch.long
    )
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(peak_allocated, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(peak_reserved, op=torch.distributed.ReduceOp.MAX)

    report: dict[str, Any] | None = None
    if trainer.is_world_process_zero():
        adapter_dir = output_dir / "adapter"
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(adapter_dir)
        write_json(output_dir / "resolved_config.json", config)
        effective_batch = (
            world_size
            * int(train_cfg["per_device_train_batch_size"])
            * int(train_cfg["gradient_accumulation_steps"])
        )
        vision_tower_frozen = any(
            "visual" in name.lower()
            or "vision_tower" in name.lower()
            or "vision_model" in name.lower()
            for name in frozen_names
        )
        multimodal_projector_frozen = any(
            "projector" in name.lower() or "merger" in name.lower()
            for name in frozen_names
        )
        report = {
            "passed": True,
            "gate_name": gate_name,
            "model": config["model"]["name_or_path"],
            "model_revision": config["model"].get("revision"),
            "dataset_path": str(dataset_path),
            "dataset_size": len(records),
            "world_size": world_size,
            "visible_gpu_count": visible_gpu_count,
            "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),
            "effective_global_batch_size": effective_batch,
            "optimizer_steps": trainer.state.global_step,
            "minimum_optimizer_steps": minimum_optimizer_steps,
            "train_runtime_seconds": result.metrics.get("train_runtime"),
            "losses": losses,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "all_losses_finite": True,
            "lora_probe_parameter": probe_name,
            "lora_probe_max_abs_delta": probe_max_abs_delta,
            "lora_parameter_changed": True,
            "parameter_audit": audit,
            "resolved_lora_target_count": len(resolved_lora_targets),
            "resolved_lora_targets": resolved_lora_targets,
            "frozen_vision_tensor_count": len(frozen_names),
            "vision_tower_frozen": vision_tower_frozen,
            "multimodal_projector_frozen": multimodal_projector_frozen,
            "peak_allocated_bytes": int(peak_allocated.item()),
            "peak_reserved_bytes": int(peak_reserved.item()),
            "checkpoint_path": str(adapter_dir),
            "log_history": trainer.state.log_history,
            "environment": environment_report(),
        }
        write_json(report_dir / report_filename, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return report
