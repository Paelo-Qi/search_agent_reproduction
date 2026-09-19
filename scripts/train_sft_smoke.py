#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.config import load_config  # noqa: E402
from opensearch_vl_repro.data import OpenSearchVLCollator, load_json_records  # noqa: E402
from opensearch_vl_repro.model import (  # noqa: E402
    add_lora,
    freeze_vision_components,
    load_base_model,
    load_processor,
    parameter_audit,
    select_probe_parameter,
)
from opensearch_vl_repro.reporting import environment_report, write_json  # noqa: E402


def main() -> None:
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    class EvidenceTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any):
            result = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            loss, outputs = result
            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError(f"non-finite training loss at step {self.state.global_step}")
            return (loss, outputs) if return_outputs else loss

        def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                logs = dict(logs)
                logs["gpu_allocated_gib"] = torch.cuda.memory_allocated(device) / 2**30
                logs["gpu_reserved_gib"] = torch.cuda.memory_reserved(device) / 2**30
                logs["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            super().log(logs, start_time=start_time)

    parser = argparse.ArgumentParser(description="Run the 20-step BF16 LoRA SFT smoke test.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sft_smoke.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 0 training gate")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError(f"Phase 0 is specified for exactly 2 GPUs; WORLD_SIZE={world_size}")

    seed = int(config["project"]["seed"])
    set_seed(seed)
    dataset_path = (PROJECT_ROOT / config["data"]["path"]).resolve()
    records = load_json_records(dataset_path)
    expected = int(config["data"].get("expected_samples", 100))
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} smoke samples, found {len(records)}")

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
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        max_steps=int(train_cfg["max_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy="no",
        bf16=bool(train_cfg.get("bf16", True)),
        tf32=bool(train_cfg.get("tf32", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 2)),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tensorboard"),
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
    if trainer.state.global_step < 10:
        raise RuntimeError(f"only {trainer.state.global_step} optimizer steps completed")

    peak_allocated = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.long)
    peak_reserved = torch.tensor(torch.cuda.max_memory_reserved(), device="cuda", dtype=torch.long)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(peak_allocated, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(peak_reserved, op=torch.distributed.ReduceOp.MAX)

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
        report = {
            "passed": True,
            "model": config["model"]["name_or_path"],
            "model_revision": config["model"].get("revision"),
            "dataset_path": str(dataset_path),
            "dataset_size": len(records),
            "world_size": world_size,
            "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),
            "effective_global_batch_size": effective_batch,
            "optimizer_steps": trainer.state.global_step,
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
            "peak_allocated_bytes": int(peak_allocated.item()),
            "peak_reserved_bytes": int(peak_reserved.item()),
            "checkpoint_path": str(adapter_dir),
            "log_history": trainer.state.log_history,
            "environment": environment_report(),
        }
        write_json(report_dir / "training.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
