#!/usr/bin/env python3
"""One-step, two-GPU DDP VRAM stress test on the longest formal SFT sample.

Both ranks intentionally use the SAME longest sample. This is a worst-case
memory test, not the formal DistributedSampler or global-batch training run.
No scheduler, checkpoint, adapter save, or shard write is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_sft_vram import (  # noqa: E402
    GIB, find_language_decoder_layers, processor_token_length,
    resolved_attention_flags, select_sample, training_record, validate_dataset,
)
from opensearch_vl_repro.data import OpenSearchVLCollator, load_json_records  # noqa: E402
from opensearch_vl_repro.model import (  # noqa: E402
    add_lora, freeze_vision_components, load_base_model, load_processor,
    move_batch, parameter_audit,
)
from opensearch_vl_repro.sft_long_training import activate_sft_training_mode  # noqa: E402
from opensearch_vl_repro.sft_train_plan import load_main_config  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/sft_main.yaml"
DEFAULT_DATA = ROOT / "data/sft_main/main_a_1k.json"


def optimizer_spec(model: Any, config: dict[str, Any]) -> tuple[list[Any], dict[str, float]]:
    """Use precisely the trainable parameters and phase-1 AdamW settings."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("no trainable LoRA parameters for AdamW")
    return parameters, {
        "lr": float(config["scheduler"]["phase_1"]["peak_lr"]),
        "weight_decay": float(config["training"].get("weight_decay", 0.0)),
    }


def select_longest_record(records: list[dict[str, Any]], processor: Any,
                          data_path: Path, *, rank: int) -> tuple[int, dict[str, Any], int]:
    """Delegate the untruncated multimodal scan to the existing diagnostic."""
    scanned = 0

    def length_of(raw: dict[str, Any]) -> int:
        nonlocal scanned
        length = processor_token_length(processor, training_record(raw), data_path)
        scanned += 1
        if rank == 0 and scanned % 100 == 0:
            print(f"[SCAN] rank=0 processor_lengths_completed={scanned}/{len(records)}", flush=True)
        return length

    index, raw, length = select_sample(
        records, sample_index=None, select_longest=True, token_length=length_of)
    return index, training_record(raw), length


def memory_stats(torch: Any, device: Any) -> dict[str, float]:
    return {
        "allocated_gib": round(torch.cuda.memory_allocated(device) / GIB, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(device) / GIB, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated(device) / GIB, 3),
        "max_reserved_gib": round(torch.cuda.max_memory_reserved(device) / GIB, 3),
    }


def print_memory(torch: Any, device: Any, rank: int, stage: str,
                 *, synchronize: bool = True) -> dict[str, float]:
    if synchronize:
        torch.cuda.synchronize(device)
    memory = memory_stats(torch, device)
    print(f"[VRAM] rank={rank} stage={stage} "
          + " ".join(f"{key}={value:.3f}" for key, value in memory.items()), flush=True)
    return memory


def oom_event(rank: int, stage: str, memory: dict[str, float]) -> dict[str, Any]:
    """Pure, unambiguous failure marker for torchrun logs."""
    return {"passed": False, "rank": rank, "oom_stage": stage, **memory}


def aggregate_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank-0 summary; only complete, matching two-rank runs can pass."""
    by_rank = {int(report["rank"]): report for report in reports}
    if len(reports) != 2 or set(by_rank) != {0, 1}:
        raise ValueError("DDP stress summary requires exactly ranks 0 and 1")
    first = by_rank[0]
    if any((report["sample_id"], report["sample_index"], report["actual_token_length"])
           != (first["sample_id"], first["sample_index"], first["actual_token_length"])
           for report in by_rank.values()):
        raise ValueError("DDP ranks selected different longest samples")
    per_rank_allocated = {str(rank): report["peak_allocated_gib"]
                          for rank, report in sorted(by_rank.items())}
    per_rank_reserved = {str(rank): report["peak_reserved_gib"]
                         for rank, report in sorted(by_rank.items())}
    return {
        "passed": all(report["optimizer_step_completed"] and math.isfinite(report["loss"])
                      for report in by_rank.values()),
        "stress_semantics": "same_longest_sample_on_both_ranks_not_formal_sampler",
        "sample_id": first["sample_id"],
        "sample_index": first["sample_index"],
        "actual_token_length": first["actual_token_length"],
        "per_rank_peak_allocated_gib": per_rank_allocated,
        "per_rank_peak_reserved_gib": per_rank_reserved,
        "overall_peak_allocated_gib": max(per_rank_allocated.values()),
        "overall_peak_reserved_gib": max(per_rank_reserved.values()),
        "loss_by_rank": {str(rank): report["loss"] for rank, report in sorted(by_rank.items())},
        "optimizer_step_completed": all(report["optimizer_step_completed"]
                                        for report in by_rank.values()),
    }


def validate_stress_config(config: dict[str, Any]) -> None:
    training = config["training"]
    if (config["model"].get("attn_implementation") != "flash_attention_2"
            or int(training["world_size"]) != 2
            or int(training["per_device_train_batch_size"]) != 1
            or int(training["gradient_accumulation_steps"]) != 4):
        raise ValueError("DDP stress requires formal FA2, 2 GPUs, micro=1, accum=4 config")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--select-longest", action="store_true",
                        help="Required: scan actual untruncated multimodal processor lengths")
    args = parser.parse_args()
    if not args.select_longest:
        parser.error("DDP stress requires --select-longest; no sample index is hard-coded")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        parser.error("launch with torchrun --nproc_per_node=2")

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("DDP stress requires exactly two visible CUDA GPUs")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"rank {rank} GPU {local_rank} does not support BF16")
    device = torch.device(f"cuda:{local_rank}")
    stage = "process_group_init"
    try:
        dist.init_process_group("nccl")
        config = load_main_config(args.config, base_eval_config=ROOT / "configs/eval_base_300.yaml")
        validate_stress_config(config)
        data_path, expected = validate_dataset(config, args.data)
        if data_path != DEFAULT_DATA.resolve() or expected != 1000:
            raise ValueError("DDP stress only accepts the formal main_a_1k shard")
        records = load_json_records(data_path)
        if len(records) != expected:
            raise ValueError(f"expected {expected} official main_a_1k records, got {len(records)}")
        seed = int(config["project"]["seed"]) + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("tf32", True))

        properties = torch.cuda.get_device_properties(device)
        print(f"[CUDA] rank={rank} device={device} name={properties.name} "
              f"total_gib={properties.total_memory / GIB:.3f}", flush=True)
        stage = "processor_scan"
        processor = load_processor(config)
        index, sample, actual_length = select_longest_record(
            records, processor, data_path, rank=rank)
        sample_id = sample.get("_sample_id", "unassigned")
        selection = {"index": index, "sample_id": sample_id, "length": actual_length}
        selections: list[Any] = [None, None]
        dist.all_gather_object(selections, selection)
        if selections[0] != selections[1]:
            raise RuntimeError(f"ranks disagree on longest sample: {selections}")
        print(f"[SAMPLE] rank={rank} index={index} sample_id={sample_id} "
              f"source={sample.get('_source')} original_index={sample.get('_source_index')} "
              f"actual_token_length={actual_length} image_count={len(sample.get('images', []))} "
              "same_sample_both_ranks=True formal_sampler=False", flush=True)

        stage = "model_load"
        base = load_base_model(config, for_training=True)
        print_memory(torch, device, rank, "model_loaded")
        stage = "freeze_vision"
        frozen_names = freeze_vision_components(base)
        print(f"[MODEL] rank={rank} frozen_vision_projector_tensors={len(frozen_names)}", flush=True)
        stage = "lora_injection"
        model, targets = add_lora(base, config)
        print(f"[MODEL] rank={rank} resolved_lora_targets={len(targets)}", flush=True)
        stage = "gradient_checkpointing"
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        stage = "model_on_cuda"
        model = model.to(device)
        activate_sft_training_mode(model)
        print_memory(torch, device, rank, "model_on_cuda")
        parameter_audit(model)
        _, language_model, layers = find_language_decoder_layers(model)
        training_count = sum(layer.training for layer in layers)
        checkpoint_count = sum(bool(getattr(layer, "gradient_checkpointing", False))
                               for layer in layers)
        flags = resolved_attention_flags(model, language_model, layers, "flash_attention_2")
        print(f"[MODEL] rank={rank} model_training={model.training} "
              f"language_model_training={language_model.training} "
              f"decoder_layers_training={training_count}/36 "
              f"decoder_layers_gradient_checkpointing={checkpoint_count}/36 "
              f"effective_attention_implementation={flags['effective_attention_implementation']} "
              f"use_cache={flags['use_cache']} text_use_cache={flags['text_use_cache']}", flush=True)
        if not flags["attention_matches_request"]:
            raise RuntimeError(f"rank {rank} did not resolve flash_attention_2: {flags}")

        stage = "ddp_wrap"
        wrapped = DistributedDataParallel(model, device_ids=[local_rank],
                                          output_device=local_rank, find_unused_parameters=False)
        print_memory(torch, device, rank, "ddp_wrapped")
        stage = "optimizer_create"
        parameters, options = optimizer_spec(model, config)
        optimizer = torch.optim.AdamW(parameters, **options)
        print(f"[OPTIMIZER] rank={rank} type=AdamW lr={options['lr']} "
              f"weight_decay={options['weight_decay']} trainable_tensors={len(parameters)}",
              flush=True)
        print_memory(torch, device, rank, "optimizer_created")

        stage = "collate"
        collator = OpenSearchVLCollator(processor, data_path, int(config["data"]["max_length"]))
        batch = collator([sample])
        if batch["input_ids"].shape[0] != 1 or "pixel_values" not in batch:
            raise RuntimeError("DDP stress requires one real multimodal training sample per rank")
        shapes = {name: list(value.shape) for name, value in batch.items()
                  if hasattr(value, "shape")}
        print(f"[BATCH] rank={rank} input_ids_shape={shapes.get('input_ids')} "
              f"pixel_values_shape={shapes.get('pixel_values')} all_shapes={shapes}", flush=True)
        stage = "batch_to_cuda"
        batch = move_batch(batch, device)
        print_memory(torch, device, rank, "batch_on_cuda")

        stage = "forward"
        print_memory(torch, device, rank, "before_forward")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = wrapped(**batch).loss
        print_memory(torch, device, rank, "after_forward")
        finite = torch.tensor(bool(torch.isfinite(loss.detach()).all()), device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.item()):
            raise FloatingPointError("non-finite DDP stress loss")
        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError("non-finite DDP stress loss")
        print(f"[LOSS] rank={rank} finite=True value={loss_value}", flush=True)

        stage = "backward"
        loss.backward()  # One micro-batch only; no accumulation or no_sync().
        print_memory(torch, device, rank, "after_backward")
        stage = "optimizer_step"
        optimizer.step()
        print_memory(torch, device, rank, "after_optimizer_step")
        stage = "zero_grad"
        optimizer.zero_grad(set_to_none=True)
        final_memory = print_memory(torch, device, rank, "after_zero_grad")
        print(f"[OPTIMIZER] rank={rank} step_completed=True zero_grad_completed=True", flush=True)

        report = {
            "rank": rank, "sample_id": sample_id, "sample_index": index,
            "actual_token_length": actual_length, "loss": loss_value,
            "peak_allocated_gib": final_memory["max_allocated_gib"],
            "peak_reserved_gib": final_memory["max_reserved_gib"],
            "optimizer_step_completed": True,
        }
        reports: list[Any] = [None, None]
        stage = "summary_gather"
        dist.all_gather_object(reports, report)
        if rank == 0:
            summary = aggregate_summary(reports)
            print("[SUMMARY] " + json.dumps(summary, sort_keys=True), flush=True)
            if not summary["passed"]:
                raise RuntimeError("DDP stress did not pass on both ranks")
        return 0
    except torch.OutOfMemoryError:
        print("[OOM] " + json.dumps(
            oom_event(rank, stage, memory_stats(torch, device)), sort_keys=True), flush=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
