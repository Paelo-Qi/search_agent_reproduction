"""Pure, testable optimizer-step and continuation planning for formal 4B SFT."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .inference.config import load_inference_config
from .sft_main_data import SHARD_SIZES
from .data import SFT_INPUT_MESSAGE_VERSION


STAGE_ORDER = tuple(SHARD_SIZES)
STAGE_NAMES = {"main_a_1k": "checkpoint-1k", "main_b_2k": "checkpoint-3k",
               "extra_1k": "checkpoint-4k", "reserve_4k": "checkpoint-8k"}


@dataclass(frozen=True)
class StagePlan:
    stage: str
    scheduler_phase: str
    shard_samples: int
    epochs: int
    world_size: int
    micro_batch: int
    gradient_accumulation: int
    effective_global_batch: int
    stage_steps: int
    phase_total_steps: int
    phase_start_step: int
    global_start_step: int
    global_target_step: int
    warmup_steps: int
    scheduler_type: str
    peak_lr: float
    lineage: tuple[str, ...]


def optimizer_steps(samples: int, epochs: int, effective_global_batch: int) -> int:
    if min(samples, epochs, effective_global_batch) <= 0:
        raise ValueError("samples, epochs, and effective global batch must be positive")
    return math.ceil(samples * epochs / effective_global_batch)


def load_main_config(path: str | Path, *, base_eval_config: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("SFT config must be a mapping")
    base = load_inference_config(base_eval_config)
    model = config["model"]
    if (model["name_or_path"], model["revision"]) != (base.model_name_or_path, base.revision):
        raise ValueError("SFT base model/revision must match formal Base Eval-300")
    if model["dtype"] != "bfloat16" or not model["freeze_vision_tower"] or not model["freeze_multimodal_projector"]:
        raise ValueError("formal SFT requires BF16 and frozen vision/projector")
    if model.get("quantization") not in (None, False):
        raise ValueError("formal SFT does not use quantization")
    if config["data"]["max_length"] != 32000 or model["image_max_pixels"] != 262144:
        raise ValueError("formal SFT image/sequence limits changed unexpectedly")
    if config["training"]["world_size"] != 2:
        raise ValueError("formal SFT requires two GPUs")
    if not config["training"].get("bf16") or not config["training"].get("gradient_checkpointing"):
        raise ValueError("formal SFT requires BF16 training and gradient checkpointing")
    lora = config["lora"]
    if (lora["rank"], lora["alpha"], float(lora["dropout"])) != (16, 32, .05):
        raise ValueError("formal SFT LoRA rank/alpha/dropout changed unexpectedly")
    if set(lora["target_modules"]) != {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }:
        raise ValueError("formal SFT LoRA target modules changed unexpectedly")
    return config


def plan_stage(config: dict[str, Any], stage: str, *, micro_batch: int | None = None,
               gradient_accumulation: int | None = None,
               phase_2_peak_lr: float | None = None) -> StagePlan:
    if stage not in STAGE_ORDER and stage != "smoke":
        raise ValueError(f"unknown SFT stage: {stage}")
    training = config["training"]
    world = int(training["world_size"])
    micro = int(micro_batch or training["per_device_train_batch_size"])
    accum = int(gradient_accumulation or training["gradient_accumulation_steps"])
    global_batch = world * micro * accum
    if world != 2 or global_batch != 8:
        raise ValueError("formal 4B SFT requires 2 GPUs and effective global batch 8")
    if stage == "smoke":
        target = int(training["max_steps"])
        phase = config["scheduler"]["phase_1"]
        warmup = math.ceil(target * float(phase["warmup_ratio"]))
        return StagePlan(stage, "smoke", int(config["data"]["expected_samples"]),
                         2, world, micro, accum, global_batch, target, target, 0, 0,
                         target, warmup, phase["type"], float(phase["peak_lr"]), (stage,))
    epochs = int(config["data"]["epochs_per_shard"])
    stage_steps = {name: optimizer_steps(size, epochs, global_batch)
                   for name, size in SHARD_SIZES.items()}
    # Exact shard boundaries avoid padded/repeated examples and give unambiguous resume offsets.
    if any(size * epochs % global_batch for size in SHARD_SIZES.values()):
        raise ValueError("shard sample-epochs must be divisible by effective global batch")
    index = STAGE_ORDER.index(stage)
    global_start = sum(stage_steps[name] for name in STAGE_ORDER[:index])
    global_target = global_start + stage_steps[stage]
    scheduler_phase = "phase_2" if stage == "reserve_4k" else "phase_1"
    if scheduler_phase == "phase_1":
        horizon = sum(stage_steps[name] for name in STAGE_ORDER[:3])
        phase_start = global_start
    else:
        horizon = stage_steps["reserve_4k"]
        phase_start = 0
    scheduler = config["scheduler"][scheduler_phase]
    peak = phase_2_peak_lr if scheduler_phase == "phase_2" else scheduler["peak_lr"]
    if peak is None or float(peak) <= 0:
        raise ValueError("Phase 2 peak LR must be explicitly chosen after the 4k checkpoint")
    if scheduler["type"] != "cosine":
        raise ValueError("only the audited cosine scheduler is supported")
    warmup = math.ceil(horizon * float(scheduler["warmup_ratio"]))
    if not 0 <= warmup < horizon:
        raise ValueError("invalid warmup horizon")
    return StagePlan(stage, scheduler_phase, SHARD_SIZES[stage], epochs, world, micro,
                     accum, global_batch, stage_steps[stage], horizon, phase_start,
                     global_start, global_target, warmup, scheduler["type"], float(peak),
                     STAGE_ORDER[:index + 1])


def cosine_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < 0 or total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("invalid scheduler step/horizon")
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return .5 * (1 + math.cos(math.pi * progress))


def validate_resume_metadata(plan: StagePlan, metadata: dict[str, Any], *,
                             model_name: str, revision: str, pool_sha256: str,
                             checkpoint_complete: bool = True) -> str:
    if not checkpoint_complete:
        raise ValueError("checkpoint is incomplete")
    if metadata.get("model") != model_name or metadata.get("model_revision") != revision:
        raise ValueError("checkpoint base model/revision mismatch")
    if metadata.get("pool_manifest_sha256") != pool_sha256:
        raise ValueError("checkpoint dataset manifest mismatch")
    if metadata.get("sft_input_message_version") != SFT_INPUT_MESSAGE_VERSION:
        raise ValueError("checkpoint SFT input/message format version mismatch")
    if (metadata.get("world_size"), metadata.get("micro_batch"),
            metadata.get("gradient_accumulation")) != (
            plan.world_size, plan.micro_batch, plan.gradient_accumulation):
        raise ValueError("checkpoint batch/DDP configuration mismatch")
    previous_stage = metadata.get("stage")
    if previous_stage == plan.stage:
        if (metadata.get("scheduler_phase") != plan.scheduler_phase or
                metadata.get("phase_total_steps") != plan.phase_total_steps or
                metadata.get("peak_lr") != plan.peak_lr):
            raise ValueError("same-stage scheduler identity mismatch")
        if not 0 <= metadata.get("global_step", -1) < plan.global_target_step:
            raise ValueError("same-stage checkpoint is past target")
        expected_phase_step = (plan.phase_start_step + metadata["global_step"]
                               - plan.global_start_step)
        if metadata.get("phase_step") != expected_phase_step:
            raise ValueError("same-stage checkpoint scheduler/global step mismatch")
        return "same_stage"
    if plan.stage == "smoke" or plan.stage == STAGE_ORDER[0]:
        raise ValueError("first stage must start from pinned Base, not another checkpoint")
    expected_previous = STAGE_ORDER[STAGE_ORDER.index(plan.stage) - 1]
    if (previous_stage != expected_previous or
            metadata.get("global_step") != plan.global_start_step or
            metadata.get("stage_complete") is not True):
        raise ValueError("checkpoint is not the exact preceding completed stage")
    if metadata.get("lineage") != list(plan.lineage[:-1]):
        raise ValueError("checkpoint shard lineage mismatch")
    if plan.scheduler_phase == "phase_1":
        if (metadata.get("scheduler_phase") != "phase_1" or
                metadata.get("phase_total_steps") != plan.phase_total_steps or
                metadata.get("phase_step") != plan.phase_start_step or
                metadata.get("peak_lr") != plan.peak_lr):
            raise ValueError("Phase 1 scheduler/global step would reset")
    elif (metadata.get("scheduler_phase") != "phase_1" or
          metadata.get("phase_step") != metadata.get("phase_total_steps") or
          metadata.get("phase_total_steps") != plan.global_start_step):
        raise ValueError("Phase 2 must start only from completed Phase 1 checkpoint-4k")
    return "next_stage"
