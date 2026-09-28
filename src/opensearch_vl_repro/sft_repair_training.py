"""Isolated two-GPU targeted repair, continuing adapter weights only from 3k."""

from __future__ import annotations

import json
import os
import random
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from .data import SFT_INPUT_MESSAGE_VERSION, load_json_records
from .model import (freeze_vision_components, load_base_model, load_processor,
                    move_batch, parameter_audit)
from .sft_long_training import (PROJECT_ROOT, activate_sft_training_mode,
                                check_checkpoint)
from .sft_protocol_diagnostics import CORRECTED_POOL_MANIFEST_SHA256
from .sft_repair_data import validate_repair_manifest
from .sft_repair_mask import MASK_VERSIONS, RepairCollator
from .sft_tool_audit import sha256_file, write_json_atomic
from .sft_train_plan import load_main_config


REPAIR_KIND = "targeted_repair_ablation"
PARENT_STAGE = "main_b_2k"
LEGACY_REPAIR_PARENT_PROTOCOL_VERSION = "runtime-image-id-grounding-v2"


def load_repair_config(path: Path, *, max_steps: int | None = None) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("repair_mode") not in MASK_VERSIONS:
        raise ValueError("invalid repair mode/config")
    steps = int(max_steps if max_steps is not None else config["max_steps"])
    is_r3 = config.get("repair_experiment") == "r3"
    if is_r3 and path.resolve() != (PROJECT_ROOT / "configs/sft_repair_r3.yaml").resolve():
        raise ValueError("R3 requires the pinned standalone config path")
    if (is_r3 and (steps != 100 or int(config.get("max_steps", 0)) != 100
                   or config["repair_mode"] != "argument_only")):
        raise ValueError("R3 requires 100 steps and R1 argument-only supervision")
    if not is_r3 and steps not in (25, 50, 100):
        raise ValueError("repair max_steps must be 25, 50, or 100")
    if (config.get("scheduler") != "constant" or config.get("save_steps") != (
            [50, 75, 100] if is_r3 else [25, 50, 100])
            or int(config.get("world_size", 0)) != 2
            or int(config.get("micro_batch", 0)) != 1
            or int(config.get("gradient_accumulation", 0)) != 4
            or float(config.get("learning_rate", 0)) != 1e-5):
        raise ValueError("repair ablation recipe changed unexpectedly")
    mode = config["repair_mode"]
    subdir = ("r3_argument_only_derived" if is_r3 else
              "r1_argument_only" if mode == "argument_only" else "r2_full_tool_call")
    expected = {"dataset": f"data/sft_repair/{subdir}/repair.json",
                "manifest": f"data/sft_repair/{subdir}/manifest.json",
                "output_dir": f"outputs/sft_repair/{subdir}",
                "parent_checkpoint": "outputs/sft_main/checkpoint-3k",
                "base_config": "configs/sft_main.yaml"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("repair paths must remain isolated from formal SFT")
    if not is_r3 and config.get("repair_experiment") not in (None, "r1", "r2"):
        raise ValueError("unknown repair experiment")
    config["max_steps"] = steps
    return config


def validate_parent_metadata(metadata: dict[str, Any], *, model: dict[str, Any],
                             parent_path: Path) -> None:
    if (parent_path.resolve() != (PROJECT_ROOT / "outputs/sft_main/checkpoint-3k").resolve()
            or metadata.get("stage") != PARENT_STAGE
            or metadata.get("stage_complete") is not True
            or metadata.get("checkpoint_complete") is not True
            or metadata.get("lineage") != ["main_a_1k", "main_b_2k"]
            or metadata.get("model") != model["name_or_path"]
            or metadata.get("model_revision") != model["revision"]
            or metadata.get("pool_manifest_sha256") != CORRECTED_POOL_MANIFEST_SHA256
            or metadata.get("sft_input_message_version") != LEGACY_REPAIR_PARENT_PROTOCOL_VERSION):
        raise ValueError("repair parent is not the complete corrected formal checkpoint-3k")


def repair_checkpoint_metadata(*, repair: dict[str, Any], base: dict[str, Any],
                               manifest_sha: str, dataset_sha: str, parent_sha: str,
                               step: int, config_sha: str | None = None) -> dict[str, Any]:
    metadata = {
        "checkpoint_kind": REPAIR_KIND, "formal_sft_stage": False,
        "checkpoint_complete": True, "repair_mode": repair["repair_mode"],
        "repair_mask_version": MASK_VERSIONS[repair["repair_mode"]],
        "parent_checkpoint": repair["parent_checkpoint"],
        "parent_checkpoint_metadata_sha256": parent_sha,
        "source_checkpoint_lineage": ["main_a_1k", "main_b_2k"],
        "model": base["model"]["name_or_path"],
        "model_revision": base["model"]["revision"],
        "repair_dataset_manifest_sha256": manifest_sha,
        "repair_dataset_sha256": dataset_sha,
        "global_repair_step": step, "learning_rate": repair["learning_rate"],
        "scheduler": {"type": "constant", "warmup_steps": 0},
        "batch": {"world_size": repair["world_size"], "micro_batch": repair["micro_batch"],
                  "gradient_accumulation": repair["gradient_accumulation"],
                  "effective_global_batch": repair["world_size"] * repair["micro_batch"]
                  * repair["gradient_accumulation"]},
        "lora": base["lora"], "file_sha256": {},
    }
    if repair.get("repair_experiment") == "r3":
        if config_sha is None or len(config_sha) != 64:
            raise ValueError("R3 checkpoint requires config SHA-256")
        metadata.update({"repair_experiment": "r3", "repair_config_sha256": config_sha,
                         "repair_max_steps": repair["max_steps"],
                         "repair_save_steps": repair["save_steps"]})
    return metadata


def validate_repair_checkpoint(path: Path, *, mode: str | None = None) -> dict[str, Any]:
    """Reject partial/tampered artifacts and any formal-SFT checkpoint identity."""
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("checkpoint_kind") != REPAIR_KIND
            or metadata.get("formal_sft_stage") is not False
            or metadata.get("checkpoint_complete") is not True
            or metadata.get("repair_mode") not in MASK_VERSIONS
            or metadata.get("repair_mask_version") != MASK_VERSIONS[metadata["repair_mode"]]
            or (mode is not None and metadata["repair_mode"] != mode)
            or "stage" in metadata or "global_step" in metadata):
        raise ValueError("not a complete isolated repair checkpoint")
    if (metadata.get("repair_experiment") not in (None, "r3")
            or ("r3_argument_only_derived" in path.parts
                and metadata.get("repair_experiment") != "r3")):
        raise ValueError("repair checkpoint experiment identity changed")
    if metadata.get("repair_experiment") == "r3":
        if (metadata.get("repair_mode") != "argument_only"
                or metadata.get("parent_checkpoint") != "outputs/sft_main/checkpoint-3k"
                or metadata.get("source_checkpoint_lineage") != ["main_a_1k", "main_b_2k"]
                or metadata.get("repair_max_steps") != 100
                or metadata.get("repair_save_steps") != [50, 75, 100]
                or metadata.get("global_repair_step") not in (50, 75, 100)
                or not isinstance(metadata.get("repair_config_sha256"), str)
                or len(metadata["repair_config_sha256"]) != 64):
            raise ValueError("R3 checkpoint recipe or corrected-3k lineage changed")
        root = Path(__file__).resolve().parents[2]
        if (metadata["repair_config_sha256"] != sha256_file(
                root / "configs/sft_repair_r3.yaml")
                or metadata.get("repair_dataset_manifest_sha256") != sha256_file(
                    root / "data/sft_repair/r3_argument_only_derived/manifest.json")):
            raise ValueError("R3 checkpoint config/manifest provenance changed")
    checksums = metadata.get("file_sha256")
    if (not isinstance(checksums, dict) or not checksums
            or not any(name.startswith("adapter/") and name.endswith(".safetensors")
                       for name in checksums)):
        raise ValueError("repair checkpoint lacks adapter artifact checksums")
    for name, expected in checksums.items():
        if not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError("unsafe repair checkpoint artifact path")
        if sha256_file(path / name) != expected:
            raise ValueError(f"repair checkpoint checksum mismatch: {name}")
    return metadata


def require_r3_mask_audit(config_path: Path, dataset_path: Path,
                          manifest_path: Path, audit_path: Path) -> dict[str, Any]:
    """Reject stale or failed real-processor audits before a GPU run starts."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    overlap_keys = ("eval300_id_overlap_count", "eval300_question_overlap_count",
                    "eval300_image_overlap_count", "dev50_id_overlap_count",
                    "dev50_question_overlap_count", "dev50_image_overlap_count",
                    "frozen_exclusion_overlap_count", "image_contract_bad_count")
    overlaps = audit.get("overlap_counts")
    if (audit.get("passed") is not True or audit.get("sample_count") != 600
            or audit.get("repair_experiment") != "r3"
            or audit.get("repair_config_sha256") != sha256_file(config_path)
            or audit.get("repair_dataset_sha256") != sha256_file(dataset_path)
            or audit.get("repair_manifest_sha256") != sha256_file(manifest_path)
            or not isinstance(overlaps, dict)
            or any(overlaps.get(key) != 0 for key in overlap_keys)):
        raise ValueError("R3 requires a passing, current full processor/mask audit")
    return audit


def _save_repair_checkpoint(path: Path, *, model: Any, processor: Any,
                            optimizer: Any, scheduler: Any,
                            metadata: dict[str, Any], dist: Any, rank: int,
                            torch: Any) -> None:
    failure = [None]
    if rank == 0:
        try:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite repair checkpoint: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
            model.module.save_pretrained(temporary / "adapter", safe_serialization=True)
            processor.save_pretrained(temporary / "adapter")
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
            metadata["file_sha256"] = {item.relative_to(temporary).as_posix(): sha256_file(item)
                                       for item in sorted(temporary.rglob("*")) if item.is_file()}
            write_json_atomic(temporary / "metadata.json", metadata)
            os.replace(temporary, path)
        except Exception as exc:
            failure[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"repair checkpoint save failed: {failure[0]}")
    dist.barrier()


def run_repair(config_path: Path, *, max_steps: int | None = None) -> None:
    if SFT_INPUT_MESSAGE_VERSION != LEGACY_REPAIR_PARENT_PROTOCOL_VERSION:
        raise RuntimeError("legacy v2 targeted repair cannot train under the v3 image_id runtime")
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    repair = load_repair_config(config_path, max_steps=max_steps)
    base = load_main_config(PROJECT_ROOT / repair["base_config"],
                            base_eval_config=PROJECT_ROOT / "configs/eval_base_300.yaml")
    dataset_path = PROJECT_ROOT / repair["dataset"]
    manifest_path = PROJECT_ROOT / repair["manifest"]
    manifest = validate_repair_manifest(dataset_path, manifest_path,
                                        mode=repair["repair_mode"])
    if (sha256_file(PROJECT_ROOT / "data/sft_main/manifest.json")
            != manifest["corrected_pool_manifest_sha256"]):
        raise ValueError("current corrected SFT pool does not match repair provenance")
    if (sha256_file(PROJECT_ROOT / "data/eval/tool_protocol_dev50/"
                    "tool_protocol_dev50_manifest.json")
            != manifest["dev50_manifest_sha256"]):
        raise ValueError("current Dev50 membership does not match repair provenance")
    records = load_json_records(dataset_path)
    expected_count = 600 if repair.get("repair_experiment") == "r3" else 300
    if len(records) != manifest["sample_count"] or len(records) != expected_count:
        raise ValueError("repair dataset count mismatch")
    if ((repair.get("repair_experiment") == "r3") !=
            (manifest.get("repair_experiment") == "r3")):
        raise ValueError("repair config/manifest experiment mismatch")
    if repair.get("repair_experiment") == "r3":
        require_r3_mask_audit(config_path, dataset_path, manifest_path,
                              PROJECT_ROOT / "reports/sft_repair/r3_mask_audit.json")
    if any(not (dataset_path.parent / image).is_file()
           for record in records for image in record["images"]):
        raise FileNotFoundError("repair media not materialized")
    parent_path = PROJECT_ROOT / repair["parent_checkpoint"]
    parent = check_checkpoint(parent_path)
    validate_parent_metadata(parent, model=base["model"], parent_path=parent_path)
    parent_sha = sha256_file(parent_path / "metadata.json")
    output_dir = PROJECT_ROOT / repair["output_dir"]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite repair run: {output_dir}")
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or int(os.environ.get("WORLD_SIZE", "1")) != 2 or torch.cuda.device_count() != 2):
        raise RuntimeError("repair requires torchrun on exactly two BF16 CUDA GPUs")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    seed = int(base["project"]["seed"]) + 1000
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(base["training"].get("tf32", True))
    if rank == 0:
        output_dir.mkdir(parents=True)
    dist.barrier()
    processor = load_processor(base)
    raw_model = load_base_model(base, for_training=True)
    freeze_vision_components(raw_model)
    model = PeftModel.from_pretrained(raw_model, parent_path / "adapter", is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = model.to(f"cuda:{local_rank}")
    activate_sft_training_mode(model)
    audit = parameter_audit(model)
    if parent.get("trainable_parameter_names") != audit["trainable_parameter_names"]:
        raise ValueError("repair parent LoRA parameter names differ from loaded adapter")
    wrapped = DistributedDataParallel(model, device_ids=[local_rank],
                                      output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters()
                                   if parameter.requires_grad),
                                  lr=float(repair["learning_rate"]),
                                  weight_decay=float(base["training"].get("weight_decay", 0.0)))
    # This is a new short experiment, not a continuation of the formal optimizer.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    sampler = DistributedSampler(records, num_replicas=2, rank=rank,
                                 shuffle=True, seed=seed, drop_last=True)
    collator = RepairCollator(processor, dataset_path, int(base["data"]["max_length"]),
                              repair["repair_mode"])
    optimizer.zero_grad(set_to_none=True)
    device = torch.device(f"cuda:{local_rank}")
    step = 0
    epoch = 0
    micro = 0
    while step < repair["max_steps"]:
        sampler.set_epoch(epoch)
        loader = DataLoader(records, batch_size=1, sampler=sampler,
                            collate_fn=collator, num_workers=0)
        for batch in loader:
            batch = move_batch(batch, device)
            micro += 1
            context = nullcontext() if micro == 4 else wrapped.no_sync()
            with context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = wrapped(**batch).loss
                finite = torch.tensor(bool(torch.isfinite(loss.detach()).all()), device=device)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite.item()):
                    raise FloatingPointError("non-finite repair loss")
                (loss / 4).backward()
            if micro < 4:
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            micro = 0
            if rank == 0:
                print(f"[repair] mode={repair['repair_mode']} step={step}/"
                      f"{repair['max_steps']} loss={loss.detach().float().item():.4f}", flush=True)
            if step in repair["save_steps"]:
                checkpoint = output_dir / f"checkpoint-step{step}"
                metadata = repair_checkpoint_metadata(
                    repair=repair, base=base, manifest_sha=sha256_file(manifest_path),
                    dataset_sha=sha256_file(dataset_path), parent_sha=parent_sha,
                    step=step, config_sha=sha256_file(config_path))
                _save_repair_checkpoint(checkpoint, model=wrapped, processor=processor,
                                        optimizer=optimizer, scheduler=scheduler,
                                        metadata=metadata, dist=dist, rank=rank, torch=torch)
            if step >= repair["max_steps"]:
                break
        epoch += 1
    dist.barrier()
    dist.destroy_process_group()
