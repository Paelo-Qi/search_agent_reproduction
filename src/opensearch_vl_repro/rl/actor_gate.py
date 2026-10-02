"""CPU/import-safe contracts for the isolated Gate A2.2 (not an RL objective)."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import yaml

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.model import VISION_NAME_FRAGMENTS, parameter_audit
from opensearch_vl_repro.rl.data import safe_image_relpath, validate_manifest
from opensearch_vl_repro.rl.quality_audit import load_quality_audit

BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
REQUIRED_CHECKS = (
    "distributed_initialized", "expected_world_size", "base_loaded", "sft_lora_loaded",
    "vision_frozen", "projector_frozen", "base_frozen", "trainable_lora_present",
    "optimizer_only_lora", "training_checkpointing_active", "fsdp2_wrapped",
    "real_multimodal_batch", "loss_finite", "backward_completed", "lora_grad_finite",
    "nonzero_lora_grad", "optimizer_step_completed", "parameters_finite",
    "lora_param_changed", "checkpoint_saved", "optimizer_state_saved",
    "original_model_destroyed", "fresh_adapter_reloaded", "adapter_fingerprint_match",
    "reload_param_match", "native_checkpoint_reloaded", "fresh_forward_finite",
)


def load_gate_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("gate config must be a mapping")
    if set(value) != {"model", "revision", "dtype", "fsdp_mode", "microbatch",
                      "gradient_checkpointing", "optimizer", "software_contract"}:
        raise ValueError("unknown or missing gate config fields")
    expected = {"model": BASE_MODEL, "revision": BASE_REVISION, "dtype": "bfloat16",
                "fsdp_mode": "fsdp2", "microbatch": 1, "gradient_checkpointing": True}
    for name, required in expected.items():
        if type(value.get(name)) is not type(required) or value[name] != required:
            raise ValueError(f"Gate A2.2 requires {name}={required}")
    optimizer = value.get("optimizer", {})
    if (not isinstance(optimizer, dict) or set(optimizer) != {"type", "learning_rate", "weight_decay"}
            or optimizer.get("type") != "AdamW"
            or optimizer.get("learning_rate") != 1e-6 or optimizer.get("weight_decay") != 0.0):
        raise ValueError("gate optimizer must be AdamW lr=1e-6 weight_decay=0")
    if value.get("software_contract") != {
        "torch_series": "2.8", "transformers": "4.57.1", "peft": "0.21.1", "verl": "0.6.1",
    }:
        raise ValueError("unsupported gate software contract; review actual APIs before changing")
    return value


def validate_software(versions: dict[str, str], gate: dict[str, Any]) -> None:
    contract = gate["software_contract"]
    if not versions["torch"].split("+")[0].startswith(contract["torch_series"] + "."):
        raise ValueError("Gate A2.2 requires torch 2.8.x")
    for name in ("transformers", "peft", "verl"):
        if versions[name].split("+")[0] != contract[name]:
            raise ValueError(f"Gate A2.2 requires {name}=={contract[name]}, got {versions[name]}")


def select_rank_sample(records: list[dict[str, Any]], *, rank: int,
                       world_size: int, max_samples: int | None = None) -> dict[str, Any]:
    if type(world_size) is not int or world_size < 2 or not 0 <= rank < world_size:
        raise ValueError("gate requires a distributed launcher with at least two ranks")
    limit = world_size if max_samples is None else max_samples
    if type(limit) is not int or limit < world_size or len(records) < world_size:
        raise ValueError("one distinct sample per rank requires max_samples >= world_size")
    # No sampler padding, repetition, trajectory/group semantics or rollout involved.
    return records[rank]


def temporary_sft_record(row: dict[str, Any], source_root: str | Path) -> dict[str, Any]:
    identity = row.get("source_sample_id")
    if (not isinstance(identity, str) or re.fullmatch(r"rl_\d{6}", identity) is None
            or row.get("prompt_id") != identity or "trajectory_group_id" in row):
        raise ValueError("invalid schema-v3 source/prompt identity")
    for name in ("question", "reference_answer"):
        if not isinstance(row.get(name), str) or not row[name].strip() or "<image>" in row[name]:
            raise ValueError(f"invalid temporary supervised {name}")
    relpaths = row.get("image_relpaths")
    if not isinstance(relpaths, list) or not relpaths:
        raise ValueError("real multimodal gate requires source image_relpaths")
    root = Path(source_root).expanduser().resolve()
    images = [(root / safe_image_relpath(item)).resolve() for item in relpaths]
    if any(not path.is_relative_to(root) or not path.is_file() for path in images):
        raise FileNotFoundError("source images missing or escaping source root")
    # The already frozen source-image identities must still describe the files
    # used by this gate. Only the selected rank's images are opened/hashed.
    from opensearch_vl_repro.agent.reliability import image_sha256
    from opensearch_vl_repro.rl.data import question_sha256
    if (row.get("question_hash") != question_sha256(row["question"])
            or row.get("image_hashes") != [image_sha256(path) for path in images]):
        raise ValueError("gate source question/image hashes differ from frozen records")
    return {"id": identity, "images": [str(path) for path in images], "conversations": [
        {"from": "human", "value": "<image>\n" * len(images) + row["question"]},
        {"from": "gpt", "value": row["reference_answer"]},
    ]}


def load_smoke_records(data: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read-only binding to the previously prepared quality-clean schema-v3 smoke."""
    records = json.loads(data.read_text(encoding="utf-8"))
    manifest = json.loads(data.with_name(data.stem + "_manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    settings = config["data"]
    quality = load_quality_audit(settings["quality_audit_dir"], main_count=settings["main_count"])
    if (not isinstance(records, list) or len(records) != settings["smoke_count"]
            or manifest.get("name") != "smoke" or manifest.get("selected_count") != len(records)
            or manifest.get("samples_sha256") != canonical_json_sha256(records)
            or manifest.get("dataset_id") != settings["dataset_id"]
            or manifest.get("dataset_revision") != settings["dataset_revision"]
            or manifest.get("selection_seed") != settings["seed"]
            or manifest.get("selection_version") != settings["selection_version"]):
        raise ValueError("gate smoke data differs from pinned schema-v3 manifest")
    ids = [row["source_sample_id"] for row in records]
    entries = [{"candidate_rank": row.get("quality_candidate_rank"), "source_sample_id": row["source_sample_id"]}
               for row in records]
    if (len(ids) != len(set(ids)) or manifest.get("membership") != ids
            or entries != list(quality.selected[:len(records)])
            or any(manifest.get(key) != value for key, value in quality.provenance.items())):
        raise ValueError("smoke membership is not the frozen first quality-ok prefix")
    return records, manifest


def validate_output_paths(output: Path, reports: Path, protected: list[Path]) -> None:
    output, reports = output.resolve(), reports.resolve()
    if output.exists() or reports.exists():
        raise FileExistsError("gate output/report already exists; use a new run path (no overwrite/resume)")
    if output.is_relative_to(reports) or reports.is_relative_to(output):
        raise ValueError("output and report paths must be separate")
    for path in protected:
        path = path.resolve()
        if any(target.is_relative_to(path) or path.is_relative_to(target) for target in (output, reports)):
            raise ValueError("gate paths overlap protected input artifacts")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def gate_identity(*, gate: dict[str, Any], sft: dict[str, Any], lineage: dict[str, Any],
                  data_manifest: dict[str, Any], sample_ids: list[str], world_size: int,
                  versions: dict[str, str], seed: int) -> dict[str, Any]:
    # Only logical inputs, never absolute source/snapshot/adapter/output locators.
    value = {"gate_version": "a2.2-v1", "objective": "temporary_supervised_not_rl",
             "base_model": gate["model"], "base_revision": gate["revision"],
             "lineage": lineage, "gate": gate, "lora": sft["lora"],
             "max_length": sft["data"]["max_length"],
             "image_max_pixels": sft["model"]["image_max_pixels"],
             "attention": sft["model"]["attn_implementation"],
             "data_manifest_sha256": data_manifest["manifest_sha256"],
             "sample_ids": sample_ids, "world_size": world_size, "software": versions, "seed": seed}
    return {**value, "gate_identity_sha256": canonical_json_sha256(value)}


def trainable_policy(model: Any, optimizer: Any = None) -> dict[str, Any]:
    audit = parameter_audit(model)
    named = list(model.named_parameters())
    vision = [(name, p) for name, p in named if any(f in name.lower() for f in VISION_NAME_FRAGMENTS)]
    projector = [(name, p) for name, p in vision if any(f in name.lower() for f in (
        "projector", "merger", "mm_projector"))]
    if not vision or not projector or any(p.requires_grad for _, p in vision):
        raise ValueError("vision tower/projector must exist and remain frozen")
    trainable = [p for _, p in named if p.requires_grad]
    if optimizer is not None:
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        if len(actual) != len(trainable) or {id(p) for p in actual} != {id(p) for p in trainable}:
            raise ValueError("optimizer must own exactly the trainable LoRA parameters")
    return audit


def local_tensor(value: Any) -> Any:
    return value.to_local() if hasattr(value, "to_local") else value


def lora_snapshot(model: Any) -> dict[str, Any]:
    return {name: local_tensor(p).detach().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def gradient_checks(model: Any) -> dict[str, bool]:
    import torch

    gradients = [local_tensor(p.grad) for p in model.parameters() if p.requires_grad and p.grad is not None]
    return {"lora_grad_finite": bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
            "nonzero_lora_grad": any(bool(torch.count_nonzero(g)) for g in gradients)}


def update_checks(before: dict[str, Any], after: dict[str, Any]) -> dict[str, bool]:
    import torch

    same_keys = bool(before) and before.keys() == after.keys()
    return {"parameters_finite": same_keys and all(bool(torch.isfinite(p).all()) for p in after.values()),
            "lora_param_changed": same_keys and any(not torch.equal(before[k], after[k]) for k in before)}


def reload_matches(saved: dict[str, Any], reloaded: dict[str, Any]) -> bool:
    import torch

    return bool(saved) and saved.keys() == reloaded.keys() and all(torch.equal(saved[k], reloaded[k]) for k in saved)


def finite_loss(value: float) -> bool:
    return math.isfinite(value)


def aggregate_reports(ranks: list[dict[str, Any]], world_size: int) -> dict[str, Any]:
    complete = (len(ranks) == world_size and {row.get("rank") for row in ranks} == set(range(world_size)))
    checks = {name: complete and all(row.get("checks", {}).get(name) is True for row in ranks)
              for name in REQUIRED_CHECKS}
    checks["all_ranks_report_success"] = complete and all(checks.values())
    return {"passed": all(checks.values()), "world_size": world_size,
            "fsdp_mode": "fsdp2", "checks": checks, "per_rank": ranks,
            "gpu_names": [row.get("gpu_name") for row in ranks],
            "peak_allocated_bytes": max((row.get("peak_allocated_bytes", 0) for row in ranks), default=0),
            "peak_reserved_bytes": max((row.get("peak_reserved_bytes", 0) for row in ranks), default=0)}
