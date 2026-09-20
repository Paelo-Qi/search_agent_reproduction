#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def evaluate_dev_status() -> dict[str, Any]:
    env = read_json(PROJECT_ROOT / "reports" / "dev_environment.json")
    dataset = read_json(PROJECT_ROOT / "data" / "sft_dev_4.meta.json")
    inspection = read_json(PROJECT_ROOT / "reports" / "dev_sample_inspection.json")
    model_load = read_json(PROJECT_ROOT / "reports" / "dev_model_load.json")
    training = read_json(PROJECT_ROOT / "reports" / "dev_training.json")
    reload_report = read_json(PROJECT_ROOT / "reports" / "dev_checkpoint_reload.json")

    audit = training.get("parameter_audit", {}) if training else {}
    losses = training.get("losses", []) if training else []
    checkpoint = Path(training.get("checkpoint_path", "")) if training else Path()
    env_gpus = env.get("gpus", []) if env else []
    checks = {
        "cuda_available": bool(env and env.get("cuda_available")),
        "bf16_supported": bool(
            len(env_gpus) == 1 and env_gpus[0].get("bf16_supported")
        ),
        "exactly_one_gpu_used": bool(
            env
            and len(env_gpus) == 1
            and training
            and training.get("world_size") == 1
            and training.get("visible_gpu_count") == 1
        ),
        "synthetic_multimodal_dataset_prepared": bool(
            dataset
            and dataset.get("kind") == "fully_local_synthetic_multimodal"
            and dataset.get("sample_count") == 4
            and dataset.get("image_count") == 4
            and dataset.get("contains_observation_trajectories")
            and dataset.get("downloads_required") is False
        ),
        "input_ids_labels_shape_valid": bool(
            inspection
            and inspection.get("label_masking", {}).get("input_ids_labels_same_shape")
        ),
        "assistant_targets_present": bool(
            inspection and inspection.get("label_masking", {}).get("has_assistant_targets")
        ),
        "vision_tensor_present": bool(inspection and inspection.get("vision_fields")),
        "model_forward_passed": bool(model_load and model_load.get("passed")),
        "lora_audit_passed": bool(
            training
            and audit.get("trainable_parameters", 0) > 0
            and not audit.get("unexpected_trainable", [])
        ),
        "vision_tower_frozen": bool(training and training.get("vision_tower_frozen")),
        "multimodal_projector_frozen": bool(
            training and training.get("multimodal_projector_frozen")
        ),
        "at_least_two_optimizer_steps": bool(
            training and training.get("optimizer_steps", 0) >= 2
        ),
        "all_losses_finite": bool(
            training
            and training.get("all_losses_finite")
            and losses
            and all(math.isfinite(float(value)) for value in losses)
        ),
        "lora_parameter_changed": bool(training and training.get("lora_parameter_changed")),
        "adapter_saved": bool(
            training and (checkpoint / "adapter_config.json").is_file()
        ),
        "adapter_reloaded_in_fresh_process": bool(
            reload_report and reload_report.get("passed") and reload_report.get("fresh_process")
        ),
    }
    passed = all(checks.values())
    return {
        "gate": "Phase 0 Dev Smoke (not the Formal Phase 0 gate)",
        "passed": passed,
        "implies_formal_phase0_passed": False,
        "execution_state": (
            "PHASE 0 DEV SMOKE PASSED" if passed else "READY FOR SINGLE-GPU EXECUTION - NOT YET PASSED"
        ),
        "checks": checks,
        "missing_or_failed": [name for name, value in checks.items() if not value],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate only Phase 0 Dev Smoke evidence.")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase0_dev_status.json",
    )
    args = parser.parse_args()
    status = evaluate_dev_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.require_complete and not status["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
