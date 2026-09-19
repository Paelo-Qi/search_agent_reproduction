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
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 0 evidence without inventing results.")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "phase0_status.json")
    args = parser.parse_args()

    env = read_json(PROJECT_ROOT / "reports" / "environment.json")
    dataset = read_json(PROJECT_ROOT / "data" / "sft_smoke_100.meta.json")
    inspection = read_json(PROJECT_ROOT / "reports" / "sample_inspection.json")
    model_load = read_json(PROJECT_ROOT / "reports" / "model_load.json")
    training = read_json(PROJECT_ROOT / "reports" / "training.json")
    reload = read_json(PROJECT_ROOT / "reports" / "checkpoint_reload.json")

    checks = {
        "environment_has_cuda": bool(env and env.get("cuda_available")),
        "exactly_two_gpus": bool(env and len(env.get("gpus", [])) == 2),
        "dataset_has_100_samples": bool(dataset and dataset.get("sample_count") == 100),
        "all_seven_sources_present": bool(dataset and len(dataset.get("source_counts", {})) == 7),
        "input_label_shapes_match": bool(
            inspection and inspection.get("label_masking", {}).get("input_ids_labels_same_shape")
        ),
        "assistant_targets_present": bool(
            inspection and inspection.get("label_masking", {}).get("has_assistant_targets")
        ),
        "vision_tensor_present": bool(inspection and inspection.get("vision_fields")),
        "model_load_forward_passed": bool(model_load and model_load.get("passed")),
        "two_gpu_training": bool(training and training.get("world_size") == 2),
        "minimum_optimizer_steps": bool(training and training.get("optimizer_steps", 0) >= 10),
        "losses_are_finite": bool(
            training
            and training.get("all_losses_finite")
            and all(math.isfinite(value) for value in training.get("losses", []))
        ),
        "only_expected_lora_parameters_trainable": bool(
            training
            and training.get("parameter_audit", {}).get("trainable_parameters", 0) > 0
            and not training.get("parameter_audit", {}).get("unexpected_trainable", [])
        ),
        "lora_parameter_changed": bool(training and training.get("lora_parameter_changed")),
        "adapter_saved": bool(
            training
            and (Path(training.get("checkpoint_path", "")) / "adapter_config.json").is_file()
        ),
        "checkpoint_reloaded_in_fresh_process": bool(
            reload and reload.get("passed") and reload.get("fresh_process")
        ),
    }
    passed = all(checks.values())
    status = {
        "passed": passed,
        "checks": checks,
        "missing_or_failed": [key for key, value in checks.items() if not value],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.require_complete and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
