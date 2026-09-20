#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.training import run_sft_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-GPU Phase 0 Dev Smoke.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sft_dev.yaml")
    args = parser.parse_args()
    run_sft_training(
        args.config,
        expected_world_size=1,
        minimum_optimizer_steps=2,
        report_filename="dev_training.json",
        gate_name="Phase 0 Dev Smoke",
    )


if __name__ == "__main__":
    main()

