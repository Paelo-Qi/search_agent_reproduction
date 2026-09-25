#!/usr/bin/env python3
"""2-GPU BF16 LoRA SFT; complete checkpoint/resume across fixed shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_long_training import run_sft_stage  # noqa: E402
from opensearch_vl_repro.sft_train_plan import STAGE_ORDER  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sft_main.yaml")
    parser.add_argument("--stage", required=True, choices=(*STAGE_ORDER, "smoke"))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--phase-2-peak-lr", type=float)
    parser.add_argument("--stop-after-steps", type=int,
                        help="Only for interrupted/resumed 4B smoke validation")
    parser.add_argument("--run-tag")
    parser.add_argument("--acknowledge-leakage", action="store_true")
    args = parser.parse_args()
    run_sft_stage(args.config, stage=args.stage, resume_from=args.resume_from,
                  micro_batch=args.micro_batch, gradient_accumulation=args.grad_accum,
                  phase_2_peak_lr=args.phase_2_peak_lr,
                  stop_after_steps=args.stop_after_steps, run_tag=args.run_tag,
                  acknowledge_leakage=args.acknowledge_leakage)


if __name__ == "__main__":
    main()
