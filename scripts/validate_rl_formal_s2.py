#!/usr/bin/env python3
"""Two-window, offline, two-rank S2 GPU validation; NOT a Smoke20 trainer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl_main.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/rl/smoke20.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path,
                        default=ROOT / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter")
    parser.add_argument("--prompt-start", type=int, default=0)
    return parser


def main():
    from opensearch_vl_repro.rl.formal_s2_validation import run_validation
    return run_validation(build_parser().parse_args(), ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
