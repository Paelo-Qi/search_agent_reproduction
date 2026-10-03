#!/usr/bin/env python3
"""Offline single-GPU A/B0/B1 merge precision diagnostic, never a Gate update."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--local-files-only", action="store_true", default=True,
                        help="Always enforced; online mode is not supported")
    parser.add_argument("--keep-diagnostic-model", action="store_true",
                        help="Retain only this diagnostic's temporary BF16 checkpoint (not for training/rollout)")
    return parser


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args = build_parser().parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")
    from opensearch_vl_repro.rl.merge_precision_diagnostic import run_diagnostic
    return run_diagnostic(args, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
