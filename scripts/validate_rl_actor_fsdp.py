#!/usr/bin/env python3
"""Explicit torchrun entry for Gate A2.2; never launches RL/rollout/training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl_main.yaml")
    parser.add_argument("--gate-config", type=Path, default=ROOT / "configs/rl_gate_a22.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/rl/smoke20.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--sft-adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, help="must allow one sample per rank; default launcher world size")
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--local-files-only", action="store_true", help="offline is enforced even without this flag")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from opensearch_vl_repro.rl.verl_actor_gate import run_gate
    return run_gate(args, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
