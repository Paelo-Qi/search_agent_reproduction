#!/usr/bin/env python3
"""Gate B entry: offline static merge + real rLLM/vLLM visual round-trip."""

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
    parser.add_argument("--gate-config", type=Path, default=ROOT / "configs/rl_gate_b.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/rl/smoke20.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--actor-adapter", type=Path, required=True)
    parser.add_argument("--actor-gate-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--local-files-only", action="store_true", help="offline is always enforced")
    return parser


def main() -> int:
    from opensearch_vl_repro.rl.rollout_gate import run_rollout_gate
    return run_rollout_gate(build_parser().parse_args(), ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
