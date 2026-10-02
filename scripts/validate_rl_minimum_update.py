#!/usr/bin/env python3
"""Gate C only: collect (one GPU), update (torchrun 2), finalize (CPU)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "update", "finalize"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl_main.yaml")
    parser.add_argument("--gate-config", type=Path, default=ROOT / "configs/rl_gate_c.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/rl/smoke20.json")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--gate-b-manifest", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, default=ROOT / "configs/judge.example.yaml")
    parser.add_argument("--search-config", type=Path, default=ROOT / "configs/search_backends.example.yaml")
    parser.add_argument("--layout-config", type=Path, default=ROOT / "configs/layout_parsing.example.yaml")
    parser.add_argument("--tool-cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260506)
    return parser


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    args = build_parser().parse_args()
    if args.stage == "update":
        from opensearch_vl_repro.rl.verl_policy_update import update
        return update(args, ROOT)
    from opensearch_vl_repro.rl.gate_c import collect, finalize
    return (collect if args.stage == "collect" else finalize)(args, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
