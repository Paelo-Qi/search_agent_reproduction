#!/usr/bin/env python3
"""Offline BF16 FA2/SDPA forensic forward, not an RL/Gate invocation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--local-files-only", action="store_true", required=True)
    return parser


def main():
    from opensearch_vl_repro.rl.bf16_sdpa_diagnostic import run_diagnostic
    return run_diagnostic(build_parser().parse_args(), ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
