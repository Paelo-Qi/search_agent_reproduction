#!/usr/bin/env python3
"""Train isolated R1/R2 targeted-repair ablations from corrected checkpoint-3k."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_repair_training import run_repair  # noqa: E402


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--max-steps", type=int, choices=(25, 50, 100))
    args = cli.parse_args()
    run_repair(args.config, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
