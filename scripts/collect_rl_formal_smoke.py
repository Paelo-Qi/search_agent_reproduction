#!/usr/bin/env python3
"""Isolated one-GPU current-policy collection worker; not a trainer."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from opensearch_vl_repro.rl.formal_smoke_cli import build_parser
    from opensearch_vl_repro.rl.formal_collection import run_collection
    return run_collection(build_parser(ROOT, worker="collect").parse_args(), ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
