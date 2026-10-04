#!/usr/bin/env python3
"""torchrun two-rank initial-policy bootstrap OR exactly one formal window."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from opensearch_vl_repro.rl.formal_smoke_cli import build_parser
    from opensearch_vl_repro.rl.formal_smoke_update import run_update_worker
    return run_update_worker(build_parser(ROOT, worker="update").parse_args(), ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
