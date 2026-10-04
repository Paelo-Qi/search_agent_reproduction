#!/usr/bin/env python3
"""CPU coordinator; workers alone own CUDA. Same run-id/command resumes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from opensearch_vl_repro.rl.formal_smoke_cli import build_parser
    from opensearch_vl_repro.rl.formal_smoke import run_smoke
    run_smoke(build_parser(ROOT).parse_args(), ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
