"""Isolated shared static merge worker."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl.formal_main_cli import build_parser
from opensearch_vl_repro.rl.formal_main_collection import run_merge_worker


if __name__ == "__main__":
    raise SystemExit(run_merge_worker(build_parser(ROOT, worker="merge").parse_args(), ROOT))
