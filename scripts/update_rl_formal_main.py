"""Fresh four-rank S2 update worker, including actual iteration-zero bootstrap."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl.formal_main_cli import build_parser
from opensearch_vl_repro.rl.formal_main_update import run_update_worker


if __name__ == "__main__":
    raise SystemExit(run_update_worker(build_parser(ROOT, worker="update").parse_args(), ROOT))
