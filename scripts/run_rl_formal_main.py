"""CPU coordinator entry; no GPU/framework initialization in this process."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl.formal_main_cli import build_parser
from opensearch_vl_repro.rl.formal_main_coordinator import run_main


if __name__ == "__main__":
    run_main(build_parser(ROOT).parse_args(), ROOT)
