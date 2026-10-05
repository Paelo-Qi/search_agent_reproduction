"""Locator-only S4 CLI; operational scheduling/pause/disk knobs are not identity."""
import argparse
from pathlib import Path


def build_parser(root, *, worker=None):
    root = Path(root)
    p = argparse.ArgumentParser(description="Formal S4 Main400 (not Smoke continuation)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--continue-from-run", help="verified original Main v4/v5 -> NEW v6 child; dynamically freeze latest")
    p.add_argument("--config", type=Path, default=root / "configs/rl_main.yaml")
    p.add_argument("--data", type=Path, default=root / "data/rl/main400.json")
    for flag in ("source-root", "source-parquet", "base-model-path", "eval-overlap-manifest", "sft-overlap-manifest"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--sft-adapter", type=Path, default=root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter")
    for flag, name in (("judge-config", "judge.example.yaml"), ("search-config", "search_backends.example.yaml"),
                       ("layout-config", "layout_parsing.example.yaml")):
        p.add_argument("--" + flag, type=Path, default=root / "configs" / name)
    p.add_argument("--tool-cache-dir", type=Path)
    p.add_argument("--reward-cache-dir", type=Path)
    if worker == "update":
        p.add_argument("--phase", choices=("bootstrap", "update"), required=True)
    elif worker == "collect":
        p.add_argument("--prompt-id", required=True)
    elif worker is None:
        p.add_argument("--collection-gpus", default="0,1,2,3")
        p.add_argument("--collection-parallelism", type=int, default=4)
        p.add_argument("--update-gpus", default="0,1,2,3")
        p.add_argument("--stop-after-window", type=int)
        p.add_argument("--max-run-gib", type=float, default=250.)
        p.add_argument("--min-free-gib", type=float, default=30.)
        p.add_argument("--merge-headroom-gib", type=float, default=18.)
        p.add_argument("--update-headroom-gib", type=float, default=32.)
    return p
