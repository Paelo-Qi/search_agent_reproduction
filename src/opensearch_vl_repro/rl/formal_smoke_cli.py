"""Shared locator-only CLI; no model/framework initialization."""
import argparse
from pathlib import Path


def build_parser(root, *, worker=None):
    root = Path(root)
    parser = argparse.ArgumentParser(description="Formal S3 Smoke20; no Main400 initialization from Smoke checkpoints")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=root / "configs/rl_smoke.yaml")
    parser.add_argument("--data", type=Path, default=root / "data/rl/smoke20.json")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--sft-adapter", type=Path, default=root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter")
    parser.add_argument("--judge-config", type=Path, default=root / "configs/judge.example.yaml")
    parser.add_argument("--search-config", type=Path, default=root / "configs/search_backends.example.yaml")
    parser.add_argument("--layout-config", type=Path, default=root / "configs/layout_parsing.example.yaml")
    parser.add_argument("--tool-cache-dir", type=Path)
    parser.add_argument("--reward-cache-dir", type=Path)
    if worker == "update":
        parser.add_argument("--phase", choices=("bootstrap", "update"), required=True)
    elif worker is None:
        parser.add_argument("--rollout-gpu", default="0")
        parser.add_argument("--update-gpus", default="0,1")
    return parser
