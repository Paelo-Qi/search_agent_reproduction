#!/usr/bin/env python3
"""Build isolated R1/R2 repair data from pinned source and local image ZIPs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_repair_data import REPAIR_SEED, build_repair_datasets  # noqa: E402


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    cli.add_argument("--pool-manifest", type=Path, default=ROOT / "data/sft_main/manifest.json")
    cli.add_argument("--eval", type=Path, default=ROOT / "data/eval/combined_eval_300.parquet")
    cli.add_argument("--dev-dir", type=Path, default=ROOT / "data/eval/tool_protocol_dev50")
    cli.add_argument("--output-root", type=Path, default=ROOT / "data/sft_repair")
    cli.add_argument("--seed", type=int, default=REPAIR_SEED)
    args = cli.parse_args()
    results = build_repair_datasets(raw_dir=args.raw_dir, pool_manifest_path=args.pool_manifest,
                                    eval_path=args.eval, dev_dir=args.dev_dir,
                                    output_root=args.output_root, seed=args.seed)
    for mode, manifest in results.items():
        print(f"{mode}: samples={manifest['sample_count']} "
              f"categories={manifest['protocol_category_counts']} "
              f"image_ids={manifest['image_target_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
