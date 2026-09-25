#!/usr/bin/env python3
"""Build the pinned 8k SFT selection and four disjoint physical shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_main_data import DEFAULT_SEED, prepare_sft_pool  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sft_main")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--extract-images", action="store_true")
    parser.add_argument("--download-images", action="store_true",
                        help="Opt in to downloading pinned official image ZIPs")
    args = parser.parse_args()
    manifest = prepare_sft_pool(args.raw_dir, args.output_dir, seed=args.seed,
                                extract_images=args.extract_images,
                                download_images=args.download_images)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "membership"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
