#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.evaluation.dev30 import prepare_dev30  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the deterministic ID-only Dev-30.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/eval/combined_eval_300.parquet")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "data/eval/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/eval/dev30")
    args = parser.parse_args(argv)
    manifest = prepare_dev30(dataset_path=args.dataset,
                             source_manifest_path=args.source_manifest,
                             output_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
