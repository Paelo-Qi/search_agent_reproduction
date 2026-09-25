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
from opensearch_vl_repro.sft_tool_audit import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sft_main")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval", type=Path, default=ROOT / "data/eval/combined_eval_300.parquet")
    parser.add_argument("--exclusions", type=Path,
                        help="Frozen JSON mapping of sample ID to exclusion reason")
    parser.add_argument("--leakage-report", type=Path,
                        help="Complete preflight leakage.json for deterministic image-overlap replacement")
    parser.add_argument("--extract-images", action="store_true")
    parser.add_argument("--download-images", action="store_true",
                        help="Opt in to downloading pinned official image ZIPs")
    args = parser.parse_args()
    exclusions = {}
    if args.exclusions:
        exclusions.update(json.loads(args.exclusions.read_text(encoding="utf-8")))
    if args.leakage_report:
        report = json.loads(args.leakage_report.read_text(encoding="utf-8"))
        current_manifest = args.output_dir / "manifest.json"
        if (report.get("complete") is not True or not current_manifest.is_file()
                or report.get("pool_manifest_sha256") != sha256_file(current_manifest)):
            raise ValueError("leakage report must be complete and bound to the current pool")
        frozen = json.loads(current_manifest.read_text(encoding="utf-8"))
        exclusions.update({item["sample_id"]: item["reason"]
                           for item in frozen.get("exclusions", [])})
        for item in report["image_overlaps"]:
            exclusions[item["training_sample_id"]] = "eval300_image_overlap"
        for item in report["question_overlaps"]:
            exclusions[item["training_sample_id"]] = "eval300_question_overlap"
    manifest = prepare_sft_pool(args.raw_dir, args.output_dir, seed=args.seed,
                                extract_images=args.extract_images,
                                download_images=args.download_images,
                                eval_path=args.eval, exclusions=exclusions)
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "membership"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
