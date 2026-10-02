"""Deterministic RL parquet selection; never downloads or modifies source data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl.data import (prepare_formal_dataset, prepare_prompts,
                                         write_dataset_artifacts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-overlap-manifest", type=Path)
    parser.add_argument("--sft-overlap-manifest", type=Path)
    parser.add_argument("--quality-audit-dir", type=Path)
    parser.add_argument("--smoke-count", type=int, default=20)
    parser.add_argument("--main-count", type=int, default=400)
    parser.add_argument("--main-shard-size", type=int, default=100)
    parser.add_argument("--expected-source-rows", type=int)
    parser.add_argument("--allow-incomplete-sft-audit", action="store_true",
                        help="development fixture only; never use for formal RL data")
    parser.add_argument("--input", type=Path, help="local JSON list of source_sample_id/question/image_paths")
    parser.add_argument("--dataset-id", default="OpenSearch-VL/Search-VL-RL-8K")
    parser.add_argument("--dataset-revision", default="8ef567289043eef004b13da83b0e7bb7f5ae2daa")
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--validation-count", type=int, default=0)
    args = parser.parse_args()
    if args.source_parquet is not None:
        try:
            if any(value is None for value in (args.source_root, args.output_dir,
                                               args.eval_overlap_manifest, args.sft_overlap_manifest,
                                               args.quality_audit_dir)):
                raise ValueError("formal parquet mode requires source-root, output-dir, overlap manifests and quality audit directory")
            artifacts = prepare_formal_dataset(
                source_parquet=args.source_parquet, source_root=args.source_root,
                dataset_id=args.dataset_id, dataset_revision=args.dataset_revision,
                seed=args.seed, smoke_count=args.smoke_count, main_count=args.main_count,
                shard_size=args.main_shard_size,
                eval_overlap_manifest=args.eval_overlap_manifest,
                sft_overlap_manifest=args.sft_overlap_manifest,
                quality_audit_dir=args.quality_audit_dir,
                allow_incomplete_sft=args.allow_incomplete_sft_audit)
            if args.expected_source_rows is not None and artifacts["main_manifest"]["source_rows"] != args.expected_source_rows:
                raise ValueError("RL source row count does not match --expected-source-rows")
            write_dataset_artifacts(artifacts, args.output_dir,
                source_parquet=args.source_parquet, source_root=args.source_root,
                eval_overlap_manifest=args.eval_overlap_manifest,
                sft_overlap_manifest=args.sft_overlap_manifest,
                quality_audit_dir=args.quality_audit_dir,
                allow_incomplete_sft=args.allow_incomplete_sft_audit)
        except (ValueError, TypeError, KeyError, FileNotFoundError, OSError) as exc:
            print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps({"passed": True, "output_dir": str(args.output_dir),
                          "source_rows": artifacts["main_manifest"]["source_rows"],
                          "main_count": len(artifacts["main"]), "smoke_count": len(artifacts["smoke"]),
                          "shard_count": len(artifacts["shards"]),
                          "sft_image_audit_complete": artifacts["main_manifest"]["sft_image_audit_complete"],
                          "main_manifest_sha256": artifacts["main_manifest"]["manifest_sha256"]}, ensure_ascii=False))
        return 0
    if args.input is None:
        print(json.dumps({"dry_run": True, "prepared": False,
                          "required_record_fields": ["source_sample_id", "question", "image_paths"],
                          "note": "No data downloaded or written."}, indent=2))
        return 0
    try:
        records = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("RL source JSON must be a list")
        samples, manifest = prepare_prompts(records, dataset_id=args.dataset_id,
            dataset_revision=args.dataset_revision, seed=args.seed, limit=args.limit,
            validation_count=args.validation_count)
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"dry_run": True, "passed": False, "error": str(exc)}))
        return 1
    print(json.dumps({"dry_run": True, "passed": True, "manifest": manifest,
                      "sample_ids": [item.sample_id for item in samples]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
