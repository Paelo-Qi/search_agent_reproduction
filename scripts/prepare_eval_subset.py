#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.eval_subset import (  # noqa: E402
    BENCHMARKS,
    DEFAULT_DATASET,
    DEFAULT_REVISION,
    DEFAULT_SEED,
    prepare_evaluation_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen 300-question Phase 1 eval set.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples-per-benchmark", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "eval")
    parser.add_argument(
        "--source-dir", type=Path, default=PROJECT_ROOT / "data" / "eval" / "source"
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports" / "eval_subset_report.json"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require all three source parquet files to exist in --source-dir.",
    )
    args = parser.parse_args()
    report = prepare_evaluation_subset(
        project_root=PROJECT_ROOT,
        dataset=args.dataset,
        revision=args.revision,
        seed=args.seed,
        samples_per_benchmark=args.samples_per_benchmark,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        report_path=args.report,
        download_missing=not args.no_download,
    )

    print("Evaluation subset preparation complete.\n")
    for spec in BENCHMARKS:
        stats = report["benchmarks"][spec.name]
        print(f"{spec.display_name}: {stats['final_count']} / {stats['source_count']}")
    print(f"\nCombined: {report['combined']['final_count']}")
    print(f"Seed: {report['seed']}\n")
    print(f"Duplicate IDs: {report['totals']['duplicate_ids_within_benchmarks']}")
    print(f"Empty questions: {report['totals']['empty_questions']}")
    print(f"Empty answers: {report['totals']['empty_answers']}")
    print(f"Image decode failures: {report['totals']['image_decode_failures']}")
    print("\nManifest:", report["manifest_path"])
    print("Report:", args.report.resolve())
    print("\nChecksums:")
    print(json.dumps({
        name: values["output_sha256"]
        for name, values in report["benchmarks"].items()
    } | {"combined": report["combined"]["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
