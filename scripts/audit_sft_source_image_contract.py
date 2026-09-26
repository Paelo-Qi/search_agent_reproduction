#!/usr/bin/env python3
"""Read-only, CPU-only image-ID contract audit of all 36,592 pinned source records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_source_image_audit import audit_source_population  # noqa: E402
from opensearch_vl_repro.sft_main_data import load_data_quality_exclusions  # noqa: E402
from opensearch_vl_repro.sft_tool_audit import (  # noqa: E402
    DATASET_ID, DATASET_REVISION, write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--expected-manifest", type=Path,
                        default=ROOT / "data/sft_main/manifest.json",
                        help="Existing pinned manifest with source hashes; read-only")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "reports/sft_source_image_contract.json")
    parser.add_argument("--verify-frozen", action="store_true",
                        help="Fail unless configs/sft_data_exclusions.json exactly matches the full audit")
    args = parser.parse_args()
    pinned = json.loads(args.expected_manifest.read_text(encoding="utf-8"))
    if (pinned.get("dataset_id") != DATASET_ID
            or pinned.get("dataset_revision") != DATASET_REVISION):
        raise ValueError("expected manifest does not identify the pinned Search-VL-SFT revision")
    expected_hashes = {source: item["sha256"] for source, item in pinned["source_files"].items()}
    report = audit_source_population(args.raw_dir, expected_source_sha256=expected_hashes)
    if args.verify_frozen:
        report["frozen_exclusions_match"] = (
            load_data_quality_exclusions() == report["recommended_exclusions"])
    write_json_atomic(args.report, report)
    print(json.dumps({key: report[key] for key in (
        "audit_version", "dataset_revision", "total_records", "records_with_image_search",
        "image_search_non_img_n_count", "derived_image_id_gap_count",
        "ungrounded_image_reference_count", "invalid_raw_format_count", "complete")}
        | {"all_bad_sample_id_count": len(report["all_bad_sample_ids"]),
           "report": str(args.report),
           "frozen_exclusions_match": report.get("frozen_exclusions_match")},
        ensure_ascii=False, indent=2))
    return 0 if not args.verify_frozen or report["frozen_exclusions_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
