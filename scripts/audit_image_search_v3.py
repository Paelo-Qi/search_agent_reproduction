#!/usr/bin/env python3
"""Read-only CPU audit of v3 SFT image-search targets and frozen memberships."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_image_grounding import audit_raw_image_contract  # noqa: E402
from opensearch_vl_repro.sft_main_data import (SHARD_SIZES, SOURCE_COUNTS,  # noqa: E402
                                                eval_question_set,
                                                load_sft_manifest,
                                                proportional_quotas,
                                                require_data_quality_exclusions)
from opensearch_vl_repro.sft_preflight import sample_questions  # noqa: E402
from opensearch_vl_repro.sft_protocol_v3 import effective_image_search_call_counts  # noqa: E402
from opensearch_vl_repro.sft_tool_audit import iter_json_array  # noqa: E402
from opensearch_vl_repro.tool_protocol_dev import validate_dev_manifest  # noqa: E402


def audit(pool_path: Path, previous_path: Path | None = None,
          dev_dir: Path | None = None,
          previous_dev_ids: Path | None = None,
          eval_path: Path | None = None) -> dict:
    if (dev_dir is None) != (previous_dev_ids is None):
        raise ValueError("Dev50 directory and historical IDs must be supplied together")
    pool = load_sft_manifest(pool_path)
    require_data_quality_exclusions(pool)
    counts = Counter()
    bad_grounding = []
    eval_questions, eval_sha = eval_question_set(eval_path) if eval_path is not None else (set(), None)
    question_overlap_count = 0
    for shard in SHARD_SIZES:
        for record in iter_json_array(pool_path.parent / pool["shards"][shard]["path"]):
            counts.update(effective_image_search_call_counts(record))
            if eval_path is not None:
                question_overlap_count += len(sample_questions(record) & eval_questions)
            result = audit_raw_image_contract(record, sample_id=record["_sample_id"])
            if not result["passed"]:
                bad_grounding.append({"sample_id": record["_sample_id"], "errors": result["errors"]})
    checks = {
        "sft_shard_sizes": all(pool["shards"][s]["count"] == SHARD_SIZES[s]
                               for s in SHARD_SIZES),
        "sft_source_quotas": pool["pool_source_counts"] == proportional_quotas(
            sum(SHARD_SIZES.values()), SOURCE_COUNTS),
        "effective_image_search_image_id_present": counts["image_search_image_id"] > 0,
        "effective_image_search_legacy_url_zero": counts["image_search_legacy_url"] == 0,
        "effective_image_search_http_target_zero": counts["image_search_http_target"] == 0,
        "effective_image_search_non_img_n_zero": counts["image_search_non_img_n"] == 0,
        "effective_image_search_extra_arguments_zero": counts["image_search_extra_arguments"] == 0,
        "grounding_passed": not bad_grounding,
    }
    if eval_path is not None:
        checks["eval300_question_identity"] = eval_sha == pool.get("eval300_question_sha256")
        checks["eval300_question_overlap_zero"] = question_overlap_count == 0
    if previous_path is not None:
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        keys = ("seed", "selection_algorithm", "membership", "exclusions",
                "replacements", "pool_source_counts", "source_population_counts")
        checks["sft_membership_and_selection_unchanged"] = all(
            pool.get(k) == previous.get(k) for k in keys)
        checks["sft_shard_source_counts_unchanged"] = all(
            pool["shards"][s]["source_counts"] == previous["shards"][s]["source_counts"]
            for s in SHARD_SIZES)
    sft_passed = all(checks.values())
    if dev_dir is not None and previous_dev_ids is not None:
        ids = json.loads((dev_dir / "ids.json").read_text(encoding="utf-8"))
        dev = json.loads((dev_dir / "tool_protocol_dev50_manifest.json").read_text(encoding="utf-8"))
        validate_dev_manifest(ids, dev, pool_manifest_path=pool_path)
        old_ids = json.loads(previous_dev_ids.read_text(encoding="utf-8"))
        checks["dev50_ordered_ids_unchanged"] = ids == old_ids and len(ids) == 50
        checks["dev50_overlap_checks_passed"] = all(dev.get(key) == 0 for key in (
            "eval300_question_overlap_count", "eval300_image_overlap_count",
            "sft_membership_overlap_count", "frozen_exclusion_overlap_count",
            "image_contract_bad_count"))
    return {"passed": all(checks.values()) and previous_path is not None
            and dev_dir is not None and previous_dev_ids is not None,
            "sft_passed": sft_passed,
            "dev50_checked": dev_dir is not None,
            "checks": checks,
            "effective_image_search_counts": dict(sorted(counts.items())),
            "grounding_failure_count": len(bad_grounding),
            "grounding_failures": bad_grounding[:20],
            "eval300_question_checked": eval_path is not None,
            "eval300_question_overlap_count": question_overlap_count if eval_path is not None else None,
            "sft_sample_count": len(pool["membership"]),
            "dev50_sample_count": len(ids) if dev_dir is not None else None}


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pool-manifest", type=Path, required=True)
    cli.add_argument("--previous-manifest", type=Path)
    cli.add_argument("--dev-dir", type=Path)
    cli.add_argument("--previous-dev-ids", type=Path)
    cli.add_argument("--eval", type=Path,
                     default=ROOT / "data/eval/combined_eval_300.parquet")
    cli.add_argument("--sft-only", action="store_true",
                     help="Audit only the v3 8k; output passed=false until Dev50 is checked")
    args = cli.parse_args()
    if not args.sft_only and not all((args.previous_manifest, args.dev_dir,
                                      args.previous_dev_ids)):
        cli.error("full audit requires --previous-manifest, --dev-dir, and --previous-dev-ids")
    if args.sft_only and (args.dev_dir is not None or args.previous_dev_ids is not None):
        cli.error("--sft-only cannot receive Dev50 paths")
    result = audit(args.pool_manifest, args.previous_manifest,
                   args.dev_dir, args.previous_dev_ids, args.eval)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    success = result["sft_passed"] if args.sft_only else result["passed"]
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
