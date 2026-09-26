#!/usr/bin/env python3
"""Read-only official SFT 8k leakage, sequence, and tool-contract audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.data import SFT_INPUT_MESSAGE_VERSION, SFT_MASK_VERSION, load_json_records  # noqa: E402
from opensearch_vl_repro.inference.eval_reader import read_eval_samples_by_ids  # noqa: E402
from opensearch_vl_repro.evaluation.eval300 import build_eval300_plan  # noqa: E402
from opensearch_vl_repro.model import load_processor  # noqa: E402
from opensearch_vl_repro.reporting import write_json  # noqa: E402
from opensearch_vl_repro.sft_main_data import SHARD_SIZES, load_sft_manifest  # noqa: E402
from opensearch_vl_repro.sft_tool_audit import sha256_file  # noqa: E402
from opensearch_vl_repro.sft_preflight import (  # noqa: E402
    formal_preflight_checks, leakage_audit, reserved_literal_audit,
    sequence_audit, tool_contract_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/sft_main")
    parser.add_argument("--eval", type=Path, default=ROOT / "data/eval/combined_eval_300.parquet")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sft_main.yaml")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/sft_preflight")
    args = parser.parse_args()
    manifest = load_sft_manifest(args.data_dir / "manifest.json")
    pool_sha256 = sha256_file(args.data_dir / "manifest.json")
    write_json(args.report_dir / "summary.json", {
        "passed": False, "reason": "SFT preflight is incomplete",
        "mask_version": SFT_MASK_VERSION,
        "sft_input_message_version": SFT_INPUT_MESSAGE_VERSION,
        "pool_manifest_sha256": pool_sha256,
    })
    by_shard = {shard: load_json_records(args.data_dir / manifest["shards"][shard]["path"])
                for shard in SHARD_SIZES}
    records = [record for shard in SHARD_SIZES for record in by_shard[shard]]
    reserved = reserved_literal_audit(by_shard)
    write_json(args.report_dir / "reserved_literals.json", reserved)
    tool_report = tool_contract_audit(records)
    write_json(args.report_dir / "tool_contract.json", tool_report)
    eval_plan = build_eval300_plan(args.eval)
    eval_samples = read_eval_samples_by_ids(args.eval, list(eval_plan.entries))
    leakage = leakage_audit(records, eval_samples, args.data_dir)
    leakage["pool_manifest_sha256"] = pool_sha256
    write_json(args.report_dir / "leakage.json", leakage)
    if leakage["missing_image_count"]:
        checks = formal_preflight_checks(tool_report, leakage, None)
        summary = {"passed": False, "reason": "SFT images are missing; sequence/image audit incomplete",
                   "mask_version": SFT_MASK_VERSION,
                   "sft_input_message_version": SFT_INPUT_MESSAGE_VERSION,
                   "tool_contract_passed": tool_report["passed"], "leakage_complete": False,
                   "sequence_complete": False, "pool_manifest_sha256": pool_sha256,
                   "eval_sha256": eval_plan.dataset_sha256,
                   "checks": checks,
                   "raw_declaration_drift_count": tool_report["raw_declaration_drift_count"],
                   "effective_declaration_drift_count": tool_report["effective_declaration_drift_count"],
                   "actual_call_drift_count": tool_report["actual_call_drift_count"],
                   "question_overlap_count": leakage["question_overlap_count"],
                   "missing_image_count": leakage["missing_image_count"]}
        write_json(args.report_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    processor = load_processor(config)
    sequence = sequence_audit(by_shard, processor, args.data_dir,
                              max_length=int(config["data"]["max_length"]))
    write_json(args.report_dir / "sequence.json", sequence)
    write_json(args.report_dir / "image_grounding.json", sequence["image_grounding"])
    checks = formal_preflight_checks(tool_report, leakage, sequence)
    summary = {
        "passed": all(checks.values()), "checks": checks,
        "mask_version": SFT_MASK_VERSION,
        "sft_input_message_version": SFT_INPUT_MESSAGE_VERSION,
        "image_grounding_failed_sample_count": sequence["image_grounding"]["failed_sample_count"],
        "tool_contract_passed": tool_report["passed"], "leakage_complete": True,
        "raw_declaration_drift_count": tool_report["raw_declaration_drift_count"],
        "effective_declaration_drift_count": tool_report["effective_declaration_drift_count"],
        "actual_call_drift_count": tool_report["actual_call_drift_count"],
        "question_overlap_count": leakage["question_overlap_count"],
        "image_overlap_count": leakage["image_overlap_count"],
        "sequence_complete": True,
        "pool_manifest_sha256": pool_sha256,
        "eval_sha256": eval_plan.dataset_sha256,
        "zero_supervised_count": sequence["full_8k"]["zero_supervised_count"],
        "partial_assistant_span_cut_count": sequence["full_8k"]["partial_assistant_span_cut_count"],
        "partial_tool_call_cut_count": sequence["full_8k"]["partial_tool_call_cut_count"],
        "complete_assistant_span_dropped_count": sequence["full_8k"]["complete_assistant_span_dropped_count"],
    }
    write_json(args.report_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
