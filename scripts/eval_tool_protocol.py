#!/usr/bin/env python3
"""Score dev50 trajectories on tool protocol only; never judge final answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_tool_audit import write_json_atomic, write_text_atomic  # noqa: E402
from opensearch_vl_repro.tool_protocol_dev import validate_dev_manifest  # noqa: E402
from opensearch_vl_repro.tool_protocol_metrics import protocol_metrics  # noqa: E402
from opensearch_vl_repro.evaluation.run_manifest import tool_contract_fingerprint  # noqa: E402
from opensearch_vl_repro.sft_tool_audit import sha256_file  # noqa: E402


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dev-dir", type=Path, default=ROOT / "data/eval/tool_protocol_dev50_imageid_v3")
    cli.add_argument("--trajectories", type=Path, required=True,
                     help="Agent BatchRunner trajectories.jsonl for exactly this dev set")
    cli.add_argument("--report-dir", type=Path, default=ROOT / "reports/tool_protocol_imageid_v3")
    cli.add_argument("--allow-partial", action="store_true")
    args = cli.parse_args()
    ids = json.loads((args.dev_dir / "ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.dev_dir / "tool_protocol_dev50_manifest.json").read_text(encoding="utf-8"))
    validate_dev_manifest(ids, manifest)
    run_manifest_path = args.trajectories.parent / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (run_manifest.get("tool_contract_fingerprint") != tool_contract_fingerprint()
            or run_manifest.get("sample_selection", {}).get(
                "runtime_tool_protocol_version") != manifest["runtime_tool_protocol_version"]
            or run_manifest.get("sample_selection", {}).get("ids_sha256") != manifest["ids_sha256"]
            or run_manifest.get("dataset_identity", {}).get("sha256") != sha256_file(
                args.dev_dir / "tool_protocol_dev50_manifest.json")):
        raise ValueError("trajectory run manifest is not bound to the v3 Dev50 contract")
    rows = [json.loads(line) for line in args.trajectories.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    report = protocol_metrics(ids, manifest, rows)
    if not report["complete"] and not args.allow_partial:
        raise ValueError("tool-protocol dev run is incomplete; use --allow-partial for diagnostics")
    write_json_atomic(args.report_dir / "tool_protocol_metrics.json", report)
    lines = ["# Tool-protocol dev metrics", "",
             f"Complete: {report['complete']} ({report['observed_count']}/{report['expected_count']})", "",
             "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {name} | {value} |" for name, value in report["metrics"].items()
                 if not isinstance(value, dict))
    lines += ["", "| Sample | Source | Status | Calls | Errors |", "|---|---|---|---|---|"]
    for row in report["samples"]:
        calls = ", ".join(item["name"] for item in row["tool_calls"])
        errors = ", ".join(str(item["error"]) for item in row["tool_calls"] if item["error"])
        lines.append(f"| {row['sample_id']} | {row['source']} | {row['status']} | {calls} | {errors} |")
    write_text_atomic(args.report_dir / "tool_protocol_metrics.md", "\n".join(lines) + "\n")
    print(f"protocol metrics: {args.report_dir / 'tool_protocol_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
