#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.sft_tool_audit import (  # noqa: E402
    DATASET_ID,
    DATASET_REVISION,
    render_markdown,
    run_audit,
    write_json_atomic,
    write_text_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SearchVL-SFT tool-contract audit.")
    parser.add_argument(
        "--input",
        default=DATASET_ID,
        help="Official dataset ID, a local source directory, or one local JSON array.",
    )
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument(
        "--docs-output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "sft_tool_contract_audit.md",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require source JSON files to exist locally; never download missing files.",
    )
    args = parser.parse_args()

    report = run_audit(
        project_root=PROJECT_ROOT,
        input_value=args.input,
        raw_dir=args.raw_dir,
        dataset=args.dataset,
        revision=args.revision,
        download_missing=not args.no_download,
    )
    output_dir = args.output_dir.resolve()
    json_path = write_json_atomic(output_dir / "sft_tool_audit.json", report)
    markdown = render_markdown(report)
    markdown_path = write_text_atomic(output_dir / "sft_tool_audit.md", markdown)
    docs_path = write_text_atomic(args.docs_output.resolve(), markdown)

    print("SearchVL-SFT tool audit complete.\n")
    print(f"Trajectories: {report['total_trajectories']}")
    print(f"Tool calls: {report['total_tool_calls']}")
    print(f"Discovered tools: {report['discovered_tool_count']}")
    print(f"Malformed trajectories: {report['malformed_trajectory_count']}\n")
    for name, stats in sorted(
        report["tools"].items(), key=lambda item: (-item[1]["total_call_count"], item[0])
    ):
        print(
            f"{name}: calls={stats['total_call_count']}, "
            f"trajectories={stats['trajectory_count']}, "
            f"usage={100.0 * stats['trajectory_usage_rate']:.2f}%"
        )
    print("\nTop transitions:")
    for row in report["tool_transitions"][:10]:
        print(f"{row['from']} -> {row['to']}: {row['count']}")
    print("\nWrote:")
    print(json_path)
    print(markdown_path)
    print(docs_path)
    if report["source_errors"]:
        print("\nSource errors:")
        print(json.dumps(report["source_errors"], ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
