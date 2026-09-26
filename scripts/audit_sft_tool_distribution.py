#!/usr/bin/env python3
"""Compare corrected main_a_1k/main_b_2k expert tool distributions without a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_protocol_diagnostics import (  # noqa: E402
    load_corrected_shards, tool_distribution,
)
from opensearch_vl_repro.sft_tool_audit import write_json_atomic, write_text_atomic  # noqa: E402


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=ROOT / "data/sft_main")
    cli.add_argument("--report-dir", type=Path, default=ROOT / "reports/sft_diagnostics")
    args = cli.parse_args()
    _, shards = load_corrected_shards(args.data_dir, ("main_a_1k", "main_b_2k"))
    report = {"complete": True, "shards": tool_distribution(shards)}
    write_json_atomic(args.report_dir / "tool_distribution.json", report)
    lines = ["# Corrected SFT expert tool distribution", "",
             "| Metric | main_a_1k | main_b_2k |", "|---|---:|---:|"]
    for label, key in (("Samples", "total_samples"), ("Tool-containing ratio", "tool_containing_sample_ratio"),
                       ("Mean tool calls", "mean_tool_calls_per_trajectory"),
                       ("No tool", "no_tool_sample_count"), ("Single tool", "single_tool_sample_count"),
                       ("Multi tool", "multi_tool_sample_count"),
                       ("image_search -> text_search ratio", "image_search_followed_by_text_search_ratio")):
        left, right = (report["shards"][name][key] for name in ("main_a_1k", "main_b_2k"))
        lines.append(f"| {label} | {left} | {right} |")
    for name in sorted(set(report["shards"]["main_a_1k"]["tool_call_counts"]) |
                       set(report["shards"]["main_b_2k"]["tool_call_counts"])):
        left = report["shards"]["main_a_1k"]["tool_call_counts"].get(name, 0)
        right = report["shards"]["main_b_2k"]["tool_call_counts"].get(name, 0)
        lines.append(f"| {name} calls | {left} | {right} |")
    write_text_atomic(args.report_dir / "tool_distribution.md", "\n".join(lines) + "\n")
    print(f"tool distribution report: {args.report_dir / 'tool_distribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
