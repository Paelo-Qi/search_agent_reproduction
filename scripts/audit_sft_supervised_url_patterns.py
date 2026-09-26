#!/usr/bin/env python3
"""Scan only real collator-supervised assistant tokens for URL patterns; no model/API."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from opensearch_vl_repro.model import load_processor  # noqa: E402
from opensearch_vl_repro.sft_protocol_diagnostics import (  # noqa: E402
    load_corrected_shards, summarize_url_rows, supervised_url_rows,
)
from opensearch_vl_repro.sft_tool_audit import write_json_atomic, write_text_atomic  # noqa: E402


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=ROOT / "data/sft_main")
    cli.add_argument("--config", type=Path, default=ROOT / "configs/sft_main.yaml")
    cli.add_argument("--report-dir", type=Path, default=ROOT / "reports/sft_diagnostics")
    cli.add_argument("--include-full-8k", action="store_true")
    args = cli.parse_args()
    names = ("main_a_1k", "main_b_2k", "extra_1k", "reserve_4k") if args.include_full_8k else (
        "main_a_1k", "main_b_2k")
    manifest, by_shard = load_corrected_shards(args.data_dir, names)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    processor = load_processor(config, local_files_only=True)
    rows = []
    for shard, records in by_shard.items():
        path = args.data_dir / manifest["shards"][shard]["path"]
        for index, record in enumerate(records, 1):
            rows.extend(supervised_url_rows(record, shard, path, processor,
                                            int(config["data"]["max_length"])))
            if index % 100 == 0:
                print(f"{shard}: {index}/{len(records)}", flush=True)
    summary = summarize_url_rows(rows)
    summary.update(complete=True, scanned_samples={key: len(value) for key, value in by_shard.items()},
                   supervised_category_counts=dict(Counter(row["category"] for row in rows
                       if not row["category"].startswith("masked_"))))
    write_json_atomic(args.report_dir / "supervised_url_patterns.json", summary)
    lines = ["# URL patterns in corrected SFT", "",
             "Counts below derive from collator labels != -100; masked roles are separate.", "",
             "| Shard | Source | Category | Tool | Pattern | Count | Samples |",
             "|---|---|---|---|---|---:|---:|"]
    for group in summary["groups"]:
        lines.append("| {shard} | {source} | {category} | {tool} | {pattern} | {count} | {sample_count} |".format(**group))
    write_text_atomic(args.report_dir / "supervised_url_patterns.md", "\n".join(lines) + "\n")
    print(f"supervised URL report: {args.report_dir / 'supervised_url_patterns.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
