#!/usr/bin/env python3
"""Compare real corrected SFT processor inputs with real Agent runtime inputs; no model/API."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry  # noqa: E402
from opensearch_vl_repro.model import load_processor  # noqa: E402
from opensearch_vl_repro.sft_protocol_diagnostics import (  # noqa: E402
    load_corrected_shards, prompt_alignment_record, representative_records,
)
from opensearch_vl_repro.sft_tool_audit import write_json_atomic, write_text_atomic  # noqa: E402


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=ROOT / "data/sft_main_imageid_v3")
    cli.add_argument("--config", type=Path, default=ROOT / "configs/sft_main_imageid_v3.yaml")
    cli.add_argument("--report-dir", type=Path, default=ROOT / "reports/sft_diagnostics_imageid_v3")
    cli.add_argument("--search-config", type=Path, default=ROOT / "configs/search_backends.example.yaml")
    cli.add_argument("--layout-config", type=Path, default=ROOT / "configs/layout_parsing.example.yaml")
    cli.add_argument("--per-shard", type=int, default=15)
    args = cli.parse_args()
    if not 10 <= args.per_shard <= 20:
        raise ValueError("--per-shard must be between 10 and 20")
    manifest, by_shard = load_corrected_shards(args.data_dir, ("main_a_1k", "main_b_2k"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    processor = load_processor(config, local_files_only=True)
    runtime_declarations = create_phase3_tool_registry(
        search_config=args.search_config,
        layout_config=args.layout_config).declarations_for_model()
    rows = []
    for shard, selected in representative_records(by_shard, args.per_shard).items():
        path = args.data_dir / manifest["shards"][shard]["path"]
        for record in selected:
            row = prompt_alignment_record(record, path, processor,
                                          int(config["data"]["max_length"]),
                                          runtime_declarations)
            row["shard"] = shard
            rows.append(row)
    counts = Counter(diff["severity"] for row in rows for diff in row["differences"])
    report = {"complete": True, "sample_count": len(rows),
              "severity_counts": dict(counts), "samples": rows}
    write_json_atomic(args.report_dir / "prompt_alignment.json", report)
    lines = ["# SFT vs Agent runtime prompt alignment", "",
             f"Samples: {len(rows)}; findings: {dict(counts)}", "",
             "Training uses completed assistant targets; Eval appends a generation header.",
             "A textual difference alone is not a protocol failure.", "",
             "| Sample | Shard | Semantic drifts | Exact matches |",
             "|---|---|---|---|"]
    for row in rows:
        semantic = ", ".join(diff["field"] for diff in row["differences"]
                             if diff["severity"] == "semantic_drift") or "none"
        exact = ", ".join(diff["field"] for diff in row["differences"]
                          if diff["severity"] == "exact_match") or "none"
        lines.append(f"| {row['sample_id']} | {row['shard']} | {semantic} | {exact} |")
    write_text_atomic(args.report_dir / "prompt_alignment.md", "\n".join(lines) + "\n")
    print(f"prompt alignment report: {args.report_dir / 'prompt_alignment.json'}; {len(rows)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
