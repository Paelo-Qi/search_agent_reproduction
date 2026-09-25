#!/usr/bin/env python3
"""Inspect one official SFT sample's role-derived targets with the real processor.

No model weights, optimizer, or training step are loaded. Run after SFT images
are materialized; this is a diagnostic, not the full formal preflight gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.data import (  # noqa: E402
    SFT_MASK_VERSION, OpenSearchVLCollator, build_messages, load_json_records,
    message_role_spans, render_prompt, supervised_positions,
)
from opensearch_vl_repro.model import load_processor  # noqa: E402
from opensearch_vl_repro.sft_main_data import load_sft_manifest  # noqa: E402
from opensearch_vl_repro.sft_preflight import token_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="livevqa:5212")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/sft_main")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sft_main.yaml")
    args = parser.parse_args()
    manifest = load_sft_manifest(args.data_dir / "manifest.json")
    member = next((row for row in manifest["membership"]
                   if row["sample_id"] == args.sample_id), None)
    if member is None:
        raise ValueError(f"sample is not in the frozen 8k pool: {args.sample_id}")
    shard = member["shard"]
    dataset_path = args.data_dir / manifest["shards"][shard]["path"]
    record = next(row for row in load_json_records(dataset_path)
                  if row["_sample_id"] == args.sample_id)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    processor = load_processor(config)
    messages, images, tools = build_messages(record, dataset_path)
    prompt = render_prompt(processor, messages, tools)
    full = processor(text=[prompt], images=[images], padding=False,
                     truncation=False, return_tensors="pt")
    full_ids = full["input_ids"][0].tolist()
    spans = message_role_spans(processor, messages, images, tools, full_ids)
    max_length = int(config["data"]["max_length"])
    batch = OpenSearchVLCollator(processor, dataset_path, max_length)([record])
    valid = batch.get("attention_mask")
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    if valid is not None:
        labels = labels[valid[0].bool()]
        input_ids = input_ids[valid[0].bool()]
    truncated_ids = input_ids.tolist()
    expected = supervised_positions(spans, len(truncated_ids))
    actual = {index for index, label in enumerate(labels.tolist()) if label != -100}
    audit = token_audit(full_ids, truncated_ids, spans, messages, processor.tokenizer)
    report = {
        "sample_id": args.sample_id, "shard": shard, "mask_version": SFT_MASK_VERSION,
        "mask_matches_structured_roles": actual == expected,
        "assistant_span_count": len(spans),
        "tool_call_span_count": audit["tool_call_span_count"],
        "supervised_token_count": len(actual),
        "token_audit": audit,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if actual == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
