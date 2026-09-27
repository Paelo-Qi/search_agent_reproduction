#!/usr/bin/env python3
"""CPU-only, real-processor repair mask audit; fail closed before GPU training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.data import build_messages, load_json_records  # noqa: E402
from opensearch_vl_repro.model import load_processor  # noqa: E402
from opensearch_vl_repro.sft_repair_data import validate_repair_manifest  # noqa: E402
from opensearch_vl_repro.sft_repair_mask import (RepairCollator, target_character_spans,  # noqa: E402
                                                  target_token_spans)
from opensearch_vl_repro.sft_repair_training import load_repair_config  # noqa: E402
from opensearch_vl_repro.sft_tool_audit import (sha256_file, write_json_atomic,  # noqa: E402
                                                 write_text_atomic)
from opensearch_vl_repro.sft_train_plan import load_main_config  # noqa: E402


def audit(config_path: Path, report_dir: Path, *, examples: int = 30) -> dict:
    if examples < 20:
        raise ValueError("mask audit requires at least 20 representative samples")
    repair = load_repair_config(config_path)
    base = load_main_config(ROOT / repair["base_config"],
                            base_eval_config=ROOT / "configs/eval_base_300.yaml")
    data_path = ROOT / repair["dataset"]
    manifest = validate_repair_manifest(data_path, ROOT / repair["manifest"],
                                        mode=repair["repair_mode"])
    if (sha256_file(ROOT / "data/sft_main/manifest.json")
            != manifest["corrected_pool_manifest_sha256"]
            or sha256_file(ROOT / "data/eval/tool_protocol_dev50/"
                           "tool_protocol_dev50_manifest.json")
            != manifest["dev50_manifest_sha256"]):
        raise ValueError("repair audit provenance no longer matches corrected pool/Dev50")
    records = load_json_records(data_path)
    by_category = {}
    for index, record in enumerate(records):
        by_category.setdefault(record["_repair_category"], []).append(index)
    example_indices = set()
    for category in sorted(by_category):
        example_indices.update(by_category[category][:min(5, len(by_category[category]))])
    for index in range(len(records)):
        if len(example_indices) >= examples:
            break
        example_indices.add(index)
    processor = load_processor(base, local_files_only=True)
    collator = RepairCollator(processor, data_path, int(base["data"]["max_length"]),
                              repair["repair_mode"])
    rows = []
    supervised_total = 0
    tool_supervised_total = 0
    token_counts = []
    per_sample_tokens = []
    multi_call = 0
    for index, record in enumerate(records):
        batch = collator([record])
        full, spans = target_token_spans(processor, record, data_path, repair["repair_mode"])
        target_positions = {position for span in spans for position in
                            range(span["start"], span["end_exclusive"])}
        valid = batch.get("attention_mask")
        valid_indices = (valid[0].nonzero(as_tuple=True)[0].tolist() if valid is not None
                         else list(range(batch["input_ids"].shape[1])))
        actual_positions = {position for position, padded in enumerate(valid_indices)
                            if int(batch["labels"][0, padded]) != -100}
        if actual_positions != target_positions:
            raise ValueError(f"unexpected assistant tokens entered repair loss: {record['_sample_id']}")
        supervised_total += len(actual_positions)
        token_counts.append(len(actual_positions))
        per_sample_tokens.append({"sample_id": record["_sample_id"],
                                  "category": record["_repair_category"],
                                  "supervised_token_count": len(actual_positions),
                                  "supervised_span_count": len(spans)})
        tool_supervised_total += len(actual_positions)
        multi_call += int(len(spans) > 1)
        if index not in example_indices:
            continue
        assistant_text = "\n".join(turn["value"] for turn in record["conversations"]
                                   if turn["from"] == "gpt")
        messages, _, _ = build_messages(record, data_path)
        masked_spans = ([{"turn_index": None, "role": "system",
                          "excerpt": messages[0]["content"][:300]}]
                        if messages[0]["role"] == "system" else [])
        targets_by_turn = {}
        for turn_index, start, end, _ in target_character_spans(record, repair["repair_mode"]):
            targets_by_turn.setdefault(turn_index, []).append((start, end))
        for turn_index, turn in enumerate(record["conversations"]):
            text = turn["value"]
            if turn["from"] != "gpt":
                masked_spans.append({"turn_index": turn_index, "role": turn["from"],
                                     "excerpt": text[:300]})
                continue
            cursor = 0
            for start, end in sorted(targets_by_turn.get(turn_index, [])):
                if cursor < start:
                    masked_spans.append({"turn_index": turn_index, "role": "gpt",
                                         "excerpt": text[cursor:start][:300]})
                cursor = end
            if cursor < len(text):
                masked_spans.append({"turn_index": turn_index, "role": "gpt",
                                     "excerpt": text[cursor:][:300]})
        first_target_turn = spans[0]["turn_index"]
        registered = [f"img_{i}" for i in range(
            1, record["conversations"][0]["value"].count("<image>") + 1)]
        for turn in record["conversations"][:first_target_turn]:
            if turn["from"] == "observation":
                registered.extend(re.findall(r"New image ID:\s*(img_[1-9][0-9]*)\b",
                                             turn["value"]))
        rows.append({"sample_id": record["_sample_id"], "source": record["_source"],
                     "repair_category": record["_repair_category"],
                     "raw_assistant_target": assistant_text,
                     "rendered_text": processor.tokenizer.decode(
                         full, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False),
                     "supervised_token_count": len(actual_positions),
                     "decoded_supervised_spans": spans,
                     "masked_spans": masked_spans,
                     "image_search_url_target": [row.get("target_image_ids") for row in
                                                 manifest["membership"] if row["sample_id"] ==
                                                 record["_sample_id"]][0],
                     "registered_image_ids": registered,
                     "span_alignment_exact": True, "unexpected_assistant_tokens_in_loss": False})
    stats = {
        "passed": True, "repair_mode": repair["repair_mode"],
        "sample_count": len(records), "source_counts": dict(sorted(Counter(
            record["_source"] for record in records).items())),
        "category_counts": dict(sorted(Counter(record["_repair_category"]
                                            for record in records).items())),
        "target_tool_counts": manifest["target_tool_counts"],
        "img_n_target_counts": manifest["image_target_counts"],
        "supervised_token_total": supervised_total,
        "mean_supervised_tokens_per_sample": supervised_total / len(records),
        "min_supervised_tokens_per_sample": min(token_counts),
        "max_supervised_tokens_per_sample": max(token_counts),
        "multi_image_search_call_samples": multi_call if repair["repair_mode"] ==
        "argument_only" else 0,
        "tool_call_supervised_token_total": tool_supervised_total,
        "direct_answer_supervised_token_total": 0,
        "direct_answer_sample_count": 0,
        "per_sample_supervision": per_sample_tokens,
        "examples": rows,
    }
    prefix = "r1" if repair["repair_mode"] == "argument_only" else "r2"
    write_json_atomic(report_dir / f"{prefix}_mask_audit.json", stats)
    lines = [f"# {prefix.upper()} targeted repair mask audit", "",
             f"Samples: {len(records)}; supervised tokens: {supervised_total}; "
             f"alignment exact: yes", "", "## Representative supervised spans", ""]
    for row in rows:
        lines.extend([f"### {row['sample_id']} ({row['repair_category']})", "",
                      f"Tokens: {row['supervised_token_count']}", "",
                      "```text", *(span["decoded"] for span in row["decoded_supervised_spans"]),
                      "```", ""])
    write_text_atomic(report_dir / f"{prefix}_mask_audit.md", "\n".join(lines) + "\n")
    return stats


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--report-dir", type=Path, default=ROOT / "reports/sft_repair")
    cli.add_argument("--examples", type=int, default=30)
    args = cli.parse_args()
    stats = audit(args.config, args.report_dir, examples=args.examples)
    print(json.dumps({key: value for key, value in stats.items()
                      if key not in ("examples", "per_sample_supervision")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
