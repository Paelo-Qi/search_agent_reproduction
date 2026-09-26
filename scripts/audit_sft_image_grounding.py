#!/usr/bin/env python3
"""CPU-only, local-only inspection of formal SFT messages and main 3k image IDs.

Requires already materialized shard media and an already cached pinned processor.
It does not download data/models, load model weights, or modify the shards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME  # noqa: E402
from opensearch_vl_repro.agent.tool_parser import ToolCallParser  # noqa: E402
from opensearch_vl_repro.data import (build_messages, load_json_records,  # noqa: E402
                                      messages_to_json_safe, render_prompt)
from opensearch_vl_repro.model import load_processor  # noqa: E402
from opensearch_vl_repro.sft_image_grounding import (  # noqa: E402
    audit_sample_image_grounding, summarize_image_grounding,
)
from opensearch_vl_repro.sft_main_data import (  # noqa: E402
    load_data_quality_exclusions, load_sft_manifest, require_data_quality_exclusions,
)


def first_direct_image_search(records: list[dict]) -> tuple[int, dict]:
    parser = ToolCallParser(TOOL_DECLARATIONS_BY_NAME)
    for index, record in enumerate(records):
        turns = record["conversations"]
        if len(turns) < 2 or turns[0]["from"] != "human" or turns[1]["from"] != "gpt":
            continue
        if any(call.name == "image_search" and call.arguments.get("url") == "img_1"
               for call in parser.parse(turns[1]["value"]).tool_calls):
            return index, record
    raise ValueError("main_a_1k contains no first-turn image_search(img_1) sample")


def selection_identity_unchanged(previous: dict, current: dict) -> bool:
    """Ignore only message-format metadata; identities and partition must match."""
    keys = ("selection_algorithm", "seed", "membership", "exclusions",
            "replacements", "pool_source_counts", "source_population_counts")
    return (all(previous.get(key) == current.get(key) for key in keys)
            and all(previous.get("shards", {}).get(name, {}).get("source_counts")
                    == current.get("shards", {}).get(name, {}).get("source_counts")
                    for name in current.get("shards", {})))


def selection_change_matches_frozen_exclusions(previous: dict, current: dict) -> bool:
    """Accept only the planned eight same-source replacements, not arbitrary drift."""
    frozen = load_data_quality_exclusions()
    old_ids = {row["sample_id"] for row in previous.get("membership", [])}
    new_ids = {row["sample_id"] for row in current.get("membership", [])}
    replacement_rows = [row for row in current.get("replacements", [])
                        if row["excluded_sample_id"] in frozen]
    replacements = {row["excluded_sample_id"]: row for row in replacement_rows}
    expected_removed = old_ids & frozen.keys()
    if (len(replacement_rows) != len(expected_removed)
            or set(replacements) != expected_removed
            or old_ids - new_ids != expected_removed
            or new_ids - old_ids != {row["replacement_sample_id"] for row in replacements.values()}):
        return False
    if any(row["reason"] != frozen[sample_id]
           or row["replacement_source"] != sample_id.split(":", 1)[0]
           or row["replacement_sample_id"].split(":", 1)[0] != row["replacement_source"]
           for sample_id, row in replacements.items()):
        return False
    return (previous.get("seed") == current.get("seed")
            and previous.get("selection_algorithm") == current.get("selection_algorithm")
            and previous.get("pool_source_counts") == current.get("pool_source_counts")
            and {name: row["count"] for name, row in previous.get("shards", {}).items()}
            == {name: row["count"] for name, row in current.get("shards", {}).items()})


def legacy_messages(sample: dict, current: list[dict]) -> list[dict]:
    """Undo only the new system grounding for a read-only before/after render."""
    if not current or current[0]["role"] != "system":
        return list(current)
    if sample.get("system"):
        return [{"role": "system", "content": sample["system"]}, *current[1:]]
    return current[1:]


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=ROOT / "data/sft_main")
    cli.add_argument("--config", type=Path, default=ROOT / "configs/sft_main.yaml")
    cli.add_argument("--output", type=Path, help="Optional JSON report path (shards are never written)")
    cli.add_argument("--previous-manifest", type=Path,
                     help="Legacy manifest to verify unchanged selection and shard partition")
    args = cli.parse_args()
    manifest = load_sft_manifest(args.data_dir / "manifest.json")
    require_data_quality_exclusions(manifest)
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    processor = load_processor(config, local_files_only=True)
    names = ("main_a_1k", "main_b_2k")
    records_by_shard = {name: load_json_records(args.data_dir / manifest["shards"][name]["path"])
                        for name in names}
    sample_index, sample = first_direct_image_search(records_by_shard["main_a_1k"])
    messages, _, tools = build_messages(sample, args.data_dir / manifest["shards"]["main_a_1k"]["path"])
    assistant_index = next(index for index, row in enumerate(messages) if row["role"] == "assistant")
    before = render_prompt(processor, messages[:assistant_index], tools)
    legacy = legacy_messages(sample, messages)
    legacy_assistant_index = next(index for index, row in enumerate(legacy)
                                  if row["role"] == "assistant")
    legacy_before = render_prompt(processor, legacy[:legacy_assistant_index], tools)
    inspection = {
        "sample_index": sample_index, "sample_id": sample["_sample_id"],
        "messages": messages_to_json_safe(messages), "tools": tools,
        "formatted_prompt": render_prompt(processor, messages, tools),
        "before_first_assistant_image_search": before,
        "img_1_registered_before_call": "Registered input images:" in before and "- img_1:" in before,
        "legacy_messages": messages_to_json_safe(legacy),
        "legacy_formatted_prompt": render_prompt(processor, legacy, tools),
        "legacy_before_first_assistant_image_search": legacy_before,
        "legacy_img_1_registered_before_call":
            "Registered input images:" in legacy_before and "- img_1:" in legacy_before,
    }
    rows = []
    for name, records in records_by_shard.items():
        path = args.data_dir / manifest["shards"][name]["path"]
        for record in records:
            effective, _, _ = build_messages(record, path)
            rows.append(audit_sample_image_grounding(record, effective))
    summary = summarize_image_grounding(rows)
    report = {"inspection": inspection, "main_3k_audit": summary}
    valid_selection = True
    if args.previous_manifest:
        previous = json.loads(args.previous_manifest.read_text(encoding="utf-8"))
        same_selection = selection_identity_unchanged(previous, manifest)
        controlled_change = selection_change_matches_frozen_exclusions(previous, manifest)
        valid_selection = same_selection or controlled_change
        report["selection_identity_unchanged"] = same_selection
        report["selection_change_matches_frozen_exclusions"] = controlled_change
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (summary["passed"] and inspection["img_1_registered_before_call"]
                 and valid_selection) else 1


if __name__ == "__main__":
    raise SystemExit(main())
