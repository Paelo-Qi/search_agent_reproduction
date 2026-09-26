from __future__ import annotations

import json

from PIL import Image
import pytest

from opensearch_vl_repro.data import build_messages
from opensearch_vl_repro.sft_image_grounding import (audit_raw_image_contract,
                                                     audit_sample_image_grounding)
from opensearch_vl_repro.sft_source_image_audit import audit_source_population
from opensearch_vl_repro.sft_tool_audit import sha256_file


def _call(name: str, argument: str) -> dict[str, str]:
    key = "url" if name == "image_search" else "image"
    return {"from": "gpt", "value": "<tool_call>" + json.dumps({
        "name": name, "arguments": {key: argument}}) + "</tool_call>"}


def _record(index: int, *, initial: int = 1) -> dict:
    return {"images": [f"image_{index}_{position}.png" for position in range(initial)],
            "tools": "[]", "conversations": [
                {"from": "human", "value": "<image>" * initial + "What is this?"},
                _call("image_search", "img_1"),
            ]}


def test_full_source_static_audit_detects_non_ids_gaps_and_early_references(tmp_path):
    rows = [_record(index) for index in range(7)]
    rows[1]["conversations"][1] = _call("image_search", "https://example.org/a.jpg")
    rows[2]["conversations"][1] = _call("image_search", "local-file.png")
    rows[3]["images"].append("derived_3.png")
    rows[3]["conversations"] = [rows[3]["conversations"][0], _call("crop", "img_1"),
        {"from": "observation", "value":
         "<image><observation>New image ID: img_3.</observation>"},
        _call("image_search", "img_3")]
    rows[4]["conversations"] = [rows[4]["conversations"][0],
        _call("image_search", "img_2"),
        {"from": "observation", "value": "No image produced."},
        {"from": "gpt", "value": "Final answer."}]
    rows[5] = _record(5, initial=2)
    rows[5]["images"].append("derived_5.png")
    rows[5]["conversations"] = [rows[5]["conversations"][0], _call("crop", "img_2"),
        {"from": "observation", "value":
         "<image><observation>New image ID: img_3.</observation>"},
        _call("image_search", "img_3")]
    rows[6]["images"].append("derived_6.png")
    rows[6]["conversations"] = [rows[6]["conversations"][0], _call("crop", "img_1"),
        {"from": "observation", "value": "<image><observation>No ID.</observation>"},
        {"from": "gpt", "value": "Final answer."}]
    raw = tmp_path / "fvqa" / "records.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps(rows), encoding="utf-8")
    report = audit_source_population(
        tmp_path, source_files={"fvqa": "fvqa/records.json"},
        source_counts={"fvqa": 7}, expected_source_sha256={"fvqa": sha256_file(raw)})
    assert report["complete"] and report["total_records"] == 7
    assert report["records_with_image_search"] == 6
    assert report["image_search_non_img_n_count"] == 2
    assert report["derived_image_id_gap_count"] == 2
    assert report["ungrounded_image_reference_count"] == 2
    assert report["sample_ids_by_kind"]["image_search_non_img_n"] == ["fvqa:1", "fvqa:2"]
    assert report["sample_ids_by_kind"]["derived_image_id_gap"] == ["fvqa:3", "fvqa:6"]
    assert report["sample_ids_by_kind"]["ungrounded_image_reference"] == ["fvqa:3", "fvqa:4"]
    assert report["all_bad_sample_ids"] == ["fvqa:1", "fvqa:2", "fvqa:3", "fvqa:4", "fvqa:6"]
    assert report["recommended_exclusions"] == {
        "fvqa:1": "image_search_non_img_n_target", "fvqa:2": "image_search_non_img_n_target",
        "fvqa:3": "derived_image_id_gap", "fvqa:4": "ungrounded_image_reference",
        "fvqa:6": "derived_image_id_gap"}
    gap = next(row for row in report["anomalies"]
               if row["sample_id"] == "fvqa:3" and row["kind"] == "derived_image_registration_gap")
    assert (gap["source"], gap["source_index"], gap["turn_index"], gap["tool"],
            gap["argument"]) == ("fvqa", 3, 2, "crop", "img_3")
    assert "New image ID: img_3" in gap["relevant_observation"]
    assert report["per_source"]["fvqa"]["bad_record_count"] == 5
    with pytest.raises(ValueError, match="pinned source SHA256 mismatch"):
        audit_source_population(
            tmp_path, source_files={"fvqa": "fvqa/records.json"},
            source_counts={"fvqa": 7}, expected_source_sha256={"fvqa": "0" * 64})


def test_source_and_final_message_audits_share_the_same_turn_contract(tmp_path):
    sample = _record(0)
    sample["_sample_id"] = "fvqa:0"
    sample["images"].append("derived_0.png")
    sample["conversations"] = [sample["conversations"][0], _call("crop", "img_1"),
        {"from": "observation", "value":
         "<image><observation>New image ID: img_3.</observation>"},
        _call("image_search", "img_3")]
    for name in sample["images"]:
        Image.new("RGB", (4, 4), "navy").save(tmp_path / name)
    raw = audit_raw_image_contract(sample)
    messages, _, _ = build_messages(sample, tmp_path / "shard.json")
    final = audit_sample_image_grounding(sample, messages)
    assert raw["errors"] == final["errors"]
    assert raw["calls"] == final["calls"]


def test_assistant_prose_cannot_reference_unregistered_image_id():
    sample = _record(0)
    sample["conversations"][1] = {"from": "gpt", "value": "Inspect img_2 next."}
    result = audit_raw_image_contract(sample, sample_id="fvqa:0")
    assert result["errors"] == [{
        "kind": "ungrounded_assistant_image_reference", "turn_index": 1,
        "tool": None, "argument": "img_2", "image_id": "img_2",
        "relevant_observation": None, "registered_ids_before": ["img_1"],
    }]
