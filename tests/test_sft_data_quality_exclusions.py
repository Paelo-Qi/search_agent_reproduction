"""CPU-only checks for frozen exclusions and deterministic same-source refill."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from opensearch_vl_repro.data import build_messages
from opensearch_vl_repro.sft_image_grounding import (audit_sample_image_grounding,
                                                     summarize_image_grounding)
from opensearch_vl_repro.sft_main_data import (load_data_quality_exclusions,
                                               prepare_sft_pool,
                                               require_data_quality_exclusions,
                                               selection_plan)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "fvqa:219": "derived_image_id_gap", "fvqa:1145": "derived_image_id_gap",
    "fvqa:2711": "derived_image_id_gap", "fvqa:3026": "derived_image_id_gap",
    "fvqa:3371": "derived_image_id_gap", "fvqa:4016": "derived_image_id_gap",
    "webqa:1884": "image_search_non_img_n_target",
    "webqa:3301": "image_search_non_img_n_target",
}


def test_frozen_eight_and_formal_manifest_gate():
    frozen = load_data_quality_exclusions()
    assert len(frozen) == 104
    assert all(frozen.get(sample_id) == reason for sample_id, reason in EXPECTED.items())
    assert list(frozen.values()).count("image_search_non_img_n_target") == 22
    assert list(frozen.values()).count("derived_image_id_gap") == 82
    manifest = {"exclusions": [{"sample_id": key, "reason": value}
                               for key, value in frozen.items()], "membership": []}
    require_data_quality_exclusions(manifest)
    manifest["exclusions"][0]["reason"] = "wrong"
    with pytest.raises(ValueError, match="data-quality exclusions mismatch"):
        require_data_quality_exclusions(manifest)
    manifest["exclusions"][0]["reason"] = frozen[manifest["exclusions"][0]["sample_id"]]
    manifest["membership"] = [{"sample_id": "fvqa:219"}]
    with pytest.raises(ValueError, match="contains a frozen"):
        require_data_quality_exclusions(manifest)


def test_formal_prepare_cli_requires_explicit_exclusions(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_sft_main.py"),
         "--output-dir", str(tmp_path / "unused")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "--exclusions" in result.stderr
    assert not (tmp_path / "unused").exists()


def test_previous_pool_comparison_accepts_only_frozen_same_source_refill():
    compare = runpy.run_path(str(ROOT / "scripts/audit_sft_image_grounding.py"))[
        "selection_change_matches_frozen_exclusions"]
    old_ids = [*EXPECTED, "fvqa:9999"]
    replacement_ids = {sample_id: f"{sample_id.split(':')[0]}:{7000 + index}"
                       for index, sample_id in enumerate(EXPECTED)}
    common = {"seed": 20260506, "selection_algorithm": "same",
              "pool_source_counts": {"fvqa": 7, "webqa": 2},
              "shards": {"main_a_1k": {"count": 9}}}
    previous = {**common, "membership": [{"sample_id": value} for value in old_ids]}
    current = {**common,
               "membership": [{"sample_id": value} for value in
                              ["fvqa:9999", *replacement_ids.values()]],
               "replacements": [{"excluded_sample_id": sample_id, "reason": reason,
                                 "replacement_sample_id": replacement_ids[sample_id],
                                 "replacement_source": sample_id.split(":")[0]}
                                for sample_id, reason in EXPECTED.items()]}
    assert compare(previous, current)
    current["replacements"][0]["replacement_source"] = "wrong"
    assert not compare(previous, current)


def test_eight_selected_exclusions_refill_same_source_without_target_rewrite(tmp_path, monkeypatch):
    from opensearch_vl_repro import sft_main_data as module

    counts = {"fvqa": 12, "webqa": 4}
    sizes = {"main_a_1k": 1, "main_b_2k": 2, "extra_1k": 1, "reserve_4k": 4}
    monkeypatch.setattr(module, "SOURCE_FILES", {
        "fvqa": "fvqa/records.json", "webqa": "webqa/records.json"})
    monkeypatch.setattr(module, "SOURCE_COUNTS", counts)
    monkeypatch.setattr(module, "SHARD_SIZES", sizes)
    monkeypatch.setattr(module, "POOL_SIZE", 8)
    plan = selection_plan(source_counts=counts, shard_sizes=sizes,
                          eligible_indices={source: list(range(count))
                                            for source, count in counts.items()})
    selected = {source: {index for shard in plan.values() for index in shard[source]}
                for source in counts}
    assert {source: len(indices) for source, indices in selected.items()} == {
        "fvqa": 6, "webqa": 2}
    exclusions = {f"{source}:{index}": ("derived_image_id_gap" if source == "fvqa"
                                        else "image_search_http_url_target")
                  for source, indices in selected.items() for index in indices}
    raw = tmp_path / "raw"
    source_records = {}
    for source, count in counts.items():
        records = []
        for index in range(count):
            target = ('<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}'
                      '</tool_call>')
            record = {"system": "Source guidance", "tools": "[]",
                      "images": [f"image_{index}.png"],
                      "conversations": [
                          {"from": "human", "value": f"<image>Question {source}:{index}?"},
                          {"from": "gpt", "value": target},
                      ]}
            if index in selected[source] and source == "fvqa":
                record["images"].append(f"derived_{index}.png")
                record["conversations"].extend([
                    {"from": "observation", "value":
                     "<image><observation>New image ID: img_3.</observation>"},
                    {"from": "gpt", "value": target.replace("img_1", "img_3")},
                ])
            elif index in selected[source]:
                record["conversations"][1]["value"] = target.replace(
                    "img_1", "https://example.invalid/image.png")
            records.append(record)
        source_records[source] = records
        path = raw / source / "records.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records), encoding="utf-8")

    legacy = prepare_sft_pool(raw, tmp_path / "legacy")
    corrected_dir = tmp_path / "corrected"
    corrected = prepare_sft_pool(raw, corrected_dir, exclusions=exclusions)
    again = prepare_sft_pool(raw, tmp_path / "again", exclusions=exclusions)
    old_ids = {row["sample_id"] for row in legacy["membership"]}
    new_ids = {row["sample_id"] for row in corrected["membership"]}
    assert old_ids == set(exclusions)
    assert new_ids.isdisjoint(exclusions)
    assert corrected["membership"] == again["membership"]
    assert corrected["shards"] == again["shards"]
    assert corrected["pool_source_counts"] == legacy["pool_source_counts"]
    assert {name: row["count"] for name, row in corrected["shards"].items()} == sizes
    assert {name: row["source_counts"] for name, row in corrected["shards"].items()} == {
        name: legacy["shards"][name]["source_counts"] for name in sizes}
    replacement_rows = {row["excluded_sample_id"]: row for row in corrected["replacements"]}
    assert set(replacement_rows) == set(exclusions)
    assert all(row["replacement_source"] == sample_id.split(":")[0]
               and row["replacement_sample_id"].split(":")[0] == row["replacement_source"]
               for sample_id, row in replacement_rows.items())
    assert {row["sample_id"]: row["reason"] for row in corrected["exclusions"]} == exclusions

    grounding_rows = []
    for shard in sizes:
        shard_path = corrected_dir / f"{shard}.json"
        records = json.loads(shard_path.read_text(encoding="utf-8"))
        for record in records:
            source = record["_source"]
            index = record["_source_index"]
            assert record["conversations"] == source_records[source][index]["conversations"]
            for relative in record["images"]:
                image = corrected_dir / relative
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), "navy").save(image)
            messages, _, _ = build_messages(record, shard_path)
            grounding_rows.append(audit_sample_image_grounding(record, messages))
    report = summarize_image_grounding(grounding_rows)
    assert report["passed"] and report["counts"].get("image_search_non_img_n", 0) == 0
    assert report["counts"]["image_search_img_n"] == 8


def test_abnormal_replacement_candidate_is_skipped_before_selection():
    counts = {"fvqa": 8}
    sizes = {"main_a_1k": 2}
    eligible = {"fvqa": list(range(8))}
    original = selection_plan(source_counts=counts, shard_sizes=sizes,
                              eligible_indices=eligible)["main_a_1k"]["fvqa"]
    first_bad = f"fvqa:{original[0]}"
    first_refill = selection_plan(source_counts=counts, shard_sizes=sizes,
                                  eligible_indices=eligible,
                                  exclusions={first_bad: "derived_image_id_gap"})[
                                      "main_a_1k"]["fvqa"]
    candidate = next(index for index in first_refill if index not in original)
    frozen = {first_bad: "derived_image_id_gap",
              f"fvqa:{candidate}": "derived_image_id_gap"}
    final = selection_plan(source_counts=counts, shard_sizes=sizes,
                           eligible_indices=eligible, exclusions=frozen)[
                               "main_a_1k"]["fvqa"]
    assert len(final) == 2 and original[0] not in final and candidate not in final
