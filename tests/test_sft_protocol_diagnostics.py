from __future__ import annotations

import json
import io
import zipfile

from PIL import Image

from opensearch_vl_repro.sft_main_data import runtime_tools
from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.evaluation.eval300 import FROZEN_EVAL300_SHA256
from opensearch_vl_repro.sft_protocol_diagnostics import CORRECTED_POOL_MANIFEST_SHA256
from opensearch_vl_repro import sft_protocol_diagnostics as diagnostics
from opensearch_vl_repro.sft_tool_audit import DATASET_ID, DATASET_REVISION, sha256_file
from opensearch_vl_repro.tool_protocol_dev import (candidate_metadata,
                                                   load_dev_source_samples,
                                                   select_dev_records,
                                                   validate_dev_manifest)
from opensearch_vl_repro.eval_subset import canonical_json_sha256


PINNED = {"version": 2,
          "runtime_tool_protocol_version": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
          "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
          "corrected_pool_manifest_sha256": CORRECTED_POOL_MANIFEST_SHA256,
          "eval300_dataset_sha256": FROZEN_EVAL300_SHA256,
          "dev30_subset_of_eval300": True}


def _call(name="image_search", argument="img_1"):
    key = "url" if name == "image_search" else "image"
    return {"from": "gpt", "value": "<tool_call>" + json.dumps(
        {"name": name, "arguments": {key: argument}}) + "</tool_call>"}


def _record(sample_id="fvqa:0", question="Question 0?", tool="image_search"):
    return {"_sample_id": sample_id, "_source": sample_id.split(":")[0],
            "images": ["test.png"], "tools": runtime_tools(), "system": "Source system.",
            "conversations": [{"from": "human", "value": "<image>" + question}, _call(tool)]}


class _Row:
    def __init__(self, values):
        self.values = list(values)

    def __ne__(self, other):
        return _Mask([index for index, value in enumerate(self.values) if value != other])

    def __getitem__(self, positions):
        return _Row([self.values[index] for index in positions])

    def tolist(self):
        return list(self.values)


class _Mask:
    def __init__(self, positions):
        self.positions = positions

    def nonzero(self, as_tuple=False):
        assert as_tuple
        return (_Indices(self.positions),)


class _Indices(list):
    def tolist(self):
        return list(self)


class _Matrix:
    def __init__(self, row):
        self.row = row
        self.shape = (1, len(row.values))

    def __getitem__(self, index):
        assert index == 0
        return self.row


class _Batch:
    def __init__(self, text, prefix=0):
        self.ids = _Row([ord(char) for char in text])
        self.labels = _Row([-100] * prefix + [ord(char) for char in text[prefix:]])

    def __getitem__(self, key):
        return _Matrix(self.ids if key == "input_ids" else self.labels)


class _Tokenizer:
    def decode(self, values, **kwargs):
        return "".join(chr(value) for value in values)


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        if kwargs.get("tokenize"):
            return {"input_ids": type("Shape", (), {"shape": (1, 12)})()}
        return "|".join(message["role"] for message in messages)


def test_prompt_alignment_uses_real_message_builders_and_reports_stable_drift(tmp_path, monkeypatch):
    Image.new("RGB", (4, 5)).save(tmp_path / "test.png")
    record = _record()
    monkeypatch.setattr(diagnostics, "OpenSearchVLCollator",
                        lambda *args: lambda features: _Batch("masked assistant", 7))
    first = diagnostics.prompt_alignment_record(record, tmp_path / "shard.json", _Processor(), 4096)
    again = diagnostics.prompt_alignment_record(record, tmp_path / "shard.json", _Processor(), 4096)
    assert first == again
    assert "Registered input images:\n- img_1: width=4, height=5" in first["training_system"]
    assert "Registered input images:\n- img_1: width=4, height=5" in first["runtime_system"]
    severity = {item["field"]: item["severity"] for item in first["differences"]}
    assert severity["registered_input_images"] == "exact_match"
    assert severity["tool_declarations"] == "exact_match"
    assert severity["system_guidance"] == "semantic_drift"
    assert first["supervised_token_count"] == len("assistant")


def test_supervised_url_scan_separates_masked_roles_and_tool_arguments(tmp_path, monkeypatch):
    Image.new("RGB", (4, 5)).save(tmp_path / "test.png")
    record = _record()
    record["system"] = "https://system.example/a.png"
    record["conversations"][0]["value"] += " https://user.example/b.jpg"
    record["conversations"].extend([
        {"from": "observation", "value": "https://observation.example/c.webp"},
        {"from": "gpt", "value": "Answer: https://assistant.example/d.jpeg"},
    ])
    supervised = ('<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}'
                  '</tool_call> Answer: https://assistant.example/d.jpeg')
    monkeypatch.setattr(diagnostics, "OpenSearchVLCollator",
                        lambda *args: lambda features: _Batch("MASK" + supervised, 4))
    rows = diagnostics.supervised_url_rows(record, "main_a_1k", tmp_path / "shard.json",
                                           _Processor(), 4096)
    assert any(row["category"] == "assistant_natural_language" and row["pattern"] == "https"
               for row in rows)
    assert any(row["category"] == "image_search_arguments" and row["pattern"] == '"url":'
               for row in rows)
    assert not any(row["category"] == "image_search_arguments" and row["pattern"] == "https"
                   for row in rows)
    assert {row["category"] for row in rows if row["pattern"] == "https"} == {
        "assistant_natural_language", "masked_system", "masked_user", "masked_tool_observation"}
    summary = diagnostics.summarize_url_rows(rows)
    assert summary["supervised_by_shard"]["main_a_1k"]["https"] == 1
    assert summary["by_category"]["masked_tool_observation"]["https"]["count"] == 1


def test_tool_distribution_counts_real_parser_calls_and_no_tool():
    direct = _record("fvqa:1")
    direct["conversations"][1] = {"from": "gpt", "value": "Direct answer"}
    chained = _record("fvqa:2")
    chained["conversations"].extend([
        {"from": "observation", "value": "Found entity"},
        {"from": "gpt", "value": '<tool_call>{"name":"text_search","arguments":{"q":"entity"}}</tool_call>'},
    ])
    result = diagnostics.tool_distribution({"main_a_1k": [direct, chained]})["main_a_1k"]
    assert result["total_samples"] == 2 and result["no_tool_sample_count"] == 1
    assert result["multi_tool_sample_count"] == 1
    assert result["image_search_followed_by_text_search_ratio"] == 1.0
    assert result["by_source"]["fvqa"]["first_tool"]["no_tool"] == 1


def test_dev_candidates_exclude_eval_question_membership_frozen_and_bad(tmp_path, monkeypatch):
    from opensearch_vl_repro import tool_protocol_dev as dev
    raw = tmp_path / "fvqa" / "records.json"
    raw.parent.mkdir()
    rows = [_record(f"fvqa:{index}", f"Question {index}?") for index in range(6)]
    rows[4]["conversations"][1] = _call("image_search", "https://bad.example/a.png")
    raw.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(dev, "SOURCE_FILES", {"fvqa": "fvqa/records.json"})
    pool = {"membership": [{"sample_id": "fvqa:0"}],
            "source_files": {"fvqa": {"sha256": sha256_file(raw)}}}
    candidates, counts = candidate_metadata(tmp_path, pool, {"question 1?"},
                                             {"fvqa:2": "derived_image_id_gap"})
    assert {row["sample_id"] for row in candidates} == {"fvqa:3", "fvqa:5"}
    assert counts["eval_question_overlap"] == 1 and counts["image_contract_bad"] == 1


def test_dev_selection_repeatable_image_overlap_safe_and_checksum():
    items = [{"sample_id": f"fvqa:{index}", "source": "fvqa", "source_index": index,
              "record": {}, "tags": ["image_search_img_1"]} for index in range(8)]
    selected, info = select_dev_records(items, image_hashes=lambda item: [
        "eval-hash" if item["sample_id"] == "fvqa:0" else item["sample_id"]],
        forbidden_image_hashes={"eval-hash"}, seed=42, size=4,
        targets={"image_search_img_1": 4})
    again, _ = select_dev_records(reversed(items), image_hashes=lambda item: [
        "eval-hash" if item["sample_id"] == "fvqa:0" else item["sample_id"]],
        forbidden_image_hashes={"eval-hash"}, seed=42, size=4,
        targets={"image_search_img_1": 4})
    ids = [item["sample_id"] for item in selected]
    assert ids == [item["sample_id"] for item in again]
    assert "fvqa:0" not in ids and info["image_overlap_candidate_skip_count"] <= 1
    samples = [{"sample_id": value} for value in ids]
    manifest = {"count": 4, "samples": samples, "ids_sha256": canonical_json_sha256(ids),
                "sample_metadata_sha256": canonical_json_sha256(samples),
                **PINNED,
                **{key: 0 for key in ("eval300_question_overlap_count", "eval300_image_overlap_count",
                                       "sft_membership_overlap_count", "frozen_exclusion_overlap_count",
                                       "image_contract_bad_count")}}
    validate_dev_manifest(ids, manifest)
    manifest["ids_sha256"] = "wrong"
    import pytest
    with pytest.raises(ValueError, match="checksum"):
        validate_dev_manifest(ids, manifest)
    manifest["ids_sha256"] = canonical_json_sha256(ids)
    manifest["corrected_pool_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="pinned provenance"):
        validate_dev_manifest(ids, manifest)


def test_dev_source_reader_uses_only_initial_images_from_local_zip(tmp_path, monkeypatch):
    from opensearch_vl_repro import tool_protocol_dev as dev
    monkeypatch.setattr(dev, "SOURCE_FILES", {"fvqa": "fvqa/records.json"})
    source = tmp_path / "fvqa"
    source.mkdir()
    record = _record("fvqa:0")
    record["images"] = ["test.png", "derived.png"]
    record["conversations"].extend([
        {"from": "observation", "value": "<image>New image ID: img_2."},
        {"from": "gpt", "value": "Done"},
    ])
    raw = source / "records.json"
    raw.write_text(json.dumps([record]), encoding="utf-8")
    with zipfile.ZipFile(source / "images.zip", "w") as archive:
        for name in ("test.png", "derived.png"):
            buffer = io.BytesIO()
            Image.new("RGB", (3, 4), "navy").save(buffer, format="PNG")
            archive.writestr(f"images/{name}", buffer.getvalue())
    ids = ["fvqa:0"]
    samples = [{"sample_id": "fvqa:0"}]
    manifest = {"count": 1, "samples": samples, "ids_sha256": canonical_json_sha256(ids),
                "sample_metadata_sha256": canonical_json_sha256(samples),
                **PINNED,
                "source_file_sha256": {"fvqa": sha256_file(raw)},
                **{key: 0 for key in ("eval300_question_overlap_count", "eval300_image_overlap_count",
                                       "sft_membership_overlap_count", "frozen_exclusion_overlap_count",
                                       "image_contract_bad_count")}}
    result = load_dev_source_samples(ids, manifest, tmp_path)
    assert result[0][0] == "fvqa:0" and result[0][1] == "Question 0?"
    assert len(result[0][2]) == 1 and result[0][2][0].size == (3, 4)
