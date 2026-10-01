import json
from pathlib import Path

import pytest
from PIL import Image

from opensearch_vl_repro.agent.reliability import image_sha256
from opensearch_vl_repro.eval_subset import sha256_file
from opensearch_vl_repro.rl.data import load_overlap_manifest, question_sha256
from scripts import build_rl_overlap_manifest as overlap_script


def sft_fixture(tmp_path: Path, monkeypatch):
    directory = tmp_path / "sft"
    directory.mkdir()
    shards = {}
    for index, name in enumerate(("main_a_1k", "main_b_2k", "extra_1k")):
        image_name = f"image_{index}.png"
        Image.new("RGB", (2, 2), (index * 50, 20, 30)).save(directory / image_name)
        shard_name = f"{name}.json"
        (directory / shard_name).write_text(json.dumps([{
            "conversations": [{"from": "human", "value": f"Question {index}?"}],
            "images": [image_name]}]), encoding="utf-8")
        shards[name] = {"path": shard_name, "sha256": sha256_file(directory / shard_name), "count": 1}
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(overlap_script, "load_sft_manifest", lambda _: {"shards": shards})
    return directory, shards


def test_only_explicit_sft_lineage_is_audited(tmp_path, monkeypatch):
    directory, shards = sft_fixture(tmp_path, monkeypatch)
    result = overlap_script.build_sft(directory, sft_shards=["main_a_1k", "main_b_2k"])
    assert result["sft_shards"] == ["main_a_1k", "main_b_2k"]
    assert result["audited_sample_count"] == 2
    assert result["image_audit_complete"] is True
    assert result["missing_image_count"] == 0
    assert result["sft_shard_files"] == [
        {"name": name, "path": shards[name]["path"], "sha256": shards[name]["sha256"], "count": 1}
        for name in result["sft_shards"]]
    assert question_sha256("Question 2?") not in result["question_hashes"]
    assert image_sha256(directory / "image_2.png") not in result["image_hashes"]
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert load_overlap_manifest(path, kind="sft") == result
    expanded = overlap_script.build_sft(directory, sft_shards=["main_a_1k", "main_b_2k", "extra_1k"])
    assert expanded["manifest_sha256"] != result["manifest_sha256"]
    assert question_sha256("Question 2?") in expanded["question_hashes"]


@pytest.mark.parametrize("shards", [
    [], ["main_a_1k", "main_a_1k"], ["not_a_shard"],
    ["main_b_2k", "main_a_1k"]])
def test_invalid_sft_scope_fails(tmp_path, monkeypatch, shards):
    directory, _ = sft_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="shards"):
        overlap_script.build_sft(directory, sft_shards=shards)
