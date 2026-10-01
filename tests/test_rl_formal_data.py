import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from opensearch_vl_repro.agent.reliability import image_sha256
from opensearch_vl_repro.rl.data import (make_overlap_manifest, preflight_dataset,
    prepare_formal_dataset, question_sha256, read_source_parquet, safe_image_relpath,
    source_sample_id, write_dataset_artifacts)


def source_fixture(tmp_path: Path):
    root = tmp_path / "source"
    (root / "images").mkdir(parents=True)
    rows = []
    for index in range(9):
        relative = f"images/{index:06d}.png"
        Image.new("RGB", (2, 2), (index * 20, 10, 30)).save(root / relative)
        rows.append({"question": f"Question {index}?", "answer": f"Answer {index}",
                     "images": [relative], "dataset": "synthetic"})
    parquet = root / "rl.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    return root, parquet, rows


def overlap_file(tmp_path, *, name, questions=frozenset(), images=frozenset(), complete=True):
    result = make_overlap_manifest(kind=name, question_hashes=set(questions), image_hashes=set(images),
                                   source_sha256="a" * 64, image_audit_complete=complete)
    path = tmp_path / f"{name}_overlap.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def plan(tmp_path):
    root, parquet, rows = source_fixture(tmp_path)
    dataset_id, revision, seed = "OpenSearch-VL/Search-VL-RL-8K", "pinned", 20260506
    import hashlib
    ranked = sorted(range(len(rows)), key=lambda index: (
        hashlib.sha256(f"{seed}:{dataset_id}:{revision}:{source_sample_id(index)}".encode()).hexdigest(), index))
    excluded_q, excluded_i, sft_q = ranked[:3]
    eval_path = overlap_file(tmp_path, name="eval",
        questions={question_sha256(rows[excluded_q]["question"])},
        images={image_sha256(root / rows[excluded_i]["images"][0])})
    sft_path = overlap_file(tmp_path, name="sft",
        questions={question_sha256(rows[sft_q]["question"]),
                   question_sha256(rows[ranked[3]]["question"])})
    kwargs = dict(source_parquet=parquet, source_root=root, dataset_id=dataset_id,
                  dataset_revision=revision, seed=seed, smoke_count=2,
                  main_count=4, shard_size=2, eval_overlap_manifest=eval_path,
                  sft_overlap_manifest=sft_path)
    return kwargs, rows, root, ranked, eval_path, sft_path


def test_source_mapping_stable_ids_and_relative_paths(tmp_path):
    root, parquet, rows = source_fixture(tmp_path)
    assert read_source_parquet(parquet) == rows
    assert source_sample_id(0) == "rl_000000"
    assert source_sample_id(7991) == "rl_007991"
    with pytest.raises(ValueError):
        safe_image_relpath("../escape.jpg")
    with pytest.raises(ValueError):
        safe_image_relpath("C:\\absolute.jpg")


def test_deterministic_backfill_shards_sft_audit_and_preflight(tmp_path):
    kwargs, rows, root, ranked, eval_path, sft_path = plan(tmp_path)
    first = prepare_formal_dataset(**kwargs)
    second = prepare_formal_dataset(**kwargs)
    assert first == second
    ids = [item["source_sample_id"] for item in first["main"]]
    assert ids == [source_sample_id(index) for index in ranked[2:6]]
    assert first["smoke"] == first["main"][:2]
    assert first["main"] == [item for shard in first["shards"] for item in shard]
    assert [len(shard) for shard in first["shards"]] == [2, 2]
    assert first["main"][0]["reference_answer"] == rows[ranked[2]]["answer"]
    assert first["main"][0]["source_dataset"] == "synthetic"
    assert first["main"][0]["image_relpaths"] == rows[ranked[2]]["images"]
    assert first["main"][0]["prompt_id"] == first["main"][0]["trajectory_group_id"] == ids[0]
    assert first["main"][0]["question_hash"] == question_sha256(rows[ranked[2]]["question"])
    assert first["main"][0]["image_hashes"] == [image_sha256(root / rows[ranked[2]]["images"][0])]
    assert first["overlap_audit"]["eval_excluded_question_count"] == 1
    assert first["overlap_audit"]["eval_excluded_image_count"] == 1
    assert first["overlap_audit"]["sft_question_overlap_ids"] == ids[:2]
    assert ids[0] in first["main_manifest"]["membership"]  # SFT overlap is audit-only
    output = tmp_path / "output"
    write_dataset_artifacts(first, output)
    assert preflight_dataset(output, eval_overlap_manifest=eval_path,
                             sft_overlap_manifest=sft_path)["passed"] is True
    with pytest.raises(FileExistsError):
        write_dataset_artifacts(first, output)


def test_preflight_fails_on_tampered_sample_or_missing_image(tmp_path):
    kwargs, _, root, _, eval_path, sft_path = plan(tmp_path)
    output = tmp_path / "output"
    write_dataset_artifacts(prepare_formal_dataset(**kwargs), output)
    main_file = output / "main4.json"
    original = main_file.read_text(encoding="utf-8")
    records = json.loads(main_file.read_text(encoding="utf-8"))
    records[0]["reference_answer"] = "tampered"
    main_file.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError):
        preflight_dataset(output, eval_overlap_manifest=eval_path,
                          sft_overlap_manifest=sft_path)
    main_file.write_text(original, encoding="utf-8")
    (root / records[0]["image_relpaths"][0]).unlink()
    with pytest.raises(FileNotFoundError):
        preflight_dataset(output, eval_overlap_manifest=eval_path,
                          sft_overlap_manifest=sft_path)


def test_incomplete_sft_audit_is_dev_only(tmp_path):
    kwargs, _, _, _, _, _ = plan(tmp_path)
    partial = overlap_file(tmp_path, name="sft", complete=False)
    kwargs["sft_overlap_manifest"] = partial
    with pytest.raises(ValueError, match="incomplete"):
        prepare_formal_dataset(**kwargs)
    result = prepare_formal_dataset(**kwargs, allow_incomplete_sft=True)
    assert result["main_manifest"]["sft_image_audit_complete"] is False
