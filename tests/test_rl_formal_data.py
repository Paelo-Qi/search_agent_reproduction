import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from opensearch_vl_repro.agent.reliability import image_sha256
from opensearch_vl_repro.rl.data import (RL_SELECTION_VERSION, make_overlap_manifest, preflight_dataset,
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
    scope = ({"sft_shards": ["main_a_1k", "main_b_2k"],
              "sft_shard_files": [
                  {"name": shard, "path": f"{shard}.json", "sha256": "b" * 64, "count": count}
                  for shard, count in (("main_a_1k", 1), ("main_b_2k", 2))],
              "audited_sample_count": 3} if name == "sft" else {})
    result = make_overlap_manifest(kind=name, question_hashes=set(questions), image_hashes=set(images),
                                   source_sha256="a" * 64, image_audit_complete=complete, **scope)
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
    assert first["main_manifest"]["selection_version"] == RL_SELECTION_VERSION
    assert all(item["selection_version"] == RL_SELECTION_VERSION
               for item in (first["smoke_manifest"], *first["shard_manifests"]))
    assert "source_root" not in first["main_manifest"]
    assert "source_parquet" not in first["main_manifest"]
    assert first["overlap_audit"]["eval_excluded_question_count"] == 1
    assert first["overlap_audit"]["eval_excluded_image_count"] == 1
    assert first["overlap_audit"]["sft_question_overlap_ids"] == ids[:2]
    assert ids[0] in first["main_manifest"]["membership"]  # SFT overlap is audit-only
    output = tmp_path / "output"
    write_dataset_artifacts(first, output)
    assert preflight_dataset(output, source_parquet=kwargs["source_parquet"], source_root=root,
                             eval_overlap_manifest=eval_path,
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
        preflight_dataset(output, source_parquet=kwargs["source_parquet"], source_root=root,
                          eval_overlap_manifest=eval_path,
                          sft_overlap_manifest=sft_path)
    main_file.write_text(original, encoding="utf-8")
    (root / records[0]["image_relpaths"][0]).unlink()
    with pytest.raises(FileNotFoundError):
        preflight_dataset(output, source_parquet=kwargs["source_parquet"], source_root=root,
                          eval_overlap_manifest=eval_path,
                          sft_overlap_manifest=sft_path)


def test_incomplete_sft_audit_is_dev_only(tmp_path):
    kwargs, _, _, _, _, _ = plan(tmp_path)
    partial = overlap_file(tmp_path, name="sft", complete=False)
    kwargs["sft_overlap_manifest"] = partial
    with pytest.raises(ValueError, match="incomplete"):
        prepare_formal_dataset(**kwargs)
    result = prepare_formal_dataset(**kwargs, allow_incomplete_sft=True)
    assert result["main_manifest"]["sft_image_audit_complete"] is False


def test_manifest_fingerprint_is_independent_of_runtime_source_path(tmp_path):
    kwargs, _, root, _, eval_path, sft_path = plan(tmp_path)
    copied_root = tmp_path / "second-machine" / "source"
    shutil.copytree(root, copied_root)
    first = prepare_formal_dataset(**kwargs)
    second = prepare_formal_dataset(**(kwargs | {
        "source_root": copied_root, "source_parquet": copied_root / "rl.parquet"}))
    assert first == second
    assert first["main_manifest"]["manifest_sha256"] == second["main_manifest"]["manifest_sha256"]
    assert first["smoke_manifest"]["manifest_sha256"] == second["smoke_manifest"]["manifest_sha256"]
    assert [m["manifest_sha256"] for m in first["shard_manifests"]] == [
        m["manifest_sha256"] for m in second["shard_manifests"]]
    output = tmp_path / "portable-output"
    write_dataset_artifacts(first, output)
    assert preflight_dataset(output, source_parquet=copied_root / "rl.parquet",
                             source_root=copied_root, eval_overlap_manifest=eval_path,
                             sft_overlap_manifest=sft_path)["passed"] is True


def test_selection_version_changes_fingerprint_and_old_version_fails_preflight(tmp_path, monkeypatch):
    import opensearch_vl_repro.rl.data as data_module

    kwargs, _, root, _, eval_path, sft_path = plan(tmp_path)
    original = prepare_formal_dataset(**kwargs)
    monkeypatch.setattr(data_module, "RL_SELECTION_VERSION", "new-selection-v2")
    changed = prepare_formal_dataset(**kwargs)
    assert changed["main_manifest"]["manifest_sha256"] != original["main_manifest"]["manifest_sha256"]
    monkeypatch.setattr(data_module, "RL_SELECTION_VERSION", RL_SELECTION_VERSION)
    output = tmp_path / "old-version-output"
    write_dataset_artifacts(changed, output)
    with pytest.raises(ValueError, match="selection version"):
        preflight_dataset(output, source_parquet=kwargs["source_parquet"], source_root=root,
                          eval_overlap_manifest=eval_path, sft_overlap_manifest=sft_path)


def test_formal_preflight_requires_overlap_scope_equal_adapter_lineage(tmp_path, monkeypatch):
    from scripts import preflight_rl as script

    kwargs, _, root, _, eval_path, sft_path = plan(tmp_path)
    config = {"data": {"manifest": "main4_manifest.json", "output_dir": "unused",
                       "dataset_id": kwargs["dataset_id"], "dataset_revision": kwargs["dataset_revision"],
                       "seed": kwargs["seed"], "selection_version": RL_SELECTION_VERSION,
                       "main_count": 4, "smoke_count": 2, "shard_size": 2, "source_rows": 9},
              "model": {"sft_config": "unused", "sft_adapter": "unused"},
              "tool": {"resolved_runtime_protocol": "test"}}
    data = prepare_formal_dataset(**kwargs)
    output = tmp_path / "formal-output"
    write_dataset_artifacts(data, output)
    monkeypatch.setattr(script, "load_rl_config", lambda _: config)
    monkeypatch.setattr(script, "load_main_config", lambda *args, **kwargs: {})
    lineage = SimpleNamespace(sft_lineage=("main_a_1k", "main_b_2k"), validate=lambda: None,
                              to_dict=lambda: {"sft_lineage": ["main_a_1k", "main_b_2k"]})
    monkeypatch.setattr(script, "build_rl_lineage", lambda **kwargs: lineage)
    monkeypatch.setattr(script, "build_rl_run_manifest", lambda *args, **kwargs: {})
    args = dict(data_dir=output, source_parquet=kwargs["source_parquet"], source_root=root,
                eval_overlap_manifest=eval_path, sft_overlap_manifest=sft_path)
    assert script.preflight(tmp_path / "config.yaml", **args)["passed"] is True
    lineage.sft_lineage = ("main_a_1k", "main_b_2k", "extra_1k")
    with pytest.raises(ValueError, match="overlap shard scope"):
        script.preflight(tmp_path / "config.yaml", **args)
