"""Audited Eval-300 v2 text replacement and frozen-version regression tests."""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from opensearch_vl_repro.eval_subset import sha256_file
from opensearch_vl_repro.evaluation import (FROZEN_EVAL300_SHA256,
                                           FROZEN_EVAL300_V2_SHA256,
                                           build_eval300_plan,
                                           build_run_manifest,
                                           expected_eval300_sha256_for_config)
from opensearch_vl_repro.evaluation.judge import load_judge_samples
from opensearch_vl_repro.evaluation.run_manifest import frozen_dataset_identity
from opensearch_vl_repro.inference import load_inference_config


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data/eval/combined_eval_300.parquet"
V2 = ROOT / "data/eval/combined_eval_300_v2.parquet"
MANIFEST_V1 = ROOT / "data/eval/manifest.json"
MANIFEST_V2 = ROOT / "data/eval/manifest_v2.json"
AUDITED = ROOT.parent / "eval_300/eval300_blind_audit_rebuilt_v1/samples.json"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_eval300_v2", ROOT / "scripts/build_eval300_v2.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_eval300_v2


def test_formal_v1_and_v2_configs_are_identical_except_dataset_path():
    v1 = yaml.safe_load((ROOT / "configs/eval_base_300.yaml").read_text(encoding="utf-8"))
    v2 = yaml.safe_load((ROOT / "configs/eval_base_300_v2.yaml").read_text(encoding="utf-8"))
    assert {key: value for key, value in v1.items() if key != "data"} == {
        key: value for key, value in v2.items() if key != "data"}
    assert v1["data"] == {"path": "data/eval/combined_eval_300.parquet"}
    assert v2["data"] == {"path": "data/eval/combined_eval_300_v2.parquet",
                          "manifest": "data/eval/manifest_v2.json"}
    assert load_inference_config(ROOT / "configs/eval_base_300.yaml").eval_manifest_path == (
        MANIFEST_V1)
    loaded_v2 = load_inference_config(ROOT / "configs/eval_base_300_v2.yaml")
    assert loaded_v2.data_path == V2 and loaded_v2.eval_manifest_path == MANIFEST_V2
    assert expected_eval300_sha256_for_config(ROOT / "configs/eval_base_300.yaml") == (
        FROZEN_EVAL300_SHA256)
    assert expected_eval300_sha256_for_config(ROOT / "configs/eval_base_300_v2.yaml") == (
        FROZEN_EVAL300_V2_SHA256)


@pytest.mark.parametrize("path,config,expected", [
    (V1, "eval_base_300.yaml", FROZEN_EVAL300_SHA256),
    (V2, "eval_base_300_v2.yaml", FROZEN_EVAL300_V2_SHA256),
])
def test_frozen_eval300_both_versions_pass_with_same_balanced_membership(path, config, expected):
    plan = build_eval300_plan(path, expected_sha256=expected_eval300_sha256_for_config(config))
    assert build_eval300_plan(path) == plan
    assert plan.dataset_sha256 == sha256_file(path) == expected
    assert len(plan.entries) == 300
    assert Counter(benchmark for benchmark, _ in plan.entries) == {
        "simplevqa": 100, "mmsearch": 100, "vdr_bench": 100}
    assert len({sample_id for _, sample_id in plan.entries}) == 300
    assert len(plan.first_batch) == 200 and len(plan.second_batch) == 100
    assert set(plan.first_batch).isdisjoint(plan.second_batch)


@pytest.mark.parametrize("path,config", [
    (V1, "eval_base_300_v2.yaml"),
    (V2, "eval_base_300.yaml"),
])
def test_wrong_eval300_version_sha_fails_closed(path, config):
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_eval300_plan(path, expected_sha256=expected_eval300_sha256_for_config(config))


def test_audited_manifest_preserves_selection_provenance_and_records_rebuild():
    old = json.loads(MANIFEST_V1.read_text(encoding="utf-8"))
    new = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    assert {key: value for key, value in old.items() if key != "combined"} == {
        key: value for key, value in new.items() if key not in ("combined", "audited_rebuild")}
    assert {key: value for key, value in old["combined"].items()
            if key not in ("output_file", "output_sha256")} == {
                key: value for key, value in new["combined"].items()
                if key not in ("output_file", "output_sha256")}
    assert new["combined"]["output_file"] == V2.name
    assert new["combined"]["output_sha256"] == FROZEN_EVAL300_V2_SHA256
    rebuild = new["audited_rebuild"]
    assert rebuild["parent_eval300_v1_sha256"] == FROZEN_EVAL300_SHA256
    assert rebuild["parent_manifest_sha256"] == sha256_file(MANIFEST_V1)
    assert rebuild["sample_count"] == 300
    assert rebuild["benchmark_counts"] == {
        "simplevqa": 100, "mmsearch": 100, "vdr_bench": 100}
    assert rebuild["question_changed_count"] == 49
    assert rebuild["reference_changed_count"] == 66


@pytest.mark.parametrize("dataset,manifest,expected", [
    (V1, MANIFEST_V1, FROZEN_EVAL300_SHA256),
    (V2, MANIFEST_V2, FROZEN_EVAL300_V2_SHA256),
])
def test_dataset_and_own_frozen_manifest_pass(dataset, manifest, expected):
    identity = frozen_dataset_identity(dataset, manifest)
    assert identity["sha256"] == expected
    assert identity["frozen_manifest_identity"]["combined_output_sha256"] == expected
    assert identity["frozen_manifest_identity"]["combined_output_file"] == dataset.name


@pytest.mark.parametrize("dataset,manifest", [
    (V1, MANIFEST_V2),
    (V2, MANIFEST_V1),
])
def test_crossed_frozen_manifest_fails_closed(dataset, manifest):
    with pytest.raises(ValueError, match="checksum does not match"):
        frozen_dataset_identity(dataset, manifest)


def test_v2_run_manifest_records_actual_audit_provenance():
    config_path = ROOT / "configs/eval_base_300_v2.yaml"
    config = load_inference_config(config_path)
    plan = build_eval300_plan(config.data_path)
    manifest = build_run_manifest(
        run_id="base-eval300-audited-v2", model_name_or_path=config.model_name_or_path,
        model_revision=config.revision, inference_config_path=config_path,
        dataset_path=config.data_path, eval_manifest_path=config.eval_manifest_path,
        start=None, limit=None, sample_selection=plan.selection_identity(),
        max_agent_turns=config.max_agent_turns,
        search_config_path=ROOT / "configs/search_backends.example.yaml",
        layout_config_path=ROOT / "configs/layout_parsing.example.yaml",
    )
    identity = manifest["dataset_identity"]
    assert identity["sha256"] == FROZEN_EVAL300_V2_SHA256
    frozen = identity["frozen_manifest_identity"]
    assert frozen["combined_output_file"] == V2.name
    assert frozen["manifest_file_sha256"] == sha256_file(MANIFEST_V2)
    assert frozen["audited_rebuild"]["parent_eval300_v1_sha256"] == FROZEN_EVAL300_SHA256
    assert frozen["audited_rebuild"]["question_changed_count"] == 49


@pytest.mark.parametrize("config_name,dataset", [
    ("eval_base_300.yaml", V2),
    ("eval_base_300_v2.yaml", V1),
])
def test_agent_cli_rejects_mismatched_config_dataset_before_model(
        tmp_path, monkeypatch, config_name, dataset):
    spec = importlib.util.spec_from_file_location(
        "run_agent_batch_eval300_v2", ROOT / "scripts/run_agent_batch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    config["data"]["path"] = str(dataset)
    config_path = tmp_path / config_name
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def no_model(*args, **kwargs):
        raise AssertionError("model must not load when Eval-300 SHA does not match")

    monkeypatch.setattr(module, "load_inference_bundle", no_model)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.main(["--run-id", "sha-mismatch-test", "--config", str(config_path),
                     "--eval300"])


def test_agent_cli_rejects_v2_wrong_manifest_before_model(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "run_agent_batch_wrong_manifest", ROOT / "scripts/run_agent_batch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((ROOT / "configs/eval_base_300_v2.yaml").read_text(
        encoding="utf-8"))
    config["data"] = {"path": str(V2), "manifest": str(MANIFEST_V1)}
    config_path = tmp_path / "eval_base_300_v2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def no_model(*args, **kwargs):
        raise AssertionError("model must not load when frozen manifest is wrong")

    monkeypatch.setattr(module, "load_inference_bundle", no_model)
    with pytest.raises(ValueError, match="checksum does not match"):
        module.main(["--run-id", "wrong-manifest", "--config", str(config_path),
                     "--eval300"])


def test_agent_cli_v2_passes_run_manifest_gate_before_model(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "run_agent_batch_valid_manifest", ROOT / "scripts/run_agent_batch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def stop_before_model(*args, **kwargs):
        raise RuntimeError("validated v2 provenance; model intentionally not loaded")

    monkeypatch.setattr(module, "load_inference_bundle", stop_before_model)
    with pytest.raises(RuntimeError, match="validated v2 provenance"):
        module.main(["--run-id", "valid-v2-provenance-test", "--config",
                     str(ROOT / "configs/eval_base_300_v2.yaml"), "--eval300"])


def test_v2_preserves_every_other_column_and_matches_audited_text():
    old = pq.read_table(V1)
    new = pq.read_table(V2)
    assert old.schema == new.schema and old.num_rows == new.num_rows == 300
    for name in old.column_names:
        if name not in ("question", "answer"):
            assert old[name].equals(new[name]), name
    old_rows, new_rows = old.to_pylist(), new.to_pylist()
    audited = {row["sample_id"]: row for row in json.loads(AUDITED.read_text(encoding="utf-8"))}
    assert len(audited) == 300
    assert {row["id"] for row in new_rows} == set(audited)
    assert all(row["benchmark"] == audited[row["id"]]["benchmark"] and
               row["question"] == audited[row["id"]]["question"] and
               row["answer"] == audited[row["id"]]["reference_answer"]
               for row in new_rows)
    assert sum(a["question"] != b["question"] for a, b in zip(old_rows, new_rows)) == 49
    assert sum(a["answer"] != b["answer"] for a, b in zip(old_rows, new_rows)) == 66


def test_judge_can_join_v2_question_and_reference_without_provider(tmp_path):
    old = {row["id"]: row for row in pq.read_table(V1, columns=["id", "question"]).to_pylist()}
    changed = next(row for row in pq.read_table(
        V2, columns=["id", "benchmark", "question", "answer"]).to_pylist()
        if row["question"] != old[row["id"]]["question"])
    trajectory = tmp_path / "trajectories.jsonl"
    trajectory.write_text(json.dumps({"sample_id": changed["id"],
                                      "benchmark": changed["benchmark"],
                                      "question": changed["question"],
                                      "status": "success", "final_answer": "candidate"}) + "\n",
                          encoding="utf-8")
    samples = load_judge_samples(trajectory, V2)
    assert len(samples) == 1
    assert samples[0].reference_answer == changed["answer"]
    with pytest.raises(ValueError, match="question differs"):
        load_judge_samples(trajectory, V1)


def test_v2_static_preflight_passes_without_model_or_api():
    spec = importlib.util.spec_from_file_location(
        "preflight_eval300_v2", ROOT / "scripts/preflight_eval300.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_preflight_report(
        config_path=ROOT / "configs/eval_base_300_v2.yaml",
        search_config_path=ROOT / "configs/search_backends.example.yaml",
        layout_config_path=ROOT / "configs/layout_parsing.example.yaml",
        judge_config_path=ROOT / "configs/judge.example.yaml",
    )
    assert report["static_validation_passed"] is True
    assert report["dataset_sha256"] == FROZEN_EVAL300_V2_SHA256
    assert report["eval_manifest_path"] == str(MANIFEST_V2)
    assert report["static_checks"]["dataset_manifest_sha256_matches"] is True
    assert report["run_id"] == "base-eval300-audited-v2"
    assert report["provider_calls"] == 0 and report["model_loaded"] is False


@pytest.mark.parametrize("config_name,dataset,manifest", [
    ("eval_base_300.yaml", V1, MANIFEST_V2),
    ("eval_base_300_v2.yaml", V2, MANIFEST_V1),
])
def test_preflight_rejects_crossed_manifest(tmp_path, config_name, dataset, manifest):
    spec = importlib.util.spec_from_file_location(
        "preflight_eval300_wrong_manifest", ROOT / "scripts/preflight_eval300.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    config["data"] = {"path": str(dataset), "manifest": str(manifest)}
    config_path = tmp_path / config_name
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum does not match"):
        module.build_preflight_report(
            config_path=config_path,
            search_config_path=ROOT / "configs/search_backends.example.yaml",
            layout_config_path=ROOT / "configs/layout_parsing.example.yaml",
            judge_config_path=ROOT / "configs/judge.example.yaml",
        )


def _fixture_rows():
    return [{"id": f"{benchmark}:{index}", "benchmark": benchmark,
             "question": f"old {index}", "answer": f"answer {index}",
             "image_packed": f"packed-{benchmark}-{index}", "other": index}
            for benchmark in ("simplevqa", "mmsearch", "vdr_bench")
            for index in range(100)]


def test_builder_ignores_audit_image_paths_and_preserves_original_columns(tmp_path):
    old_rows = _fixture_rows()
    v1, audited_path, output = (tmp_path / name for name in
                                ("v1.parquet", "audited.json", "v2.parquet"))
    pq.write_table(pa.Table.from_pylist(old_rows), v1)
    audited = [{"sample_id": row["id"], "benchmark": row["benchmark"],
                "question": row["question"] + " corrected",
                "reference_answer": row["answer"] + " corrected",
                "images": [{"path": "../path/that/does/not/exist.jpg"}]}
               for row in reversed(old_rows)]
    audited_path.write_text(json.dumps(audited), encoding="utf-8")
    report = _builder()(v1, audited_path, output, expected_v1_sha256=sha256_file(v1))
    assert report["question_changed_count"] == report["reference_changed_count"] == 300
    assert report["image_packed_unchanged"] is True
    result = pq.read_table(output)
    original = pq.read_table(v1)
    assert result["image_packed"].equals(original["image_packed"])
    assert result["other"].equals(original["other"])
    assert result.column("id").equals(original.column("id"))
    assert result["question"][0].as_py() == "old 0 corrected"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _builder()(v1, audited_path, output, expected_v1_sha256=sha256_file(v1))


@pytest.mark.parametrize("problem", ["duplicate_id", "wrong_benchmark", "missing_id"])
def test_builder_rejects_bad_audited_population_before_output(tmp_path, problem):
    rows = _fixture_rows()
    v1, audited_path, output = (tmp_path / name for name in
                                ("v1.parquet", "audited.json", "v2.parquet"))
    pq.write_table(pa.Table.from_pylist(rows), v1)
    audited = [{"sample_id": row["id"], "benchmark": row["benchmark"],
                "question": row["question"], "reference_answer": row["answer"]}
               for row in rows]
    if problem == "duplicate_id":
        audited[1]["sample_id"] = audited[0]["sample_id"]
    elif problem == "wrong_benchmark":
        audited[0]["benchmark"], audited[100]["benchmark"] = (
            audited[100]["benchmark"], audited[0]["benchmark"])
    else:
        audited[0]["sample_id"] = "unknown:0"
    audited_path.write_text(json.dumps(audited), encoding="utf-8")
    with pytest.raises(ValueError):
        _builder()(v1, audited_path, output, expected_v1_sha256=sha256_file(v1))
    assert not output.exists()
