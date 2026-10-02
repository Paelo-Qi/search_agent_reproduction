import json
import shutil
import sys

import pytest

from opensearch_vl_repro.rl.data import preflight_dataset, prepare_formal_dataset, write_dataset_artifacts
from opensearch_vl_repro.rl.quality_audit import QUALITY_EXCLUSION_REASONS, load_quality_audit
from rl_quality_helpers import make_quality_bundle
from test_rl_formal_data import plan


def small_bundle(tmp_path):
    identities = [f"rl_{index:06d}" for index in range(1, 6)]
    statuses = ["manual_review", "ok", "exclude", "ok", "ok"]
    path = make_quality_bundle(tmp_path / "quality", identities, statuses, main_count=2)
    return path, identities, statuses


def audit_rows(path):
    return [json.loads(line) for line in (path / "rl_quality_audit_v1.jsonl").read_text(
        encoding="utf-8").splitlines()]


def write_rows(path, rows):
    (path / "rl_quality_audit_v1.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_valid_bundle_and_manual_review_semantics(tmp_path):
    assert QUALITY_EXCLUSION_REASONS == {
        "exclude_question_ambiguous", "exclude_question_invalid",
        "exclude_reference_incorrect", "exclude_reference_incomplete",
        "exclude_image_mismatch", "exclude_unanswerable",
        "exclude_outdated_reference", "exclude_multiple_valid_answers", "exclude_other"}
    path, identities, _ = small_bundle(tmp_path)
    result = load_quality_audit(path, main_count=2)
    assert [item["source_sample_id"] for item in result.selected] == [identities[1], identities[3]]
    assert [item["source_sample_id"] for item in result.reserve] == [identities[4]]
    assert result.rows[0]["status"] == "manual_review"
    assert result.rows[0]["reason"] is None
    assert result.provenance["quality_excluded_count"] == 1


@pytest.mark.parametrize("change", [
    lambda rows: rows[1].update(candidate_rank=1),
    lambda rows: rows[1].update(candidate_rank=4),
    lambda rows: rows[1].update(source_sample_id=rows[0]["source_sample_id"]),
    lambda rows: rows[1].update(status="unknown"),
    lambda rows: rows[2].update(reason="not_in_frozen_enum"),
    lambda rows: rows[1].update(reason="exclude_other"),
    lambda rows: rows[0].update(reason="exclude_other"),
])
def test_bad_audit_row_fails_closed(tmp_path, change):
    path, _, _ = small_bundle(tmp_path)
    rows = audit_rows(path)
    change(rows)
    write_rows(path, rows)
    with pytest.raises(ValueError):
        load_quality_audit(path, main_count=2)


def test_allowlist_and_summary_are_recomputed_not_trusted(tmp_path):
    path, _, _ = small_bundle(tmp_path)
    file = path / "rl_quality_ok_allowlist_v1.json"
    value = json.loads(file.read_text(encoding="utf-8"))
    value["selected_main400"][0], value["selected_main400"][1] = (
        value["selected_main400"][1], value["selected_main400"][0])
    file.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="selected_main400"):
        load_quality_audit(path, main_count=2)

    path2, _, _ = small_bundle(tmp_path / "second")
    summary_file = path2 / "rl_quality_audit_summary_v1.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["manual_review_count"] += 1
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        load_quality_audit(path2, main_count=2)

    path3, _, _ = small_bundle(tmp_path / "third")
    file3 = path3 / "rl_quality_ok_allowlist_v1.json"
    value3 = json.loads(file3.read_text(encoding="utf-8"))
    value3["selected_main400"][1] = value3["reserve_ok"][0]
    value3["reserve_ok"] = [value3["selected_main400"][1]]
    file3.write_text(json.dumps(value3), encoding="utf-8")
    with pytest.raises(ValueError, match="first-N ok"):
        load_quality_audit(path3, main_count=2)


def test_all_five_files_are_required_and_cross_checked(tmp_path):
    path, _, _ = small_bundle(tmp_path)
    exclusion = path / "rl_quality_exclusions_v1.json"
    value = json.loads(exclusion.read_text(encoding="utf-8"))
    value["excluded"][0]["reason"] = "exclude_other"
    exclusion.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exclusions/manual"):
        load_quality_audit(path, main_count=2)
    (path / "rl_quality_manual_review_v1.json").unlink()
    with pytest.raises(FileNotFoundError, match="manual_review"):
        load_quality_audit(path, main_count=2)


def test_candidate_rank_source_id_must_match_recomputed_eval_clean_stream(tmp_path):
    kwargs, _, _, ranked, _, _ = plan(tmp_path)
    actual_ids = [f"rl_{index:06d}" for index in ranked[2:]]
    swapped = actual_ids.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    bad_dir = make_quality_bundle(tmp_path / "wrong_stream", swapped,
        ["manual_review", "ok", "exclude", "ok", "ok", "ok", "ok"], main_count=4)
    assert load_quality_audit(bad_dir, main_count=4).provenance["audited_count"] == 7
    with pytest.raises(ValueError, match="candidate stream mismatch"):
        prepare_formal_dataset(**(kwargs | {"quality_audit_dir": bad_dir}))


def test_quality_bytes_change_manifest_hash_but_path_does_not(tmp_path):
    kwargs, _, _, _, _, _ = plan(tmp_path)
    first = prepare_formal_dataset(**kwargs)
    copied = tmp_path / "copied-quality"
    shutil.copytree(kwargs["quality_audit_dir"], copied)
    same = prepare_formal_dataset(**(kwargs | {"quality_audit_dir": copied}))
    assert same["main_manifest"]["manifest_sha256"] == first["main_manifest"]["manifest_sha256"]
    changed = make_quality_bundle(tmp_path / "changed-quality",
        [row["source_sample_id"] for row in load_quality_audit(copied, main_count=4).rows],
        [row["status"] for row in load_quality_audit(copied, main_count=4).rows],
        main_count=4, note="reviewed again")
    second = prepare_formal_dataset(**(kwargs | {"quality_audit_dir": changed}))
    assert second["main"] == first["main"]
    assert second["main_manifest"]["manifest_sha256"] != first["main_manifest"]["manifest_sha256"]
    output = tmp_path / "published"
    write_dataset_artifacts(first, output, **{key: kwargs[key] for key in (
        "source_parquet", "source_root", "eval_overlap_manifest", "sft_overlap_manifest",
        "quality_audit_dir")})
    with pytest.raises(ValueError, match="quality audit fingerprint"):
        preflight_dataset(output, source_parquet=kwargs["source_parquet"],
            source_root=kwargs["source_root"], eval_overlap_manifest=kwargs["eval_overlap_manifest"],
            sft_overlap_manifest=kwargs["sft_overlap_manifest"], quality_audit_dir=changed)


def test_staged_publication_cleans_up_on_validation_failure(tmp_path, monkeypatch):
    import opensearch_vl_repro.rl.data as data

    kwargs, _, _, _, _, _ = plan(tmp_path)
    artifacts = prepare_formal_dataset(**kwargs)
    output = tmp_path / "published"
    monkeypatch.setattr(data, "preflight_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("staged validation failed")))
    with pytest.raises(ValueError, match="staged validation failed"):
        write_dataset_artifacts(artifacts, output, **{key: kwargs[key] for key in (
            "source_parquet", "source_root", "eval_overlap_manifest", "sft_overlap_manifest",
            "quality_audit_dir")})
    assert not output.exists()
    assert not list(tmp_path.glob(".published.staging-*"))


def test_formal_cli_requires_quality_directory(tmp_path, monkeypatch, capsys):
    from scripts import prepare_rl_data as script

    monkeypatch.setattr(sys, "argv", ["prepare_rl_data.py", "--source-parquet", "missing.parquet",
        "--source-root", str(tmp_path), "--output-dir", str(tmp_path / "output"),
        "--eval-overlap-manifest", "eval.json", "--sft-overlap-manifest", "sft.json"])
    assert script.main() == 1
    assert "quality audit directory" in capsys.readouterr().out
