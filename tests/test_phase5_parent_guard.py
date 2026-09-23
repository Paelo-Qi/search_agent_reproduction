from __future__ import annotations

import json
from pathlib import Path

import pytest

from opensearch_vl_repro.evaluation.judge import load_judge_config
from opensearch_vl_repro.evaluation.parent_run import (
    ParentRunValidationError, validate_parent_run_ready_for_judge,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_status(run_dir: Path, statuses: dict[str, str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    value = {"version": 1, "samples": {
        sample_id: {"benchmark": "synthetic", "status": status, "attempts": 1,
                    "error_type": None, "error": None}
        for sample_id, status in statuses.items()
    }}
    (run_dir / "status.json").write_text(json.dumps(value), encoding="utf-8")


def _write_trajectories(run_dir: Path, statuses: dict[str, str]) -> None:
    lines = [json.dumps({"sample_id": sample_id, "benchmark": "synthetic",
                         "question": f"q-{sample_id}", "status": status,
                         "final_answer": "answer" if status == "success" else None})
             for sample_id, status in statuses.items()]
    (run_dir / "trajectories.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_default_judge_config_uses_deepseek_flash():
    config = load_judge_config(ROOT / "configs/judge.example.yaml")
    assert config.provider == "deepseek"
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-flash"


def test_parent_pending_is_rejected_before_judge_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    statuses = {**{f"success-{index}": "success" for index in range(10)},
                **{f"pending-{index}": "pending" for index in range(20)}}
    _write_status(run_dir, statuses)
    judge_dir = run_dir / "judge"
    with pytest.raises(ParentRunValidationError, match="pending: 20"):
        validate_parent_run_ready_for_judge(run_dir)
    assert not judge_dir.exists()


def test_parent_running_is_rejected(tmp_path):
    run_dir = tmp_path / "run"
    _write_status(run_dir, {"A": "success", "B": "running"})
    with pytest.raises(ParentRunValidationError, match="running: 1"):
        validate_parent_run_ready_for_judge(run_dir)


def test_guard_failure_does_not_modify_existing_judge_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    _write_status(run_dir, {"A": "pending"})
    judge_dir = run_dir / "judge"
    judge_dir.mkdir()
    for name in ("judge_manifest.json", "judge_status.json",
                 "judge_results.jsonl", "judge_summary.json"):
        (judge_dir / name).write_bytes(f"unchanged-{name}".encode())
    before = {path.name: path.read_bytes() for path in judge_dir.iterdir()}
    with pytest.raises(ParentRunValidationError):
        validate_parent_run_ready_for_judge(run_dir)
    assert {path.name: path.read_bytes() for path in judge_dir.iterdir()} == before


def test_parent_complete_with_agent_failures_is_allowed(tmp_path):
    run_dir = tmp_path / "run"
    statuses = {**{f"success-{index}": "success" for index in range(28)},
                "failed-0": "failed", "failed-1": "failed"}
    _write_status(run_dir, statuses)
    _write_trajectories(run_dir, statuses)
    assert validate_parent_run_ready_for_judge(run_dir) == {
        "total": 30, "pending": 0, "running": 0, "success": 28, "failed": 2,
    }


@pytest.mark.parametrize("kind", ["missing", "invalid_json", "missing_samples", "bad_status"])
def test_missing_or_invalid_parent_status_is_rejected(tmp_path, kind):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status_path = run_dir / "status.json"
    if kind == "invalid_json":
        status_path.write_text("{broken", encoding="utf-8")
    elif kind == "missing_samples":
        status_path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    elif kind == "bad_status":
        status_path.write_text(json.dumps({"samples": {"A": {"status": "unknown"}}}),
                               encoding="utf-8")
    with pytest.raises(ParentRunValidationError, match="refusing to start Judge"):
        validate_parent_run_ready_for_judge(run_dir)
    assert not (run_dir / "judge").exists()


@pytest.mark.parametrize("trajectory_statuses", [
    {"A": "success", "B": "failed"},
    {"A": "success", "B": "failed", "C": "success", "EXTRA": "success"},
])
def test_parent_status_and_trajectory_ids_must_match(tmp_path, trajectory_statuses):
    run_dir = tmp_path / "run"
    statuses = {"A": "success", "B": "failed", "C": "success"}
    _write_status(run_dir, statuses)
    _write_trajectories(run_dir, trajectory_statuses)
    with pytest.raises(ParentRunValidationError, match="sample IDs are inconsistent"):
        validate_parent_run_ready_for_judge(run_dir)
    assert not (run_dir / "judge").exists()


def test_parent_status_and_trajectory_outcomes_must_match(tmp_path):
    run_dir = tmp_path / "run"
    _write_status(run_dir, {"A": "success"})
    _write_trajectories(run_dir, {"A": "failed"})
    with pytest.raises(ParentRunValidationError, match="outcomes are inconsistent"):
        validate_parent_run_ready_for_judge(run_dir)
