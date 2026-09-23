from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
import requests

from opensearch_vl_repro.agent.runtime import AgentTrajectory, AgentTurn
from opensearch_vl_repro.evaluation import (
    BatchRunner, BatchSample, FIRST_BATCH_COUNTS, FROZEN_EVAL300_SHA256,
    SECOND_BATCH_COUNTS, build_eval300_plan, create_run_manifest,
)
from opensearch_vl_repro.evaluation.judge_runner import JudgeRunner


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/eval/combined_eval_300.parquet"


def _counts(entries):
    return Counter(benchmark for benchmark, _ in entries)


def test_frozen_eval300_balanced_partition_is_complete_and_deterministic():
    first = build_eval300_plan(DATASET)
    second = build_eval300_plan(DATASET)

    assert first == second
    assert first.dataset_sha256 == FROZEN_EVAL300_SHA256
    assert len(first.entries) == 300
    assert _counts(first.entries) == {"simplevqa": 100, "mmsearch": 100, "vdr_bench": 100}
    assert len(first.first_batch) == 200 and _counts(first.first_batch) == FIRST_BATCH_COUNTS
    assert len(first.second_batch) == 100 and _counts(first.second_batch) == SECOND_BATCH_COUNTS
    assert set(first.first_batch).isdisjoint(first.second_batch)
    assert set(first.first_batch) | set(first.second_batch) == set(first.entries)
    assert len({sample_id for _, sample_id in first.entries}) == 300


def _run_manifest(plan):
    return create_run_manifest(
        run_id="base-eval300-v1",
        model_name_or_path="Qwen/Qwen3-VL-4B-Instruct",
        model_revision="model-revision",
        inference_config_fingerprint="inference-fingerprint",
        dataset_path=plan.dataset_path,
        dataset_identity={"sha256": plan.dataset_sha256},
        start=None, limit=None, sample_selection=plan.selection_identity(),
        max_agent_turns=8, search_config_fingerprint="search-fingerprint",
        layout_config_fingerprint="layout-fingerprint",
        checkpoint={"kind": "remote", "identifier": "Qwen/Qwen3-VL-4B-Instruct"},
        tool_fingerprint="tool-fingerprint", created_at="2026-01-01T00:00:00+00:00",
    )


class Eval300Runtime:
    def __init__(self, fail_once_id):
        self.fail_once_id = fail_once_id
        self.calls = Counter()

    def run(self, *, question, images, sample_id, benchmark):
        self.calls[sample_id] += 1
        failed = sample_id == self.fail_once_id and self.calls[sample_id] == 1
        tool = {"simplevqa": "web_search", "mmsearch": "crop",
                "vdr_bench": "layout_parsing"}[benchmark]
        metadata = ({"cache_hit": False, "attempt_count": 1}
                    if tool in {"web_search", "layout_parsing"} else {})
        turn = AgentTurn(
            assistant_output=f'{tool}({{"q":"evidence"}})',
            tool_call={"name": tool, "arguments": {"q": "evidence"}},
            observation="<observation>evidence</observation>",
            status="success", metadata=metadata,
        )
        return AgentTrajectory(
            sample_id=sample_id, benchmark=benchmark, turns=[turn],
            final_answer=None if failed else "answer",
            status="max_agent_turns_exceeded" if failed else "success",
            image_ids=[], error="failed once" if failed else None,
        )


def test_full_manifest_partial_invocation_resume_and_retry_failed(
    tmp_path, monkeypatch, capsys,
):
    # Keep this 300-sample semantics test fast; durability itself is covered by
    # the existing BatchRunner tests.
    monkeypatch.setattr("opensearch_vl_repro.evaluation.batch_runner.os.fsync", lambda _: None)
    plan = build_eval300_plan(DATASET)
    samples = [BatchSample(sample_id, benchmark, f"question {sample_id}", [])
               for benchmark, sample_id in plan.entries]
    failed_id = plan.first_batch[0][1]
    runtime = Eval300Runtime(failed_id)
    run_dir = tmp_path / "base-eval300-v1"
    manifest = _run_manifest(plan)
    runner = BatchRunner(runtime, run_dir, run_manifest=manifest)

    first = runner.run(samples, max_samples=200)
    first_output = capsys.readouterr().out
    assert "Run ID: base-eval300-v1" in first_output
    assert "Total samples: 300" in first_output
    assert "Eligible this invocation: 200" in first_output
    assert "Already successful: 0" in first_output
    assert "Already failed: 0" in first_output
    assert "Pending: 300" in first_output
    assert "Retry failed: false" in first_output
    assert first["total"] == 300 and first["processed"] == 200
    assert first["pending"] == 100 and first["success"] == 199 and first["failed"] == 1
    assert first["per_benchmark"]["simplevqa"]["processed"] == 67
    assert first["per_benchmark"]["mmsearch"]["processed"] == 67
    assert first["per_benchmark"]["vdr_bench"]["processed"] == 66
    assert first["total_tool_calls"] == 200
    assert first["tool_calls"]["web_search"] == 67
    assert first["tool_calls"]["crop"] == 67
    assert first["tool_calls"]["layout_parsing"] == 66
    assert first["tool_stats"]["web_search"]["real_tool_executions"] == 67
    assert first["per_benchmark"]["vdr_bench"]["tool_stats"]["layout_parsing"][
        "real_tool_executions"
    ] == 66
    persisted = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert persisted["sample_selection"]["sample_count"] == 300
    assert "max_samples" not in json.dumps(persisted)

    calls_after_first_batch = runtime.calls.copy()
    repeated_first = BatchRunner(runtime, run_dir, run_manifest=_run_manifest(plan)).run(
        samples, max_samples=200,
    )
    assert repeated_first["pending"] == 100
    assert runtime.calls == calls_after_first_batch

    resumed = BatchRunner(runtime, run_dir, run_manifest=_run_manifest(plan)).run(samples)
    assert resumed["total"] == 300 and resumed["pending"] == 0
    assert resumed["success"] == 299 and resumed["failed"] == 1
    assert len(runtime.calls) == 300 and set(runtime.calls.values()) == {1}

    retried = BatchRunner(runtime, run_dir, run_manifest=_run_manifest(plan)).run(
        samples, retry_failed=True,
    )
    assert retried["success"] == 300 and retried["failed"] == 0
    assert runtime.calls[failed_id] == 2
    assert all(count == 1 for sample_id, count in runtime.calls.items()
               if sample_id != failed_id)


def test_agent_and_judge_summaries_include_per_benchmark_and_e2e_metrics():
    state = {"samples": {
        "A": {"benchmark": "simplevqa", "status": "success"},
        "B": {"benchmark": "simplevqa", "status": "failed"},
        "C": {"benchmark": "mmsearch", "status": "pending"},
    }}
    records = {
        "A": {"benchmark": "simplevqa", "trajectory": {"turns": [{
            "tool_call": {"name": "web_search", "arguments": {}},
            "error": None, "metadata": {"cache_hit": False},
        }]}},
        "B": {"benchmark": "simplevqa", "trajectory": {"turns": [{
            "tool_call": {"name": "image_search", "arguments": {}},
            "error": "timeout", "metadata": {"cache_hit": False},
        }]}},
    }
    summary = BatchRunner._summary(state, records)
    assert summary["processed"] == 2 and summary["completion_rate"] == pytest.approx(2 / 3)
    assert summary["provider_errors"] == 1 and summary["cache_misses"] == 2
    assert summary["tool_stats"]["image_search"]["errors"] == 1
    assert summary["per_benchmark"]["simplevqa"]["total_tool_calls"] == 2
    assert summary["per_benchmark"]["mmsearch"]["pending"] == 1

    judge_state = {"samples": {
        "A": {"benchmark": "simplevqa", "status": "success"},
        "B": {"benchmark": "simplevqa", "status": "success"},
        "C": {"benchmark": "mmsearch", "status": "failed"},
    }}
    judge_records = {
        "A": {"verdict": "correct", "error_type": None},
        "B": {"verdict": "incorrect", "error_type": None},
        "C": {"verdict": None, "error_type": "upstream_agent_failure"},
    }
    judged = JudgeRunner._summary(judge_state, judge_records)
    assert judged["accuracy_among_successful_judges"] == pytest.approx(0.5)
    assert judged["end_to_end_accuracy"] == pytest.approx(1 / 3)
    assert judged["per_benchmark"]["simplevqa"]["end_to_end_accuracy"] == 0.5
    assert judged["per_benchmark"]["mmsearch"]["upstream_failed"] == 1


def _load_preflight_module():
    path = ROOT / "scripts/preflight_eval300.py"
    spec = importlib.util.spec_from_file_location("phase6_preflight_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_preflight_is_offline_and_does_not_construct_provider_calls(monkeypatch):
    def forbidden_request(*args, **kwargs):
        raise AssertionError("preflight must not make HTTP requests")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden_request)
    for name in ("SERPER_API_KEY", "JINA_API_KEY", "SERPAPI_API_KEY",
                 "PADDLEOCR_ACCESS_TOKEN", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(name, "present-for-offline-test")
    module = _load_preflight_module()
    report = module.build_preflight_report(
        config_path=ROOT / "configs/eval_base_300.yaml",
        search_config_path=ROOT / "configs/search_backends.example.yaml",
        layout_config_path=ROOT / "configs/layout_parsing.example.yaml",
        judge_config_path=ROOT / "configs/judge.example.yaml",
    )
    assert report["offline"] is True and report["provider_calls"] == 0
    assert report["model_loaded"] is False and report["static_validation_passed"] is True
    assert report["agent_api_env_ready"] is True and report["judge_api_env_ready"] is True
