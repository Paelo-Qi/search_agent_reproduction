from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry
from opensearch_vl_repro.agent.reliability import RetryPolicy
from opensearch_vl_repro.agent.runtime import AgentRuntime, AgentTrajectory, AgentTurn
from opensearch_vl_repro.agent.search_providers import (
    SearchBackendError, SearchResult, SerperSearchBackend, load_search_config,
)
from opensearch_vl_repro.agent.search_tools import SearchTools
from opensearch_vl_repro.evaluation.batch_runner import BatchRunner, BatchSample
from opensearch_vl_repro.evaluation.judge import JudgeConfig, JudgeResult, JudgeSample
from opensearch_vl_repro.evaluation.judge_runner import JudgeRunner, build_judge_manifest
from opensearch_vl_repro.evaluation.systemic_errors import find_systemic_tool_error


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_agent_batch_test", ROOT / "scripts/run_agent_batch.py")
assert SPEC is not None and SPEC.loader is not None
agent_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent_cli)


def _samples() -> list[BatchSample]:
    return [BatchSample(name, "synthetic", f"question {name}",
                        [Image.new("RGB", (3, 3))]) for name in "ABCD"]


def _manifest() -> dict:
    return {"manifest_version": 1, "run_id": "systemic-test",
            "run_config_fingerprint": "fixed-protocol"}


def _records(path: Path) -> dict[str, dict]:
    return {item["sample_id"]: item for item in
            map(json.loads, path.read_text(encoding="utf-8").splitlines())}


class SequencedRuntime:
    def __init__(self, failures: dict[str, str]):
        self.failures = failures
        self.calls: Counter[str] = Counter()

    def run(self, *, question, images, sample_id, benchmark):
        self.calls[sample_id] += 1
        error = self.failures.get(sample_id) if self.calls[sample_id] == 1 else None
        turn = AgentTurn(
            assistant_output='web_search({"q":"evidence"})',
            tool_call={"name": "web_search", "arguments": {"q": "evidence"}},
            observation="<observation>result</observation>",
            status="error" if error else "success", error=error,
            metadata={"provider": "serper", "error_type": error,
                      "attempt_count": 3 if error else 1, "cache_hit": False},
        )
        # A systemic turn may be followed by a final answer; batch must still stop.
        status = "tool_error" if error and error not in {
            "quota_error", "authentication_error", "configuration_error"} else "success"
        return AgentTrajectory(sample_id, benchmark, [turn],
                               "answer" if status == "success" else None,
                               status, ["img_1"])


def test_quota_stop_is_durable_and_same_run_id_resumes_without_retry_failed(tmp_path, capsys):
    runtime = SequencedRuntime({"C": "quota_error"})
    run_dir = tmp_path / "agent"
    manifest = _manifest()
    first = BatchRunner(runtime, run_dir, run_manifest=manifest).run(_samples())
    assert list(runtime.calls) == ["A", "B", "C"]
    assert (first["success"], first["failed"], first["pending"]) == (2, 0, 2)
    assert first["interruption"]["error_type"] == "quota_error"
    assert first["interruption"]["tool"] == "web_search"
    assert first["interruption"]["provider"] == "serper"
    assert agent_cli.batch_exit_code(first, max_samples=3) == 1
    assert "Agent batch stopped early due to systemic error: quota_error" in capsys.readouterr().out
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = _records(run_dir / "trajectories.jsonl")
    assert [status["samples"][name]["status"] for name in "ABCD"] == [
        "success", "success", "pending", "pending"]
    assert status["samples"]["C"]["attempts"] == 1
    assert status["last_interruption"]["sample_id"] == "C"
    assert summary["interruption"]["sample_id"] == "C"
    assert set(records) == {"A", "B", "C"}
    assert records["C"]["status"] == "failed"
    assert records["C"]["trajectory_status"] == "success"
    assert records["C"]["trajectory"]["turns"][0]["error"] == "quota_error"
    manifest_bytes = (run_dir / "run_manifest.json").read_bytes()

    resumed = BatchRunner(runtime, run_dir, run_manifest=manifest).run(_samples())
    assert (resumed["success"], resumed["failed"], resumed["pending"]) == (4, 0, 0)
    assert "interruption" not in resumed
    assert runtime.calls == Counter({"A": 1, "B": 1, "C": 2, "D": 1})
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_bytes
    records = _records(run_dir / "trajectories.jsonl")
    assert records["C"]["status"] == "success"
    assert records["C"]["attempt_history"][0]["systemic_error"]["error_type"] == "quota_error"
    assert records["C"]["attempt_history"][0]["trajectory"]["turns"][0]["error"] == "quota_error"
    assert agent_cli.batch_exit_code(resumed, max_samples=None) == 0


@pytest.mark.parametrize("error_type", ["authentication_error", "configuration_error"])
def test_other_systemic_errors_stop_without_running_later_samples(tmp_path, error_type):
    runtime = SequencedRuntime({"B": error_type})
    summary = BatchRunner(runtime, tmp_path / "run", run_manifest=_manifest()).run(_samples())
    assert list(runtime.calls) == ["A", "B"]
    assert summary["pending"] == 3 and summary["failed"] == 0
    assert summary["interruption"]["error_type"] == error_type


@pytest.mark.parametrize("error_type", [
    "timeout", "network_error", "provider_error", "invalid_response",
])
def test_ordinary_provider_errors_keep_per_sample_failure_semantics(tmp_path, error_type):
    runtime = SequencedRuntime({"B": error_type})
    summary = BatchRunner(runtime, tmp_path / "run", run_manifest=_manifest()).run(_samples())
    assert list(runtime.calls) == list("ABCD")
    assert (summary["success"], summary["failed"], summary["pending"]) == (3, 1, 0)
    assert "interruption" not in summary


def test_running_status_with_persisted_systemic_trajectory_recovers_as_pending(tmp_path):
    runtime = SequencedRuntime({"B": "quota_error"})
    run_dir = tmp_path / "run"
    runner = BatchRunner(runtime, run_dir, run_manifest=_manifest())
    runner.run(_samples())
    status_path = run_dir / "status.json"
    state = json.loads(status_path.read_text(encoding="utf-8"))
    state["samples"]["B"]["status"] = "running"  # Crash between trajectory and status replace.
    status_path.write_text(json.dumps(state), encoding="utf-8")
    summary = BatchRunner(runtime, run_dir, run_manifest=_manifest()).run(_samples())
    assert summary["success"] == 4 and runtime.calls["B"] == 2


def test_legacy_failed_quota_turn_is_migrated_without_touching_success(tmp_path):
    runtime = SequencedRuntime({"B": "quota_error"})
    run_dir = tmp_path / "run"
    BatchRunner(runtime, run_dir, run_manifest=_manifest()).run(_samples())
    state_path = run_dir / "status.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["samples"]["B"]["status"] = "failed"  # Old BatchRunner semantics.
    state_path.write_text(json.dumps(state), encoding="utf-8")
    trajectory_path = run_dir / "trajectories.jsonl"
    records = _records(trajectory_path)
    records["B"].pop("systemic_error")  # Legacy record only has the tool turn.
    trajectory_path.write_text("\n".join(json.dumps(item) for item in records.values()) + "\n",
                               encoding="utf-8")
    summary = BatchRunner(runtime, run_dir, run_manifest=_manifest()).run(_samples())
    assert summary["success"] == 4
    assert runtime.calls == Counter({"A": 1, "B": 2, "C": 1, "D": 1})
    assert _records(trajectory_path)["B"]["attempt_history"][0]["trajectory"]["turns"][0][
        "error"] == "quota_error"


class FakeResponse:
    def __init__(self, status: int):
        self.status_code = status

    def json(self):
        return {"organic": [{"title": "Result", "link": "https://example.test", "snippet": "text"}]}


class FakeSession:
    def __init__(self, statuses: list[int]):
        self.statuses = list(statuses)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse(self.statuses.pop(0))


class ToolThenAnswer:
    def generate(self, *, messages, tools):
        if messages[-1]["role"] == "tool":
            return "Final answer."
        return 'web_search({"q":"same evidence"})'


def _real_runtime(tmp_path, monkeypatch, statuses):
    monkeypatch.setenv("SERPER_API_KEY", "test-only-token")
    settings = load_search_config(ROOT / "configs/search_backends.example.yaml")
    session = FakeSession(statuses)
    backend = SerperSearchBackend(
        settings.serper, session=session,
        retry=RetryPolicy(max_attempts=2, backoff_seconds=(), sleeper=lambda _: None),
    )
    tools = SearchTools(settings, serper=backend)
    registry = create_phase3_tool_registry(search_tools=tools, cache_dir=tmp_path / "cache")
    return AgentRuntime(model=ToolThenAnswer(), tool_registry=registry,
                        max_agent_turns=2), session


def test_transient_429_success_and_cache_hit_do_not_stop_batch(tmp_path, monkeypatch):
    runtime, session = _real_runtime(tmp_path, monkeypatch, [429, 200])
    summary = BatchRunner(runtime, tmp_path / "run", run_manifest=_manifest()).run(_samples()[:2])
    assert session.calls == 2  # Provider's bounded retry, then cache hit.
    assert summary["success"] == 2 and summary["pending"] == 0
    assert summary["cache_hits"] == 1
    assert "interruption" not in summary
    records = _records(tmp_path / "run/trajectories.jsonl")
    assert records["A"]["trajectory"]["turns"][0]["metadata"]["attempt_count"] == 2
    assert records["B"]["trajectory"]["turns"][0]["metadata"]["cache_hit"] is True


def test_exhausted_429_stops_only_after_retry_and_resumes_through_cache(tmp_path, monkeypatch):
    runtime, session = _real_runtime(tmp_path, monkeypatch, [429, 429, 200])
    run_dir = tmp_path / "run"
    first = BatchRunner(runtime, run_dir, run_manifest=_manifest()).run(_samples()[:2])
    assert session.calls == 2 and first["pending"] == 2
    assert first["interruption"]["attempt_count"] == 2
    assert first["interruption"]["provider"] == "serper"
    record = _records(run_dir / "trajectories.jsonl")["A"]
    assert record["trajectory"]["turns"][0]["metadata"]["cache_hit"] is False
    assert record["trajectory"]["turns"][0]["error"] == "quota_error"
    resumed = BatchRunner(runtime, run_dir, run_manifest=_manifest()).run(_samples()[:2])
    assert resumed["success"] == 2 and resumed["pending"] == 0
    assert session.calls == 3  # A retries; B uses the successful cache entry.
    assert resumed["cache_hits"] == 1


def test_jina_systemic_error_is_not_snippet_fallback_and_is_detectable(tmp_path):
    settings = load_search_config(ROOT / "configs/search_backends.example.yaml")

    class Serper:
        last_attempt_count = 1
        def search(self, *args, **kwargs):
            return (SearchResult("title", "https://example.test", "snippet"),)

    class Reader:
        last_attempt_count = 3
        def read(self, url):
            error = SearchBackendError("quota_error", "reader exhausted", retryable=True)
            error.attempt_count = 3
            raise error

    result = SearchTools(settings, serper=Serper(), reader=Reader()).text_search(
        {"q": "evidence"}, None)
    assert result.status == "error" and result.error_type == "quota_error"
    assert result.metadata["provider"] == "jina_reader"
    trajectory = AgentTrajectory("X", "synthetic", [AgentTurn(
        "text_search", {"name": "text_search", "arguments": {"q": "evidence"}},
        result.observation, result.status, result.error_type, result.metadata,
    )], "answer", "success", ["img_1"])
    assert find_systemic_tool_error(trajectory)["provider"] == "jina_reader"


def test_systemic_detector_reads_tool_metadata_error_type_not_only_trajectory_status():
    trajectory = AgentTrajectory("X", "synthetic", [AgentTurn(
        "web_search", {"name": "web_search", "arguments": {}},
        "failed", "error", None,
        {"error_type": "authentication_error", "provider": "serper", "attempt_count": 1},
    )], "answer", "success", ["img_1"])
    error = find_systemic_tool_error(trajectory)
    assert error["error_type"] == "authentication_error"
    assert error["tool"] == "web_search" and error["provider"] == "serper"


def test_judge_quota_is_pending_then_retries_without_retry_failed(tmp_path):
    class Provider:
        def __init__(self):
            self.calls = Counter()
        def judge(self, sample):
            self.calls[sample.sample_id] += 1
            if sample.sample_id == "C" and self.calls["C"] == 1:
                return JudgeResult("error", error_type="quota_error", reason="quota",
                                   metadata={"provider": "deepseek", "attempt_count": 3})
            return JudgeResult("success", "correct", metadata={"provider": "deepseek"})

    samples = [JudgeSample(name, "synthetic", f"q{name}", "ref", "answer")
               for name in "ABCD"]
    manifest = build_judge_manifest(
        parent_manifest={"run_id": "agent-run", "run_config_fingerprint": "fixed"},
        config=JudgeConfig("deepseek", "https://example.test", "deepseek-flash"),
        samples=samples, created_at="fixed",
    )
    provider = Provider()
    run_dir = tmp_path / "judge"
    first = JudgeRunner(provider, run_dir, judge_manifest=manifest).run(samples)
    assert provider.calls == Counter({"A": 1, "B": 1, "C": 1})
    assert (first["success"], first["failed"], first["pending"]) == (2, 0, 2)
    assert first["interruption"]["error_type"] == "quota_error"
    status = json.loads((run_dir / "judge_status.json").read_text(encoding="utf-8"))
    assert status["samples"]["C"]["status"] == "pending"
    assert _records(run_dir / "judge_results.jsonl")["C"]["error_type"] == "quota_error"
    resumed = JudgeRunner(provider, run_dir, judge_manifest=manifest).run(samples)
    assert resumed["success"] == 4 and resumed["pending"] == 0
    assert provider.calls == Counter({"A": 1, "B": 1, "C": 2, "D": 1})
    assert resumed["correct"] == 4 and resumed["incorrect"] == 0
    record = _records(run_dir / "judge_results.jsonl")["C"]
    assert record["attempt_history"][0]["error_type"] == "quota_error"


def test_legacy_judge_failed_quota_result_is_auto_retried(tmp_path):
    class Provider:
        def __init__(self): self.calls = Counter()
        def judge(self, sample):
            self.calls[sample.sample_id] += 1
            if sample.sample_id == "A" and self.calls["A"] == 1:
                return JudgeResult("error", error_type="quota_error", reason="quota")
            return JudgeResult("success", "correct")

    samples = [JudgeSample(name, "synthetic", "q", "ref", "answer") for name in "AB"]
    manifest = build_judge_manifest(
        parent_manifest={"run_id": "agent-run", "run_config_fingerprint": "fixed"},
        config=JudgeConfig("deepseek", "https://example.test", "deepseek-flash"),
        samples=samples, created_at="fixed",
    )
    provider = Provider()
    run_dir = tmp_path / "judge"
    JudgeRunner(provider, run_dir, judge_manifest=manifest).run(samples)
    state_path = run_dir / "judge_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["samples"]["A"]["status"] = "failed"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    summary = JudgeRunner(provider, run_dir, judge_manifest=manifest).run(samples)
    assert summary["success"] == 2 and provider.calls == Counter({"A": 2, "B": 1})
    assert _records(run_dir / "judge_results.jsonl")["A"]["attempt_history"][0][
        "error_type"] == "quota_error"
