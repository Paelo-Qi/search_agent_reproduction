from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
from PIL import Image

from opensearch_vl_repro.agent.reliability import RetryPolicy
from opensearch_vl_repro.eval_subset import canonical_json_sha256, sha256_file
from opensearch_vl_repro.evaluation.dev30 import load_selection_manifest, prepare_dev30
from opensearch_vl_repro.evaluation.judge import (
    DeepSeekJudge, JudgeConfig, JudgeResult, JudgeSample, build_judge_messages,
    load_judge_samples, parse_judge_response,
)
from opensearch_vl_repro.evaluation.judge_runner import JudgeRunner, build_judge_manifest
from opensearch_vl_repro.evaluation.run_manifest import (
    RunManifestMismatchError, create_run_manifest,
)
from opensearch_vl_repro.inference.eval_reader import read_eval_samples_by_ids


def _eval300(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for benchmark in ("simplevqa", "mmsearch", "vdr_bench"):
        rows.extend({"id": f"{benchmark}-{index:03d}", "benchmark": benchmark,
                     "question": f"question {benchmark} {index}", "answer": f"answer {index}"}
                    for index in range(100))
    path = tmp_path / "eval.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"manifest_version": 2, "dataset": "fixture",
        "dataset_revision": "r1", "combined": {"output_sha256": sha256_file(path)}}),
        encoding="utf-8")
    return path, manifest


def test_dev30_is_repeatable_balanced_unique_and_ordered(tmp_path):
    dataset, source = _eval300(tmp_path)
    first = prepare_dev30(dataset_path=dataset, source_manifest_path=source,
                          output_dir=tmp_path / "one", created_at_factory=lambda: "t1")
    second = prepare_dev30(dataset_path=dataset, source_manifest_path=source,
                           output_dir=tmp_path / "two", created_at_factory=lambda: "t2")
    entries, loaded = load_selection_manifest(tmp_path / "one/dev30_manifest.json")
    assert [item[0] for item in entries] == ["simplevqa"] * 10 + ["mmsearch"] * 10 + ["vdr_bench"] * 10
    assert len(entries) == len({item[1] for item in entries}) == 30
    assert first["selection_manifest_checksum"] == second["selection_manifest_checksum"]
    assert first["combined"]["canonical_id_checksum"] == second["combined"]["canonical_id_checksum"]
    assert loaded["seed"] == 20260506
    for name in ("simplevqa", "mmsearch", "vdr_bench"):
        assert len(first["benchmarks"][name]["selected_ids"]) == 10


@pytest.mark.parametrize(("raw", "expected"), [
    ('{"verdict":"correct","reason":"yes"}', "correct"),
    ('{"verdict":"incorrect","reason":"no"}', "incorrect"),
    ('```json\n{"verdict":"correct","reason":"yes"}\n```', "correct"),
])
def test_judge_parse_valid(raw, expected):
    assert parse_judge_response(raw)[0] == expected


@pytest.mark.parametrize("raw", [
    '{"verdict":"maybe"}', "not json", "", "[]", "```json\n```",
])
def test_judge_parse_invalid(raw):
    with pytest.raises(ValueError):
        parse_judge_response(raw)


class Response:
    def __init__(self, status=200, content='{"verdict":"correct","reason":"ok"}'):
        self.status_code, self.text = status, content

    def json(self):
        return {"choices": [{"message": {"content": self.text}}]}


class Session:
    def __init__(self, outcomes):
        self.outcomes, self.calls, self.payloads = list(outcomes), 0, []

    def post(self, *args, **kwargs):
        self.calls += 1
        self.payloads.append(kwargs["json"])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _judge(session, monkeypatch, attempts=3):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deep-secret")
    config = JudgeConfig("deepseek", "https://example.test", "deepseek-flash",
                         max_attempts=attempts)
    return DeepSeekJudge(config, session=session,
        retry=RetryPolicy(max_attempts=attempts, backoff_seconds=(0, 0), sleeper=lambda _: None))


@pytest.mark.parametrize("outcomes,calls", [
    ([requests.ConnectionError(), Response()], 2),
    ([requests.Timeout(), requests.Timeout(), Response()], 3),
    ([Response(429, "quota"), Response()], 2),
    ([Response(500, "server"), Response()], 2),
])
def test_judge_retries_transient(monkeypatch, outcomes, calls):
    session = Session(outcomes)
    result = _judge(session, monkeypatch).judge(JudgeSample("1", "b", "q", "a", "a"))
    assert result.status == "success" and session.calls == calls
    messages = session.payloads[0]["messages"]
    assert messages == build_judge_messages(JudgeSample("1", "b", "q", "a", "a"))
    user_data = json.loads(messages[1]["content"])
    assert set(user_data) == {"sample_id", "benchmark", "question",
                              "reference_answer", "model_answer"}


@pytest.mark.parametrize("response,error", [
    (Response(401, "bad"), "authentication_error"),
    (Response(403, "bad"), "authentication_error"),
    (Response(400, "bad"), "invalid_request"),
])
def test_judge_does_not_retry_non_transient(monkeypatch, response, error):
    session = Session([response])
    result = _judge(session, monkeypatch).judge(JudgeSample("1", "b", "q", "a", "a"))
    assert result.status == "error" and result.error_type == error and session.calls == 1


@pytest.mark.parametrize("first", [
    "",
    '{"verdict":"correct","reason":"truncated',
    '{"verdict":"maybe","reason":"invalid schema"}',
])
def test_invalid_structured_response_reissues_request_and_recovers(monkeypatch, first):
    final = '{"verdict":"correct","reason":"valid"}'
    session = Session([Response(200, first), Response(200, final)])
    result = _judge(session, monkeypatch).judge(JudgeSample("1", "b", "q", "a", "a"))
    assert result.status == "success" and result.verdict == "correct"
    assert result.raw_response == final
    assert result.metadata["attempt_count"] == 2 and session.calls == 2


def test_invalid_response_exhaustion_keeps_final_raw_and_shared_attempt_budget(monkeypatch):
    outcomes = [Response(200, ""), Response(200, ""), Response(200, "final invalid JSON")]
    session = Session(outcomes)
    result = _judge(session, monkeypatch).judge(JudgeSample("1", "b", "q", "a", "a"))
    assert result.status == "error" and result.error_type == "invalid_response"
    assert result.metadata["attempt_count"] == 3 and session.calls == 3
    assert result.raw_response == "final invalid JSON"


def test_transport_and_structured_failures_share_one_attempt_budget(monkeypatch):
    valid = '{"verdict":"correct","reason":"ok"}'
    session = Session([Response(500, "server"), Response(200, ""), Response(200, valid)])
    result = _judge(session, monkeypatch).judge(JudgeSample("1", "b", "q", "a", "a"))
    assert result.status == "success" and result.metadata["attempt_count"] == 3
    assert session.calls == 3


def test_valid_incorrect_verdict_is_success_without_retry(monkeypatch):
    raw = '{"verdict":"incorrect","reason":"wrong entity"}'
    session = Session([Response(200, raw)])
    result = _judge(session, monkeypatch).judge(JudgeSample("1", "b", "q", "a", "wrong"))
    assert result.status == "success" and result.verdict == "incorrect"
    assert result.metadata["attempt_count"] == 1 and session.calls == 1


def test_missing_key_is_configuration_error_without_call(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    session = Session([])
    result = DeepSeekJudge(JudgeConfig("deepseek", "https://x", "m"), session=session).judge(
        JudgeSample("1", "b", "q", "a", "a"))
    assert result.error_type == "configuration_error" and session.calls == 0


def test_invalid_local_input_does_not_call_provider(monkeypatch):
    session = Session([])
    result = _judge(session, monkeypatch).judge(JudgeSample("1", "b", "", "a", "a"))
    assert result.error_type == "invalid_input" and session.calls == 0


class FakeProvider:
    def __init__(self):
        self.calls = {}

    def judge(self, sample):
        self.calls[sample.sample_id] = self.calls.get(sample.sample_id, 0) + 1
        if sample.sample_id == "A":
            return JudgeResult("success", "correct", "ok", metadata={"attempt_count": 1})
        if sample.sample_id == "B":
            return JudgeResult("success", "incorrect", "wrong", metadata={"attempt_count": 1})
        if sample.sample_id == "C" and self.calls["C"] == 1:
            return JudgeResult("error", reason="network", error_type="network_error",
                               metadata={"attempt_count": 3})
        return JudgeResult("success", "correct", "ok", metadata={"attempt_count": 1})


def _samples():
    return [JudgeSample(name, "simplevqa", f"q{name}", f"r{name}", f"m{name}")
            for name in "ABCD"]


def _manifest(samples):
    return build_judge_manifest(
        parent_manifest={"run_id": "agent-run", "run_config_fingerprint": "parent-fp"},
        config=JudgeConfig("deepseek", "https://example.test", "deepseek-flash"),
        samples=samples, created_at="fixed",
    )


def test_judge_resume_retry_failed_and_accuracy_denominator(tmp_path):
    samples, provider = _samples(), FakeProvider()
    runner = JudgeRunner(provider, tmp_path / "judge", judge_manifest=_manifest(samples))
    first = runner.run(samples, max_samples=3)
    assert (first["success"], first["failed"], first["pending"]) == (2, 1, 1)
    resumed = runner.run(samples)
    assert provider.calls == {"A": 1, "B": 1, "C": 1, "D": 1}
    assert resumed["accuracy_among_successful_judges"] == pytest.approx(2 / 3)
    final = runner.run(samples, retry_failed=True)
    assert final["success"] == 4 and provider.calls["C"] == 2
    assert final["accuracy_among_successful_judges"] == pytest.approx(0.75)


def test_systemic_fail_fast_and_upstream_failure_no_provider_call(tmp_path, capsys):
    class Provider:
        def __init__(self): self.calls = []
        def judge(self, sample):
            self.calls.append(sample.sample_id)
            if sample.sample_id == "B":
                return JudgeResult("error", error_type="quota_error", reason="quota")
            return JudgeResult("success", "correct")
    provider = Provider()
    samples = _samples()
    summary = JudgeRunner(provider, tmp_path / "systemic", judge_manifest=_manifest(samples)).run(samples)
    assert provider.calls == ["A", "B"]
    assert (summary["success"], summary["failed"], summary["pending"]) == (1, 1, 2)
    output = capsys.readouterr().out
    assert "Judge progress: 2/4" in output
    assert "Judge stopped early due to systemic error: quota_error" in output

    upstream = [JudgeSample("X", "vdr_bench", "q", "r", None, "failed")]
    other = Provider()
    summary = JudgeRunner(other, tmp_path / "upstream", judge_manifest=_manifest(upstream)).run(upstream)
    assert other.calls == [] and summary["upstream_failed"] == 1
    assert summary["incorrect"] == 0
    record = json.loads((tmp_path / "upstream/judge_results.jsonl").read_text(encoding="utf-8"))
    assert record["attempt_count"] == 0 and record["error_type"] == "upstream_agent_failure"


def test_judge_progress_every_five_and_final_partial_group(tmp_path, capsys):
    class AlwaysCorrect:
        def judge(self, sample):
            return JudgeResult("success", "correct", metadata={"attempt_count": 1})
    samples = [JudgeSample(str(index), "simplevqa", "q", "r", "m")
               for index in range(12)]
    JudgeRunner(AlwaysCorrect(), tmp_path / "progress",
                judge_manifest=_manifest(samples)).run(samples)
    output = capsys.readouterr().out
    assert "Total samples: 12" in output
    assert "Eligible this invocation: 12" in output
    assert "Judge progress: 5/12" in output
    assert "Judge progress: 10/12" in output
    assert "Judge progress: 12/12" in output


def test_retry_failed_progress_uses_eligible_denominator(tmp_path, capsys):
    class FailsLastFiveOnce:
        def __init__(self): self.calls = {}
        def judge(self, sample):
            count = self.calls.get(sample.sample_id, 0) + 1
            self.calls[sample.sample_id] = count
            if int(sample.sample_id) >= 25 and count == 1:
                return JudgeResult("error", error_type="invalid_response",
                                   metadata={"attempt_count": 3})
            return JudgeResult("success", "correct", metadata={"attempt_count": 1})
    samples = [JudgeSample(str(index), "simplevqa", "q", "r", "m")
               for index in range(30)]
    provider = FailsLastFiveOnce()
    runner = JudgeRunner(provider, tmp_path / "retry-progress",
                         judge_manifest=_manifest(samples))
    runner.run(samples)
    capsys.readouterr()
    runner.run(samples, retry_failed=True)
    output = capsys.readouterr().out
    assert "Total samples: 30" in output
    assert "Eligible this invocation: 5" in output
    assert "Already successful: 25" in output
    assert "Retry failed: true" in output
    assert "Judge progress: 5/5" in output and "Judge progress: 5/30" not in output


def test_reference_answer_is_joined_only_for_judge(tmp_path):
    image = Image.new("RGB", (2, 2), "red")
    buffer = io.BytesIO(); image.save(buffer, format="PNG")
    dataset = tmp_path / "eval.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "x", "benchmark": "simplevqa",
        "question": "q", "answer": "TOP SECRET REFERENCE", "image_packed": buffer.getvalue()}]), dataset)
    agent_samples = read_eval_samples_by_ids(dataset, [("simplevqa", "x")])
    assert not hasattr(agent_samples[0], "answer")
    trajectory = tmp_path / "trajectories.jsonl"
    trajectory.write_text(json.dumps({"sample_id": "x", "benchmark": "simplevqa",
        "question": "q", "status": "success", "final_answer": "model"}) + "\n", encoding="utf-8")
    judged = load_judge_samples(trajectory, dataset)
    assert judged[0].reference_answer == "TOP SECRET REFERENCE"


def test_selection_identity_is_part_of_agent_run_manifest():
    selection = {"selection_mode": "id_manifest", "selection_manifest_checksum": "m",
                 "selected_ids_checksum": "i", "sample_count": 30}
    manifest = create_run_manifest(
        run_id="dev30", model_name_or_path="model", model_revision="r",
        inference_config_fingerprint="inference", dataset_path="eval.parquet",
        dataset_identity={"sha256": "data"}, start=None, limit=None,
        max_agent_turns=8, search_config_fingerprint="search",
        layout_config_fingerprint="layout", checkpoint={"kind": "fake"},
        tool_fingerprint="tools", sample_selection=selection,
    )
    assert manifest["sample_selection"] == selection


def test_judge_manifest_mismatch_refuses_resume_without_mutation(tmp_path):
    samples = _samples()
    runner = JudgeRunner(FakeProvider(), tmp_path / "judge", judge_manifest=_manifest(samples))
    runner.run(samples, max_samples=1)
    before = {path.name: path.read_bytes() for path in (tmp_path / "judge").iterdir()}
    changed = dict(_manifest(samples), judge_model="other")
    with pytest.raises(RunManifestMismatchError):
        JudgeRunner(FakeProvider(), tmp_path / "judge", judge_manifest=changed).run(samples)
    assert {path.name: path.read_bytes() for path in (tmp_path / "judge").iterdir()} == before


def test_judge_stale_running_becomes_interrupted_and_needs_retry(tmp_path):
    samples, provider = _samples()[:1], FakeProvider()
    directory = tmp_path / "judge"
    runner = JudgeRunner(provider, directory, judge_manifest=_manifest(samples))
    runner._prepare_manifest()
    directory.joinpath("judge_status.json").write_text(json.dumps({"version": 1,
        "samples": {"A": {"benchmark": "simplevqa", "status": "running",
        "attempts": 1, "error_type": None, "error": None}}}), encoding="utf-8")
    assert runner.run(samples)["failed"] == 1 and provider.calls == {}
    assert runner.run(samples, retry_failed=True)["success"] == 1


def test_judge_artifacts_redact_deepseek_key(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "never-persist-this")
    class EchoProvider:
        def judge(self, sample):
            return JudgeResult("success", "correct", "never-persist-this",
                               raw_response='{"verdict":"correct","reason":"never-persist-this"}')
    samples = [JudgeSample("x", "simplevqa", "q", "r", "m")]
    JudgeRunner(EchoProvider(), tmp_path / "judge", judge_manifest=_manifest(samples)).run(samples)
    serialized = "".join(path.read_text(encoding="utf-8")
                         for path in (tmp_path / "judge").iterdir())
    assert "never-persist-this" not in serialized and "[REDACTED]" in serialized
