"""Actual shared HTTP classification/retry/cache, with CPU-only fixture sessions."""
import copy
import json

import pytest
import requests

from opensearch_vl_repro.agent.reliability import RetryPolicy
from opensearch_vl_repro.evaluation.judge import DeepSeekJudge, JudgeConfig, parse_judge_response
from opensearch_vl_repro.rl.live_workflow import ProviderInterruption
from opensearch_vl_repro.rl.query_judge import parse_query_response
from opensearch_vl_repro.rl.reward_judges import cached_request, live_rewards, reward_cache_identity
from test_rl_gate_c import apply, new_adapter, tool


class Response:
    def __init__(self, status, content): self.status_code, self.text, self.content = status, content, content
    def json(self): return {"choices": [{"message": {"content": self.content}}]}


class Session:
    def __init__(self, responses): self.responses, self.calls = iter(responses), []
    def post(self, *args, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception): raise value
        return value


def client(responses):
    session = Session(responses)
    config = JudgeConfig("deepseek", "https://api.deepseek.com", "deepseek-flash")
    return DeepSeekJudge(config, session=session, retry=RetryPolicy(max_attempts=2, sleeper=lambda _: None)), session


def identity(value, kind="accuracy"):
    return reward_cache_identity(kind=kind, config=value.config, question="Q", reference="R",
                                  trajectory={"tool_trace": [], "final_answer": "A"}, messages=[{"role": "user", "content": "Q"}])


def test_accuracy_query_separate_live_requests_formula_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "cpu-secret")
    value, session = client([Response(200, '{"verdict":"correct","reason":"yes"}'), Response(200, '{"score":0.7,"reason":"useful"}')])
    adapter = new_adapter(); apply(adapter, tool()); apply(adapter, "A red detail.")
    row = {"source_sample_id": "s", "question": "real question", "reference_answer": "Red"}
    trajectory = adapter.finalize_episode(termination="env_done")
    fatal = {"fatal": False, "fatal_step": None}
    result = live_rewards(client=value, directory=tmp_path, row=row, trajectory=trajectory, fatal=fatal)
    assert result["total"] == pytest.approx(result["format"] * (.8 + .2 * .7))
    assert len(session.calls) == 2 and session.calls[0]["json"]["messages"] != session.calls[1]["json"]["messages"]
    assert all(c["json"]["thinking"] == {"type": "disabled"} for c in session.calls)
    assert result["accuracy_judge"]["real_provider_request"] and result["query_judge"]["real_provider_request"]
    query = json.loads(session.calls[1]["json"]["messages"][1]["content"])
    assert len(query["tool_trace"]) == 1 and query["tool_trace"][0]["tool_call"]["name"] == "crop"
    second = live_rewards(client=value, directory=tmp_path, row=row, trajectory=trajectory, fatal=fatal)
    assert len(session.calls) == 2 and second["accuracy_judge"]["cache_hit"] and second["query_judge"]["cache_hit"]
    assert "cpu-secret" not in '\n'.join(p.read_text() for p in tmp_path.rglob("*.json"))


@pytest.mark.parametrize("status,reason", [(401, "auth_failed"), (403, "auth_failed"), (429, "quota_exhausted"), (500, "judge_unavailable")])
def test_exhausted_http_failures_no_reward_no_cache(monkeypatch, tmp_path, status, reason):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "cpu-key")
    value, session = client([Response(status, "bad")] * 2)
    with pytest.raises(ProviderInterruption) as error:
        cached_request(value, tmp_path, identity(value), [{"role": "user", "content": "Q"}], parse_judge_response)
    assert error.value.reason == reason and not list(tmp_path.rglob("*.json"))
    assert len(session.calls) == (1 if status in {401, 403} else 2)


@pytest.mark.parametrize("failure,reason", [(requests.Timeout("timeout"), "judge_unavailable"),
    (requests.ConnectionError("network"), "judge_unavailable"), (Response(200, "not JSON"), "malformed_provider_response")])
def test_exhausted_transport_and_malformed_parse_fail_closed(monkeypatch, tmp_path, failure, reason):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "cpu-key")
    value, session = client([failure] * 2)
    with pytest.raises(ProviderInterruption) as error:
        cached_request(value, tmp_path, identity(value), [], parse_judge_response)
    assert error.value.reason == reason and len(session.calls) == 2 and not list(tmp_path.rglob("*.json"))


def test_transient_429_success_then_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "cpu-key")
    value, session = client([Response(429, "transient"), Response(200, '{"verdict":"incorrect"}')])
    result, audit = cached_request(value, tmp_path, identity(value), [], parse_judge_response)
    assert result[0] == "incorrect" and audit["attempt_count"] == 2
    cached_request(value, tmp_path, identity(value), [], parse_judge_response)
    assert len(session.calls) == 2


def test_missing_key_interruption_and_cache_keys_bind_judge_context(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    value, _ = client([])
    with pytest.raises(ProviderInterruption, match="provider_misconfigured"):
        cached_request(value, tmp_path, identity(value), [], parse_judge_response)
    from opensearch_vl_repro.eval_subset import canonical_json_sha256
    first = identity(value)
    for changed in ({**first, "kind": "query"}, {**first, "question": "changed"},
                    {**first, "reference_answer": "different"}, {**first, "trajectory": {"tool_trace": ["different"]}},
                    {**first, "judge_config": {**first["judge_config"], "model": "changed"}}, {**first, "prompt_version": 999}):
        assert canonical_json_sha256(first) != canonical_json_sha256(changed)


def test_fatal_no_answer_still_live_judged_null_not_fabricated(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "cpu-key")
    value, session = client([Response(200, '{"verdict":"incorrect","reason":"no answer"}'), Response(200, '{"score":0.3,"reason":"some prior evidence"}')])
    adapter = new_adapter(); apply(adapter, tool())
    for _ in range(3): apply(adapter, '<tool_call>{broken}</tool_call>')
    trajectory = adapter.finalize_episode(termination="env_done"); trajectory.status = "max_agent_turns_exceeded"
    result = live_rewards(client=value, directory=tmp_path, row={"source_sample_id":"s", "question":"Q", "reference_answer":"R"},
                          trajectory=trajectory, fatal={"fatal":True,"fatal_step":3})
    assert len(session.calls) == 2 and result["accuracy"] == 0 and result["format"] > 0
    payload = json.loads(session.calls[0]["json"]["messages"][1]["content"])
    assert payload["model_answer"] is None and payload["no_final_answer"] is True and payload["fatal"]["fatal"]
