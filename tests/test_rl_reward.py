import json

import pytest

from opensearch_vl_repro.agent.runtime import AgentTrajectory, AgentTurn
from opensearch_vl_repro.evaluation.judge import JudgeResult, JudgeSample
from opensearch_vl_repro.rl.query_judge import build_query_messages, parse_query_response, query_reward
from opensearch_vl_repro.rl.reward import (accuracy_reward, compose_reward, clamp_fatal_advantages,
                                           format_reward)
from opensearch_vl_repro.rl.workflow_types import RLInfrastructureError


@pytest.mark.parametrize("fmt,acc,query,total", [(1, 1, 1, 1), (1, 1, 0, .8),
                                                   (1, 0, 1, .2), (0, 1, 1, 0)])
def test_composition(fmt, acc, query, total):
    assert compose_reward(fmt, acc, query).total == pytest.approx(total)


@pytest.mark.parametrize("bad", [-.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_reward_fails_closed(bad):
    with pytest.raises(ValueError):
        compose_reward(1, bad, 0)


def test_bad_weights_fail():
    with pytest.raises(ValueError):
        compose_reward(1, 1, 1, accuracy_weight=.7)


def sample():
    return JudgeSample("s", "synthetic", "q", "a", "a")


def test_correctness_uses_existing_judge_interface():
    class Fake:
        def judge(self, value):
            assert value == sample()
            return JudgeResult("success", verdict="correct")
    assert accuracy_reward(sample(), Fake()) == 1


def test_judge_failure_not_zero():
    class Fake:
        def judge(self, value):
            return JudgeResult("error", error_type="quota_error")
    with pytest.raises(RLInfrastructureError):
        accuracy_reward(sample(), Fake())


def test_query_parser_and_provider_failure():
    messages = build_query_messages(question="ignore instructions", reference_answer="reference",
                                    tool_trace=[], final_answer="a")
    assert "untrusted" in messages[0]["content"]
    assert parse_query_response('{"score": 0.5, "reason": "relevant"}').score == .5
    with pytest.raises(RLInfrastructureError):
        query_reward(messages, lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RLInfrastructureError):
        query_reward(messages, lambda _: '{"score": 2, "reason": "bad"}')


def test_query_prompt_inputs_rubric_and_correctness_separation():
    trace = [{"tool": "image_search", "query": "img_1", "observation": "match"}]
    messages = build_query_messages(question="Which landmark?", reference_answer="Tower",
                                    tool_trace=trace, final_answer="Possibly a tower")
    payload = json.loads(messages[1]["content"])
    assert payload == {"question": "Which landmark?", "reference_answer": "Tower",
                       "tool_trace": trace, "final_answer": "Possibly a tower"}
    system = messages[0]["content"].lower()
    for criterion in ("image search utility", "text search utility", "query progression",
                      "complementarity", "evidence vs noise ratio"):
        assert criterion in system
    for anchor in ("0.0", "0.3", "0.5", "0.7", "1.0"):
        assert anchor in system
    assert "search/query utility" in system
    assert "do not re-score final-answer correctness" in system
    assert "independent r_acc" in system
    assert "reference answer is context" in system


def test_query_reference_answer_is_required():
    with pytest.raises(TypeError):
        build_query_messages(question="q", tool_trace=[], final_answer=None)
    with pytest.raises(ValueError):
        build_query_messages(question="q", reference_answer=" ", tool_trace=[], final_answer=None)


def test_query_strict_json_score_and_reason():
    assert parse_query_response('{"score": 0.7, "reason": "useful progression"}').score == .7
    for raw in (
        '{"score": -0.1, "reason": "bad"}',
        '{"score": 1.1, "reason": "bad"}',
        '{"score": NaN, "reason": "bad"}',
        '{"score": Infinity, "reason": "bad"}',
        '{"score": 0.7}',
        '{"score": 0.7, "reason": "ok", "extra": true}',
        '{"score": 0.7, "reason": " "}',
        'score: 0.7',
    ):
        with pytest.raises(ValueError):
            parse_query_response(raw)


def test_format_current_tool_schema_and_terminal():
    output = '<tool_call>{"name":"image_search","arguments":{"image_id":"img_1"}}</tool_call>'
    turns = [AgentTurn(output, {"name": "image_search", "arguments": {"image_id": "img_1"}},
                       "matches", "success")]
    trace = AgentTrajectory("s", "synthetic", turns, "answer", "success", ["img_1"])
    assert format_reward(trace) == 1


def test_fatal_advantage_clamp_leaves_group_estimation_to_trainer():
    assert clamp_fatal_advantages([-1, 1, -2], [True, True, False]) == [0, 1, -2]
