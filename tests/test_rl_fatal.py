import pytest

from opensearch_vl_repro.agent.runtime import AgentTrajectory, AgentTurn
from opensearch_vl_repro.rl.fatal import classify_turn, detect_fatal
from opensearch_vl_repro.rl.workflow_types import RLInfrastructureError


def trajectory(*errors):
    turns = [AgentTurn("x", None, "Detail: crop width and height must be positive" if error == "ValueError" else "",
                       "success" if error is None else "error", error=error)
             for error in errors]
    return AgentTrajectory("s", "synthetic", turns, "answer", "success", ["img_1"])


@pytest.mark.parametrize("errors,start", [
    ((None, "invalid_tool_call", "invalid_tool_call", None), None),
    ((None, "invalid_tool_call", "unknown_image_id", "duplicate_tool_call"), 1),
    (("invalid_tool_call", "ValueError", "unknown_image_id"), 0),
    (("invalid_tool_call", None, "ValueError", "unknown_image_id"), None),
    (("invalid_tool_call", "ValueError", None, "invalid_tool_call", "ValueError", "unknown_image_id"), 3),
])
def test_fatal_cascades(errors, start):
    result = detect_fatal(trajectory(*errors))
    assert result.fatal == (start is not None)
    assert result.start_index == start
    assert result.preserved_prefix_length == (start if start is not None else len(errors))


@pytest.mark.parametrize("error", ["quota_error", "authentication_error", "configuration_error",
                                    "timeout", "network_error", "provider_error", "unknown_backend"])
def test_infrastructure_never_becomes_model_error(error):
    with pytest.raises(RLInfrastructureError):
        detect_fatal(trajectory("invalid_tool_call", "ValueError", error, "unknown_image_id"))


def test_empty_search_is_neutral_and_breaks_cascade():
    assert classify_turn(trajectory("no_results").turns[0]) == "neutral"
    assert not detect_fatal(trajectory("invalid_tool_call", "ValueError", "no_results", "unknown_image_id")).fatal


def test_successful_empty_result_resets_cascade():
    turns = trajectory("invalid_tool_call", "ValueError", None, "unknown_image_id")
    turns.turns[2].observation = "Search Results: []"
    assert not detect_fatal(turns).fatal


def test_invalid_threshold_fails():
    with pytest.raises(ValueError):
        detect_fatal(trajectory(), threshold=2)


def test_unattributed_value_error_is_infrastructure():
    with pytest.raises(RLInfrastructureError):
        classify_turn(AgentTurn("x", None, "Detail: failed to decode image", "error", error="ValueError"))


def test_unreadable_registered_image_is_not_model_argument_error():
    with pytest.raises(RLInfrastructureError):
        classify_turn(AgentTurn("x", None, "image_search failed (invalid_argument): registered image is unreadable.",
                                "error", error="invalid_argument"))
    assert classify_turn(AgentTurn("x", None, "image_search failed (invalid_argument): image_id must reference a registered img_n.",
                                   "error", error="invalid_argument")) == "model_error"


def test_abnormal_termination_preserves_all_existing_steps():
    trace = trajectory(None, None, None)
    trace.status = "max_agent_turns_exceeded"
    trace.final_answer = None
    result = detect_fatal(trace)
    assert result.fatal is True
    assert result.start_index == 3
    assert result.preserved_prefix_length == 3
    assert result.reason == "abnormal_termination:max_agent_turns_exceeded"


def test_direct_answer_without_tool_turns_is_not_fatal():
    result = detect_fatal(trajectory())
    assert result.fatal is False
    assert result.start_index is None
    assert result.preserved_prefix_length == 0


def test_isolated_model_error_does_not_move_abnormal_fatal_start():
    trace = trajectory("invalid_tool_call", None, None)
    trace.status = "max_agent_turns_exceeded"
    result = detect_fatal(trace)
    assert result.fatal is True
    assert result.start_index == len(trace.turns) == 3
    assert result.preserved_prefix_length == 3


@pytest.mark.parametrize("status", ["success", "max_agent_turns_exceeded"])
def test_error_cascade_start_precedes_termination_status(status):
    trace = trajectory(None, "invalid_tool_call", "unknown_image_id", "duplicate_tool_call")
    trace.status = status
    result = detect_fatal(trace)
    assert result.fatal is True
    assert result.start_index == result.preserved_prefix_length == 1
    assert result.reason == "consecutive_model_tool_errors"


def test_model_generation_error_still_fails_closed():
    trace = trajectory(None)
    trace.status = "model_error"
    with pytest.raises(RLInfrastructureError):
        detect_fatal(trace)
