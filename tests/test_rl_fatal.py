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
