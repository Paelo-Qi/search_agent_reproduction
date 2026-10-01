"""Consecutive model-caused tool failures, with infrastructure fail-closed."""

from __future__ import annotations

from opensearch_vl_repro.agent.runtime import AgentTrajectory, AgentTurn
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS
from opensearch_vl_repro.evaluation.systemic_errors import SYSTEMIC_ERROR_TYPES

from .workflow_types import FatalInfo, RLInfrastructureError

MODEL_ERRORS = frozenset({
    "invalid_tool_call", "unknown_tool", "unknown_image_id", "duplicate_tool_call",
})
INFRA_ERRORS = frozenset(SYSTEMIC_ERROR_TYPES | {
    "timeout", "network_error", "provider_error", "invalid_response",
    "internal_error", "model_error", "RuntimeError", "OSError",
})


def _local_validation_error(turn: AgentTurn) -> bool:
    """Only known schema/visual-argument ValueErrors are model-attributed."""
    detail = (turn.observation or "").split("Detail: ", 1)
    if len(detail) != 2:
        return False
    message = detail[1].splitlines()[0]
    schema_prefixes = tuple(f"{declaration.name}: " for declaration in TOOL_DECLARATIONS)
    local_prefixes = ("x must ", "y must ", "width must ", "height must ",
                      "amount must ", "scale must ", "crop width and height must ",
                      "crop region is empty", "upscaled image exceeds ")
    return message.startswith(schema_prefixes + local_prefixes)


def _search_argument_error(turn: AgentTurn) -> bool:
    observation = turn.observation or ""
    return any(fragment in observation for fragment in (
        "query must not be empty", "top_k must be a finite integer",
        "top_k must be an integer from", "image_id must reference a registered img_n",
    ))


def classify_turn(turn: AgentTurn) -> str:
    """Return success/neutral/model_error, or raise for non-attributed failure."""
    if turn.status == "success":
        return "success"
    error = turn.metadata.get("error_type") or turn.error
    if error in INFRA_ERRORS:
        raise RLInfrastructureError(f"tool infrastructure failure: {error}")
    if error == "no_results":
        return "neutral"
    if error == "ValueError" and _local_validation_error(turn):
        return "model_error"
    if error == "invalid_argument" and _search_argument_error(turn):
        return "model_error"
    if error in MODEL_ERRORS:
        return "model_error"
    raise RLInfrastructureError(f"unclassified tool failure: {error!r}")


def detect_fatal(trajectory: AgentTrajectory, *, threshold: int = 3) -> FatalInfo:
    if threshold != 3:
        raise ValueError("RL fatal threshold must be 3")
    if trajectory.status == "model_error":
        raise RLInfrastructureError(f"rollout model failure: {trajectory.error}")
    consecutive = 0
    for index, turn in enumerate(trajectory.turns):
        if classify_turn(turn) in {"success", "neutral"}:
            consecutive = 0
            continue
        consecutive += 1
        if consecutive == threshold:
            start = index - threshold + 1
            return FatalInfo(True, start, "consecutive_model_tool_errors", start, threshold)
    return FatalInfo(False, None, None, len(trajectory.turns), threshold)
