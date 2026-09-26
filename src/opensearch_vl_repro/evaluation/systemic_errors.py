"""Shared systemic-provider interruption detection and attempt audit retention."""

from __future__ import annotations

from typing import Any


SYSTEMIC_ERROR_TYPES = frozenset({
    "authentication_error", "configuration_error", "quota_error",
})


def find_systemic_tool_error(trajectory: Any) -> dict[str, Any] | None:
    """Inspect actual tool turns, even when the trajectory later says success."""
    turns = trajectory.get("turns", ()) if isinstance(trajectory, dict) else trajectory.turns
    for index, turn in enumerate(turns):
        call = turn.get("tool_call") if isinstance(turn, dict) else turn.tool_call
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            continue
        metadata = turn.get("metadata") if isinstance(turn, dict) else turn.metadata
        metadata = metadata if isinstance(metadata, dict) else {}
        errors = (turn.get("error") if isinstance(turn, dict) else turn.error,
                  metadata.get("error_type"))
        error_type = next((value for value in errors
                           if isinstance(value, str) and value in SYSTEMIC_ERROR_TYPES), None)
        if error_type is None:
            continue
        tool = call["name"]
        provider = metadata.get("provider") or metadata.get("backend")
        if provider is None:
            provider = {"web_search": "serper", "image_search": "serpapi_google_lens"}.get(tool)
        return {"error_type": error_type, "turn_index": index, "tool": tool,
                "provider": provider, "attempt_count": metadata.get("attempt_count")}
    return None


def systemic_error_in_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Accept both new interrupted records and legacy failed trajectories."""
    known = record.get("systemic_error")
    if (isinstance(known, dict) and isinstance(known.get("error_type"), str)
            and known["error_type"] in SYSTEMIC_ERROR_TYPES):
        return known
    trajectory = record.get("trajectory")
    return find_systemic_tool_error(trajectory) if isinstance(trajectory, dict) else None


def carry_interruption_history(record: dict[str, Any],
                               previous: dict[str, Any] | None) -> dict[str, Any]:
    """Keep prior systemic result/trajectory when a pending sample is retried."""
    if previous is None:
        return record
    history = list(previous.get("attempt_history") or [])
    previous_type = previous.get("error_type")
    if (systemic_error_in_record(previous) is not None
            or isinstance(previous_type, str) and previous_type in SYSTEMIC_ERROR_TYPES):
        history.append({key: value for key, value in previous.items()
                        if key != "attempt_history"})
    if history:
        record["attempt_history"] = history
    return record
