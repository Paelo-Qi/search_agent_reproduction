"""Read-only audit of the runtime image IDs visible in constructed SFT messages."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .agent.runtime import IMAGE_REFERENCE_ARGUMENTS
from .agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from .agent.tool_parser import ToolCallParser
from .data import IMAGE_MARKER


_NEW_IMAGE_ID = re.compile(r"New image ID:\s*(img_[1-9][0-9]*)\b")
_REGISTERED_LINE = re.compile(r"^- (img_[1-9][0-9]*): width=(\d+), height=(\d+)$", re.MULTILINE)
_RUNTIME_ID = re.compile(r"img_[1-9][0-9]*\Z")
_ASSISTANT_RUNTIME_ID = re.compile(r"(?<![A-Za-z0-9_])img_[1-9][0-9]*(?![A-Za-z0-9_])")


def audit_raw_image_contract(sample: dict[str, Any], *, sample_id: str | None = None) -> dict[str, Any]:
    """Audit source turns before selection; no image files or processor required."""
    identity = sample_id or sample.get("_sample_id", "unknown")
    turns = sample.get("conversations")
    if not isinstance(turns, list):
        turns = []
    first = turns[0] if turns and isinstance(turns[0], dict) else {}
    initial_count = first.get("value", "").count(IMAGE_MARKER) if isinstance(first.get("value"), str) else 0
    visible = {f"img_{index}" for index in range(1, initial_count + 1)}
    next_index = initial_count + 1
    parser = ToolCallParser(TOOL_DECLARATIONS_BY_NAME)
    errors: list[dict[str, Any]] = []
    calls = Counter()
    last_tool: str | None = None
    last_observation: str | None = None
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict) or not isinstance(turn.get("value"), str):
            continue
        value = turn["value"]
        if turn.get("from") == "observation":
            markers = value.count(IMAGE_MARKER)
            ids = _NEW_IMAGE_ID.findall(value)
            if markers != len(ids):
                errors.append({"kind": "derived_image_marker_id_mismatch",
                               "turn_index": turn_index, "tool": last_tool,
                               "argument": None, "markers": markers, "ids": ids,
                               "relevant_observation": value})
            else:
                for image_id in ids:
                    expected_id = f"img_{next_index}"
                    if image_id != expected_id:
                        errors.append({"kind": "derived_image_registration_gap",
                                       "turn_index": turn_index, "tool": last_tool,
                                       "argument": image_id, "expected": expected_id,
                                       "actual": image_id, "relevant_observation": value})
                    else:
                        visible.add(image_id)
                        next_index += 1
            last_observation = value
        elif turn.get("from") == "gpt":
            parsed = parser.parse(value)
            ungrounded_tool_refs: set[str] = set()
            for call in parsed.tool_calls:
                argument = IMAGE_REFERENCE_ARGUMENTS.get(call.name)
                if argument is None:
                    continue
                reference = call.arguments.get(argument)
                if call.name == "image_search":
                    calls["image_search_total"] += 1
                    if isinstance(reference, str) and _RUNTIME_ID.fullmatch(reference):
                        calls["image_search_img_n"] += 1
                    else:
                        calls["image_search_non_img_n"] += 1
                        errors.append({"kind": "non_runtime_image_search_reference",
                                       "turn_index": turn_index, "tool": call.name,
                                       "argument": reference, "reference": reference,
                                       "relevant_observation": last_observation,
                                       "registered_ids_before": sorted(visible)})
                if isinstance(reference, str) and _RUNTIME_ID.fullmatch(reference):
                    calls["runtime_id_references"] += 1
                    if reference not in visible:
                        ungrounded_tool_refs.add(reference)
                        errors.append({"kind": "ungrounded_tool_image_reference",
                                       "turn_index": turn_index, "tool": call.name,
                                       "argument": reference, "image_id": reference,
                                       "relevant_observation": last_observation,
                                       "registered_ids_before": sorted(visible)})
                    else:
                        calls["grounded_runtime_id_references"] += 1
                        if call.name == "image_search":
                            calls["grounded_image_search_img_n"] += 1
            if parsed.tool_calls:
                last_tool = parsed.tool_calls[-1].name
            for image_id in sorted(set(_ASSISTANT_RUNTIME_ID.findall(value)) - visible
                                   - ungrounded_tool_refs):
                errors.append({"kind": "ungrounded_assistant_image_reference",
                               "turn_index": turn_index,
                               "tool": parsed.tool_calls[0].name if len(parsed.tool_calls) == 1 else None,
                               "argument": image_id, "image_id": image_id,
                               "relevant_observation": last_observation,
                               "registered_ids_before": sorted(visible)})
    return {"sample_id": identity, "passed": not errors,
            "initial_image_count": initial_count, "calls": dict(calls), "errors": errors}


def audit_sample_image_grounding(
    sample: dict[str, Any], messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check actual message order; never infer derived IDs from filenames/paths."""
    raw = audit_raw_image_contract(sample)
    sample_id = raw["sample_id"]
    initial_count = raw["initial_image_count"]
    errors = list(raw["errors"])
    if not initial_count:
        errors.append({"kind": "missing_initial_image", "turn_index": 0})
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    registered = _REGISTERED_LINE.findall(system if isinstance(system, str) else "")
    expected = [f"img_{index}" for index in range(1, initial_count + 1)]
    if ("Registered input images:" not in system
            or [row[0] for row in registered] != expected):
        errors.append({"kind": "initial_registration_mismatch", "turn_index": 0,
                       "expected": expected, "actual": [row[0] for row in registered]})
    else:
        first_user = next((row for row in messages if row["role"] == "user"), None)
        parts = first_user["content"] if first_user is not None else []
        images = [part["image"] for part in parts
                  if isinstance(part, dict) and part.get("type") == "image"] if isinstance(parts, list) else []
        sizes = [(str(image.width), str(image.height)) for image in images]
        if sizes != [(width, height) for _, width, height in registered]:
            errors.append({"kind": "initial_image_order_or_size_mismatch", "turn_index": 0})

    return {"sample_id": sample_id, "passed": not errors,
            "initial_image_count": initial_count, "calls": raw["calls"], "errors": errors}


def summarize_image_grounding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        counts.update(row["calls"])
    failures = [{"sample_id": row["sample_id"], "errors": row["errors"]}
                for row in rows if not row["passed"]]
    return {"passed": not failures, "sample_count": len(rows),
            "failed_sample_count": len(failures), "failures": failures,
            "counts": dict(counts)}
