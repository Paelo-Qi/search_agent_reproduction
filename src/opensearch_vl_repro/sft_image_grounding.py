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


def audit_sample_image_grounding(
    sample: dict[str, Any], messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check actual message order; never infer derived IDs from filenames/paths."""
    sample_id = sample.get("_sample_id", "unknown")
    turns = sample["conversations"]
    errors: list[dict[str, Any]] = []
    calls = Counter()
    initial_count = turns[0]["value"].count(IMAGE_MARKER)
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

    visible = set(expected) if not errors else set()
    next_index = initial_count + 1
    parser = ToolCallParser(TOOL_DECLARATIONS_BY_NAME)
    for turn_index, turn in enumerate(turns):
        if turn["from"] == "observation":
            markers = turn["value"].count(IMAGE_MARKER)
            ids = _NEW_IMAGE_ID.findall(turn["value"])
            if markers != len(ids):
                errors.append({"kind": "derived_image_marker_id_mismatch",
                               "turn_index": turn_index, "markers": markers, "ids": ids})
                continue
            for image_id in ids:
                expected_id = f"img_{next_index}"
                if image_id != expected_id:
                    errors.append({"kind": "derived_image_registration_gap",
                                   "turn_index": turn_index, "expected": expected_id,
                                   "actual": image_id})
                else:
                    visible.add(image_id)
                    next_index += 1
        elif turn["from"] == "gpt":
            parsed = parser.parse(turn["value"])
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
                                       "reference": reference})
                if isinstance(reference, str) and _RUNTIME_ID.fullmatch(reference):
                    calls["runtime_id_references"] += 1
                    if reference not in visible:
                        errors.append({"kind": "ungrounded_tool_image_reference",
                                       "turn_index": turn_index, "tool": call.name,
                                       "image_id": reference})
                    else:
                        calls["grounded_runtime_id_references"] += 1
                        if call.name == "image_search":
                            calls["grounded_image_search_img_n"] += 1
    return {"sample_id": sample_id, "passed": not errors,
            "initial_image_count": initial_count, "calls": dict(calls), "errors": errors}


def summarize_image_grounding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        counts.update(row["calls"])
    failures = [{"sample_id": row["sample_id"], "errors": row["errors"]}
                for row in rows if not row["passed"]]
    return {"passed": not failures, "sample_count": len(rows),
            "failed_sample_count": len(failures), "failures": failures,
            "counts": dict(counts)}
