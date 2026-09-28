"""Read-only protocol metrics for a fixed tool-protocol dev trajectory run."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .agent.runtime import IMAGE_REFERENCE_ARGUMENTS
from .tool_protocol_dev import validate_dev_manifest


REGISTERED_ID = re.compile(r"img_[1-9][0-9]*\Z")
HTTP_URL = re.compile(r"https?://", re.I)


def _initial_ids(trajectory: dict[str, Any]) -> set[str]:
    images = trajectory.get("images") or []
    initial = {str(image["image_id"]) for image in images
               if isinstance(image, dict) and image.get("kind") == "initial"}
    if initial:
        return initial
    derived = {str(image.get("image_id")) for turn in trajectory.get("turns", [])
               for image in turn.get("derived_images", []) if isinstance(image, dict)}
    return {str(value) for value in trajectory.get("image_ids", [])} - derived


def protocol_metrics(ids: list[str], manifest: dict[str, Any],
                     trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    validate_dev_manifest(ids, manifest)
    lookup = {}
    for record in trajectories:
        identity = str(record["sample_id"])
        if identity in lookup:
            raise ValueError(f"duplicate trajectory sample ID: {identity}")
        if identity not in ids:
            raise ValueError(f"trajectory outside tool-protocol dev set: {identity}")
        lookup[identity] = record
    meta = {row["sample_id"]: row for row in manifest["samples"]}
    totals = Counter()
    call_distribution = Counter()
    rows = []
    for sample_id in ids:
        record = lookup.get(sample_id)
        if record is None:
            rows.append({"sample_id": sample_id, "source": meta[sample_id]["source"],
                         "expected_protocol_category": meta[sample_id].get(
                             "expected_protocol_category", meta[sample_id]["protocol_tags"]),
                         "status": "missing", "tool_calls": []})
            continue
        trajectory = record.get("trajectory") or record
        registered = _initial_ids(trajectory)
        if not registered:
            raise ValueError(f"trajectory has no registered initial image IDs: {sample_id}")
        sample_calls = []
        for turn in trajectory.get("turns", []):
            call = turn.get("tool_call")
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            arguments = call.get("arguments") or {}
            error = turn.get("error")
            metadata = turn.get("metadata") or {}
            totals["total_tool_calls"] += 1
            totals[f"tool:{name}"] += 1
            if error == "duplicate_tool_call":
                totals["duplicate_tool_call_count"] += 1
            if error == "unknown_image_id":
                totals["unknown_image_id_count"] += 1
                if metadata.get("provider_called") is False:
                    totals["provider_not_called_due_to_bad_image_id"] += 1
            argument_name = IMAGE_REFERENCE_ARGUMENTS.get(name)
            reference = arguments.get(argument_name) if argument_name and isinstance(arguments, dict) else None
            if name == "image_search" and isinstance(arguments, dict):
                if "url" in arguments:
                    totals["image_search_legacy_url_argument_count"] += 1
                if "image_id" in arguments:
                    totals["image_search_image_id_argument_count"] += 1
            valid = None
            if argument_name:
                totals["image_tool_argument_count"] += 1
                valid = isinstance(reference, str) and bool(REGISTERED_ID.fullmatch(reference)) and reference in registered
                if valid:
                    totals["registered_image_id_count"] += 1
                if isinstance(reference, str) and HTTP_URL.search(reference):
                    totals["http_image_argument_hallucination_count"] += 1
                    if name == "image_search":
                        totals["image_search_http_image_id_hallucination_count"] += 1
                if name in ("image_search", "layout_parsing", "crop"):
                    totals[f"{name}_argument_count"] += 1
                    if valid:
                        totals[f"{name}_valid_argument_count"] += 1
            sample_calls.append({"name": name, "arguments": arguments, "image_reference": reference,
                                 "registered_image_id_valid": valid, "error": error,
                                 "provider_called": metadata.get("provider_called")})
            for image in turn.get("derived_images", []):
                image_id = image.get("image_id") if isinstance(image, dict) else None
                if isinstance(image_id, str):
                    registered.add(image_id)
        call_distribution[len(sample_calls)] += 1
        if not sample_calls:
            totals["no_tool_behavior_count"] += 1
        expected_tool = meta[sample_id].get("expected_tool_label")
        if expected_tool:
            totals["tool_selection_labeled_count"] += 1
            actual = sample_calls[0]["name"] if sample_calls else "no_tool"
            if actual == expected_tool:
                totals["tool_selection_agreement_count"] += 1
        rows.append({"sample_id": sample_id, "source": meta[sample_id]["source"],
                     "expected_protocol_category": meta[sample_id].get(
                         "expected_protocol_category", meta[sample_id]["protocol_tags"]),
                     "expected_tool_label": expected_tool,
                     "status": trajectory.get("status", record.get("status")),
                     "tool_calls": sample_calls})
    def rate(numerator: str, denominator: str) -> float | None:
        return totals[numerator] / totals[denominator] if totals[denominator] else None
    metrics = {key: totals[key] for key in (
        "total_tool_calls", "image_tool_argument_count", "registered_image_id_count",
        "http_image_argument_hallucination_count", "unknown_image_id_count",
        "image_search_http_image_id_hallucination_count",
        "image_search_legacy_url_argument_count", "image_search_image_id_argument_count",
        "provider_not_called_due_to_bad_image_id", "duplicate_tool_call_count",
        "no_tool_behavior_count", "tool_selection_labeled_count",
        "tool_selection_agreement_count")}
    metrics.update({
        "registered_image_id_rate": rate("registered_image_id_count", "image_tool_argument_count"),
        "http_image_argument_hallucination_rate": rate(
            "http_image_argument_hallucination_count", "image_tool_argument_count"),
        "unknown_image_id_rate": rate("unknown_image_id_count", "image_tool_argument_count"),
        "tool_selection_agreement": rate("tool_selection_agreement_count", "tool_selection_labeled_count"),
        "tool_call_count_distribution": dict(sorted(call_distribution.items())),
    })
    for name in ("image_search", "layout_parsing", "crop"):
        metrics[f"{name}_valid_argument_rate"] = rate(f"{name}_valid_argument_count",
                                                        f"{name}_argument_count")
        metrics[f"{name}_argument_count"] = totals[f"{name}_argument_count"]
    return {"complete": len(lookup) == len(ids), "expected_count": len(ids),
            "observed_count": len(lookup), "metrics": metrics, "samples": rows}
