"""Read-only readiness guard for starting Judge on a parent Agent run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_PARENT_STATUSES = {"pending", "running", "success", "failed"}


class ParentRunValidationError(RuntimeError):
    """The parent Agent artifacts are incomplete, invalid, or inconsistent."""


def _load_parent_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ParentRunValidationError(
            "Parent Agent status.json is missing; refusing to start Judge."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParentRunValidationError(
            "Parent Agent status.json is invalid; refusing to start Judge."
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("samples"), dict):
        raise ParentRunValidationError(
            "Parent Agent status.json has no valid samples mapping; refusing to start Judge."
        )
    for sample_id, item in value["samples"].items():
        if (not isinstance(sample_id, str) or not sample_id
                or not isinstance(item, dict)
                or item.get("status") not in VALID_PARENT_STATUSES):
            raise ParentRunValidationError(
                "Parent Agent status.json contains an unsupported sample status; "
                "refusing to start Judge."
            )
    return value


def _load_trajectory_statuses(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ParentRunValidationError(
            "Parent Agent trajectories.jsonl is missing; refusing to start Judge."
        )
    records: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("trajectory record is not an object")
            sample_id = value.get("sample_id")
            status = value.get("status")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("trajectory record has no sample_id")
            if sample_id in records:
                raise ValueError("trajectory records contain a duplicate sample_id")
            if status not in {"success", "failed"}:
                raise ValueError("trajectory record has an unsupported status")
            records[sample_id] = status
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ParentRunValidationError(
            "Parent Agent trajectories.jsonl is invalid; refusing to start Judge."
        ) from exc
    return records


def validate_parent_run_ready_for_judge(run_dir: str | Path) -> dict[str, int]:
    """Require a finished Agent run and exact status/trajectory agreement."""
    directory = Path(run_dir).expanduser().resolve()
    state = _load_parent_status(directory / "status.json")
    samples = state["samples"]
    counts = {name: 0 for name in ("pending", "running", "success", "failed")}
    for item in samples.values():
        counts[item["status"]] += 1
    summary = {"total": len(samples), **counts}
    if counts["pending"] or counts["running"]:
        raise ParentRunValidationError(
            "Parent Agent run is not complete; refusing to start Judge.\n"
            f"- pending: {counts['pending']}\n"
            f"- running: {counts['running']}"
        )

    trajectories = _load_trajectory_statuses(directory / "trajectories.jsonl")
    if set(samples) != set(trajectories):
        raise ParentRunValidationError(
            "Parent Agent status/trajectory sample IDs are inconsistent; "
            "refusing to start Judge."
        )
    if any(trajectories[sample_id] != samples[sample_id]["status"]
           for sample_id in samples):
        raise ParentRunValidationError(
            "Parent Agent status/trajectory outcomes are inconsistent; "
            "refusing to start Judge."
        )
    return summary
