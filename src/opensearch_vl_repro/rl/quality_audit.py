"""Strict, local-only validation of the frozen RL quality audit bundle."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opensearch_vl_repro.eval_subset import sha256_file


QUALITY_AUDIT_VERSION = "rl-quality-audit-v1"
QUALITY_ELIGIBILITY_RULE = "status == ok"
QUALITY_FILENAMES = {
    "audit": "rl_quality_audit_v1.jsonl",
    "allowlist": "rl_quality_ok_allowlist_v1.json",
    "exclusions": "rl_quality_exclusions_v1.json",
    "manual_review": "rl_quality_manual_review_v1.json",
    "summary": "rl_quality_audit_summary_v1.json",
}
# Frozen v1 exclusion reasons; manual_review has no exclusion reason.
QUALITY_EXCLUSION_REASONS = frozenset({
    "exclude_question_ambiguous", "exclude_question_invalid",
    "exclude_reference_incorrect", "exclude_reference_incomplete",
    "exclude_image_mismatch", "exclude_unanswerable",
    "exclude_outdated_reference", "exclude_multiple_valid_answers",
    "exclude_other",
})


@dataclass(frozen=True)
class QualityAudit:
    rows: tuple[dict[str, Any], ...]
    selected: tuple[dict[str, Any], ...]
    reserve: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]


def _mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"quality audit file must be an object: {path.name}")
    return value


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _entries(value: Any, *, name: str, rows_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"quality {name} must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"candidate_rank", "source_sample_id"}:
            raise ValueError(f"quality {name} must contain rank/ID objects")
        identity = item.get("source_sample_id")
        row = rows_by_id.get(identity) if isinstance(identity, str) else None
        if row is None or row["status"] != "ok":
            raise ValueError(f"quality {name} contains a non-ok or unknown source ID")
        if type(item.get("candidate_rank")) is not int or item["candidate_rank"] != row["candidate_rank"]:
            raise ValueError(f"quality {name} rank/ID entry is invalid")
        result.append({"candidate_rank": row["candidate_rank"], "source_sample_id": identity})
    if len({item["source_sample_id"] for item in result}) != len(result):
        raise ValueError(f"quality {name} has duplicate source IDs")
    if [item["candidate_rank"] for item in result] != sorted(item["candidate_rank"] for item in result):
        raise ValueError(f"quality {name} must follow candidate rank")
    return result


def load_quality_audit(directory: str | Path, *, main_count: int,
                       expected_version: str = QUALITY_AUDIT_VERSION) -> QualityAudit:
    """Validate three frozen files together; never infer eligibility from an allowlist alone."""
    directory = Path(directory)
    paths = {name: directory / filename for name, filename in QUALITY_FILENAMES.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"required quality audit file missing: {path}")
    if not _integer(main_count, minimum=1):
        raise ValueError("quality main_count must be positive")
    rows = []
    with paths["audit"].open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid quality JSONL line {number}") from exc
            if not isinstance(row, dict) or not {"candidate_rank", "source_sample_id", "status", "reason", "note"} <= row.keys():
                raise ValueError(f"quality audit row {number} has missing fields")
            rank, identity, status, reason, note = (row[key] for key in (
                "candidate_rank", "source_sample_id", "status", "reason", "note"))
            if (not _integer(rank, minimum=1) or rank != number
                    or not isinstance(identity, str) or re.fullmatch(r"rl_\d{6}", identity) is None
                    or not isinstance(status, str) or status not in {"ok", "exclude", "manual_review"}
                    or not isinstance(note, str)
                    or (status == "exclude" and (not isinstance(reason, str)
                                                   or reason not in QUALITY_EXCLUSION_REASONS))
                    or (status != "exclude" and reason is not None)):
                raise ValueError(f"invalid quality audit row {number}")
            rows.append(row)
    identities = [row["source_sample_id"] for row in rows]
    if not rows or len(set(identities)) != len(identities):
        raise ValueError("quality audit has no rows or duplicate source IDs")
    allowlist, summary = _mapping(paths["allowlist"]), _mapping(paths["summary"])
    exclusions, manual = _mapping(paths["exclusions"]), _mapping(paths["manual_review"])
    required = {"schema_version", "audit_version", "audited_candidate_rank_start",
                "audited_candidate_rank_end", "audited_count", "eligibility_rule",
                "total_ok_count", "selected_main_count", "reserve_ok_count",
                "selected_main400", "reserve_ok"}
    if not required <= allowlist.keys():
        raise ValueError("quality allowlist has missing fields")
    if (type(allowlist["schema_version"]) is not int or allowlist["schema_version"] != 1
            or allowlist["audit_version"] != expected_version
            or type(allowlist["audited_candidate_rank_start"]) is not int
            or allowlist["audited_candidate_rank_start"] != 1
            or type(allowlist["audited_candidate_rank_end"]) is not int
            or allowlist["audited_candidate_rank_end"] != len(rows)
            or type(allowlist["audited_count"]) is not int
            or allowlist["audited_count"] != len(rows)
            or allowlist["eligibility_rule"] != QUALITY_ELIGIBILITY_RULE
            or type(allowlist["selected_main_count"]) is not int
            or allowlist["selected_main_count"] != main_count):
        raise ValueError("quality allowlist identity, range or eligibility mismatch")
    rows_by_id = {row["source_sample_id"]: row for row in rows}
    selected = _entries(allowlist["selected_main400"], name="selected_main400", rows_by_id=rows_by_id)
    reserve = _entries(allowlist["reserve_ok"], name="reserve_ok", rows_by_id=rows_by_id)
    ok_rows = [{"candidate_rank": row["candidate_rank"], "source_sample_id": row["source_sample_id"]}
               for row in rows if row["status"] == "ok"]
    if (len(selected) != main_count or selected != ok_rows[:main_count]
            or reserve != ok_rows[main_count:]
            or type(allowlist["total_ok_count"]) is not int
            or allowlist["total_ok_count"] != len(ok_rows)
            or type(allowlist["reserve_ok_count"]) is not int
            or allowlist["reserve_ok_count"] != len(reserve)):
        raise ValueError("quality allowlist differs from recomputed first-N ok stream")
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("ok", "exclude", "manual_review")}
    expected_exclusions = [{key: row[key] for key in ("candidate_rank", "source_sample_id", "reason", "note")}
                           for row in rows if row["status"] == "exclude"]
    expected_manual = [{key: row[key] for key in ("candidate_rank", "source_sample_id", "note")}
                       for row in rows if row["status"] == "manual_review"]
    if (type(exclusions.get("schema_version")) is not int or exclusions.get("schema_version") != 1
            or exclusions.get("audit_version") != expected_version
            or exclusions.get("excluded_count") != len(expected_exclusions)
            or exclusions.get("excluded") != expected_exclusions
            or type(manual.get("schema_version")) is not int or manual.get("schema_version") != 1
            or manual.get("audit_version") != expected_version
            or manual.get("manual_review_count") != len(expected_manual)
            or manual.get("samples") != expected_manual):
        raise ValueError("quality exclusions/manual-review files differ from full audit")
    reason_counts = {reason: sum(row["reason"] == reason for row in rows)
                     for reason in QUALITY_EXCLUSION_REASONS}
    reported_reason_counts = summary.get("reason_counts")
    if (not isinstance(reported_reason_counts, dict)
            or any(reason not in QUALITY_EXCLUSION_REASONS or not _integer(count)
                   for reason, count in reported_reason_counts.items())
            or any(reported_reason_counts.get(reason, 0) != count for reason, count in reason_counts.items())):
        raise ValueError("quality summary exclusion reason counts mismatch")
    expected_summary = {"audited_count": len(rows), "ok_count": counts["ok"],
                        "excluded_count": counts["exclude"],
                        "manual_review_count": counts["manual_review"],
                        "final_main_count": main_count, "reserve_ok_count": len(reserve),
                        "complete": True, "schema_version": 1,
                        "audit_version": expected_version,
                        "candidate_rank_start": 1, "candidate_rank_end": len(rows),
                        "eligibility_rule": QUALITY_ELIGIBILITY_RULE,
                        "selected_main400_first_rank": selected[0]["candidate_rank"],
                        "selected_main400_last_rank": selected[-1]["candidate_rank"],
                        "selected_main400_membership": [item["source_sample_id"] for item in selected]}
    if (type(summary.get("complete")) is not bool
            or any(summary.get(key) != value or (type(value) is int and type(summary.get(key)) is not int)
                   for key, value in expected_summary.items())):
        raise ValueError("quality summary differs from full audit/allowlist")
    source_manifests = summary.get("source_candidate_manifest_sha256_values")
    if (not isinstance(source_manifests, list) or not source_manifests
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in source_manifests)):
        raise ValueError("quality source candidate manifest fingerprints are invalid")
    provenance = {
        "quality_audit_version": expected_version,
        "quality_audit_sha256": sha256_file(paths["audit"]),
        "quality_allowlist_sha256": sha256_file(paths["allowlist"]),
        "quality_exclusions_sha256": sha256_file(paths["exclusions"]),
        "quality_manual_review_sha256": sha256_file(paths["manual_review"]),
        "quality_summary_sha256": sha256_file(paths["summary"]),
        "audited_candidate_rank_start": 1,
        "audited_candidate_rank_end": len(rows),
        "audited_count": len(rows),
        "quality_ok_count": counts["ok"],
        "quality_excluded_count": counts["exclude"],
        "quality_manual_review_count": counts["manual_review"],
        "quality_eligibility_rule": QUALITY_ELIGIBILITY_RULE,
        "selected_from_quality_ok": True,
    }
    return QualityAudit(tuple(rows), tuple(selected), tuple(reserve), provenance)
