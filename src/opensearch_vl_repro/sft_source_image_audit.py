"""CPU-only image-ID contract audit over all pinned Search-VL-SFT sources."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .data import validate_raw_sample
from .sft_image_grounding import audit_raw_image_contract
from .sft_main_data import SOURCE_COUNTS
from .sft_tool_audit import (DATASET_ID, DATASET_REVISION, SOURCE_FILES,
                             iter_json_array, sha256_file)


AUDIT_VERSION = "source-image-contract-v1"
DERIVED_ERROR_KINDS = {"derived_image_registration_gap", "derived_image_marker_id_mismatch"}
UNGROUNDED_ERROR_KINDS = {"ungrounded_tool_image_reference",
                          "ungrounded_assistant_image_reference"}
NON_IMG_N_ERROR_KIND = "non_runtime_image_search_reference"


def reason_for_errors(errors: list[dict[str, Any]]) -> str | None:
    kinds = {error["kind"] for error in errors}
    if NON_IMG_N_ERROR_KIND in kinds:
        return "image_search_non_img_n_target"
    if kinds & (DERIVED_ERROR_KINDS | UNGROUNDED_ERROR_KINDS):
        return "derived_image_id_gap"
    return None


def audit_source_population(
    raw_dir: str | Path, *,
    source_files: Mapping[str, str] = SOURCE_FILES,
    source_counts: Mapping[str, int] = SOURCE_COUNTS,
    expected_source_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Stream each source once; never open media, a processor, GPU, or network."""
    root = Path(raw_dir)
    totals = Counter()
    per_source: dict[str, dict[str, Any]] = {}
    by_reason: dict[str, set[str]] = {
        "image_search_non_img_n_target": set(), "derived_image_id_gap": set(),
    }
    kind_ids: dict[str, set[str]] = {
        "image_search_non_img_n": set(), "derived_image_id_gap": set(),
        "ungrounded_image_reference": set(),
    }
    anomalies: list[dict[str, Any]] = []
    exclusions: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    for source, relative in source_files.items():
        if source not in source_counts:
            raise ValueError(f"missing pinned source count: {source}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned source JSON is missing: {path}")
        digest = sha256_file(path)
        if expected_source_sha256 is not None and digest != expected_source_sha256.get(source):
            raise ValueError(f"pinned source SHA256 mismatch: {source}")
        source_hashes[source] = digest
        counts = Counter()
        source_bad: set[str] = set()
        for index, record in enumerate(iter_json_array(path)):
            sample_id = f"{source}:{index}"
            counts["total_records"] += 1
            if not isinstance(record, dict):
                counts["invalid_raw_format_count"] += 1
                continue
            try:
                validate_raw_sample(record)
            except (TypeError, ValueError):
                counts["invalid_raw_format_count"] += 1
            result = audit_raw_image_contract(record, sample_id=sample_id)
            if result["calls"].get("image_search_total", 0):
                counts["records_with_image_search"] += 1
            counts["image_search_non_img_n_count"] += result["calls"].get("image_search_non_img_n", 0)
            reason = reason_for_errors(result["errors"])
            if reason:
                exclusions[sample_id] = reason
                by_reason[reason].add(sample_id)
                source_bad.add(sample_id)
            for error in result["errors"]:
                kind = error["kind"]
                if kind == NON_IMG_N_ERROR_KIND:
                    kind_ids["image_search_non_img_n"].add(sample_id)
                elif kind in DERIVED_ERROR_KINDS:
                    counts["derived_image_id_gap_count"] += 1
                    kind_ids["derived_image_id_gap"].add(sample_id)
                elif kind in UNGROUNDED_ERROR_KINDS:
                    counts["ungrounded_image_reference_count"] += 1
                    kind_ids["ungrounded_image_reference"].add(sample_id)
                anomalies.append({"sample_id": sample_id, "source": source,
                                  "source_index": index, **error})
        if counts["total_records"] != source_counts[source]:
            raise ValueError(f"{source} has {counts['total_records']} records, expected {source_counts[source]}")
        counts["bad_record_count"] = len(source_bad)
        per_source[source] = dict(counts)
        totals.update(counts)
    return {
        "audit_version": AUDIT_VERSION, "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION, "source_file_sha256": source_hashes,
        "total_records": totals["total_records"],
        "records_with_image_search": totals["records_with_image_search"],
        "image_search_non_img_n_count": totals["image_search_non_img_n_count"],
        "derived_image_id_gap_count": totals["derived_image_id_gap_count"],
        "ungrounded_image_reference_count": totals["ungrounded_image_reference_count"],
        "invalid_raw_format_count": totals["invalid_raw_format_count"],
        "per_source": per_source, "anomalies": anomalies,
        "sample_ids_by_kind": {kind: sorted(ids) for kind, ids in kind_ids.items()},
        "sample_ids_by_reason": {reason: sorted(ids) for reason, ids in by_reason.items()},
        "all_bad_sample_ids": sorted(exclusions),
        "recommended_exclusions": dict(sorted(exclusions.items())),
        "complete": True,
    }
