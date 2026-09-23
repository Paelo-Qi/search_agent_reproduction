"""Deterministic ID-only Dev-30 selection from the frozen Eval-300."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from opensearch_vl_repro.eval_subset import (
    BENCHMARKS, DEFAULT_SEED, ID_CHECKSUM_ALGORITHM, canonical_json_sha256,
    select_stable_ids, sha256_file, write_json_atomic,
)


DEV30_SEED = DEFAULT_SEED
DEV30_MANIFEST_VERSION = 1
DEV30_SCRIPT_VERSION = 1
DEV30_SAMPLES_PER_BENCHMARK = 10
DEV30_SELECTION_STRATEGY = "stable_id_sort_then_seeded_sample_without_replacement"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _semantic_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items()
            if key not in {"created_at", "selection_manifest_checksum"}}


def prepare_dev30(
    *, dataset_path: str | Path, source_manifest_path: str | Path,
    output_dir: str | Path, seed: int = DEV30_SEED,
    created_at_factory: Callable[[], str] | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    dataset = Path(dataset_path).expanduser().resolve()
    source_path = Path(source_manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    source = _read_json(source_path)
    expected_sha = source.get("combined", {}).get("output_sha256")
    actual_sha = sha256_file(dataset)
    if not expected_sha or expected_sha != actual_sha:
        raise ValueError("Eval-300 parquet does not match its frozen source manifest")
    table = pq.read_table(dataset, columns=["id", "benchmark"])
    rows = table.to_pylist()
    seen_pairs = {(str(row["benchmark"]), str(row["id"])) for row in rows}
    if len(seen_pairs) != len(rows):
        raise ValueError("Eval-300 contains duplicate benchmark/sample ID pairs")

    selected_by_benchmark: dict[str, list[str]] = {}
    benchmark_manifest: dict[str, Any] = {}
    combined: list[dict[str, str]] = []
    for spec in BENCHMARKS:
        candidates = [str(row["id"]) for row in rows
                      if str(row["benchmark"]) == spec.name]
        if len(candidates) != 100:
            raise ValueError(f"{spec.name}: expected 100 frozen Eval-300 IDs, got {len(candidates)}")
        selected = select_stable_ids(candidates, DEV30_SAMPLES_PER_BENCHMARK, seed)
        selected_by_benchmark[spec.name] = selected
        ids_file = f"{spec.name}_10_ids.json"
        write_json_atomic(output / ids_file, selected)
        checksum = canonical_json_sha256(selected)
        benchmark_manifest[spec.name] = {
            "benchmark": spec.display_name,
            "selected_count": len(selected),
            "selected_ids": selected,
            "ids_file": ids_file,
            "canonical_id_checksum": checksum,
        }
        combined.extend({"benchmark": spec.name, "sample_id": item} for item in selected)

    if len(combined) != 30 or len({item["sample_id"] for item in combined}) != 30:
        raise ValueError("Dev-30 must contain exactly 30 globally unique sample IDs")
    write_json_atomic(output / "dev30_ids.json", combined)
    combined_checksum = canonical_json_sha256(combined)
    manifest = {
        "manifest_version": DEV30_MANIFEST_VERSION,
        "id_checksum_algorithm": ID_CHECKSUM_ALGORITHM,
        "seed": seed,
        "selection_strategy": DEV30_SELECTION_STRATEGY,
        "source_eval_manifest_version": source.get("manifest_version"),
        "source_dataset": source.get("dataset"),
        "source_dataset_revision": source.get("dataset_revision"),
        "source_combined_eval_sha256": actual_sha,
        "benchmarks": benchmark_manifest,
        "combined": {
            "total_count": len(combined),
            "ids_file": "dev30_ids.json",
            "benchmark_order": [spec.name for spec in BENCHMARKS],
            "canonical_id_checksum": combined_checksum,
        },
        "script_version": DEV30_SCRIPT_VERSION,
        "created_at": (created_at_factory or
                       (lambda: datetime.now(timezone.utc).isoformat()))(),
    }
    manifest["selection_manifest_checksum"] = canonical_json_sha256(
        _semantic_identity(manifest)
    )
    write_json_atomic(output / "dev30_manifest.json", manifest)
    return manifest


def load_selection_manifest(path: str | Path) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("manifest_version") != DEV30_MANIFEST_VERSION:
        raise ValueError("unsupported Dev-30 manifest version")
    expected = manifest.get("selection_manifest_checksum")
    if expected != canonical_json_sha256(_semantic_identity(manifest)):
        raise ValueError("Dev-30 selection manifest checksum mismatch")
    entries: list[tuple[str, str]] = []
    for name in manifest.get("combined", {}).get("benchmark_order", []):
        block = manifest.get("benchmarks", {}).get(name, {})
        ids = block.get("selected_ids")
        if not isinstance(ids, list):
            raise ValueError(f"Dev-30 benchmark {name!r} has no selected ID list")
        entries.extend((name, str(sample_id)) for sample_id in ids)
    if len(entries) != manifest.get("combined", {}).get("total_count"):
        raise ValueError("Dev-30 manifest count mismatch")
    expected_ids = manifest["combined"].get("canonical_id_checksum")
    payload = [{"benchmark": benchmark, "sample_id": sample_id}
               for benchmark, sample_id in entries]
    if canonical_json_sha256(payload) != expected_ids:
        raise ValueError("Dev-30 combined ID checksum mismatch")
    return entries, manifest
