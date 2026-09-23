"""Frozen Eval-300 validation and deterministic balanced invocation planning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from opensearch_vl_repro.eval_subset import canonical_json_sha256, sha256_file


FROZEN_EVAL300_SHA256 = "f9d0ca74f98d3f73cd6ad1f60b7f2294c5e5d59d4956b60ad95084dc8ee36cf4"
EVAL300_SELECTION_VERSION = 1
BENCHMARK_ORDER = ("simplevqa", "mmsearch", "vdr_bench")
EXPECTED_BENCHMARK_COUNTS = {name: 100 for name in BENCHMARK_ORDER}
FIRST_BATCH_COUNTS = {"simplevqa": 67, "mmsearch": 67, "vdr_bench": 66}
SECOND_BATCH_COUNTS = {"simplevqa": 33, "mmsearch": 33, "vdr_bench": 34}


@dataclass(frozen=True)
class Eval300Plan:
    dataset_path: Path
    dataset_sha256: str
    entries: tuple[tuple[str, str], ...]
    first_batch: tuple[tuple[str, str], ...]
    second_batch: tuple[tuple[str, str], ...]

    def selection_identity(self) -> dict[str, Any]:
        """Return full-run identity; invocation limits intentionally do not appear here."""
        ordered = [{"benchmark": benchmark, "sample_id": sample_id}
                   for benchmark, sample_id in self.entries]
        return {
            "selection_mode": "full_eval300_balanced_batches",
            "selection_version": EVAL300_SELECTION_VERSION,
            "dataset_sha256": self.dataset_sha256,
            "sample_count": len(self.entries),
            "ordered_ids_checksum": canonical_json_sha256(ordered),
            "first_batch_counts": dict(FIRST_BATCH_COUNTS),
            "second_batch_counts": dict(SECOND_BATCH_COUNTS),
        }


def _interleave(groups: Mapping[str, list[str]], counts: Mapping[str, int],
                offsets: Mapping[str, int] | None = None) -> list[tuple[str, str]]:
    offsets = offsets or {name: 0 for name in BENCHMARK_ORDER}
    output: list[tuple[str, str]] = []
    longest = max(counts.values(), default=0)
    for index in range(longest):
        for benchmark in BENCHMARK_ORDER:
            if index < counts[benchmark]:
                output.append((benchmark, groups[benchmark][offsets[benchmark] + index]))
    return output


def build_eval300_plan(
    dataset_path: str | Path,
    *,
    expected_sha256: str = FROZEN_EVAL300_SHA256,
) -> Eval300Plan:
    """Validate the frozen universe and order it as balanced 200 + remaining 100."""
    import pyarrow.parquet as pq

    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen Eval-300 parquet is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "frozen Eval-300 SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    table = pq.read_table(path, columns=["id", "benchmark"])
    rows = [(str(row["benchmark"]), str(row["id"])) for row in table.to_pylist()]
    if len(rows) != 300:
        raise ValueError(f"Eval-300 must contain exactly 300 rows, got {len(rows)}")
    if len(set(rows)) != len(rows) or len({sample_id for _, sample_id in rows}) != len(rows):
        raise ValueError("Eval-300 sample IDs must be globally unique")
    counts = Counter(benchmark for benchmark, _ in rows)
    if dict(counts) != EXPECTED_BENCHMARK_COUNTS:
        raise ValueError(
            "Eval-300 benchmark counts must be 100/100/100; "
            f"got {dict(sorted(counts.items()))}"
        )

    groups = {
        benchmark: sorted(sample_id for row_benchmark, sample_id in rows
                          if row_benchmark == benchmark)
        for benchmark in BENCHMARK_ORDER
    }
    first = _interleave(groups, FIRST_BATCH_COUNTS)
    second = _interleave(groups, SECOND_BATCH_COUNTS, FIRST_BATCH_COUNTS)
    entries = first + second
    if len(first) != 200 or len(second) != 100 or len(entries) != 300:
        raise AssertionError("internal Eval-300 batch planning count mismatch")
    if set(first) & set(second) or set(entries) != set(rows):
        raise AssertionError("internal Eval-300 partition mismatch")
    return Eval300Plan(path, actual_sha256, tuple(entries), tuple(first), tuple(second))
