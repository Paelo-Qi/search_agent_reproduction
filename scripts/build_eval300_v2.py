#!/usr/bin/env python3
"""Build audited Eval-300 v2 from v1 packed images and audited text only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.eval_subset import sha256_file  # noqa: E402
from opensearch_vl_repro.evaluation.eval300 import (EXPECTED_BENCHMARK_COUNTS,  # noqa: E402
                                                    FROZEN_EVAL300_SHA256)


def _validate_population(rows: list[dict], *, id_field: str, label: str) -> dict[str, dict]:
    if len(rows) != 300:
        raise ValueError(f"{label} must have exactly 300 records, got {len(rows)}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} records must be objects")
    ids = [row.get(id_field) for row in rows]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError(f"{label} has a missing or invalid sample ID")
    if len(set(ids)) != 300:
        raise ValueError(f"{label} sample IDs must be globally unique")
    counts = Counter(row.get("benchmark") for row in rows)
    if dict(counts) != EXPECTED_BENCHMARK_COUNTS:
        raise ValueError(f"{label} benchmark counts must be 100/100/100: {dict(counts)}")
    return {row[id_field]: row for row in rows}


def build_eval300_v2(v1_path: Path, audit_json_path: Path, output_path: Path, *,
                     expected_v1_sha256: str = FROZEN_EVAL300_SHA256,
                     progress: Callable[[str], None] | None = None) -> dict:
    """Preserve the original Arrow table except its question and answer columns."""
    v1_path = v1_path.expanduser().resolve()
    audit_json_path = audit_json_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path == v1_path or output_path == audit_json_path:
        raise ValueError("Eval-300 v2 output must not overwrite either input")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing Eval-300 v2: {output_path}")
    if progress:
        progress("validating pinned Eval-300 v1 and audited JSON")
    v1_sha256 = sha256_file(v1_path)
    if v1_sha256 != expected_v1_sha256:
        raise ValueError(f"Eval-300 v1 SHA256 mismatch: {v1_sha256}")
    old = pq.read_table(v1_path)
    required = {"id", "benchmark", "question", "answer", "image_packed"}
    if not required.issubset(old.column_names):
        raise ValueError(f"Eval-300 v1 is missing columns: {sorted(required - set(old.column_names))}")
    old_rows = old.select(["id", "benchmark", "question", "answer"]).to_pylist()
    old_by_id = _validate_population(old_rows, id_field="id", label="Eval-300 v1")
    audited = json.loads(audit_json_path.read_text(encoding="utf-8"))
    if not isinstance(audited, list):
        raise ValueError("audited Eval-300 JSON must be a list")
    audited_by_id = _validate_population(audited, id_field="sample_id", label="audited Eval-300")
    if set(old_by_id) != set(audited_by_id):
        missing = sorted(set(old_by_id) - set(audited_by_id))
        extra = sorted(set(audited_by_id) - set(old_by_id))
        raise ValueError(f"audited sample ID set differs from v1: missing={missing[:5]}, extra={extra[:5]}")
    questions, answers = [], []
    for row in old_rows:
        source = audited_by_id[row["id"]]
        if source["benchmark"] != row["benchmark"]:
            raise ValueError(f"benchmark changed for {row['id']}")
        for key in ("question", "reference_answer"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f"audited {key} must be a nonempty string: {row['id']}")
        questions.append(source["question"])
        answers.append(source["reference_answer"])
    updated = old.set_column(old.schema.get_field_index("question"), "question",
                             pa.array(questions, type=old.schema.field("question").type))
    updated = updated.set_column(old.schema.get_field_index("answer"), "answer",
                                 pa.array(answers, type=old.schema.field("answer").type))
    for name in old.column_names:
        if name not in ("question", "answer") and not old[name].equals(updated[name]):
            raise AssertionError(f"Eval-300 v2 unexpectedly changed column: {name}")
    if old.schema != updated.schema:
        raise AssertionError("Eval-300 v2 changed the Arrow schema or metadata")

    if progress:
        progress("writing audited text with original packed images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.",
                                                 suffix=".tmp", dir=output_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(updated, temporary, compression="zstd")
        if progress:
            progress("verifying written parquet and computing SHA256")
        if not pq.read_table(temporary).equals(updated, check_metadata=True):
            raise AssertionError("written Eval-300 v2 differs from validated in-memory table")
        v2_sha256 = sha256_file(temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output_path": str(output_path), "v1_sha256": v1_sha256,
        "audited_json_sha256": sha256_file(audit_json_path),
        "v2_sha256": v2_sha256, "total_records": len(old_rows),
        "benchmark_counts": dict(sorted(Counter(row["benchmark"] for row in old_rows).items())),
        "question_changed_count": sum(row["question"] != question
                                      for row, question in zip(old_rows, questions, strict=True)),
        "reference_changed_count": sum(row["answer"] != answer
                                       for row, answer in zip(old_rows, answers, strict=True)),
        "image_packed_unchanged": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=ROOT / "data/eval/combined_eval_300.parquet")
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/eval/combined_eval_300_v2.parquet")
    args = parser.parse_args(argv)
    print(json.dumps(build_eval300_v2(
        args.v1, args.audit_json, args.output,
        progress=lambda stage: print(stage, file=sys.stderr, flush=True)),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
