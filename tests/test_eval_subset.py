from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from opensearch_vl_repro.eval_subset import (
    BENCHMARKS,
    DEFAULT_SEED,
    prepare_evaluation_subset,
    select_stable_ids,
    validate_subset_images,
)


def packed_test_image() -> str:
    image = Image.new("RGB", (3, 2), color=(20, 80, 140))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return json.dumps([{"filename": "fixture.png", "data": encoded}])


def source_table(prefix: str, count: int = 120, reverse: bool = False) -> pa.Table:
    packed = packed_test_image()
    rows = [
        {
            "id": index if prefix == "simplevqa" else f"{prefix}_{index:04d}",
            "question": f"Question {index}?",
            "answer": f"Answer {index}",
            "images": ["fixture.png"],
            "image_caption": f"Caption {index}",
            "image_packed": packed,
        }
        for index in range(count)
    ]
    if prefix == "vdr_bench":
        for index, row in enumerate(rows):
            row["question_original"] = f"Original question {index}?"
    random.Random(91).shuffle(rows)
    if reverse:
        rows.reverse()
    return pa.Table.from_pylist(rows)


def write_fixture_sources(source_dir: Path, reverse: bool = False) -> None:
    source_dir.mkdir(parents=True)
    for spec in BENCHMARKS:
        pq.write_table(source_table(spec.name, reverse=reverse), source_dir / spec.source_file)


def run_fixture(root: Path, reverse: bool = False) -> dict:
    source_dir = root / "sources"
    output_dir = root / "output"
    write_fixture_sources(source_dir, reverse=reverse)
    return prepare_evaluation_subset(
        project_root=root,
        dataset="fixture/eval",
        revision="fixture-revision",
        seed=DEFAULT_SEED,
        samples_per_benchmark=100,
        output_dir=output_dir,
        source_dir=source_dir,
        report_path=root / "report.json",
        download_missing=False,
    )


def test_stable_sampling_is_independent_of_source_row_order() -> None:
    ids = [f"id_{index:03d}" for index in range(150)]
    shuffled = list(ids)
    random.Random(17).shuffle(shuffled)
    assert select_stable_ids(ids, 100, DEFAULT_SEED) == select_stable_ids(
        shuffled, 100, DEFAULT_SEED
    )


def test_eval_subset_outputs_are_complete_frozen_and_reproducible(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_report = run_fixture(first_root)
    second_report = run_fixture(second_root, reverse=True)

    assert first_report["combined"]["final_count"] == 300
    assert first_report["combined"]["benchmark_counts"] == {
        "simplevqa": 100,
        "mmsearch": 100,
        "vdr_bench": 100,
    }
    assert first_report["totals"]["image_decode_successes"] == 300
    assert first_report["totals"]["image_decode_failures"] == 0

    combined = pq.read_table(first_root / "output" / "combined_eval_300.parquet")
    assert combined.num_rows == 300
    assert set(combined["benchmark"].to_pylist()) == {
        "simplevqa",
        "mmsearch",
        "vdr_bench",
    }
    assert combined.schema.field("id").type == pa.string()
    assert "question_original" not in combined.column_names
    assert "question_original" in pq.read_schema(
        first_root / "output" / "vdr_bench_100.parquet"
    ).names
    assert all(str(value).strip() for value in combined["id"].to_pylist())
    assert all(str(value).strip() for value in combined["question"].to_pylist())
    assert all(str(value).strip() for value in combined["answer"].to_pylist())

    for spec in BENCHMARKS:
        first_ids = json.loads((first_root / "output" / spec.ids_file).read_text(encoding="utf-8"))
        second_ids = json.loads((second_root / "output" / spec.ids_file).read_text(encoding="utf-8"))
        subset = pq.read_table(first_root / "output" / spec.output_file)
        assert first_ids == second_ids
        assert len(first_ids) == len(set(first_ids)) == subset.num_rows == 100
        assert set(first_ids) == set(str(value) for value in subset["id"].to_pylist())
        assert set(subset["benchmark"].to_pylist()) == {spec.name}


def test_existing_id_manifest_is_reused_instead_of_resampled(tmp_path: Path) -> None:
    report = run_fixture(tmp_path)
    ids_path = tmp_path / "output" / BENCHMARKS[0].ids_file
    frozen_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    rerun = prepare_evaluation_subset(
        project_root=tmp_path,
        dataset=report["dataset"],
        revision=report["dataset_revision"],
        seed=report["seed"],
        samples_per_benchmark=100,
        output_dir=tmp_path / "output",
        source_dir=tmp_path / "sources",
        report_path=tmp_path / "report.json",
        download_missing=False,
    )
    assert rerun["combined"]["final_count"] == 300
    assert json.loads(ids_path.read_text(encoding="utf-8")) == frozen_ids

    ids_path.write_text(json.dumps(frozen_ids[:-1] + ["tampered-id"]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ID manifest checksum"):
        prepare_evaluation_subset(
            project_root=tmp_path,
            dataset=report["dataset"],
            revision=report["dataset_revision"],
            seed=report["seed"],
            samples_per_benchmark=100,
            output_dir=tmp_path / "output",
            source_dir=tmp_path / "sources",
            report_path=tmp_path / "report.json",
            download_missing=False,
        )


def test_image_decoder_accepts_official_packed_shape() -> None:
    table = source_table("image", count=1).append_column(
        "benchmark", pa.array(["simplevqa"])
    )
    stats = validate_subset_images(table, "simplevqa")
    assert stats["image_decode_successes"] == 1
    assert stats["image_decode_failures"] == 0
    assert stats["decoded_image_count"] == 1
