from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_DATASET = "Osilly/Vision-DeepResearch-Eval"
DEFAULT_REVISION = "deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9"
DEFAULT_SEED = 20260506
SAMPLING_STRATEGY = "fixed_random_sampling_without_stratification"
SCRIPT_VERSION = 1
REQUIRED_FIELDS = ("id", "question", "answer")


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    source_file: str
    output_file: str
    ids_file: str
    display_name: str


BENCHMARKS = (
    BenchmarkSpec(
        "simplevqa",
        "simplevqa_300.parquet",
        "simplevqa_100.parquet",
        "simplevqa_100_ids.json",
        "SimpleVQA",
    ),
    BenchmarkSpec(
        "mmsearch",
        "mmsearch_vision_171.parquet",
        "mmsearch_100.parquet",
        "mmsearch_100_ids.json",
        "MMSearch",
    ),
    BenchmarkSpec(
        "vdr_bench",
        "testmini500_v1.parquet",
        "vdr_bench_100.parquet",
        "vdr_bench_100_ids.json",
        "VDR-Bench",
    ),
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
    return output


def download_source_file(
    dataset: str,
    revision: str,
    source_file: str,
    destination: str | Path,
) -> Path:
    """Download one pinned source parquet without using datasets.load_dataset."""

    output = Path(destination)
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = urllib.parse.quote(dataset, safe="/")
    remote_path = urllib.parse.quote(f"data/{source_file}", safe="/")
    url = (
        f"https://huggingface.co/datasets/{dataset_path}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/{remote_path}?download=true"
    )
    temporary = output.with_name(f".{output.name}.download")
    request = urllib.request.Request(url, headers={"User-Agent": "OpenSearch-VL-Reproduction/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set, bytes, bytearray)):
        return bool(value)
    # Numeric ID 0 is a valid, non-empty identifier.
    return True


def validate_source_table(table: Any, benchmark: str, minimum_count: int) -> dict[str, Any]:
    missing_fields = [name for name in REQUIRED_FIELDS if name not in table.column_names]
    if missing_fields:
        raise ValueError(f"{benchmark}: missing required fields: {missing_fields}")
    if table.num_rows < minimum_count:
        raise ValueError(
            f"{benchmark}: expected at least {minimum_count} rows, found {table.num_rows}"
        )

    columns = {name: table[name].to_pylist() for name in REQUIRED_FIELDS}
    empty_counts = {
        name: sum(not _nonempty(value) for value in values)
        for name, values in columns.items()
    }
    if any(empty_counts.values()):
        raise ValueError(f"{benchmark}: empty required values: {empty_counts}")

    canonical_ids = [str(value) for value in columns["id"]]
    duplicate_count = len(canonical_ids) - len(set(canonical_ids))
    if duplicate_count:
        raise ValueError(f"{benchmark}: duplicate IDs: {duplicate_count}")
    return {
        "duplicate_ids": duplicate_count,
        "empty_ids": empty_counts["id"],
        "empty_questions": empty_counts["question"],
        "empty_answers": empty_counts["answer"],
    }


def select_stable_ids(ids: Iterable[Any], count: int, seed: int) -> list[str]:
    """Sample from stable ID order, independent of parquet row order."""

    stable_ids = sorted(str(value) for value in ids)
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError("IDs must be unique before sampling")
    if len(stable_ids) < count:
        raise ValueError(f"cannot select {count} IDs from {len(stable_ids)} rows")
    selected = random.Random(seed).sample(stable_ids, count)
    return sorted(selected)


def select_rows_by_frozen_ids(table: Any, selected_ids: list[str], benchmark: str) -> Any:
    import pyarrow as pa

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"{benchmark}: frozen ID list contains duplicates")
    positions = {str(value): index for index, value in enumerate(table["id"].to_pylist())}
    missing = [sample_id for sample_id in selected_ids if sample_id not in positions]
    if missing:
        raise ValueError(f"{benchmark}: frozen IDs missing from source: {missing[:5]}")
    ordered_ids = sorted(selected_ids)
    subset = table.take(pa.array([positions[sample_id] for sample_id in ordered_ids]))
    # The official sources use int64 IDs for SimpleVQA and strings for the
    # other two benchmarks. Normalize IDs to strings so the ID manifests,
    # per-benchmark parquet files, and combined file share one stable type.
    id_index = subset.schema.get_field_index("id")
    subset = subset.set_column(
        id_index,
        "id",
        pa.array([str(value) for value in subset["id"].to_pylist()], type=pa.string()),
    )
    benchmark_values = pa.array([benchmark] * subset.num_rows, type=pa.string())
    if "benchmark" in subset.column_names:
        subset = subset.drop(["benchmark"])
    subset = subset.append_column("benchmark", benchmark_values)
    ordered_columns = ["id", "benchmark"] + [
        name for name in subset.column_names if name not in {"id", "benchmark"}
    ]
    return subset.select(ordered_columns)


def _packed_image_payloads(value: Any) -> list[bytes]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return [bytes(value)]
    if not isinstance(value, list) or not value:
        raise ValueError("image_packed must contain at least one image")

    payloads: list[bytes] = []
    for item in value:
        data = item
        if isinstance(item, dict):
            data = item.get("data", item.get("bytes"))
        if isinstance(data, str):
            if data.startswith("data:"):
                data = data.split(",", 1)[1]
            compact = "".join(data.split())
            payloads.append(base64.b64decode(compact, validate=True))
        elif isinstance(data, (bytes, bytearray, memoryview)):
            payloads.append(bytes(data))
        else:
            raise ValueError(f"unsupported packed image payload: {type(data).__name__}")
    return payloads


def validate_subset_images(table: Any, benchmark: str) -> dict[str, Any]:
    if "images" not in table.column_names or "image_packed" not in table.column_names:
        raise ValueError(f"{benchmark}: images and image_packed are required for image validation")

    successes = 0
    decoded_images = 0
    failures: list[dict[str, str]] = []
    for row in table.select(["id", "images", "image_packed"]).to_pylist():
        sample_id = str(row["id"])
        try:
            if not isinstance(row["images"], list) or not row["images"]:
                raise ValueError("images is empty")
            payloads = _packed_image_payloads(row["image_packed"])
            for payload in payloads:
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    if image.width <= 0 or image.height <= 0:
                        raise ValueError("decoded image has invalid dimensions")
                decoded_images += 1
            successes += 1
        except Exception as exc:  # The report needs the offending ID and decoder reason.
            failures.append({"id": sample_id, "error": f"{type(exc).__name__}: {exc}"})

    result = {
        "image_decode_successes": successes,
        "image_decode_failures": len(failures),
        "decoded_image_count": decoded_images,
        "failures": failures,
    }
    if failures:
        raise ValueError(f"{benchmark}: {len(failures)} selected samples failed image decoding")
    return result


def schema_description(table: Any) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in table.schema
    ]


def write_parquet_atomic(table: Any, path: str | Path) -> Path:
    import pyarrow.parquet as pq

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, output)
    return output


def current_git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_existing_freeze(
    output_dir: Path,
    dataset: str,
    revision: str,
    seed: int,
    samples_per_benchmark: int,
) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    id_paths = [output_dir / spec.ids_file for spec in BENCHMARKS]
    if not manifest_path.exists():
        if any(path.exists() for path in id_paths):
            raise RuntimeError("ID manifests exist without manifest.json; refusing an ambiguous resample")
        return None
    manifest = _read_json(manifest_path)
    expected = {
        "dataset": dataset,
        "dataset_revision": revision,
        "seed": seed,
        "samples_per_benchmark": samples_per_benchmark,
        "sampling_strategy": SAMPLING_STRATEGY,
    }
    mismatches = {
        key: {"existing": manifest.get(key), "requested": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"existing frozen evaluation manifest conflicts with request: {mismatches}")
    missing_ids = [str(path) for path in id_paths if not path.is_file()]
    if missing_ids:
        raise RuntimeError(f"frozen manifest is missing ID files: {missing_ids}")
    for spec, ids_path in zip(BENCHMARKS, id_paths):
        expected_sha256 = manifest.get("benchmarks", {}).get(spec.name, {}).get("ids_sha256")
        actual_sha256 = sha256_file(ids_path)
        if not expected_sha256 or actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{spec.name}: frozen ID manifest checksum does not match manifest.json"
            )
    return manifest


def prepare_evaluation_subset(
    *,
    project_root: str | Path,
    dataset: str = DEFAULT_DATASET,
    revision: str = DEFAULT_REVISION,
    seed: int = DEFAULT_SEED,
    samples_per_benchmark: int = 100,
    output_dir: str | Path,
    source_dir: str | Path,
    report_path: str | Path,
    download_missing: bool = True,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    sources = Path(source_dir).resolve()
    report_output = Path(report_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    existing_manifest = _validate_existing_freeze(
        output, dataset, revision, seed, samples_per_benchmark
    )

    source_tables: dict[str, Any] = {}
    source_info: dict[str, dict[str, Any]] = {}
    for spec in BENCHMARKS:
        source_path = sources / spec.source_file
        if not source_path.is_file():
            if not download_missing:
                raise FileNotFoundError(f"source parquet is missing: {source_path}")
            download_source_file(dataset, revision, spec.source_file, source_path)
        table = pq.read_table(source_path)
        validation = validate_source_table(table, spec.name, samples_per_benchmark)
        source_tables[spec.name] = table
        source_sha256 = sha256_file(source_path)
        if existing_manifest is not None:
            expected_source_sha256 = existing_manifest["benchmarks"][spec.name][
                "source_sha256"
            ]
            if source_sha256 != expected_source_sha256:
                raise RuntimeError(
                    f"{spec.name}: source parquet checksum differs from the frozen manifest"
                )
        source_info[spec.name] = {
            "source_file": spec.source_file,
            "source_path": str(source_path),
            "source_count": table.num_rows,
            "source_schema": schema_description(table),
            "source_sha256": source_sha256,
            **validation,
        }

    subsets: dict[str, Any] = {}
    selected_ids_by_benchmark: dict[str, list[str]] = {}
    image_stats: dict[str, dict[str, Any]] = {}
    for spec in BENCHMARKS:
        table = source_tables[spec.name]
        ids_path = output / spec.ids_file
        if existing_manifest is not None:
            selected_ids = _read_json(ids_path)
            if not isinstance(selected_ids, list) or len(selected_ids) != samples_per_benchmark:
                raise ValueError(
                    f"{spec.name}: frozen ID manifest must contain exactly "
                    f"{samples_per_benchmark} IDs"
                )
            selected_ids = [str(value) for value in selected_ids]
        else:
            selected_ids = select_stable_ids(table["id"].to_pylist(), samples_per_benchmark, seed)
        subset = select_rows_by_frozen_ids(table, selected_ids, spec.name)
        image_stats[spec.name] = validate_subset_images(subset, spec.name)
        subsets[spec.name] = subset
        selected_ids_by_benchmark[spec.name] = sorted(selected_ids)

    # VDR-Bench currently has an extra question_original field. Preserve it in
    # vdr_bench_100.parquet, but use only genuine common fields in the combined
    # file instead of inventing null values for the other benchmarks.
    first_columns = list(subsets[BENCHMARKS[0].name].column_names)
    common_columns = [
        name
        for name in first_columns
        if all(name in subsets[spec.name].column_names for spec in BENCHMARKS)
    ]
    required_combined = {"id", "benchmark", "question", "answer", "images", "image_packed"}
    if not required_combined <= set(common_columns):
        raise ValueError(
            f"combined output is missing required common fields: "
            f"{sorted(required_combined - set(common_columns))}"
        )
    combined_tables = [subsets[spec.name].select(common_columns) for spec in BENCHMARKS]
    combined_schemas = [table.schema for table in combined_tables]
    if any(
        not schema.equals(combined_schemas[0], check_metadata=False)
        for schema in combined_schemas[1:]
    ):
        raise ValueError("common source fields have incompatible types after ID normalization")
    combined = pa.concat_tables(combined_tables)
    combined = combined.sort_by([("benchmark", "ascending"), ("id", "ascending")])
    expected_combined_count = samples_per_benchmark * len(BENCHMARKS)
    if combined.num_rows != expected_combined_count:
        raise AssertionError(
            f"combined row count is {combined.num_rows}, expected {expected_combined_count}"
        )

    generated: dict[str, dict[str, Any]] = {}
    for spec in BENCHMARKS:
        ids_path = write_json_atomic(output / spec.ids_file, selected_ids_by_benchmark[spec.name])
        parquet_path = write_parquet_atomic(subsets[spec.name], output / spec.output_file)
        generated[spec.name] = {
            "output_file": spec.output_file,
            "output_path": str(parquet_path),
            "output_sha256": sha256_file(parquet_path),
            "ids_file": spec.ids_file,
            "ids_path": str(ids_path),
            "ids_sha256": sha256_file(ids_path),
            "selected_count": subsets[spec.name].num_rows,
            "selected_schema": schema_description(subsets[spec.name]),
        }
    combined_path = write_parquet_atomic(combined, output / "combined_eval_300.parquet")

    generated_at = datetime.now(timezone.utc).isoformat()
    git_commit = current_git_commit(root)
    implementation_sha256 = sha256_file(Path(__file__).resolve())
    manifest = {
        "dataset": dataset,
        "dataset_revision": revision,
        "seed": seed,
        "samples_per_benchmark": samples_per_benchmark,
        "sampling_strategy": SAMPLING_STRATEGY,
        "script_version": SCRIPT_VERSION,
        "script_git_commit": git_commit,
        "implementation_sha256": implementation_sha256,
        "generated_at_utc": generated_at,
        "benchmarks": {
            spec.name: {
                "source_file": spec.source_file,
                "source_count": source_info[spec.name]["source_count"],
                "source_schema": source_info[spec.name]["source_schema"],
                "source_sha256": source_info[spec.name]["source_sha256"],
                "selected_count": generated[spec.name]["selected_count"],
                "output_file": spec.output_file,
                "output_sha256": generated[spec.name]["output_sha256"],
                "ids_file": spec.ids_file,
                "ids_sha256": generated[spec.name]["ids_sha256"],
                "image_decode_successes": image_stats[spec.name]["image_decode_successes"],
                "image_decode_failures": image_stats[spec.name]["image_decode_failures"],
            }
            for spec in BENCHMARKS
        },
        "combined": {
            "output_file": "combined_eval_300.parquet",
            "selected_count": combined.num_rows,
            "output_sha256": sha256_file(combined_path),
            "sort_order": ["benchmark", "id"],
            "columns": common_columns,
            "omitted_non_common_fields": {
                spec.name: sorted(set(subsets[spec.name].column_names) - set(common_columns))
                for spec in BENCHMARKS
            },
        },
    }
    manifest_path = write_json_atomic(output / "manifest.json", manifest)

    report = {
        "passed": True,
        "dataset": dataset,
        "dataset_revision": revision,
        "seed": seed,
        "target_samples_per_benchmark": samples_per_benchmark,
        "sampling_strategy": SAMPLING_STRATEGY,
        "manifest_path": str(manifest_path),
        "benchmarks": {
            spec.name: {
                **source_info[spec.name],
                **generated[spec.name],
                **image_stats[spec.name],
                "final_count": subsets[spec.name].num_rows,
            }
            for spec in BENCHMARKS
        },
        "combined": {
            "final_count": combined.num_rows,
            "benchmark_counts": {
                spec.name: subsets[spec.name].num_rows for spec in BENCHMARKS
            },
            "output_path": str(combined_path),
            "output_sha256": sha256_file(combined_path),
            "schema": schema_description(combined),
        },
        "totals": {
            "duplicate_ids_within_benchmarks": sum(
                info["duplicate_ids"] for info in source_info.values()
            ),
            "empty_questions": sum(info["empty_questions"] for info in source_info.values()),
            "empty_answers": sum(info["empty_answers"] for info in source_info.values()),
            "image_decode_successes": sum(
                stats["image_decode_successes"] for stats in image_stats.values()
            ),
            "image_decode_failures": sum(
                stats["image_decode_failures"] for stats in image_stats.values()
            ),
        },
        "generated_at_utc": generated_at,
        "script_git_commit": git_commit,
        "implementation_sha256": implementation_sha256,
    }
    write_json_atomic(report_output, report)
    return report
