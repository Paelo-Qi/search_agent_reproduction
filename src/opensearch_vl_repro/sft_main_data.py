"""Pinned, proportional Search-VL-SFT selection and disjoint training shards."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .data import validate_raw_sample
from .sft_tool_audit import DATASET_ID, DATASET_REVISION, SOURCE_FILES, iter_json_array, sha256_file


SOURCE_COUNTS = {
    "fvqa": 4413, "livevqa": 13326, "palace": 2973, "webqa": 3804,
    "wiki_art": 5093, "wiki_en": 3503, "wiki_zh": 3480,
}
SHARD_SIZES = {"main_a_1k": 1000, "main_b_2k": 2000, "extra_1k": 1000,
               "reserve_4k": 4000}
POOL_SIZE = sum(SHARD_SIZES.values())
DEFAULT_SEED = 20260506


def _stable_rank(seed: int, purpose: str, source: str, index: int) -> bytes:
    return hashlib.sha256(f"{seed}:{purpose}:{source}:{index}".encode("utf-8")).digest()


def proportional_quotas(total: int, capacities: Mapping[str, int]) -> dict[str, int]:
    """Largest remainder, alphabetical tie-break, with exact integer totals."""
    capacity_total = sum(capacities.values())
    if total < 0 or total > capacity_total or capacity_total <= 0:
        raise ValueError("requested total exceeds available source capacity")
    names = sorted(capacities)
    floors = {name: total * capacities[name] // capacity_total for name in names}
    remainder = total - sum(floors.values())
    order = sorted(names, key=lambda name: (-(total * capacities[name] % capacity_total), name))
    for name in order[:remainder]:
        floors[name] += 1
    return floors


def selection_plan(seed: int = DEFAULT_SEED,
                   source_counts: Mapping[str, int] = SOURCE_COUNTS,
                   shard_sizes: Mapping[str, int] = SHARD_SIZES,
                   eligible_indices: Mapping[str, list[int]] | None = None,
                   ) -> dict[str, dict[str, list[int]]]:
    """Select once per source, then partition that fixed pool without replacement."""
    if sum(shard_sizes.values()) > sum(source_counts.values()):
        raise ValueError("shard sizes exceed source population")
    pool_quotas = proportional_quotas(sum(shard_sizes.values()), source_counts)
    shuffled = {}
    for source in sorted(source_counts):
        population = (range(source_counts[source]) if eligible_indices is None
                      else eligible_indices[source])
        if len(population) < pool_quotas[source]:
            raise ValueError(f"{source} has too few trainable samples for its fixed quota")
        selected = sorted(population,
                          key=lambda index: _stable_rank(seed, "pool", source, index)
                          )[:pool_quotas[source]]
        selected.sort(key=lambda index: _stable_rank(seed, "partition", source, index))
        shuffled[source] = selected
    remaining = dict(pool_quotas)
    offsets = {source: 0 for source in source_counts}
    plan = {}
    for shard, size in shard_sizes.items():
        quotas = proportional_quotas(size, remaining)
        plan[shard] = {}
        for source in sorted(source_counts):
            begin = offsets[source]
            end = begin + quotas[source]
            plan[shard][source] = shuffled[source][begin:end]
            offsets[source] = end
            remaining[source] -= quotas[source]
    if any(remaining.values()):
        raise AssertionError("pool partition left unassigned samples")
    identities = [(source, index) for by_source in plan.values()
                  for source, indices in by_source.items() for index in indices]
    if len(identities) != len(set(identities)):
        raise AssertionError("SFT shards overlap")
    return plan


def stable_sample_id(source: str, index: int) -> str:
    if source not in SOURCE_FILES or index < 0:
        raise ValueError("invalid SFT source/index")
    return f"{source}:{index}"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _image_path(source: str, reference: str) -> str:
    suffix = PurePosixPath(reference.replace("\\", "/")).suffix.lower() or ".img"
    digest = hashlib.sha256(f"{source}:{reference}".encode("utf-8")).hexdigest()[:20]
    return f"media/{source}/{digest}{suffix}"


def _zip_member(reference: str, names: set[str], basenames: Mapping[str, list[str]]) -> str:
    normalized = str(PurePosixPath(reference.replace("\\", "/"))).lstrip("./")
    candidates = (normalized, normalized.split("/", 1)[-1],
                  f"images/{PurePosixPath(normalized).name}")
    for candidate in candidates:
        if candidate in names:
            return candidate
    matches = basenames.get(PurePosixPath(normalized).name, [])
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot uniquely resolve image {reference!r} in source archive")


def materialize_images(records: list[dict[str, Any]], raw_dir: Path, output_dir: Path,
                       *, download_missing: bool = False) -> int:
    """Extract only selected images; no archive download unless explicitly requested."""
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_source.setdefault(record["_source"], []).append(record)
    extracted = 0
    for source, selected in by_source.items():
        archive_path = raw_dir / source / "images.zip"
        if not archive_path.is_file():
            if not download_missing:
                raise FileNotFoundError(f"missing {archive_path}; pass --download-images explicitly")
            from huggingface_hub import hf_hub_download

            archive_path = Path(hf_hub_download(
                repo_id=DATASET_ID, repo_type="dataset", revision=DATASET_REVISION,
                filename=f"{source}/images.zip", local_dir=raw_dir,
            ))
        with zipfile.ZipFile(archive_path) as archive:
            names = {item.filename for item in archive.infolist() if not item.is_dir()}
            basenames: dict[str, list[str]] = {}
            for name in names:
                basenames.setdefault(PurePosixPath(name).name, []).append(name)
            for record in selected:
                for reference, local in zip(record["_source_images"], record["images"]):
                    target = output_dir / local
                    if target.is_file():
                        continue
                    member = _zip_member(reference, names, basenames)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source_handle, target.open("wb") as target_handle:
                        shutil.copyfileobj(source_handle, target_handle)
                    extracted += 1
    return extracted


def prepare_sft_pool(raw_dir: str | Path, output_dir: str | Path,
                     *, seed: int = DEFAULT_SEED, extract_images: bool = False,
                     download_images: bool = False) -> dict[str, Any]:
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    if download_images and not extract_images:
        raise ValueError("--download-images requires --extract-images")
    eligible_indices = {}
    validity = {}
    source_files = {}
    for source, relative in SOURCE_FILES.items():
        path = raw_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned source JSON is missing: {path}")
        count = 0
        eligible = []
        for index, raw in enumerate(iter_json_array(path)):
            count += 1
            try:
                validate_raw_sample(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            eligible.append(index)
        if count != SOURCE_COUNTS[source]:
            raise ValueError(f"{source} has {count} records, expected pinned {SOURCE_COUNTS[source]}")
        eligible_indices[source] = eligible
        validity[source] = {"eligible": len(eligible), "excluded_invalid": count - len(eligible)}
        source_files[source] = {"path": relative, "sha256": sha256_file(path), "count": count}
    plan = selection_plan(seed, source_counts=SOURCE_COUNTS,
                          shard_sizes=SHARD_SIZES, eligible_indices=eligible_indices)
    wanted = {source: {index: shard for shard, by_source in plan.items()
                       for index in by_source[source]} for source in SOURCE_FILES}
    records: dict[str, list[dict[str, Any]]] = {shard: [] for shard in SHARD_SIZES}
    for source, relative in SOURCE_FILES.items():
        path = raw_dir / relative
        for index, raw in enumerate(iter_json_array(path)):
            shard = wanted[source].get(index)
            if shard is None:
                continue
            validate_raw_sample(raw)
            record = dict(raw)
            original_images = list(raw["images"])
            record.update({
                "_sample_id": stable_sample_id(source, index), "_source": source,
                "_source_index": index, "_source_file": relative,
                "_source_images": original_images,
                "_raw_sha256": hashlib.sha256(_json_bytes(raw)).hexdigest(),
                "images": [_image_path(source, image) for image in original_images],
            })
            records[shard].append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = [record for shard in SHARD_SIZES for record in records[shard]]
    if extract_images:
        materialize_images(all_records, raw_dir, output_dir,
                           download_missing=download_images)
    outputs = {}
    membership = []
    for shard, expected in SHARD_SIZES.items():
        selected = records[shard]
        if len(selected) != expected:
            raise AssertionError(f"{shard}: got {len(selected)}, expected {expected}")
        selected.sort(key=lambda record: (record["_source"], record["_source_index"]))
        selected.sort(key=lambda record: _stable_rank(
            seed, f"order:{shard}", record["_source"], record["_source_index"]))
        path = output_dir / f"{shard}.json"
        path.write_bytes(_json_bytes(selected))
        outputs[shard] = {
            "path": path.name, "count": len(selected), "sha256": sha256_file(path),
            "source_counts": dict(sorted(Counter(item["_source"] for item in selected).items())),
        }
        membership.extend({
            "sample_id": item["_sample_id"], "source": item["_source"],
            "source_index": item["_source_index"], "raw_sha256": item["_raw_sha256"],
            "shard": shard,
        } for item in selected)
    ids = [item["sample_id"] for item in membership]
    disjoint = len(ids) == len(set(ids)) == POOL_SIZE
    if not disjoint:
        raise AssertionError("SFT pool sample identities are not disjoint")
    images_ready = all((output_dir / image).is_file()
                       for record in all_records for image in record["images"])
    manifest = {
        "version": 1, "selection_algorithm": "sha256-rank-v1",
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "seed": seed, "total_selected_count": len(all_records),
        "source_population_counts": dict(SOURCE_COUNTS), "source_validity": validity,
        "source_files": source_files,
        "pool_source_counts": dict(sorted(Counter(item["_source"] for item in all_records).items())),
        "shards": outputs, "membership": membership,
        "disjointness_verified": disjoint,
    }
    (output_dir / "manifest.json").write_bytes(_json_bytes(manifest))
    (output_dir / "media_status.json").write_bytes(_json_bytes({"images_ready": images_ready}))
    return {**manifest, "images_ready": images_ready}


def load_sft_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("selection_algorithm") != "sha256-rank-v1":
        raise ValueError("SFT selection algorithm/version mismatch")
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("SFT dataset identity/revision mismatch")
    if manifest.get("total_selected_count") != POOL_SIZE or manifest.get("disjointness_verified") is not True:
        raise ValueError("SFT pool count/disjointness mismatch")
    ids = [row["sample_id"] for row in manifest["membership"]]
    if len(ids) != len(set(ids)) or len(ids) != POOL_SIZE:
        raise ValueError("SFT membership is duplicated or incomplete")
    for row in manifest["membership"]:
        if (row["sample_id"] != stable_sample_id(row["source"], row["source_index"])
                or row["shard"] not in SHARD_SIZES):
            raise ValueError("SFT membership source/index/shard mismatch")
    if dict(Counter(row["source"] for row in manifest["membership"])) != manifest["pool_source_counts"]:
        raise ValueError("SFT pool source counts do not match membership")
    for shard, expected in SHARD_SIZES.items():
        entry = manifest["shards"][shard]
        file = path.parent / entry["path"]
        if entry["count"] != expected or sha256_file(file) != entry["sha256"]:
            raise ValueError(f"SFT shard identity mismatch: {shard}")
        members = [row for row in manifest["membership"] if row["shard"] == shard]
        if len(members) != expected or dict(Counter(row["source"] for row in members)) != entry["source_counts"]:
            raise ValueError(f"SFT shard membership counts mismatch: {shard}")
        expected_rows = {(row["sample_id"], row["source"], row["source_index"],
                          row["raw_sha256"]) for row in members}
        actual_rows = [(row["_sample_id"], row["_source"], row["_source_index"],
                        row["_raw_sha256"]) for row in iter_json_array(file)]
        if len(actual_rows) != expected or set(actual_rows) != expected_rows:
            raise ValueError(f"SFT shard records do not match membership: {shard}")
    return manifest
