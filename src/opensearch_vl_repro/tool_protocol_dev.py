"""Deterministic, image-verified protocol dev set outside SFT and Eval-300."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from .agent.reliability import image_sha256
from .agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from .data import validate_raw_sample
from .eval_subset import canonical_json_sha256
from .evaluation.eval300 import FROZEN_EVAL300_SHA256, build_eval300_plan
from .inference.eval_reader import read_eval_samples_by_ids
from .sft_image_grounding import audit_raw_image_contract
from .sft_main_data import (_zip_member, load_data_quality_exclusions, load_sft_manifest,
                            require_data_quality_exclusions)
from .sft_preflight import normalized_question, sample_questions
from .sft_protocol_diagnostics import parsed_calls, protocol_tags
from .sft_tool_audit import (DATASET_ID, DATASET_REVISION, SOURCE_FILES,
                             iter_json_array, sha256_file)


DEV_SEED = 20260927
DEV_SIZE = 50
CATEGORY_TARGETS = {
    "image_search_img_1": 12, "image_search_derived": 5,
    "layout_parsing": 8, "crop": 7, "other_image_tool": 5,
    "no_tool": 6, "non_image_search_visual": 3, "multi_tool": 4,
}


def require_unchanged_dev_ids(ids: list[str], historical_ids: list[str]) -> None:
    """The v3 contract changes tools, never the frozen ordered Dev50 membership."""
    if len(ids) != DEV_SIZE or ids != historical_ids:
        raise ValueError("v3 Dev50 ordered IDs differ from historical Dev50")


def _rank(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:tool-protocol-dev50:{identity}".encode()).hexdigest()


def candidate_metadata(raw_dir: Path, pool_manifest: dict[str, Any],
                       eval_questions: set[str], frozen: dict[str, str],
                       ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Stream pinned JSON; no image bytes and no source mutation."""
    membership = {row["sample_id"] for row in pool_manifest["membership"]}
    excluded = set(frozen)
    counts = Counter()
    candidates = []
    for source, relative in SOURCE_FILES.items():
        path = raw_dir / relative
        if sha256_file(path) != pool_manifest["source_files"][source]["sha256"]:
            raise ValueError(f"pinned source hash mismatch: {source}")
        for index, raw in enumerate(iter_json_array(path)):
            sample_id = f"{source}:{index}"
            counts["total_source_records"] += 1
            if sample_id in membership or sample_id in excluded:
                counts["excluded_membership_or_frozen"] += 1
                continue
            try:
                validate_raw_sample(raw)
            except (TypeError, ValueError):
                counts["invalid_raw"] += 1
                continue
            if sample_questions(raw) & eval_questions:
                counts["eval_question_overlap"] += 1
                continue
            if not audit_raw_image_contract(raw, sample_id=sample_id)["passed"]:
                counts["image_contract_bad"] += 1
                continue
            tags = protocol_tags(raw)
            candidates.append({"sample_id": sample_id, "source": source,
                               "source_index": index, "record": raw,
                               "tags": sorted(tags)})
            counts["clean_candidates_before_image_hash"] += 1
    return candidates, dict(counts)


def select_dev_records(candidates: Iterable[dict[str, Any]], *,
                       image_hashes: Callable[[dict[str, Any]], list[str]],
                       forbidden_image_hashes: set[str], seed: int = DEV_SEED,
                       size: int = DEV_SIZE,
                       targets: dict[str, int] = CATEGORY_TARGETS,
                       ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy stratified selection, deterministic even if input row order changes."""
    ranked = sorted(candidates, key=lambda item: (_rank(seed, item["sample_id"]), item["sample_id"]))
    available = Counter(tag for item in ranked for tag in item["tags"])
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    source_counts = Counter()
    overlap_skips: set[str] = set()
    hash_cache: dict[str, list[str]] = {}

    def eligible(item: dict[str, Any]) -> bool:
        sample_id = item["sample_id"]
        if sample_id in selected_ids or sample_id in overlap_skips:
            return False
        if sample_id not in hash_cache:
            hash_cache[sample_id] = image_hashes(item)
            if not hash_cache[sample_id]:
                raise ValueError(f"candidate has no decodable images: {sample_id}")
        if set(hash_cache[sample_id]) & forbidden_image_hashes:
            overlap_skips.add(sample_id)
            return False
        return True

    def choose(tag: str | None) -> bool:
        choices = [item for item in ranked if tag is None or tag in item["tags"]]
        choices.sort(key=lambda item: (source_counts[item["source"]],
                                       _rank(seed, item["sample_id"]), item["sample_id"]))
        for item in choices:
            if eligible(item):
                selected.append({**item, "selection_category": tag or "fill"})
                selected_ids.add(item["sample_id"])
                source_counts[item["source"]] += 1
                return True
        return False

    unmet = Counter()
    for tag, requested in targets.items():
        for _ in range(requested):
            if len(selected) >= size:
                break
            if not choose(tag):
                unmet[tag] += 1
    while len(selected) < size and choose(None):
        pass
    if len(selected) != size:
        raise ValueError(f"only {len(selected)} image-verified clean candidates for dev{size}")
    return selected, {"available_tag_counts": dict(sorted(available.items())),
                      "selected_tag_counts": dict(sorted(Counter(
                          tag for item in selected for tag in item["tags"]).items())),
                      "unmet_requested_slots": dict(sorted(unmet.items())),
                      "image_overlap_candidate_skip_count": len(overlap_skips),
                      "source_counts": dict(sorted(source_counts.items()))}


def source_zip_image_hasher(raw_dir: Path) -> Callable[[dict[str, Any]], list[str]]:
    """Read source ZIP members lazily; never extract/copy images into Git."""
    stack = ExitStack()
    archives: dict[str, tuple[zipfile.ZipFile, set[str], dict[str, list[str]]]] = {}
    for source in SOURCE_FILES:
        path = raw_dir / source / "images.zip"
        if not path.is_file():
            stack.close()
            raise FileNotFoundError(f"source image archive missing: {path}")
        archive = stack.enter_context(zipfile.ZipFile(path))
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        basenames: dict[str, list[str]] = {}
        for name in names:
            basenames.setdefault(Path(name).name, []).append(name)
        archives[source] = archive, names, basenames

    def hashes(item: dict[str, Any]) -> list[str]:
        archive, names, basenames = archives[item["source"]]
        result = []
        for reference in item["record"]["images"]:
            member = _zip_member(reference, names, basenames)
            with archive.open(member) as handle:
                payload = handle.read()
            with Image.open(io.BytesIO(payload)) as image:
                result.append(image_sha256(image.convert("RGB")))
        return result

    hashes.close = stack.close  # type: ignore[attr-defined]
    return hashes


def build_dev_manifest(raw_dir: Path, pool_manifest_path: Path, eval_path: Path,
                       *, image_hasher: Callable[[dict[str, Any]], list[str]] | None = None,
                       seed: int = DEV_SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_sha = sha256_file(pool_manifest_path)
    pool = load_sft_manifest(pool_manifest_path)
    if pool.get("runtime_tool_protocol_version") != RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION:
        raise ValueError("Dev50 requires a v3 image_id SFT pool")
    require_data_quality_exclusions(pool)
    if (pool.get("dataset_id") != DATASET_ID or pool.get("dataset_revision") != DATASET_REVISION):
        raise ValueError("SFT pool does not use the pinned source revision")
    eval_plan = build_eval300_plan(eval_path)
    dev30_ids_path = eval_path.parent / "dev30/dev30_ids.json"
    if not dev30_ids_path.is_file():
        raise FileNotFoundError(f"Dev30 ID manifest is required for subset validation: {dev30_ids_path}")
    dev30_entries = {(str(row["benchmark"]), str(row["sample_id"]))
                     for row in json.loads(dev30_ids_path.read_text(encoding="utf-8"))}
    if not dev30_entries.issubset(set(eval_plan.entries)):
        raise ValueError("Dev30 contains IDs outside frozen Eval-300")
    eval_samples = read_eval_samples_by_ids(eval_path, list(eval_plan.entries))
    eval_questions = {normalized_question(sample.question) for sample in eval_samples}
    eval_hashes = {image_sha256(image) for sample in eval_samples for image in sample.images}
    frozen = load_data_quality_exclusions()
    candidates, scan = candidate_metadata(raw_dir, pool, eval_questions, frozen)
    owned_hasher = image_hasher is None
    hasher = image_hasher or source_zip_image_hasher(raw_dir)
    try:
        selected, selection = select_dev_records(
            candidates, image_hashes=hasher, forbidden_image_hashes=eval_hashes, seed=seed)
    finally:
        if owned_hasher:
            hasher.close()  # type: ignore[attr-defined]
    ids = [item["sample_id"] for item in selected]
    metadata = [{"sample_id": item["sample_id"], "source": item["source"],
                 "source_index": item["source_index"], "protocol_tags": item["tags"],
                 "expected_protocol_category": item["selection_category"],
                 "reference_tool_sequence": [call["name"] for call in parsed_calls(item["record"])],
                 "expected_tool_label": None}
                for item in selected]
    membership = {row["sample_id"] for row in pool["membership"]}
    if set(ids) & (membership | set(frozen)):
        raise AssertionError("dev set overlaps formal SFT membership or frozen exclusions")
    manifest = {"version": 2, "purpose": "tool-protocol-regression-not-final-QA",
                "runtime_tool_protocol_version": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
                "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
                "seed": seed, "count": len(ids), "corrected_pool_manifest_sha256": pool_sha,
                "eval300_dataset_sha256": eval_plan.dataset_sha256,
                "dev30_subset_of_eval300": True,
                "source_file_sha256": {source: row["sha256"] for source, row in pool["source_files"].items()},
                "ids_sha256": canonical_json_sha256(ids),
                "sample_metadata_sha256": canonical_json_sha256(metadata),
                "eval300_question_overlap_count": 0, "eval300_image_overlap_count": 0,
                "sft_membership_overlap_count": 0, "frozen_exclusion_overlap_count": 0,
                "image_contract_bad_count": 0,
                "source_scan": scan, "selection": selection, "samples": metadata}
    return ids, manifest


def validate_dev_manifest(ids: list[str], manifest: dict[str, Any], *,
                          pool_manifest_path: Path | None = None) -> None:
    if (manifest.get("version") != 2
            or manifest.get("runtime_tool_protocol_version") != RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
            or manifest.get("dataset_id") != DATASET_ID
            or manifest.get("dataset_revision") != DATASET_REVISION
            or not isinstance(manifest.get("corrected_pool_manifest_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", manifest["corrected_pool_manifest_sha256"]) is None
            or manifest.get("eval300_dataset_sha256") != FROZEN_EVAL300_SHA256
            or manifest.get("dev30_subset_of_eval300") is not True):
        raise ValueError("tool-protocol dev manifest pinned provenance mismatch")
    if pool_manifest_path is not None:
        if sha256_file(pool_manifest_path) != manifest["corrected_pool_manifest_sha256"]:
            raise ValueError("Dev50 belongs to a different v3 SFT pool manifest")
        load_sft_manifest(pool_manifest_path)
    if (len(ids) != manifest.get("count") or len(ids) != len(set(ids))
            or ids != [row["sample_id"] for row in manifest.get("samples", [])]
            or manifest.get("ids_sha256") != canonical_json_sha256(ids)
            or manifest.get("sample_metadata_sha256") != canonical_json_sha256(manifest["samples"])):
        raise ValueError("tool-protocol dev manifest checksum or ordered IDs mismatch")
    for key in ("eval300_question_overlap_count", "eval300_image_overlap_count",
                "sft_membership_overlap_count", "frozen_exclusion_overlap_count",
                "image_contract_bad_count"):
        if manifest.get(key) != 0:
            raise ValueError(f"tool-protocol dev manifest failed {key}")


def load_dev_source_samples(ids: list[str], manifest: dict[str, Any], raw_dir: Path,
                            ) -> list[tuple[str, str, list[Image.Image]]]:
    """Load only initial source images from local ZIPs in fixed dev-ID order."""
    validate_dev_manifest(ids, manifest)
    wanted = set(ids)
    found: dict[str, tuple[str, str, list[Image.Image]]] = {}
    with ExitStack() as stack:
        for source, relative in SOURCE_FILES.items():
            path = raw_dir / relative
            if sha256_file(path) != manifest["source_file_sha256"][source]:
                raise ValueError(f"pinned source hash mismatch: {source}")
            archive = stack.enter_context(zipfile.ZipFile(raw_dir / source / "images.zip"))
            names = {item.filename for item in archive.infolist() if not item.is_dir()}
            basenames: dict[str, list[str]] = {}
            for name in names:
                basenames.setdefault(Path(name).name, []).append(name)
            for index, record in enumerate(iter_json_array(path)):
                sample_id = f"{source}:{index}"
                if sample_id not in wanted:
                    continue
                first = record["conversations"][0]["value"]
                initial_count = first.count("<image>")
                images = []
                for reference in record["images"][:initial_count]:
                    member = _zip_member(reference, names, basenames)
                    with archive.open(member) as handle:
                        payload = handle.read()
                    with Image.open(io.BytesIO(payload)) as image:
                        images.append(image.convert("RGB").copy())
                if not images:
                    raise ValueError(f"dev sample has no initial images: {sample_id}")
                found[sample_id] = (sample_id, first.replace("<image>", " ").strip(), images)
    if set(found) != wanted:
        raise ValueError(f"dev samples missing from pinned sources: {sorted(wanted-set(found))}")
    return [found[sample_id] for sample_id in ids]
