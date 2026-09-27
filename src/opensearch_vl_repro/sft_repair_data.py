"""Frozen-source, image-verified targeted-repair ablation datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .agent.reliability import image_sha256
from .data import validate_raw_sample
from .eval_subset import canonical_json_sha256
from .evaluation.eval300 import build_eval300_plan
from .inference.eval_reader import read_eval_samples_by_ids
from .sft_main_data import (_image_path, canonicalize_tool_declarations,
                            load_data_quality_exclusions, load_sft_manifest,
                            materialize_images, require_data_quality_exclusions)
from .sft_preflight import normalized_question, sample_questions
from .sft_protocol_diagnostics import (CORRECTED_POOL_MANIFEST_SHA256,
                                       parsed_calls)
from .sft_tool_audit import DATASET_ID, DATASET_REVISION, sha256_file
from .tool_protocol_dev import (candidate_metadata, source_zip_image_hasher,
                                validate_dev_manifest)


REPAIR_VERSION = "targeted-repair-data-v1"
SELECTION_VERSION = "sha256-rank-stratified-image-verified-v1"
REPAIR_SEED = 20260927
TARGETS = {
    "argument_only": {"img_1": 180, "img_2": 70, "img_3_or_later": 50},
    "full_tool_call": {"image_search": 220, "other_image_tool": 40, "no_tool": 40},
}
OTHER_IMAGE_TOOLS = frozenset(("layout_parsing", "crop", "sharpen",
                               "super_resolution", "perspective_correct"))


def _rank(seed: int, mode: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:sft-repair:{mode}:{sample_id}".encode()).hexdigest()


def repair_category(record: dict[str, Any], mode: str) -> tuple[str, dict[str, Any]] | None:
    """Choose auditable original target turns; never rewrite expert text."""
    calls = parsed_calls(record)
    all_image_calls = [call for call in calls if call["name"] == "image_search"]
    image_calls = [call for call in calls if call["name"] == "image_search"
                   and isinstance(call["arguments"].get("url"), str)
                   and call["arguments"]["url"].startswith("img_")]
    if mode == "argument_only":
        if (not image_calls or len(image_calls) != len(all_image_calls)
                or any(set(call["arguments"]) != {"url"} for call in image_calls)):
            return None
        ids = [int(call["arguments"]["url"][4:]) for call in image_calls
               if call["arguments"]["url"][4:].isdigit()]
        if len(ids) != len(image_calls):
            return None
        category = ("img_3_or_later" if any(value >= 3 for value in ids) else
                    "img_2" if 2 in ids else "img_1")
        return category, {"target_turn_index": image_calls[0]["turn_index"],
                          "target_tool": "image_search", "target_image_ids": [
                              call["arguments"]["url"] for call in image_calls],
                          "target_call_policy": "all_valid_image_search_calls"}
    if mode != "full_tool_call":
        raise ValueError(f"unknown repair mode: {mode}")
    if image_calls:
        target = image_calls[0]
        category = "image_search"
    elif any(call["name"] in OTHER_IMAGE_TOOLS for call in calls):
        target = next(call for call in calls if call["name"] in OTHER_IMAGE_TOOLS)
        category = "other_image_tool"
    elif not calls:
        if not any(re.search(r"<response>.*?</response>", turn["value"], re.S)
                   for turn in record["conversations"] if turn["from"] == "gpt"):
            return None
        target = {"turn_index": max(index for index, turn in enumerate(record["conversations"])
                                    if turn["from"] == "gpt" and re.search(
                                        r"<response>.*?</response>", turn["value"], re.S)),
                  "name": None}
        category = "no_tool"
    else:
        return None
    return category, {"target_turn_index": target["turn_index"],
                      "target_tool": target["name"],
                      "target_image_ids": ([target["arguments"]["url"]]
                                           if target["name"] == "image_search" else []),
                      "target_call_policy": "first_category_call"}


def select_repair_records(candidates: list[dict[str, Any]], *, mode: str,
                          image_hashes: Callable[[dict[str, Any]], list[str]],
                          forbidden_hashes: set[str], seed: int = REPAIR_SEED,
                          targets: dict[str, int] | None = None,
                          ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    goals = targets or TARGETS[mode]
    ranked = []
    for item in candidates:
        chosen = repair_category(item["record"], mode)
        if chosen:
            category, target = chosen
            ranked.append({**item, "repair_category": category, **target})
    ranked.sort(key=lambda item: (_rank(seed, mode, item["sample_id"]), item["sample_id"]))
    available = Counter(item["repair_category"] for item in ranked)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    overlap_skips: set[str] = set()

    def choose(category: str | None) -> bool:
        for item in ranked:
            identity = item["sample_id"]
            if identity in used or identity in overlap_skips:
                continue
            if category is not None and item["repair_category"] != category:
                continue
            hashes = image_hashes(item)
            if not hashes:
                raise ValueError(f"repair candidate has no decodable image: {identity}")
            if set(hashes) & forbidden_hashes:
                overlap_skips.add(identity)
                continue
            selected.append({**item, "image_sha256": hashes})
            used.add(identity)
            return True
        return False

    unmet = Counter()
    for category, count in goals.items():
        for _ in range(count):
            if not choose(category):
                unmet[category] += 1
    while len(selected) < sum(goals.values()) and choose(None):
        pass
    if len(selected) != sum(goals.values()):
        raise ValueError(f"only {len(selected)}/{sum(goals.values())} clean repair candidates")
    return selected, {"available_category_counts": dict(sorted(available.items())),
                      "selected_category_counts": dict(sorted(Counter(
                          item["repair_category"] for item in selected).items())),
                      "unmet_requested_slots": dict(sorted(unmet.items())),
                      "image_overlap_candidate_skip_count": len(overlap_skips)}


def filter_repair_candidates(candidates: list[dict[str, Any]], *,
                             excluded_ids: set[str], forbidden_questions: set[str],
                             ) -> list[dict[str, Any]]:
    """Exclude Dev50, Eval/Dev question duplicates before image verification."""
    return [item for item in candidates if item["sample_id"] not in excluded_ids
            and not sample_questions(item["record"]) & forbidden_questions]


def build_repair_datasets(*, raw_dir: Path, pool_manifest_path: Path,
                          eval_path: Path, dev_dir: Path, output_root: Path,
                          seed: int = REPAIR_SEED,
                          image_hasher: Callable[[dict[str, Any]], list[str]] | None = None,
                          ) -> dict[str, dict[str, Any]]:
    """Build both modes only after all pinned provenance and overlap checks pass."""
    pool_sha = sha256_file(pool_manifest_path)
    if pool_sha != CORRECTED_POOL_MANIFEST_SHA256:
        raise ValueError(f"corrected pool SHA mismatch: {pool_sha}")
    pool = load_sft_manifest(pool_manifest_path)
    require_data_quality_exclusions(pool)
    dev_ids_path = dev_dir / "ids.json"
    dev_manifest_path = dev_dir / "tool_protocol_dev50_manifest.json"
    dev_ids = json.loads(dev_ids_path.read_text(encoding="utf-8"))
    dev_manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
    validate_dev_manifest(dev_ids, dev_manifest)
    dev_manifest_sha = sha256_file(dev_manifest_path)
    eval_plan = build_eval300_plan(eval_path)
    dev30_path = eval_path.parent / "dev30/dev30_ids.json"
    dev30_entries = {(str(row["benchmark"]), str(row["sample_id"]))
                     for row in json.loads(dev30_path.read_text(encoding="utf-8"))}
    if not dev30_entries.issubset(set(eval_plan.entries)):
        raise ValueError("Dev30 is no longer a subset of frozen Eval300")
    eval_samples = read_eval_samples_by_ids(eval_path, list(eval_plan.entries))
    eval_questions = {normalized_question(sample.question) for sample in eval_samples}
    eval_hashes = {image_sha256(image) for sample in eval_samples for image in sample.images}
    frozen = load_data_quality_exclusions()
    candidates, scan = candidate_metadata(raw_dir, pool, eval_questions, frozen)
    by_id = {item["sample_id"]: item for item in candidates}
    if not set(dev_ids).issubset(by_id):
        raise ValueError("Dev50 IDs missing from clean pinned source candidates")
    dev_questions = set().union(*(sample_questions(by_id[identity]["record"])
                                  for identity in dev_ids))
    owned_hasher = image_hasher is None
    hasher = image_hasher or source_zip_image_hasher(raw_dir)
    try:
        dev_hashes = {value for identity in dev_ids
                      for value in hasher(by_id[identity])}
        forbidden = eval_hashes | dev_hashes
        dev_set = set(dev_ids)
        candidates = filter_repair_candidates(candidates, excluded_ids=dev_set,
                                               forbidden_questions=dev_questions)
        plans = {mode: select_repair_records(
            candidates, mode=mode, image_hashes=hasher,
            forbidden_hashes=forbidden, seed=seed) for mode in TARGETS}
    finally:
        if owned_hasher:
            hasher.close()  # type: ignore[attr-defined]
    # Prepare deterministic records before touching the output tree.
    prepared = {}
    for mode, (selected, selection) in plans.items():
        name = "r1_argument_only" if mode == "argument_only" else "r2_full_tool_call"
        output_dir = output_root / name
        records = []
        membership = []
        for item in selected:
            raw = item["record"]
            validate_raw_sample(raw)
            record = canonicalize_tool_declarations(raw)
            references = list(raw["images"])
            record.update({"_sample_id": item["sample_id"], "_source": item["source"],
                           "_source_index": item["source_index"],
                           "_source_images": references,
                           "_repair_category": item["repair_category"],
                           "_repair_target_turn_index": item["target_turn_index"],
                           "_repair_target_tool": item["target_tool"],
                           "_repair_target_call_policy": item["target_call_policy"],
                           "images": [_image_path(item["source"], ref) for ref in references]})
            records.append(record)
            membership.append({"sample_id": item["sample_id"], "source": item["source"],
                               "source_index": item["source_index"],
                               "category": item["repair_category"],
                               "target_turn_index": item["target_turn_index"],
                               "target_tool": item["target_tool"],
                               "target_image_ids": item.get("target_image_ids", []),
                               "image_sha256": item["image_sha256"]})
        ids = {row["sample_id"] for row in membership}
        if ids & (set(dev_ids) | set(frozen) | {row["sample_id"] for row in pool["membership"]}):
            raise AssertionError("repair membership overlaps forbidden source IDs")
        if any(sample_questions(record) & (eval_questions | dev_questions) for record in records):
            raise AssertionError("repair question overlap")
        if any(set(row["image_sha256"]) & forbidden for row in membership):
            raise AssertionError("repair image overlap")
        payload = (json.dumps(records, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n").encode("utf-8")
        prepared[mode] = (output_dir, records, payload, membership, selection)

    results = {}
    for mode, (output_dir, records, payload, membership, selection) in prepared.items():
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite repair dataset: {output_dir}")
        output_dir.mkdir(parents=True)
        materialize_images(records, raw_dir, output_dir, download_missing=False)
        dataset_path = output_dir / "repair.json"
        dataset_path.write_bytes(payload)
        image_targets = Counter(image_id for row in membership for image_id in row["target_image_ids"])
        manifest = {
            "repair_version": REPAIR_VERSION, "repair_mode": mode,
            "selection_algorithm": SELECTION_VERSION, "seed": seed,
            "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
            "corrected_pool_manifest_sha256": pool_sha,
            "dev50_manifest_sha256": dev_manifest_sha,
            "dev50_ids_sha256": canonical_json_sha256(dev_ids),
            "dev50_sample_count": len(dev_ids), "dev30_sample_count": len(dev30_entries),
            "frozen_exclusion_population_count": len(frozen),
            "eval300_dataset_sha256": eval_plan.dataset_sha256,
            "sample_count": len(records), "source_counts": dict(sorted(Counter(
                row["source"] for row in membership).items())),
            "protocol_category_counts": selection["selected_category_counts"],
            "target_tool_counts": dict(sorted(Counter(
                row["target_tool"] or "no_tool" for row in membership).items())),
            "image_target_counts": dict(sorted(image_targets.items())),
            "eval300_id_overlap_count": 0, "eval300_question_overlap_count": 0,
            "eval300_image_overlap_count": 0, "dev30_overlap_count": 0,
            "dev50_id_overlap_count": 0, "dev50_question_overlap_count": 0,
            "dev50_image_overlap_count": 0, "sft8k_overlap_count": 0,
            "frozen_exclusion_overlap_count": 0, "image_contract_bad_count": 0,
            "repair_json_sha256": hashlib.sha256(payload).hexdigest(),
            "membership_sha256": canonical_json_sha256(membership),
            "membership": membership, "selection": selection, "source_scan": scan,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results[mode] = manifest
    return results


def validate_repair_manifest(dataset_path: Path, manifest_path: Path, *,
                             mode: str | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("repair_version") != REPAIR_VERSION
            or manifest.get("selection_algorithm") != SELECTION_VERSION
            or manifest.get("dataset_id") != DATASET_ID
            or manifest.get("dataset_revision") != DATASET_REVISION
            or manifest.get("corrected_pool_manifest_sha256") != CORRECTED_POOL_MANIFEST_SHA256
            or manifest.get("repair_mode") not in TARGETS
            or (mode is not None and manifest["repair_mode"] != mode)):
        raise ValueError("repair manifest provenance/mode mismatch")
    if sha256_file(dataset_path) != manifest.get("repair_json_sha256"):
        raise ValueError("repair dataset checksum mismatch")
    members = manifest.get("membership", [])
    if (len(members) != manifest.get("sample_count")
            or len({row["sample_id"] for row in members}) != len(members)
            or canonical_json_sha256(members) != manifest.get("membership_sha256")):
        raise ValueError("repair membership checksum mismatch")
    for key in ("eval300_id_overlap_count", "eval300_question_overlap_count",
                "eval300_image_overlap_count", "dev30_overlap_count",
                "dev50_id_overlap_count", "dev50_question_overlap_count",
                "dev50_image_overlap_count", "sft8k_overlap_count",
                "frozen_exclusion_overlap_count", "image_contract_bad_count"):
        if manifest.get(key) != 0:
            raise ValueError(f"repair dataset failed isolation: {key}")
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if [row["_sample_id"] for row in records] != [row["sample_id"] for row in members]:
        raise ValueError("repair records and membership diverged")
    return manifest
