"""Frozen-source, image-verified targeted-repair ablation datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .agent.reliability import image_sha256
from .data import (canonicalize_sft_source_system, load_json_records,
                   validate_raw_sample)
from .eval_subset import canonical_json_sha256
from .evaluation.eval300 import build_eval300_plan
from .inference.eval_reader import read_eval_samples_by_ids
from .sft_main_data import (_image_path, canonicalize_tool_declarations,
                            load_data_quality_exclusions, load_sft_manifest,
                            materialize_images, require_data_quality_exclusions)
from .sft_image_grounding import audit_raw_image_contract
from .sft_preflight import normalized_question, sample_questions
from .sft_protocol_diagnostics import (CORRECTED_POOL_MANIFEST_SHA256,
                                       parsed_calls)
from .sft_repair_mask import MASK_VERSIONS, TOOL_BLOCK, target_character_spans
from .sft_tool_audit import DATASET_ID, DATASET_REVISION, iter_json_array, sha256_file
from .tool_protocol_dev import (candidate_metadata, source_zip_image_hasher,
                                validate_dev_manifest)


REPAIR_VERSION = "targeted-repair-data-v1"
SELECTION_VERSION = "sha256-rank-stratified-image-verified-v1"
REPAIR_SEED = 20260927
R3_SIZE = 600
R3_NAME = "r3_argument_only_derived"
R3_SELECTION_VERSION = "corrected-8k-derived-priority-sha256-v1"
R3_CATEGORIES = ("img_3_or_later", "img_2", "img_1")
TARGETS = {
    "argument_only": {"img_1": 180, "img_2": 70, "img_3_or_later": 50},
    "full_tool_call": {"image_search": 220, "other_image_tool": 80},
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


def r3_candidate_diagnostic(record: dict[str, Any]
                            ) -> tuple[tuple[str, dict[str, Any]] | None, str | None]:
    """Reuse R1 targets and explain every CPU-only legality rejection."""
    for turn in record["conversations"]:
        if turn["from"] != "gpt":
            continue
        for block in TOOL_BLOCK.finditer(turn["value"]):
            if not re.search(r'"name"\s*:\s*"image_search"', block.group()):
                continue
            try:
                parsed = json.loads(block.group()[len("<tool_call>"):-len("</tool_call>")])
            except (TypeError, ValueError):
                return None, "malformed_image_search_block"
            args = parsed.get("arguments") if isinstance(parsed, dict) else None
            if (not isinstance(args, dict) or set(args) != {"url"}
                    or not isinstance(args["url"], str)
                    or re.fullmatch(r"img_[1-9][0-9]*", args["url"]) is None):
                return None, "non_runtime_image_search_argument"
    if not audit_raw_image_contract(record)["passed"]:
        return None, "image_contract_bad"
    try:
        canonicalize_sft_source_system(record.get("system") or "")
    except ValueError:
        return None, "legacy_direct_url_system"
    chosen = repair_category(record, "argument_only")
    if chosen is None:
        return None, "no_valid_image_search_target"
    category, target = chosen
    try:
        spans = target_character_spans({**record, "_repair_target_turn_index":
                                        target["target_turn_index"], "_repair_target_tool":
                                        "image_search"}, "argument_only")
    except ValueError:
        return None, "unmaskable_argument"
    ids = [call["arguments"]["url"] for call in parsed_calls(record)
           if call["name"] == "image_search"]
    if len(spans) != len(ids) or ids != target["target_image_ids"]:
        return None, "target_occurrence_mismatch"
    return (category, target), None


def r3_candidate(record: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    return r3_candidate_diagnostic(record)[0]


def select_r3_records(candidates: list[dict[str, Any]], *,
                      image_hashes: Callable[[dict[str, Any]], list[str]],
                      forbidden_hashes: set[str], seed: int = REPAIR_SEED,
                      size: int = R3_SIZE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exhaust legal derived-ID categories before filling with img_1."""
    if size <= 0:
        raise ValueError("R3 size must be positive")
    ranked = sorted(candidates, key=lambda item: (
        R3_CATEGORIES.index(item["repair_category"]),
        _rank(seed, "r3", item["sample_id"]), item["sample_id"]))
    available = Counter(item["repair_category"] for item in ranked)
    selected = []
    rejects = Counter()
    for item in ranked:
        if len(selected) == size:
            break
        hashes = image_hashes(item)
        if not hashes:
            rejects["missing_decodable_image"] += 1
            continue
        if set(hashes) & forbidden_hashes:
            rejects["eval_or_dev_image_overlap"] += 1
            continue
        selected.append({**item, "image_sha256": hashes})
    if len(selected) != size:
        raise ValueError(f"only {len(selected)}/{size} legal, image-disjoint R3 candidates")
    return selected, {
        "available_category_counts": dict(sorted(available.items())),
        "selected_category_counts": dict(sorted(Counter(
            item["repair_category"] for item in selected).items())),
        "selection_rejections": dict(sorted(rejects.items())),
        "priority_order": list(R3_CATEGORIES),
    }


def validate_r3_selection_membership(selected: list[dict[str, Any]], *,
                                     pool_ids: set[str], forbidden_ids: set[str],
                                     forbidden_hashes: set[str],
                                     size: int = R3_SIZE) -> None:
    ids = [row["sample_id"] for row in selected]
    if (len(ids) != size or len(set(ids)) != size
            or not set(ids).issubset(pool_ids) or set(ids) & forbidden_ids
            or any(set(row["image_sha256"]) & forbidden_hashes for row in selected)):
        raise ValueError("R3 selection is incomplete, duplicated, or overlaps forbidden data")


def build_repair_datasets(*, raw_dir: Path, pool_manifest_path: Path,
                          eval_path: Path, dev_dir: Path, output_root: Path,
                          seed: int = REPAIR_SEED,
                          modes: tuple[str, ...] = tuple(TARGETS),
                          image_hasher: Callable[[dict[str, Any]], list[str]] | None = None,
                          ) -> dict[str, dict[str, Any]]:
    """Build both modes only after all pinned provenance and overlap checks pass."""
    if not modes or any(mode not in TARGETS for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("repair modes must be a non-empty unique subset of R1/R2")
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
            forbidden_hashes=forbidden, seed=seed) for mode in modes}
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
                row["target_tool"] for row in membership).items())),
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
        if mode == "full_tool_call":
            manifest["requested_category_targets"] = dict(TARGETS[mode])
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results[mode] = manifest
    return results


def build_r3_dataset(*, raw_dir: Path, pool_manifest_path: Path,
                     eval_path: Path, dev_dir: Path, output_root: Path,
                     seed: int = REPAIR_SEED,
                     image_hasher: Callable[[dict[str, Any]], list[str]] | None = None,
                     ) -> dict[str, Any]:
    """Build R3 solely from the pinned corrected 8k, never from R1/R2 or Dev50."""
    output_dir = output_root / R3_NAME
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite R3 dataset: {output_dir}")
    pool_sha = sha256_file(pool_manifest_path)
    if pool_sha != CORRECTED_POOL_MANIFEST_SHA256:
        raise ValueError("R3 requires the exact corrected SFT pool")
    pool = load_sft_manifest(pool_manifest_path)
    require_data_quality_exclusions(pool)
    members = {row["sample_id"]: row for row in pool["membership"]}
    frozen = load_data_quality_exclusions()

    dev_ids = json.loads((dev_dir / "ids.json").read_text(encoding="utf-8"))
    dev_manifest_path = dev_dir / "tool_protocol_dev50_manifest.json"
    dev_manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))
    validate_dev_manifest(dev_ids, dev_manifest)
    eval_plan = build_eval300_plan(eval_path)
    eval_samples = read_eval_samples_by_ids(eval_path, list(eval_plan.entries))
    eval_questions = {normalized_question(sample.question) for sample in eval_samples}
    eval_ids = {str(sample.sample_id) for sample in eval_samples}
    eval_hashes = {image_sha256(image) for sample in eval_samples for image in sample.images}

    # The existing pinned-source scanner locates Dev50's original records and
    # verifies source-file hashes. It excludes the corrected 8k by design.
    external_candidates, external_scan = candidate_metadata(raw_dir, pool,
                                                             eval_questions, frozen)
    by_id = {item["sample_id"]: item for item in external_candidates}
    if not set(dev_ids).issubset(by_id):
        raise ValueError("Dev50 source records are missing from pinned clean candidates")
    dev_questions = set().union(*(sample_questions(by_id[identity]["record"])
                                  for identity in dev_ids))
    owned_hasher = image_hasher is None
    hasher = image_hasher or source_zip_image_hasher(raw_dir)
    try:
        dev_hashes = {value for identity in dev_ids for value in hasher(by_id[identity])}
        forbidden_hashes = eval_hashes | dev_hashes
        candidates = []
        rejected = Counter()
        seen = set()
        for shard in pool["shards"]:
            for record in iter_json_array(pool_manifest_path.parent /
                                          pool["shards"][shard]["path"]):
                sample_id = record["_sample_id"]
                if sample_id in seen or sample_id not in members:
                    raise ValueError("R3 source is not unique corrected-pool membership")
                seen.add(sample_id)
                if sample_id in frozen:
                    rejected["frozen_exclusion"] += 1
                    continue
                if sample_id in dev_ids:
                    rejected["dev50_id_overlap"] += 1
                    continue
                if sample_id in eval_ids or sample_questions(record) & eval_questions:
                    rejected["eval300_id_or_question_overlap"] += 1
                    continue
                if sample_questions(record) & dev_questions:
                    rejected["dev50_question_overlap"] += 1
                    continue
                chosen, reason = r3_candidate_diagnostic(record)
                if chosen is None:
                    rejected[reason or "unknown_rejection"] += 1
                    continue
                category, target = chosen
                candidates.append({"sample_id": sample_id, "source": record["_source"],
                                   "source_index": record["_source_index"],
                                   "record": {"images": record["_source_images"]},
                                   "corrected_record": record,
                                   "repair_category": category, **target})
        if seen != set(members):
            raise ValueError("R3 scan did not cover all corrected-pool members")
        selected, selection = select_r3_records(
            candidates, image_hashes=hasher, forbidden_hashes=forbidden_hashes,
            seed=seed)
    finally:
        if owned_hasher:
            hasher.close()  # type: ignore[attr-defined]

    records = []
    membership = []
    for item in selected:
        original = item["corrected_record"]
        record = {**original,
                  "_repair_category": item["repair_category"],
                  "_repair_target_turn_index": item["target_turn_index"],
                  "_repair_target_tool": "image_search",
                  "_repair_target_call_policy": "all_valid_image_search_calls"}
        records.append(record)
        membership.append({"sample_id": item["sample_id"], "source": item["source"],
                           "source_index": item["source_index"],
                           "category": item["repair_category"],
                           "target_turn_index": item["target_turn_index"],
                           "target_tool": "image_search",
                           "target_image_ids": item["target_image_ids"],
                           "image_sha256": item["image_sha256"],
                           "corrected_shard": members[item["sample_id"]]["shard"]})
    validate_r3_selection_membership(membership, pool_ids=set(members),
                                     forbidden_ids=set(frozen) | set(dev_ids) | eval_ids,
                                     forbidden_hashes=forbidden_hashes)
    occurrence = Counter("img_3_or_later" if int(image_id[4:]) >= 3 else image_id
                         for row in membership for image_id in row["target_image_ids"])
    coverage = Counter(category for row in membership for category in R3_CATEGORIES
                       if any(("img_3_or_later" if int(image_id[4:]) >= 3 else image_id)
                              == category for image_id in row["target_image_ids"]))
    for category in R3_CATEGORIES:
        occurrence.setdefault(category, 0)
        coverage.setdefault(category, 0)
    payload = (json.dumps(records, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("utf-8")
    manifest = {
        "repair_version": REPAIR_VERSION, "repair_experiment": "r3",
        "repair_mode": "argument_only", "repair_mask_version": MASK_VERSIONS["argument_only"],
        "selection_algorithm": R3_SELECTION_VERSION, "seed": seed,
        "source_pool": "corrected_sft_8k", "parent_checkpoint":
        "outputs/sft_main/checkpoint-3k",
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "corrected_pool_manifest_sha256": pool_sha,
        "dev50_manifest_sha256": sha256_file(dev_manifest_path),
        "dev50_ids_sha256": canonical_json_sha256(dev_ids),
        "eval300_dataset_sha256": eval_plan.dataset_sha256,
        "frozen_exclusion_population_count": len(frozen),
        "sample_count": len(records), "source_counts": dict(sorted(Counter(
            row["source"] for row in membership).items())),
        "protocol_category_counts": selection["selected_category_counts"],
        "sample_image_target_coverage_counts": dict(sorted(coverage.items())),
        "image_target_occurrence_category_counts": dict(sorted(occurrence.items())),
        "total_supervised_image_search_argument_occurrences": sum(occurrence.values()),
        "multi_image_search_call_samples": sum(len(row["target_image_ids"]) > 1
                                                for row in membership),
        "target_tool_counts": {"image_search": len(records)},
        "image_target_counts": dict(sorted(Counter(
            image_id for row in membership for image_id in row["target_image_ids"]).items())),
        "eval300_id_overlap_count": 0, "eval300_question_overlap_count": 0,
        "eval300_image_overlap_count": 0, "dev50_id_overlap_count": 0,
        "dev50_question_overlap_count": 0, "dev50_image_overlap_count": 0,
        "frozen_exclusion_overlap_count": 0, "image_contract_bad_count": 0,
        "http_url_supervised_count": 0, "ungrounded_supervised_count": 0,
        "repair_json_sha256": hashlib.sha256(payload).hexdigest(),
        "membership_sha256": canonical_json_sha256(membership),
        "membership": membership, "selection": selection,
        "candidate_rejections": dict(sorted(rejected.items())),
        "external_source_scan": external_scan,
    }
    output_dir.mkdir(parents=True)
    materialize_images(records, raw_dir, output_dir, download_missing=False)
    (output_dir / "repair.json").write_bytes(payload)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_repair_manifest(dataset_path: Path, manifest_path: Path, *,
                             mode: str | None = None,
                             pool_manifest_path: Path | None = None,
                             dev_dir: Path | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    is_r3 = manifest.get("repair_experiment") == "r3"
    if (manifest.get("repair_version") != REPAIR_VERSION
            or manifest.get("selection_algorithm") != (
                R3_SELECTION_VERSION if is_r3 else SELECTION_VERSION)
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
    isolation_keys = ("eval300_id_overlap_count", "eval300_question_overlap_count",
                "eval300_image_overlap_count", "dev30_overlap_count",
                "dev50_id_overlap_count", "dev50_question_overlap_count",
                "dev50_image_overlap_count", "sft8k_overlap_count",
                "frozen_exclusion_overlap_count", "image_contract_bad_count")
    if is_r3:
        isolation_keys = tuple(key for key in isolation_keys
                               if key not in ("dev30_overlap_count", "sft8k_overlap_count"))
    for key in isolation_keys:
        if manifest.get(key) != 0:
            raise ValueError(f"repair dataset failed isolation: {key}")
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if [row["_sample_id"] for row in records] != [row["sample_id"] for row in members]:
        raise ValueError("repair records and membership diverged")
    if is_r3:
        if (manifest.get("repair_mode") != "argument_only"
                or manifest.get("repair_mask_version") != MASK_VERSIONS["argument_only"]
                or manifest.get("source_pool") != "corrected_sft_8k"
                or manifest.get("parent_checkpoint") != "outputs/sft_main/checkpoint-3k"
                or len(records) != R3_SIZE
                or manifest.get("http_url_supervised_count") != 0
                or manifest.get("ungrounded_supervised_count") != 0):
            raise ValueError("R3 recipe, size, or image grounding changed")
        pool_path = pool_manifest_path or (Path(__file__).resolve().parents[2]
                                           / "data/sft_main/manifest.json")
        if sha256_file(pool_path) != CORRECTED_POOL_MANIFEST_SHA256:
            raise ValueError("R3 corrected-pool manifest SHA mismatch")
        pool = load_sft_manifest(pool_path)
        require_data_quality_exclusions(pool)
        pool_members = {row["sample_id"]: row for row in pool["membership"]}
        frozen = load_data_quality_exclusions()
        if not set(row["sample_id"] for row in members).issubset(pool_members):
            raise ValueError("R3 contains a sample outside corrected 8k")
        dev_root = dev_dir or (Path(__file__).resolve().parents[2]
                               / "data/eval/tool_protocol_dev50")
        dev_ids = json.loads((dev_root / "ids.json").read_text(encoding="utf-8"))
        if (canonical_json_sha256(dev_ids) != manifest.get("dev50_ids_sha256")
                or set(dev_ids) & {row["sample_id"] for row in members}
                or set(frozen) & {row["sample_id"] for row in members}):
            raise ValueError("R3 overlaps Dev50/frozen exclusions or their provenance changed")
        expected_records = {}
        wanted = {row["sample_id"] for row in members}
        for shard in pool["shards"]:
            for row in iter_json_array(pool_path.parent / pool["shards"][shard]["path"]):
                if row["_sample_id"] in wanted:
                    expected_records[row["_sample_id"]] = row
        coverage = Counter()
        occurrences = Counter()
        for record, member in zip(records, members):
            sample_id = member["sample_id"]
            source_record = expected_records.get(sample_id)
            original = {key: value for key, value in record.items()
                        if key not in ("_repair_category", "_repair_target_turn_index",
                                       "_repair_target_tool", "_repair_target_call_policy")}
            if (original != source_record or member.get("source") != record.get("_source")
                    or member.get("source_index") != record.get("_source_index")
                    or member.get("corrected_shard") != pool_members[sample_id]["shard"]):
                raise ValueError("R3 expert record differs from the corrected source pool")
            chosen = r3_candidate(record)
            if (chosen is None or chosen[0] != member.get("category")
                    or chosen[1]["target_image_ids"] != member.get("target_image_ids")
                    or chosen[1]["target_turn_index"] != member.get("target_turn_index")
                    or record.get("_repair_target_turn_index") !=
                    chosen[1]["target_turn_index"]
                    or record.get("_repair_target_tool") != "image_search"
                    or record.get("_repair_target_call_policy") !=
                    "all_valid_image_search_calls"):
                raise ValueError("R3 target is ungrounded, unmaskable, or changed")
            for image_id in member["target_image_ids"]:
                occurrences["img_3_or_later" if int(image_id[4:]) >= 3 else image_id] += 1
            for category in R3_CATEGORIES:
                if any(("img_3_or_later" if int(image_id[4:]) >= 3 else image_id)
                       == category for image_id in member["target_image_ids"]):
                    coverage[category] += 1
        for category in R3_CATEGORIES:
            occurrences.setdefault(category, 0)
            coverage.setdefault(category, 0)
        if (dict(sorted(occurrences.items())) !=
                manifest.get("image_target_occurrence_category_counts")
                or dict(sorted(coverage.items())) !=
                manifest.get("sample_image_target_coverage_counts")
                or sum(occurrences.values()) !=
                manifest.get("total_supervised_image_search_argument_occurrences")):
            raise ValueError("R3 sample coverage or target occurrences changed")
    if manifest["repair_mode"] == "full_tool_call":
        if manifest.get("requested_category_targets") != TARGETS["full_tool_call"]:
            raise ValueError("R2 repair target recipe changed; rebuild its dataset")
        allowed = set(TARGETS["full_tool_call"])
        def valid_target(category: Any, tool: Any) -> bool:
            return ((category == "image_search" and tool == "image_search")
                    or (category == "other_image_tool" and tool in OTHER_IMAGE_TOOLS))
        if (any(not valid_target(row.get("category"), row.get("target_tool"))
                for row in members)
                or any(not valid_target(record.get("_repair_category"),
                                        record.get("_repair_target_tool")) for record in records)
                or any(key not in allowed or count <= 0 for key, count in
                       manifest.get("protocol_category_counts", {}).items())
                or "no_tool" in manifest.get("target_tool_counts", {})):
            raise ValueError("R2 repair manifest contains a no-tool or unsupported target")
    return manifest
