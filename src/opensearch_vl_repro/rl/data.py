"""Local-only RL prompt schema, deterministic selection and overlap interface."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from opensearch_vl_repro.agent.reliability import image_sha256
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.sft_preflight import normalized_question


@dataclass(frozen=True)
class RLPrompt:
    sample_id: str
    source_sample_id: str
    question: str
    image_paths: tuple[str, ...]
    question_hash: str
    image_hashes: tuple[str, ...]
    split: str

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "image_paths": list(self.image_paths),
                "image_hashes": list(self.image_hashes)}


def question_sha256(question: str) -> str:
    normalized = normalized_question(question)
    if not normalized:
        raise ValueError("RL question is empty")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def prepare_prompts(records: list[dict[str, Any]], *, dataset_id: str,
                    dataset_revision: str, seed: int, limit: int,
                    validation_count: int = 0) -> tuple[list[RLPrompt], dict[str, Any]]:
    if not dataset_id or not dataset_revision or not isinstance(seed, int) or limit < 1 or validation_count < 0:
        raise ValueError("RL source provenance/selection settings are invalid")
    if validation_count >= limit:
        raise ValueError("validation_count must be smaller than selection limit")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        source_id = record.get("source_sample_id")
        if not isinstance(source_id, str) or not source_id or source_id in indexed:
            raise ValueError("RL source IDs must be unique, nonempty strings")
        indexed[source_id] = record
    if len(indexed) < limit:
        raise ValueError("insufficient RL source records")
    ranked = sorted(indexed, key=lambda source_id: (hashlib.sha256(
        f"{seed}:{dataset_id}:{dataset_revision}:{source_id}".encode()).hexdigest(), source_id))
    selected: list[RLPrompt] = []
    for position, source_id in enumerate(ranked[:limit]):
        record = indexed[source_id]
        question = record.get("question")
        paths = record.get("image_paths")
        if not isinstance(question, str) or not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise ValueError("RL prompt needs question and local image_paths")
        if not all(Path(path).is_file() for path in paths):
            raise FileNotFoundError("RL prompt image path is missing")
        selected.append(RLPrompt(
            sample_id=f"{dataset_id}:{source_id}", source_sample_id=source_id,
            question=question, image_paths=tuple(paths),
            question_hash=question_sha256(question),
            image_hashes=tuple(image_sha256(path) for path in paths),
            split="validation" if position < validation_count else "train",
        ))
    payload = [item.to_dict() for item in selected]
    manifest = {"schema_version": 1, "dataset_id": dataset_id,
                "dataset_revision": dataset_revision, "selection_seed": seed,
                "selected_count": len(payload), "validation_count": validation_count,
                "membership": [item.sample_id for item in selected],
                "samples_sha256": canonical_json_sha256(payload)}
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    return selected, manifest


def overlap_audit(samples: list[RLPrompt], *, known_question_hashes: set[str],
                  known_image_hashes: set[str]) -> dict[str, list[str]]:
    """Caller supplies frozen Eval/SFT hash sets; does not alter those datasets."""
    return {
        "question_overlap_ids": [item.sample_id for item in samples if item.question_hash in known_question_hashes],
        "image_overlap_ids": [item.sample_id for item in samples if set(item.image_hashes) & known_image_hashes],
    }
