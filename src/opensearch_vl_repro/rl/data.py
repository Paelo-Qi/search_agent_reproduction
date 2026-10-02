"""Local-only RL prompt schema, deterministic selection and overlap interface."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from pathlib import PurePosixPath
import json

from opensearch_vl_repro.rl.quality_audit import load_quality_audit

from opensearch_vl_repro.agent.reliability import image_sha256
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.sft_preflight import normalized_question


RL_DATA_SCHEMA_VERSION = 3
RL_OVERLAP_SCHEMA_VERSION = 2
RL_SELECTION_VERSION = "sha256-rank-eval-backfill-v1"


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


# Formal RL source rows are indexed before ranking. Never use question text or
# selection position as an identifier: both can change without source movement.
def source_sample_id(row_index: int) -> str:
    if not isinstance(row_index, int) or row_index < 0:
        raise ValueError("source row index must be nonnegative")
    return f"rl_{row_index:06d}"


def safe_image_relpath(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("image path must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {".", ".."} for part in value.split("/")) or value.startswith("/"):
        raise ValueError("image path escapes source root")
    return path.as_posix()


def read_source_parquet(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    required = {"question", "answer", "images", "dataset"}
    if set(parquet.schema_arrow.names) != required:
        raise ValueError(f"RL parquet columns must be {sorted(required)}")
    rows = parquet.read(columns=sorted(required)).to_pylist()
    if not rows:
        raise ValueError("RL source parquet is empty")
    for index, row in enumerate(rows):
        if (not isinstance(row["question"], str) or not row["question"].strip()
                or not isinstance(row["answer"], str) or not row["answer"].strip()
                or not isinstance(row["dataset"], str) or not row["dataset"].strip()
                or not isinstance(row["images"], list) or not row["images"]):
            raise ValueError(f"invalid RL source row {index}")
        row["images"] = [safe_image_relpath(value) for value in row["images"]]
    return rows


def _hash_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = canonical_json_sha256(result)
    return result


def validate_manifest(manifest: dict[str, Any], *, schema_version: int = RL_DATA_SCHEMA_VERSION) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != schema_version:
        raise ValueError("invalid RL manifest schema")
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or expected != canonical_json_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ValueError("RL manifest SHA256 mismatch")


def load_overlap_manifest(path: str | Path, *, kind: str,
                          allow_incomplete_sft: bool = False) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(value, schema_version=RL_OVERLAP_SCHEMA_VERSION)
    if value.get("kind") != kind or not isinstance(value.get("question_hashes"), list) or not isinstance(value.get("image_hashes"), list):
        raise ValueError(f"invalid {kind} overlap manifest")
    for name in ("question_hashes", "image_hashes"):
        hashes = value[name]
        if hashes != sorted(set(hashes)) or any(not isinstance(item, str) or len(item) != 64 for item in hashes):
            raise ValueError(f"invalid {kind} {name}")
    if kind == "sft":
        from opensearch_vl_repro.sft_main_data import SHARD_SIZES

        shards, files = value.get("sft_shards"), value.get("sft_shard_files")
        if (not isinstance(shards, list) or not shards or len(shards) != len(set(shards))
                or any(name not in SHARD_SIZES for name in shards)
                or shards != [name for name in SHARD_SIZES if name in shards]
                or not isinstance(files, list) or any(not isinstance(item, dict) for item in files)
                or [item.get("name") for item in files] != shards
                or any(not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str)
                       or len(item["sha256"]) != 64 or not isinstance(item.get("count"), int)
                       for item in files)
                or value.get("audited_sample_count") != sum(item["count"] for item in files)):
            raise ValueError("SFT overlap shard scope/provenance is invalid")
    if value.get("image_audit_complete") is not True and not (kind == "sft" and allow_incomplete_sft):
        raise ValueError(f"{kind} image overlap manifest is incomplete")
    return value


def make_overlap_manifest(*, kind: str, question_hashes: set[str],
                          image_hashes: set[str], source_sha256: str,
                          image_audit_complete: bool, missing_image_count: int = 0,
                          sft_shards: list[str] | None = None,
                          sft_shard_files: list[dict[str, Any]] | None = None,
                          audited_sample_count: int | None = None) -> dict[str, Any]:
    if kind not in {"eval", "sft"} or len(source_sha256) != 64 or missing_image_count < 0:
        raise ValueError("invalid overlap manifest provenance")
    payload = {"schema_version": RL_OVERLAP_SCHEMA_VERSION, "kind": kind,
               "source_sha256": source_sha256,
               "image_audit_complete": image_audit_complete,
               "missing_image_count": missing_image_count,
               "question_hashes": sorted(question_hashes),
               "image_hashes": sorted(image_hashes)}
    if kind == "sft":
        if sft_shards is None or sft_shard_files is None or audited_sample_count is None:
            raise ValueError("SFT overlap requires explicit shard scope and provenance")
        payload.update(sft_shards=list(sft_shards), sft_shard_files=list(sft_shard_files),
                       audited_sample_count=audited_sample_count)
    return _hash_manifest(payload)


def prepare_formal_dataset(*, source_parquet: str | Path, source_root: str | Path,
                           dataset_id: str, dataset_revision: str, seed: int,
                           smoke_count: int, main_count: int, shard_size: int,
                           eval_overlap_manifest: str | Path,
                           sft_overlap_manifest: str | Path,
                           quality_audit_dir: str | Path,
                           allow_incomplete_sft: bool = False) -> dict[str, Any]:
    """Build complete in-memory artifacts; caller persists only after success."""
    from opensearch_vl_repro.eval_subset import sha256_file

    if (not dataset_id or not dataset_revision or not isinstance(seed, int)
            or not 0 < smoke_count <= main_count or shard_size < 1
            or main_count % shard_size):
        raise ValueError("RL selection counts, seed or source provenance are invalid")
    source_path, root = Path(source_parquet).resolve(), Path(source_root).resolve()
    revision_file = root / "source_revision.txt"
    if revision_file.is_file() and revision_file.read_text(encoding="utf-8").splitlines() != [dataset_id, dataset_revision]:
        raise ValueError("RL source_revision.txt does not match requested dataset identity")
    rows = read_source_parquet(source_path)
    if len(rows) < main_count:
        raise ValueError(f"RL source has {len(rows)} rows, cannot select {main_count}")
    eval_manifest = load_overlap_manifest(eval_overlap_manifest, kind="eval")
    sft_manifest = load_overlap_manifest(sft_overlap_manifest, kind="sft",
                                         allow_incomplete_sft=allow_incomplete_sft)
    quality = load_quality_audit(quality_audit_dir, main_count=main_count)
    eval_questions, eval_images = set(eval_manifest["question_hashes"]), set(eval_manifest["image_hashes"])
    sft_questions, sft_images = set(sft_manifest["question_hashes"]), set(sft_manifest["image_hashes"])
    ranked = sorted(range(len(rows)), key=lambda index: (
        hashlib.sha256(f"{seed}:{dataset_id}:{dataset_revision}:{source_sample_id(index)}".encode()).hexdigest(), index))
    selected, exclusions, audited_candidates = [], [], []
    for index in ranked:
        row = rows[index]
        relpaths = row["images"]
        resolved = []
        for relpath in relpaths:
            path = (root / relpath).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"RL source image missing or outside root: {relpath}")
            resolved.append(path)
        qhash = question_sha256(row["question"])
        ihashes = [image_sha256(path) for path in resolved]
        question_hit, image_hit = qhash in eval_questions, bool(set(ihashes) & eval_images)
        if question_hit or image_hit:
            exclusions.append({"source_sample_id": source_sample_id(index),
                               "question_overlap": question_hit, "image_overlap": image_hit})
            continue
        identity = source_sample_id(index)
        candidate_rank = len(audited_candidates) + 1
        expected = quality.rows[candidate_rank - 1]
        if expected["candidate_rank"] != candidate_rank or expected["source_sample_id"] != identity:
            raise ValueError(f"quality audit candidate stream mismatch at rank {candidate_rank}: {identity}")
        audited_candidates.append({"candidate_rank": candidate_rank, "source_sample_id": identity})
        if expected["status"] == "ok" and len(selected) < main_count:
            selected.append({"source_sample_id": identity, "prompt_id": identity,
                             "question": row["question"], "quality_candidate_rank": candidate_rank,
                             "reference_answer": row["answer"], "image_relpaths": relpaths,
                             "source_dataset": row["dataset"], "question_hash": qhash,
                             "image_hashes": ihashes})
        if candidate_rank == len(quality.rows):
            break
    if len(audited_candidates) != len(quality.rows) or len(selected) != main_count:
        raise ValueError("Eval-clean candidate stream exhausted before quality audit/main selection")
    if audited_candidates != [{"candidate_rank": row["candidate_rank"],
                                "source_sample_id": row["source_sample_id"]} for row in quality.rows]:
        raise ValueError("quality audit candidate stream differs from recomputed Eval-clean stream")
    if [{"candidate_rank": row["quality_candidate_rank"], "source_sample_id": row["source_sample_id"]}
            for row in selected] != list(quality.selected):
        raise ValueError("recomputed first-N quality-ok membership differs from frozen allowlist")
    sft_q_hits = [row["source_sample_id"] for row in selected if row["question_hash"] in sft_questions]
    sft_i_hits = [row["source_sample_id"] for row in selected if set(row["image_hashes"]) & sft_images]
    ids = [row["source_sample_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("RL selected source IDs are not unique")
    counts = {"eval_excluded_question_count": sum(item["question_overlap"] for item in exclusions),
              "eval_excluded_image_count": sum(item["image_overlap"] for item in exclusions),
              "sft_question_overlap_count": len(sft_q_hits), "sft_image_overlap_count": len(sft_i_hits)}
    common = {"schema_version": RL_DATA_SCHEMA_VERSION,
              "dataset_id": dataset_id, "dataset_revision": dataset_revision,
              "source_parquet_sha256": sha256_file(source_path), "source_rows": len(rows),
              "selection_version": RL_SELECTION_VERSION, "selection_seed": seed,
              "smoke_count": smoke_count, "main_count": main_count,
              "shard_size": shard_size, "eval_overlap_manifest_sha256": eval_manifest["manifest_sha256"],
              "sft_overlap_manifest_sha256": sft_manifest["manifest_sha256"],
              "sft_image_audit_complete": sft_manifest["image_audit_complete"],
              **quality.provenance,
              "quality_candidate_stream_sha256": canonical_json_sha256(audited_candidates),
              "quality_selected_main_membership_sha256": canonical_json_sha256(ids), **counts}
    def dataset_manifest(records: list[dict[str, Any]], *, name: str,
                         shard_index: int | None = None) -> dict[str, Any]:
        sample_ids = [row["source_sample_id"] for row in records]
        return _hash_manifest({**common, "name": name, "selected_count": len(records),
                               "shard_index": shard_index, "parent_dataset": "main" if name != "main" else None,
                               "membership": sample_ids,
                               "sample_ids_sha256": canonical_json_sha256(sample_ids),
                               "samples_sha256": canonical_json_sha256(records)})
    shards = [selected[i:i + shard_size] for i in range(0, main_count, shard_size)]
    shard_manifests = [dataset_manifest(shard, name=f"shard_{index:03d}", shard_index=index)
                       for index, shard in enumerate(shards)]
    main_manifest = dataset_manifest(selected, name="main")
    main_manifest = _hash_manifest({key: value for key, value in main_manifest.items() if key != "manifest_sha256"}
                                   | {"shard_order": [item["name"] for item in shard_manifests],
                                      "shard_manifest_sha256": [item["manifest_sha256"] for item in shard_manifests]})
    smoke = selected[:smoke_count]
    audit = _hash_manifest({"schema_version": RL_DATA_SCHEMA_VERSION, "excluded": exclusions,
                            "quality_candidate_stream_sha256": canonical_json_sha256(audited_candidates),
                            "quality_selected_main_membership_sha256": canonical_json_sha256(ids),
                            "sft_question_overlap_ids": sft_q_hits, "sft_image_overlap_ids": sft_i_hits,
                            "eval_final_question_overlap_count": 0, "eval_final_image_overlap_count": 0,
                            "sft_image_audit_complete": sft_manifest["image_audit_complete"], **counts})
    return {"main": selected, "smoke": smoke, "shards": shards,
            "main_manifest": main_manifest, "smoke_manifest": dataset_manifest(smoke, name="smoke"),
            "shard_manifests": shard_manifests, "overlap_audit": audit,
            "eval_overlap_manifest": eval_manifest, "sft_overlap_manifest": sft_manifest}


def write_dataset_artifacts(artifacts: dict[str, Any], output_dir: str | Path, *,
                            source_parquet: str | Path, source_root: str | Path,
                            eval_overlap_manifest: str | Path,
                            sft_overlap_manifest: str | Path,
                            quality_audit_dir: str | Path,
                            allow_incomplete_sft: bool = False) -> None:
    """Validate a sibling staging tree, then publish it with one rename."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RL output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    main_count = len(artifacts["main"])
    smoke_count = len(artifacts["smoke"])
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as staging_name:
        staging = Path(staging_name)
        written = {}
        def write(relative: str, value: Any) -> None:
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            written[relative] = value
        write(f"main{main_count}.json", artifacts["main"])
        write(f"main{main_count}_manifest.json", artifacts["main_manifest"])
        write(f"smoke{smoke_count}.json", artifacts["smoke"])
        write(f"smoke{smoke_count}_manifest.json", artifacts["smoke_manifest"])
        for index, (shard, manifest) in enumerate(zip(artifacts["shards"], artifacts["shard_manifests"], strict=True)):
            name = f"shard_{index:03d}"
            write(f"main{main_count}_shards/{name}.json", shard)
            write(f"main{main_count}_shards/{name}_manifest.json", manifest)
        write("overlap_audit.json", artifacts["overlap_audit"])
        for relative, value in written.items():
            if json.loads((staging / relative).read_text(encoding="utf-8")) != value:
                raise ValueError(f"staged RL artifact differs from in-memory selection: {relative}")
        for manifest in (artifacts["main_manifest"], artifacts["smoke_manifest"],
                         *artifacts["shard_manifests"], artifacts["overlap_audit"]):
            validate_manifest(manifest)
        if (artifacts["smoke"] != artifacts["main"][:smoke_count]
                or artifacts["main"] != [row for shard in artifacts["shards"] for row in shard]):
            raise ValueError("staged RL smoke/shards differ from main")
        preflight_dataset(staging, source_parquet=source_parquet, source_root=source_root,
                          eval_overlap_manifest=eval_overlap_manifest,
                          sft_overlap_manifest=sft_overlap_manifest,
                          quality_audit_dir=quality_audit_dir,
                          allow_incomplete_sft=allow_incomplete_sft)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite RL output directory: {output}")
        staging.replace(output)


def preflight_dataset(output_dir: str | Path, *, source_parquet: str | Path,
                      source_root: str | Path, eval_overlap_manifest: str | Path,
                      sft_overlap_manifest: str | Path, quality_audit_dir: str | Path,
                      allow_incomplete_sft: bool = False) -> dict[str, Any]:
    from opensearch_vl_repro.eval_subset import sha256_file

    output = Path(output_dir)
    manifests = list(output.glob("main*_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("RL output must have exactly one main manifest")
    main_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    validate_manifest(main_manifest)
    if main_manifest.get("selection_version") != RL_SELECTION_VERSION:
        raise ValueError("RL selection version mismatch")
    if "source_root" in main_manifest or "source_parquet" in main_manifest:
        raise ValueError("absolute source locator must not appear in deterministic manifest")
    main_count, smoke_count, shard_size = (main_manifest[key] for key in ("main_count", "smoke_count", "shard_size"))
    quality = load_quality_audit(quality_audit_dir, main_count=main_count)
    if any(main_manifest.get(key) != value for key, value in quality.provenance.items()):
        raise ValueError("RL quality audit fingerprint/provenance mismatch")
    def read(relative: str) -> Any:
        return json.loads((output / relative).read_text(encoding="utf-8"))
    main, smoke = read(f"main{main_count}.json"), read(f"smoke{smoke_count}.json")
    smoke_manifest = read(f"smoke{smoke_count}_manifest.json")
    validate_manifest(smoke_manifest)
    if (len(main) != main_count or len(smoke) != smoke_count or smoke != main[:smoke_count]
            or main_manifest.get("selected_count") != main_count
            or main_manifest.get("membership") != [row["source_sample_id"] for row in main]
            or main_manifest.get("sample_ids_sha256") != canonical_json_sha256(main_manifest["membership"])
            or main_manifest.get("samples_sha256") != canonical_json_sha256(main)
            or smoke_manifest.get("selected_count") != smoke_count
            or smoke_manifest.get("membership") != main_manifest["membership"][:smoke_count]
            or smoke_manifest.get("sample_ids_sha256") != canonical_json_sha256(smoke_manifest["membership"])
            or smoke_manifest.get("samples_sha256") != canonical_json_sha256(smoke)):
        raise ValueError("RL smoke/main count, prefix or checksum mismatch")
    common_keys = ("dataset_id", "dataset_revision", "source_parquet_sha256",
                   "selection_version",
                   "selection_seed", "smoke_count", "main_count", "shard_size",
                   "quality_candidate_stream_sha256",
                   "quality_selected_main_membership_sha256", *quality.provenance)
    if any(smoke_manifest.get(key) != main_manifest.get(key) for key in common_keys):
        raise ValueError("RL smoke/main provenance differs")
    root = Path(source_root).resolve()
    source_path = Path(source_parquet).resolve()
    revision_file = root / "source_revision.txt"
    if revision_file.is_file() and revision_file.read_text(encoding="utf-8").splitlines() != [
            main_manifest["dataset_id"], main_manifest["dataset_revision"]]:
        raise ValueError("RL source revision file differs from manifest")
    if sha256_file(source_path) != main_manifest["source_parquet_sha256"]:
        raise ValueError("RL source parquet SHA256 mismatch")
    source_rows = read_source_parquet(source_path)
    if len(source_rows) != main_manifest["source_rows"]:
        raise ValueError("RL source row count mismatch")
    ids = []
    for row in main:
        identity = row["source_sample_id"]
        if not isinstance(identity, str) or not identity.startswith("rl_") or not identity[3:].isdigit():
            raise ValueError("RL source ID format is invalid")
        index = int(identity[3:])
        if index >= len(source_rows) or source_sample_id(index) != identity:
            raise ValueError("RL source ID does not map to parquet row")
        source = source_rows[index]
        if (row["prompt_id"] != identity or "trajectory_group_id" in row
                or row["question"] != source["question"]
                or row["reference_answer"] != source["answer"]
                or row["source_dataset"] != source["dataset"]
                or row["image_relpaths"] != source["images"]
                or row["question_hash"] != question_sha256(row["question"])):
            raise ValueError(f"RL sample mapping/question hash mismatch: {identity}")
        paths = [(root / safe_image_relpath(relpath)).resolve() for relpath in row["image_relpaths"]]
        if any(not path.is_relative_to(root) or not path.is_file() for path in paths):
            raise FileNotFoundError(f"RL sample image missing/outside root: {identity}")
        if row["image_hashes"] != [image_sha256(path) for path in paths]:
            raise ValueError(f"RL sample image hash mismatch: {identity}")
        ids.append(identity)
    if len(set(ids)) != main_count:
        raise ValueError("RL source IDs are duplicated")
    if (main_manifest.get("quality_selected_main_membership_sha256") != canonical_json_sha256(ids)
            or [{"candidate_rank": row.get("quality_candidate_rank"), "source_sample_id": row["source_sample_id"]}
                for row in main] != list(quality.selected)):
        raise ValueError("RL final main differs from first-N quality-ok membership")
    shard_order = main_manifest.get("shard_order")
    if shard_order != [f"shard_{index:03d}" for index in range(main_count // shard_size)]:
        raise ValueError("RL shard order is invalid")
    combined = []
    for index, name in enumerate(shard_order):
        shard = read(f"main{main_count}_shards/{name}.json")
        manifest = read(f"main{main_count}_shards/{name}_manifest.json")
        validate_manifest(manifest)
        if (len(shard) != shard_size or manifest["shard_index"] != index
                or manifest["parent_dataset"] != "main"
                or manifest["membership"] != [item["source_sample_id"] for item in shard]
                or manifest["samples_sha256"] != canonical_json_sha256(shard)
                or manifest["sample_ids_sha256"] != canonical_json_sha256(manifest["membership"])
                or manifest["manifest_sha256"] != main_manifest["shard_manifest_sha256"][index]):
            raise ValueError(f"RL shard invalid: {name}")
        if any(manifest.get(key) != main_manifest.get(key) for key in common_keys):
            raise ValueError(f"RL shard provenance differs: {name}")
        combined.extend(shard)
    if combined != main:
        raise ValueError("RL shard concatenation differs from main")
    eval_manifest = load_overlap_manifest(eval_overlap_manifest, kind="eval")
    sft_manifest = load_overlap_manifest(sft_overlap_manifest, kind="sft",
                                         allow_incomplete_sft=allow_incomplete_sft)
    if (main_manifest["eval_overlap_manifest_sha256"] != eval_manifest["manifest_sha256"]
            or main_manifest["sft_overlap_manifest_sha256"] != sft_manifest["manifest_sha256"]
            or main_manifest["sft_image_audit_complete"] != sft_manifest["image_audit_complete"]):
        raise ValueError("RL overlap manifest provenance mismatch")
    eval_q, eval_i = set(eval_manifest["question_hashes"]), set(eval_manifest["image_hashes"])
    sft_q, sft_i = set(sft_manifest["question_hashes"]), set(sft_manifest["image_hashes"])
    if any(row["question_hash"] in eval_q or set(row["image_hashes"]) & eval_i for row in main):
        raise ValueError("RL final main has Eval overlap")
    audit = read("overlap_audit.json")
    validate_manifest(audit)
    if (audit["sft_question_overlap_ids"] != [row["source_sample_id"] for row in main if row["question_hash"] in sft_q]
            or audit["sft_image_overlap_ids"] != [row["source_sample_id"] for row in main if set(row["image_hashes"]) & sft_i]
            or audit["sft_question_overlap_count"] != main_manifest["sft_question_overlap_count"]
            or audit["sft_image_overlap_count"] != main_manifest["sft_image_overlap_count"]
            or audit["eval_excluded_question_count"] != main_manifest["eval_excluded_question_count"]
            or audit["eval_excluded_image_count"] != main_manifest["eval_excluded_image_count"]
            or audit["eval_excluded_question_count"] != sum(item["question_overlap"] for item in audit["excluded"])
            or audit["eval_excluded_image_count"] != sum(item["image_overlap"] for item in audit["excluded"])
            or audit["eval_final_question_overlap_count"] != 0
            or audit["eval_final_image_overlap_count"] != 0):
        raise ValueError("RL overlap audit mismatch")
    expected = prepare_formal_dataset(
        source_parquet=source_path, source_root=root,
        dataset_id=main_manifest["dataset_id"], dataset_revision=main_manifest["dataset_revision"],
        seed=main_manifest["selection_seed"], smoke_count=smoke_count,
        main_count=main_count, shard_size=shard_size,
        eval_overlap_manifest=eval_overlap_manifest, sft_overlap_manifest=sft_overlap_manifest,
        quality_audit_dir=quality_audit_dir, allow_incomplete_sft=allow_incomplete_sft)
    if (main != expected["main"] or smoke != expected["smoke"]
            or main_manifest != expected["main_manifest"]
            or smoke_manifest != expected["smoke_manifest"]
            or audit != expected["overlap_audit"]
            or [read(f"main{main_count}_shards/{name}_manifest.json") for name in shard_order]
               != expected["shard_manifests"]):
        raise ValueError("RL published data differs from recomputed quality-filtered candidate stream")
    return {"passed": True, "main_count": main_count, "smoke_count": smoke_count,
            "shard_count": len(shard_order), "shard_size": shard_size,
            "source_rows": len(source_rows), "dataset_id": main_manifest["dataset_id"],
            "dataset_revision": main_manifest["dataset_revision"],
            "selection_seed": main_manifest["selection_seed"],
            "selection_version": main_manifest["selection_version"],
            "quality_audit_version": quality.provenance["quality_audit_version"],
            "quality_ok_count": quality.provenance["quality_ok_count"],
            "audited_count": quality.provenance["audited_count"],
            "sft_shards": sft_manifest["sft_shards"],
            "main_manifest_sha256": main_manifest["manifest_sha256"],
            "sft_image_audit_complete": sft_manifest["image_audit_complete"],
            "eval_final_question_overlap_count": 0, "eval_final_image_overlap_count": 0}
