"""Deterministic batch run identity and resume compatibility checks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from opensearch_vl_repro.agent.reliability import (
    CACHE_SCHEMA_VERSION, LAYOUT_BEHAVIOR_VERSION, SEARCH_BEHAVIOR_VERSION,
    canonical_json,
)
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS
from opensearch_vl_repro.eval_subset import sha256_file


RUN_MANIFEST_VERSION = 1


class RunManifestMismatchError(RuntimeError):
    pass


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _safe_config_identity(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            normalized = str(key).lower()
            secret_field = (
                normalized in {"api_key", "access_token", "token", "secret", "password"}
                or (normalized.endswith(("_api_key", "_access_token", "_token"))
                    and not normalized.endswith("_env"))
            )
            if secret_field:
                output[key] = "[REDACTED]"
            else:
                output[key] = _safe_config_identity(item)
        return output
    if isinstance(value, list):
        return [_safe_config_identity(item) for item in value]
    return value


def _config_fingerprint(path: str | Path) -> str:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return _sha256_json(_safe_config_identity(value))


def tool_contract_fingerprint() -> str:
    return _sha256_json([declaration.as_chat_template_tool()
                         for declaration in TOOL_DECLARATIONS])


def _sampled_file_fingerprint(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def checkpoint_identity(model_name_or_path: str, revision: str | None) -> dict[str, Any]:
    candidate = Path(model_name_or_path).expanduser()
    if not candidate.exists():
        return {"kind": "remote", "identifier": model_name_or_path,
                "revision": revision or None}
    resolved = candidate.resolve()
    artifacts: dict[str, dict[str, Any]] = {}
    if resolved.is_dir():
        relevant_suffixes = {".json", ".safetensors", ".bin", ".pt", ".pth"}
        for path in sorted(item for item in resolved.rglob("*")
                           if item.is_file() and item.suffix.lower() in relevant_suffixes):
            relative = path.relative_to(resolved).as_posix()
            size = path.stat().st_size
            artifacts[relative] = {
                "size": size,
                "fingerprint": (sha256_file(path) if size <= 16 * 1024 * 1024
                                else _sampled_file_fingerprint(path)),
                "fingerprint_kind": "full_sha256" if size <= 16 * 1024 * 1024
                                    else "size_first_last_1m_sha256",
            }
    elif resolved.is_file():
        # Do not hash multi-GB weight files. A directly supplied small config
        # file can still be identified cheaply.
        size = resolved.stat().st_size
        artifacts[resolved.name] = {
            "size": size,
            "fingerprint": (sha256_file(resolved) if size <= 16 * 1024 * 1024
                            else _sampled_file_fingerprint(resolved)),
            "fingerprint_kind": "full_sha256" if size <= 16 * 1024 * 1024
                                else "size_first_last_1m_sha256",
        }
    return {
        "kind": "local", "resolved_path": str(resolved),
        "revision": revision or None,
        "checkpoint_fingerprint": _sha256_json(artifacts),
        "artifact_files": sorted(artifacts),
    }


def frozen_dataset_identity(dataset_path: str | Path,
                            eval_manifest_path: str | Path | None = None) -> dict[str, Any]:
    resolved = Path(dataset_path).expanduser().resolve()
    actual_sha256 = sha256_file(resolved)
    identity: dict[str, Any] = {"sha256": actual_sha256}
    if eval_manifest_path is not None and Path(eval_manifest_path).is_file():
        manifest_path = Path(eval_manifest_path).expanduser().resolve()
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        combined = raw.get("combined", {}) if isinstance(raw, dict) else {}
        frozen = {
            "manifest_version": raw.get("manifest_version"),
            "dataset": raw.get("dataset"),
            "dataset_revision": raw.get("dataset_revision"),
            "combined_output_file": combined.get("output_file"),
            "combined_output_sha256": combined.get("output_sha256"),
        }
        expected_sha256 = frozen["combined_output_sha256"]
        if expected_sha256 and expected_sha256 != actual_sha256:
            raise ValueError(
                "evaluation dataset checksum does not match its frozen manifest"
            )
        identity.update(
            frozen_manifest_identity=frozen,
            frozen_manifest_fingerprint=_sha256_json(frozen),
        )
    return identity


def create_run_manifest(
    *,
    run_id: str,
    model_name_or_path: str,
    model_revision: str | None,
    inference_config_fingerprint: str,
    dataset_path: str | Path,
    dataset_identity: dict[str, Any],
    start: int | None,
    limit: int | None,
    max_agent_turns: int,
    search_config_fingerprint: str,
    layout_config_fingerprint: str,
    checkpoint: dict[str, Any] | None = None,
    tool_fingerprint: str | None = None,
    created_at: str | None = None,
    sample_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint or checkpoint_identity(model_name_or_path, model_revision)
    tool_fingerprint = tool_fingerprint or tool_contract_fingerprint()
    if sample_selection is None:
        if start is None or limit is None:
            raise ValueError("continuous selection requires start and limit")
        sample_selection = {"start": start, "limit": limit}
    identity = {
        "model_name_or_path": model_name_or_path,
        "model_revision": model_revision or None,
        "checkpoint_identity": checkpoint,
        "inference_config_fingerprint": inference_config_fingerprint,
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "dataset_identity": dataset_identity,
        "sample_selection": sample_selection,
        "max_agent_turns": max_agent_turns,
        "search_config_fingerprint": search_config_fingerprint,
        "layout_config_fingerprint": layout_config_fingerprint,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "search_behavior_version": SEARCH_BEHAVIOR_VERSION,
        "layout_behavior_version": LAYOUT_BEHAVIOR_VERSION,
        "tool_contract_fingerprint": tool_fingerprint,
    }
    return {
        "manifest_version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        **identity,
        "run_config_fingerprint": _sha256_json(identity),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def build_run_manifest(
    *, run_id: str, model_name_or_path: str, model_revision: str | None,
    inference_config_path: str | Path, dataset_path: str | Path,
    eval_manifest_path: str | Path | None, start: int | None, limit: int | None,
    max_agent_turns: int, search_config_path: str | Path,
    layout_config_path: str | Path, sample_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return create_run_manifest(
        run_id=run_id,
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        checkpoint=checkpoint_identity(model_name_or_path, model_revision),
        inference_config_fingerprint=_config_fingerprint(inference_config_path),
        dataset_path=dataset_path,
        dataset_identity=frozen_dataset_identity(dataset_path, eval_manifest_path),
        start=start, limit=limit, sample_selection=sample_selection,
        max_agent_turns=max_agent_turns,
        search_config_fingerprint=_config_fingerprint(search_config_path),
        layout_config_fingerprint=_config_fingerprint(layout_config_path),
    )


IDENTITY_FIELDS = (
    "run_id", "model_name_or_path", "model_revision", "checkpoint_identity",
    "inference_config_fingerprint", "dataset_path", "dataset_identity", "sample_selection",
    "max_agent_turns", "search_config_fingerprint", "layout_config_fingerprint",
    "cache_schema_version", "search_behavior_version", "layout_behavior_version",
    "tool_contract_fingerprint",
)


def manifest_mismatches(persisted: dict[str, Any], current: dict[str, Any]) -> list[str]:
    mismatches = [field for field in IDENTITY_FIELDS
                  if persisted.get(field) != current.get(field)]
    if persisted.get("manifest_version") != RUN_MANIFEST_VERSION:
        mismatches.insert(0, "manifest_version")
    if persisted.get("run_config_fingerprint") != current.get("run_config_fingerprint"):
        if not mismatches:
            mismatches.append("run_config_fingerprint")
    return mismatches
