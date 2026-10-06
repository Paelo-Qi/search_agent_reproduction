"""Validated formal SFT / Formal RL Main adapter identities for inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from opensearch_vl_repro.sft_tool_audit import sha256_file
from opensearch_vl_repro.data import SFT_INPUT_MESSAGE_VERSION
from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION


def adapter_file_identity(path: str | Path) -> dict[str, Any]:
    """Hash PEFT artifacts without claiming they are a formal SFT checkpoint."""
    adapter = Path(path)
    config_path = adapter / "adapter_config.json"
    weights = sorted(adapter.glob("*.safetensors"))
    if not config_path.is_file() or not weights:
        raise ValueError("adapter requires adapter_config.json and safetensors weights")
    files = {file.relative_to(adapter).as_posix(): sha256_file(file)
             for file in [config_path, *weights]}
    return {"file_sha256": files, "adapter_fingerprint": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()}


def adapter_identity(path: str | Path, *, base_model: str,
                     base_revision: str) -> dict[str, Any]:
    adapter = Path(path).expanduser().resolve()
    checkpoint = adapter.parent
    config_path = adapter / "adapter_config.json"
    metadata_path = checkpoint / "metadata.json"
    # Preserve SFT precedence and its exact identity/resume semantics. Never
    # fall back to RL if existing SFT metadata fails validation.
    if not metadata_path.is_file() and (checkpoint / "checkpoint.json").is_file():
        return _rl_adapter_identity(adapter, base_model=base_model, base_revision=base_revision)
    if not config_path.is_file() or not metadata_path.is_file():
        raise ValueError("formal SFT adapter requires adapter_config.json and parent checkpoint metadata; "
                         "Formal RL requires parent checkpoint.json plus adapter/")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("checkpoint_complete") is not True or
            metadata.get("model") != base_model or
            metadata.get("model_revision") != base_revision or
            metadata.get("sft_input_message_version") != SFT_INPUT_MESSAGE_VERSION or
            metadata.get("runtime_tool_protocol_version") != RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION):
        raise ValueError("adapter checkpoint base model/revision or runtime protocol mismatch")
    if config.get("base_model_name_or_path") != base_model:
        raise ValueError("PEFT adapter_config base_model_name_or_path does not match")
    weights = sorted(adapter.glob("*.safetensors"))
    if not weights:
        raise ValueError("formal SFT adapter has no safetensors weights")
    artifact_identity = adapter_file_identity(adapter)
    files = artifact_identity["file_sha256"]
    expected = metadata.get("file_sha256", {})
    for relative, checksum in files.items():
        if expected.get(f"adapter/{relative}") != checksum:
            raise ValueError(f"adapter checkpoint checksum mismatch: {relative}")
    fingerprint = artifact_identity["adapter_fingerprint"]
    return {
        "kind": "peft_lora_adapter", "path": str(adapter),
        "base_model": base_model, "base_revision": base_revision,
        "adapter_fingerprint": fingerprint,
        "adapter_config_fingerprint": files["adapter_config.json"],
        "training_cumulative_stage": metadata.get("stage"),
        "source_checkpoint_lineage": metadata.get("lineage"),
        "checkpoint_metadata_fingerprint": sha256_file(metadata_path),
        "runtime_tool_protocol_version": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
    }


def _rl_adapter_identity(adapter: Path, *, base_model: str,
                         base_revision: str) -> dict[str, Any]:
    # Lazy import: checkpoint's SFT-lineage helpers also use adapter_identity.
    from opensearch_vl_repro.rl import checkpoint as cp

    manifest_path = adapter.parent / "checkpoint.json"
    if adapter.name != "adapter" or manifest_path.is_symlink():
        raise ValueError("Formal RL eval bundle requires checkpoint.json + adapter/ without redirects")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Formal RL checkpoint manifest must be an object")
    try:
        cp.validate_checkpoint_manifest(manifest)
        cp.validate_training_run_identity(manifest["run"])
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid Formal RL checkpoint manifest") from exc
    if (manifest["eligibility"] != cp.checkpoint_eligibility("main_checkpoint")
            or manifest["evidence_scope"] != "runtime"):
        raise ValueError("Formal RL eval requires a runtime main_checkpoint, not CPU/Smoke/Gate evidence")
    semantics = manifest["run"]["semantics"]
    if (semantics["base_model"]["name"] != base_model
            or semantics["base_model"]["revision"] != base_revision):
        raise ValueError("Formal RL checkpoint base model/revision mismatch")
    if (semantics["tool_protocol_version"] != RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
            or semantics["image_protocol_version"] != SFT_INPUT_MESSAGE_VERSION):
        raise ValueError("Formal RL checkpoint runtime tool/image protocol mismatch")
    iteration, step = manifest["policy_iteration"], manifest["global_optimizer_step"]
    cp.require_counter(iteration, 1)
    cp.require_counter(step, 1)
    if iteration != step:
        raise ValueError("Formal RL policy iteration/global optimizer step mismatch")
    role = manifest["artifact_role_files"]["adapter"]
    if ("adapter/adapter_config.json" not in role
            or not all(name.startswith("adapter/") for name in role)
            or not any(name.startswith("adapter/") and name.endswith(".safetensors") for name in role)):
        raise ValueError("Formal RL adapter role requires config and safetensors under adapter/")
    # Verify ONLY the sealed adapter role, including declared README/index files.
    # Do not use read_verified_checkpoint(), which requires native/AdamW/RNG bytes.
    cp.verify_artifacts(adapter, {name[len("adapter/"):]: checksum for name, checksum in role.items()})
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("base_model_name_or_path") != base_model:
        raise ValueError("PEFT adapter_config base_model_name_or_path does not match")
    artifacts = adapter_file_identity(adapter)
    return {
        "kind": "peft_lora_adapter", "training_origin": "formal_rl_main", "path": str(adapter),
        "base_model": base_model, "base_revision": base_revision,
        "adapter_fingerprint": artifacts["adapter_fingerprint"],
        "adapter_config_fingerprint": artifacts["file_sha256"]["adapter_config.json"],
        "adapter_role_fingerprint": manifest["artifact_roles"]["adapter"],
        "checkpoint_manifest_fingerprint": sha256_file(manifest_path),
        "checkpoint_identity": manifest["checkpoint_manifest_sha256"],
        "policy_iteration": iteration, "global_optimizer_step": step,
        "run_identity_sha256": manifest["run"]["run_identity_sha256"],
        "source_checkpoint_lineage": manifest["source_lineage"]["lineage"],
        "source_sft": manifest["source_lineage"],
        "runtime_tool_protocol_version": semantics["tool_protocol_version"],
        "runtime_image_protocol_version": semantics["image_protocol_version"],
    }
