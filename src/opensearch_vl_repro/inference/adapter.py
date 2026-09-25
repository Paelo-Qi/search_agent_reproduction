"""Validated formal SFT adapter identity for inference and Eval run manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from opensearch_vl_repro.sft_tool_audit import sha256_file


def adapter_identity(path: str | Path, *, base_model: str,
                     base_revision: str) -> dict[str, Any]:
    adapter = Path(path).expanduser().resolve()
    checkpoint = adapter.parent
    config_path = adapter / "adapter_config.json"
    metadata_path = checkpoint / "metadata.json"
    if not config_path.is_file() or not metadata_path.is_file():
        raise ValueError("formal SFT adapter requires adapter_config.json and parent checkpoint metadata")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("checkpoint_complete") is not True or
            metadata.get("model") != base_model or
            metadata.get("model_revision") != base_revision):
        raise ValueError("adapter checkpoint base model/revision does not match inference config")
    if config.get("base_model_name_or_path") != base_model:
        raise ValueError("PEFT adapter_config base_model_name_or_path does not match")
    weights = sorted(adapter.glob("*.safetensors"))
    if not weights:
        raise ValueError("formal SFT adapter has no safetensors weights")
    files = {file.relative_to(adapter).as_posix(): sha256_file(file)
             for file in [config_path, *weights]}
    expected = metadata.get("file_sha256", {})
    for relative, checksum in files.items():
        if expected.get(f"adapter/{relative}") != checksum:
            raise ValueError(f"adapter checkpoint checksum mismatch: {relative}")
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "kind": "peft_lora_adapter", "path": str(adapter),
        "base_model": base_model, "base_revision": base_revision,
        "adapter_fingerprint": fingerprint,
        "adapter_config_fingerprint": files["adapter_config.json"],
        "training_cumulative_stage": metadata.get("stage"),
        "source_checkpoint_lineage": metadata.get("lineage"),
        "checkpoint_metadata_fingerprint": sha256_file(metadata_path),
    }
