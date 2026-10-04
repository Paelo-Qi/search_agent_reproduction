"""CPU-only pinned snapshot validation, extracted from Gate C.

Relative file names and content hashes are identity; the directory is a locator.
Gate C retains its original checks. Formal loaders additionally reject empty weights.
"""
from __future__ import annotations

import json
from pathlib import Path

from opensearch_vl_repro.sft_tool_audit import sha256_file


def offline_snapshot_files(directory, *, revision, strict=False):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("pinned offline base snapshot required")
    if not (directory / "config.json").is_file():
        raise FileNotFoundError("pinned offline base snapshot config.json required")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or config.get("text_config", {}).get("num_hidden_layers") != 36:
        raise ValueError("offline snapshot is not Qwen3-VL-4B")
    if (config.get("_commit_hash") not in {None, revision}
            or strict and "_commit_hash" in config and config["_commit_hash"] != revision):
        raise ValueError("offline snapshot declares a different pinned revision")
    weights = sorted(directory.glob("model*.safetensors"))
    if not weights or (strict and any(not p.is_file() or p.stat().st_size == 0 for p in weights)):
        raise ValueError("pinned offline base snapshot weights missing or empty" if strict
                         else "pinned offline base snapshot weights missing")
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix in {".json", ".safetensors"}}
