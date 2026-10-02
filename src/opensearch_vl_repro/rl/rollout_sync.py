"""Offline actor PEFT -> static merged HF checkpoint; no rollout or RL update."""

from __future__ import annotations

import copy
import gc
import json
import os
import shutil
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Callable

from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.inference.adapter import adapter_file_identity
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, REQUIRED_CHECKS, atomic_json
from opensearch_vl_repro.rl.checkpoint import RLLineage, build_rl_lineage, validate_sft_overlap_scope
from opensearch_vl_repro.sft_tool_audit import sha256_file

GATE_B_VERSION = "actor-rollout-roundtrip-b-v1"


def prepare_qwen_vl_processor_inputs(processor: Any, messages: list[dict[str, Any]],
                                     tools: list[dict[str, Any]], *,
                                     max_images: int | None = None) -> tuple[str, list[Any], Any]:
    """Shared Gate input materialization; preserve string content and PIL order.

    Transformers 4.57.1's multimodal tokenize=True path assumes every message
    content is a block list. Render our unchanged messages first, then encode
    the rendered prompt with the actual PIL objects, as in the vLLM bridge.
    """
    from PIL import Image

    images = [part.get("image") for message in messages if isinstance(message.get("content"), list)
              for part in message["content"] if isinstance(part, dict) and part.get("type") == "image"]
    if not images or any(not isinstance(image, Image.Image) for image in images):
        raise RuntimeError("Gate processor requires at least one actual PIL multimodal image part")
    if max_images is not None and len(images) > max_images:
        raise RuntimeError("Gate processor image count exceeds the permitted multimodal limit")
    prompt = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[images], return_tensors="pt", truncation=False)
    return prompt, images, inputs


def validate_actor_adapter(*, adapter: Path, gate_manifest: Path | None,
                           rl_config: dict[str, Any], sft_config: dict[str, Any],
                           source_sft_adapter: Path) -> dict[str, Any]:
    source = build_rl_lineage(config=rl_config, sft_config=sft_config,
                              adapter_path=source_sft_adapter, run_id="gate-b-source-sft")
    validate_sft_overlap_scope(source, ["main_a_1k", "main_b_2k"])
    if (source.base_model, source.base_revision) != (BASE_MODEL, BASE_REVISION):
        raise ValueError("Gate B source SFT must use the pinned Qwen base/revision")
    artifact = adapter_file_identity(adapter)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    lora = sft_config["lora"]
    if (config.get("base_model_name_or_path") != BASE_MODEL or config.get("peft_type") != "LORA"
            or config.get("r") != lora["rank"] or config.get("lora_alpha") != lora["alpha"]
            or float(config.get("lora_dropout", -1)) != lora["dropout"]
            or set(config.get("target_modules", [])) != set(lora["target_modules"])):
        raise ValueError("actor adapter model/LoRA semantics mismatch")
    result = {"actor_adapter_fingerprint": artifact["adapter_fingerprint"],
              "source_sft_adapter_fingerprint": source.sft_adapter_fingerprint,
              "source_sft_lineage": list(source.sft_lineage), "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
              "actor_gate_manifest_sha256": None, "actor_gate_identity_sha256": None,
              "formal_rl_initialization_allowed": False}
    if gate_manifest is None:
        # Explicit formal SFT fallback, never infer an actor artifact from its name.
        formal = build_rl_lineage(config=rl_config, sft_config=sft_config, adapter_path=adapter,
                                  run_id="gate-b-explicit-sft-fallback")
        validate_sft_overlap_scope(formal, ["main_a_1k", "main_b_2k"])
        if formal.sft_adapter_fingerprint != source.sft_adapter_fingerprint:
            raise ValueError("fallback must be the verified source checkpoint-3k adapter")
        return {**result, "actor_source_kind": "formal_sft_checkpoint_fallback"}
    manifest = json.loads(gate_manifest.read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})
    lineage = RLLineage.from_dict(identity.get("lineage", {}))
    validate_sft_overlap_scope(lineage, ["main_a_1k", "main_b_2k"])
    if (manifest.get("passed") is not True or manifest.get("fsdp_mode") != "fsdp2"
            or (manifest.get("model"), manifest.get("revision")) != (BASE_MODEL, BASE_REVISION)
            or (identity.get("base_model"), identity.get("base_revision")) != (BASE_MODEL, BASE_REVISION)
            or identity.get("gate_version") != "a2.2-v1"
            or identity.get("objective") != "temporary_supervised_not_rl"
            or type(manifest.get("world_size")) is not int or manifest["world_size"] < 2
            or identity.get("world_size") != manifest["world_size"]
            or identity.get("gate_identity_sha256") != canonical_json_sha256(
                {k: v for k, v in identity.items() if k != "gate_identity_sha256"})
            or any(manifest.get("checks", {}).get(key) is not True for key in REQUIRED_CHECKS)
            or manifest.get("checks", {}).get("all_ranks_report_success") is not True
            or manifest.get("input_adapter_fingerprint") != source.sft_adapter_fingerprint
            or lineage.sft_adapter_fingerprint != source.sft_adapter_fingerprint
            or lineage.sft_adapter_config_fingerprint != source.sft_adapter_config_fingerprint
            or lineage.sft_checkpoint_metadata_fingerprint != source.sft_checkpoint_metadata_fingerprint
            or (lineage.base_model, lineage.base_revision) != (BASE_MODEL, BASE_REVISION)
            or manifest.get("output_adapter_fingerprint") != artifact["adapter_fingerprint"]):
        raise ValueError("A2.2 PASS/identity/adapter/source lineage validation failed")
    hashes = manifest.get("checkpoint_file_sha256", {})
    if (manifest.get("checkpoint_fingerprint") != canonical_json_sha256(hashes)
            or any(hashes.get("adapter/" + name) != checksum for name, checksum in artifact["file_sha256"].items())):
        raise ValueError("actor adapter differs from A2.2 saved checkpoint checksums")
    return {**result, "actor_source_kind": "a22_temporary_updated_actor",
            "actor_gate_manifest_sha256": sha256_file(gate_manifest),
            "actor_gate_identity_sha256": identity["gate_identity_sha256"], "actor_fsdp_mode": "fsdp2"}


def merge_identity(*, actor: dict[str, Any], versions: dict[str, str],
                   file_hashes: dict[str, str]) -> dict[str, Any]:
    # No directory locators, timestamps, or raw A2.2 manifest hash (which contains
    # informational machine locators) enter deterministic identity.
    value = {"gate_version": GATE_B_VERSION, "base_model": actor["base_model"],
             "base_revision": actor["base_revision"], "actor_adapter_fingerprint": actor["actor_adapter_fingerprint"],
             "actor_source_kind": actor["actor_source_kind"],
             "actor_gate_identity_sha256": actor["actor_gate_identity_sha256"],
             "source_sft_adapter_fingerprint": actor["source_sft_adapter_fingerprint"],
             "source_sft_lineage": actor["source_sft_lineage"], "merge_method": "peft.merge_and_unload",
             "runtime_image_protocol_version": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
             "software_versions": versions, "merged_file_sha256": file_hashes,
             "merged_config_fingerprint": file_hashes["config.json"], "formal_rl_initialization_allowed": False}
    return {**value, "merged_checkpoint_fingerprint": canonical_json_sha256(value)}


def validate_merged_files(directory: Path) -> dict[str, str]:
    if (directory / "adapter_config.json").exists() or list(directory.glob("adapter*.safetensors")):
        raise ValueError("static merged checkpoint must not contain PEFT artifacts")
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    if any(not (directory / name).is_file() for name in required):
        raise ValueError("merged model/processor/tokenizer configuration incomplete")
    weights = sorted(directory.glob("model*.safetensors"))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise ValueError("merged checkpoint weights missing or empty")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        entries = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not entries or any(Path(name).name != name or not (directory / name).is_file() for name in entries.values()):
            raise ValueError("merged weight index references missing/unsafe shards")
    elif len(weights) != 1:
        raise ValueError("multiple merged shards require a weight index")
    return {path.relative_to(directory).as_posix(): sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def required_merge_space(base: Path, adapter: Path) -> int:
    weights = list(base.glob("*.safetensors")) or list(base.glob("pytorch_model*.bin"))
    size = sum(path.stat().st_size for path in weights)
    if size <= 0:
        raise ValueError("offline base snapshot weights missing")
    return 2 * size + sum(path.stat().st_size for path in adapter.iterdir() if path.is_file())


def check_disk_space(base: Path, adapter: Path, output_parent: Path) -> int:
    required = required_merge_space(base, adapter)
    if shutil.disk_usage(output_parent).free < required:
        raise OSError(f"insufficient disk for static merge: required={required} bytes")
    return required


def require_plain_merged_model(model: Any, peft_class: Any) -> None:
    if (isinstance(model, peft_class) or getattr(model, "_hf_peft_config_loaded", False)
            or any("lora_" in name for name, _ in model.named_parameters())):
        raise RuntimeError("merged model still has an active PEFT/LoRA representation")


def merge_actor_adapter(*, base_snapshot: Path, adapter: Path, actor: dict[str, Any],
                        sft_config: dict[str, Any], output: Path, versions: dict[str, str],
                        validation_messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                        on_stage: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Reusable static handoff; publish a complete merged folder only after reload."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if output.exists():
        raise FileExistsError("merged output exists; overwrite forbidden")
    for source in (base_snapshot, adapter):
        if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
            raise ValueError("merged output overlaps protected base/adapter")
    if adapter_file_identity(adapter)["adapter_fingerprint"] != actor["actor_adapter_fingerprint"]:
        raise ValueError("actor adapter changed before merge")
    check_disk_space(base_snapshot, adapter, output.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.merge-", dir=output.parent))
    stage = on_stage or (lambda name: None)
    # Failed staging artifacts remain hidden/unpublished for forensic inspection.
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    from opensearch_vl_repro.model import load_base_model, load_processor, move_batch

    config = copy.deepcopy(sft_config)
    config["model"]["name_or_path"] = str(base_snapshot)
    stage("merge_hf_load")
    processor = load_processor(config, local_files_only=True)
    model = load_base_model(config, for_training=False).to("cuda:0")
    wrapped = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, local_files_only=True)
    if not isinstance(wrapped, PeftModel) or not any("lora_" in n for n, _ in wrapped.named_parameters()):
        raise RuntimeError("merge input has no actual PEFT LoRA representation")
    stage("peft_static_merge")
    merged = wrapped.merge_and_unload(safe_merge=True)
    require_plain_merged_model(merged, PeftModel)
    merged.to(dtype=torch.bfloat16)
    merged.config._name_or_path = actor["base_model"]
    processor.tokenizer.name_or_path = actor["base_model"]
    processor.tokenizer.init_kwargs["name_or_path"] = actor["base_model"]
    stage("merged_checkpoint_save")
    merged.save_pretrained(str(staging), safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(str(staging))
    ref = weakref.ref(merged)
    del merged, wrapped, model
    gc.collect()
    torch.cuda.empty_cache()
    if ref() is not None:
        raise RuntimeError("merge-time HF model was not destroyed")
    stage("merged_checkpoint_validation")
    file_hashes = validate_merged_files(staging)
    for path in staging.glob("model*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if not list(tensors.keys()) or any("lora_" in key for key in tensors.keys()):
                raise ValueError("merged weights contain no tensors or active LoRA keys")
    stage("fresh_hf_reload")
    fresh_config = copy.deepcopy(config)
    fresh_config["model"]["name_or_path"] = str(staging)
    fresh = load_base_model(fresh_config, for_training=False).to("cuda:0").eval()
    fresh.requires_grad_(False)
    require_plain_merged_model(fresh, PeftModel)
    fresh_processor = load_processor(fresh_config, local_files_only=True)
    inputs = prepare_qwen_vl_processor_inputs(fresh_processor, validation_messages, tools)[2]
    inputs = move_batch(inputs, "cuda:0")
    stage("fresh_hf_forward")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = fresh(**inputs)
        if not bool(torch.isfinite(result.logits).all()):
            raise RuntimeError("fresh plain merged Qwen multimodal forward is nonfinite")
    fresh_ref = weakref.ref(fresh)
    del result, inputs, fresh, fresh_processor, processor
    gc.collect()
    torch.cuda.empty_cache()
    if fresh_ref() is not None:
        raise RuntimeError("fresh validation HF model was not destroyed before vLLM")
    if adapter_file_identity(adapter)["adapter_fingerprint"] != actor["actor_adapter_fingerprint"]:
        raise ValueError("source actor adapter was modified during merge")
    identity = merge_identity(actor=actor, versions=versions, file_hashes=file_hashes)
    manifest = {"identity": identity, "actor_provenance": actor, "created_at_unix": time.time(),
                "merge_complete": True, "fresh_hf_forward_finite": True, "no_active_peft": True,
                "merge_hf_destroyed": True, "reload_hf_destroyed": True}
    atomic_json(staging / "merge_manifest.json", manifest)
    if output.exists():
        raise FileExistsError("merged output appeared during merge; refusing publication")
    staging.rename(output)
    return manifest
