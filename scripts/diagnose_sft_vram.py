#!/usr/bin/env python3
"""Single-GPU, forward-only VRAM trace for official 4B SFT smoke index 85.

This script deliberately uses the production processor, collator, model/LoRA
helpers, and BF16 forward path. It never starts DDP, backward, or an optimizer.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.data import OpenSearchVLCollator, load_json_records  # noqa: E402
from opensearch_vl_repro.model import (  # noqa: E402
    add_lora, freeze_vision_components, load_base_model, load_processor,
    move_batch, parameter_audit,
)
from opensearch_vl_repro.sft_main_data import canonicalize_tool_declarations  # noqa: E402
from opensearch_vl_repro.sft_train_plan import load_main_config  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/sft_4b_smoke_fa2_tmp.yaml"
SMOKE_DATA = ROOT / "data/sft_4b_smoke_100.json"
SAMPLE_INDEX = 85
GIB = 1024 ** 3


def select_official_sample(config: dict[str, Any], records: list[dict[str, Any]],
                           *, project_root: Path = ROOT) -> dict[str, Any]:
    configured = (project_root / config["data"]["path"]).resolve()
    expected = (project_root / "data/sft_4b_smoke_100.json").resolve()
    if configured != expected:
        raise ValueError(f"config must use the official 4B smoke file: {expected}")
    if len(records) != 100 or int(config["data"]["expected_samples"]) != 100:
        raise ValueError("4B smoke diagnostic requires exactly 100 official records")
    return canonicalize_tool_declarations(records[SAMPLE_INDEX])


def find_language_decoder_layers(model: Any, expected: int = 36) -> tuple[str, Any, list[Any]]:
    """Find the actual Qwen language decoder stack through PEFT wrappers."""
    candidates = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if "language_model" in name and layers is not None and len(layers) == expected:
            candidates.append((name, module, list(layers)))
    if len(candidates) != 1:
        paths = [name for name, _, _ in candidates]
        raise RuntimeError(f"expected one {expected}-layer language decoder stack; found {paths}")
    return candidates[0]


def parameter_summary(model: Any) -> dict[str, Any]:
    dtype_counts: Counter[str] = Counter()
    total = trainable = lora_trainable = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        dtype_counts[str(parameter.dtype)] += count
        if parameter.requires_grad:
            trainable += count
            if "lora_" in name.lower():
                lora_trainable += count
    return {"dtype_parameter_counts": dict(sorted(dtype_counts.items())),
            "total_parameters": total, "trainable_parameters": trainable,
            "lora_trainable_parameters": lora_trainable}


def resolved_model_flags(model: Any, language_model: Any,
                         requested_attention: str) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    language_config = getattr(language_model, "config", None)
    def attention(value: Any) -> str | None:
        return getattr(value, "_attn_implementation", None)
    resolved = {"model": attention(config), "text_config": attention(text_config),
                "language_model": attention(language_config)}
    effective = next((resolved[key] for key in ("language_model", "text_config", "model")
                      if resolved[key] is not None), None)
    return {
        "requested_attention_implementation": requested_attention,
        "resolved_attention_implementations": resolved,
        "effective_attention_implementation": effective,
        "attention_matches_request": effective == requested_attention,
        "model_is_gradient_checkpointing": getattr(model, "is_gradient_checkpointing", None),
        "language_model_gradient_checkpointing": getattr(language_model,
                                                           "gradient_checkpointing", None),
        "model_training": getattr(model, "training", None),
        "language_model_training": getattr(language_model, "training", None),
        "use_cache": getattr(config, "use_cache", None),
        "text_use_cache": getattr(text_config, "use_cache", None),
        "output_hidden_states": getattr(config, "output_hidden_states", None),
        "text_output_hidden_states": getattr(text_config, "output_hidden_states", None),
        "output_attentions": getattr(config, "output_attentions", None),
        "text_output_attentions": getattr(text_config, "output_attentions", None),
    }


def cuda_memory(torch: Any, device: Any) -> dict[str, float]:
    return {
        "allocated_gib": round(torch.cuda.memory_allocated(device) / GIB, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(device) / GIB, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated(device) / GIB, 3),
    }


def print_memory(torch: Any, device: Any, stage: str) -> dict[str, float]:
    value = cuda_memory(torch, device)
    print(f"[VRAM] stage={stage} " + " ".join(f"{key}={item:.3f}"
                                                for key, item in value.items()), flush=True)
    return value


def register_layer_hooks(layers: list[Any], torch: Any, device: Any
                         ) -> tuple[list[Any], dict[str, Any]]:
    state: dict[str, Any] = {"last_entered": None, "last_exited": None}
    handles = []
    for index, layer in enumerate(layers):
        def before(module: Any, inputs: tuple[Any, ...], layer_index: int = index) -> None:
            hidden = inputs[0] if inputs else None
            shape = list(hidden.shape) if hasattr(hidden, "shape") else None
            value = {"layer": layer_index, "hidden_states_shape": shape,
                     **cuda_memory(torch, device)}
            state["last_entered"] = value
            if layer_index % 4 == 0 or layer_index == len(layers) - 1:
                print(f"[LAYER] enter {value}", flush=True)

        def after(module: Any, inputs: tuple[Any, ...], output: Any,
                  layer_index: int = index) -> None:
            value = {"layer": layer_index, **cuda_memory(torch, device)}
            state["last_exited"] = value
            if layer_index % 4 == 0 or layer_index == len(layers) - 1:
                print(f"[LAYER] exit {value}", flush=True)

        handles.append(layer.register_forward_pre_hook(before))
        handles.append(layer.register_forward_hook(after))
    return handles, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"config is missing: {args.config}; pass --config for your AutoDL copy")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("run as one process on exactly one visible GPU, not with torchrun")

    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("VRAM diagnosis requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("VRAM diagnosis requires BF16-capable CUDA")
    device = torch.device("cuda:0")
    config = load_main_config(args.config, base_eval_config=ROOT / "configs/eval_base_300.yaml")
    records = load_json_records(SMOKE_DATA)
    sample = select_official_sample(config, records)
    print(f"[SAMPLE] index={SAMPLE_INDEX} path={SMOKE_DATA} "
          f"sample_id={sample.get('_sample_id', 'unassigned')}", flush=True)

    # Rank 1 encountered index 85. Match its seed without launching DDP.
    seed = int(config["project"]["seed"]) + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("tf32", True))

    processor = load_processor(config)
    model = load_base_model(config, for_training=True)
    print_memory(torch, device, "model_loaded")
    frozen_names = freeze_vision_components(model)
    print(f"[MODEL] frozen_vision_projector_tensors={len(frozen_names)}", flush=True)
    print_memory(torch, device, "vision_projector_frozen")
    model, targets = add_lora(model, config)
    print(f"[MODEL] resolved_lora_targets={len(targets)}", flush=True)
    print_memory(torch, device, "lora_injected")
    if config["training"].get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model = model.to(device)
    print_memory(torch, device, "model_on_cuda")
    audit = parameter_audit(model)
    print(f"[MODEL] parameter_summary={parameter_summary(model)}", flush=True)
    print(f"[MODEL] parameter_audit_total={audit['total_parameters']} "
          f"trainable={audit['trainable_parameters']}", flush=True)
    decoder_path, language_model, layers = find_language_decoder_layers(model)
    print(f"[MODEL] decoder_path={decoder_path} decoder_layers={len(layers)}", flush=True)
    requested_attention = config["model"].get("attn_implementation", "sdpa")
    flags = resolved_model_flags(model, language_model, requested_attention)
    first_attention = getattr(layers[0], "self_attn", None)
    flags["first_attention_class"] = (type(first_attention).__module__ + "."
                                      + type(first_attention).__name__
                                      if first_attention is not None else None)
    flags["first_attention_config_implementation"] = getattr(
        getattr(first_attention, "config", None), "_attn_implementation", None)
    if flags["first_attention_config_implementation"] is not None:
        flags["effective_attention_implementation"] = flags["first_attention_config_implementation"]
        flags["attention_matches_request"] = (
            flags["effective_attention_implementation"] == requested_attention)
    print(f"[MODEL] flags={flags}", flush=True)
    if not flags["attention_matches_request"]:
        raise RuntimeError("resolved attention implementation does not match requested config")

    collator = OpenSearchVLCollator(processor, SMOKE_DATA,
                                   int(config["data"]["max_length"]))
    batch = collator([sample])
    print_memory(torch, device, "collated_on_cpu")
    valid_length = int(batch.get("attention_mask", torch.ones_like(batch["input_ids"]))[0].sum())
    shapes = {name: list(value.shape) for name, value in batch.items()
              if hasattr(value, "shape")}
    print(f"[BATCH] sample_index={SAMPLE_INDEX} actual_token_length={valid_length} "
          f"max_length={config['data']['max_length']} "
          f"input_ids_shape={shapes.get('input_ids')} "
          f"pixel_values_shape={shapes.get('pixel_values')} all_shapes={shapes}", flush=True)
    batch = move_batch(batch, device)
    print_memory(torch, device, "batch_on_cuda")
    torch.cuda.reset_peak_memory_stats(device)
    handles, hook_state = register_layer_hooks(layers, torch, device)
    print_memory(torch, device, "before_forward")
    try:
        # No eval()/no_grad(): match the formal trainer's forward semantics.
        with torch.enable_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = model(**batch)
        print_memory(torch, device, "forward_complete")
        loss = getattr(result, "loss", None)
        print(f"[RESULT] forward_passed=True loss={float(loss.detach().float().item()) if loss is not None else None}",
              flush=True)
        return 0
    except torch.OutOfMemoryError:
        print(f"[OOM] sample_index={SAMPLE_INDEX} last_entered={hook_state['last_entered']} "
              f"last_exited={hook_state['last_exited']}", flush=True)
        print_memory(torch, device, "forward_oom")
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    raise SystemExit(main())
