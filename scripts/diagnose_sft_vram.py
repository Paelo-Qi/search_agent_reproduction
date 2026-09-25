#!/usr/bin/env python3
"""Single-GPU VRAM stress test for official 4B SFT smoke or formal shards.

This script deliberately uses the production processor, collator, model/LoRA
helpers, and BF16 forward path. Optional backward never steps an optimizer.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.data import (  # noqa: E402
    OpenSearchVLCollator, build_messages, load_json_records, render_prompt,
)
from opensearch_vl_repro.model import (  # noqa: E402
    add_lora, freeze_vision_components, load_base_model, load_processor,
    move_batch, parameter_audit,
)
from opensearch_vl_repro.sft_main_data import (  # noqa: E402
    SHARD_SIZES, canonicalize_tool_declarations,
)
from opensearch_vl_repro.sft_long_training import activate_sft_training_mode  # noqa: E402
from opensearch_vl_repro.sft_train_plan import load_main_config  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/sft_4b_smoke_fa2_tmp.yaml"
SMOKE_DATA = ROOT / "data/sft_4b_smoke_100.json"
SAMPLE_INDEX = 85
GIB = 1024 ** 3


def validate_dataset(config: dict[str, Any], data_path: str | Path,
                     *, project_root: Path = ROOT) -> tuple[Path, int]:
    """Allow only the independent smoke set or a fixed formal SFT shard."""
    path = (project_root / data_path).resolve()
    smoke = (project_root / "data/sft_4b_smoke_100.json").resolve()
    if path == smoke:
        configured = config["data"].get("path")
        if configured is None or (project_root / configured).resolve() != smoke:
            raise ValueError("config must use the official 4B smoke file")
        if int(config["data"].get("expected_samples", -1)) != 100:
            raise ValueError("4B smoke diagnostic requires exactly 100 official records")
        return path, 100
    pool_dir = config["data"].get("pool_dir")
    if pool_dir is None:
        raise ValueError("formal shard requires a formal SFT config with data.pool_dir")
    allowed = {(project_root / pool_dir / f"{name}.json").resolve(): count
               for name, count in SHARD_SIZES.items()}
    if path not in allowed:
        raise ValueError(f"data must be the official smoke file or a formal SFT shard: {path}")
    return path, allowed[path]


def training_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Use the runtime tool schema without mutating a raw or prepared record."""
    record = canonicalize_tool_declarations(raw)
    if "_source_tools" in raw:
        record["_source_tools"] = raw["_source_tools"]
    return record


def processor_token_length(processor: Any, record: dict[str, Any],
                           dataset_path: Path) -> int:
    """Use the preflight/collator message template and untruncated processor IDs."""
    messages, images, tools = build_messages(record, dataset_path)
    prompt = render_prompt(processor, messages, tools)
    encoded = processor(text=[prompt], images=[images], padding=False,
                        truncation=False, return_tensors="pt")
    return len(encoded["input_ids"][0])


def select_sample(records: list[dict[str, Any]], *, sample_index: int | None,
                  select_longest: bool, token_length: Any) -> tuple[int, dict[str, Any], int]:
    """Choose by actual processor length; ties keep the first dataset index."""
    if not records:
        raise ValueError("diagnostic dataset is empty")
    if select_longest:
        if sample_index is not None:
            raise ValueError("--sample-index and --select-longest are mutually exclusive")
        best_index, best_length = 0, -1
        for index, record in enumerate(records):
            length = int(token_length(record))
            if length > best_length:
                best_index, best_length = index, length
        return best_index, records[best_index], best_length
    index = SAMPLE_INDEX if sample_index is None else sample_index
    if not 0 <= index < len(records):
        raise ValueError(f"sample index {index} is outside 0..{len(records) - 1}")
    record = records[index]
    return index, record, int(token_length(record))


def should_run_backward(with_backward: bool, loss_value: float | None) -> bool:
    """Keep the optional backward decision testable without a GPU."""
    if not with_backward:
        return False
    if loss_value is None or not math.isfinite(loss_value):
        raise ValueError("backward requires a finite forward loss")
    return True


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


def resolved_attention_flags(model: Any, language_model: Any, layers: list[Any],
                             requested_attention: str) -> dict[str, Any]:
    """Check both model config and the first decoder's actual attention config."""
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
    return flags


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
    parser.add_argument("--data", type=Path, default=SMOKE_DATA)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-index", type=int)
    selection.add_argument("--select-longest", action="store_true")
    parser.add_argument("--with-backward", action="store_true")
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
    properties = torch.cuda.get_device_properties(device)
    print(f"[CUDA] device={properties.name} total_gib={properties.total_memory / GIB:.3f}",
          flush=True)
    config = load_main_config(args.config, base_eval_config=ROOT / "configs/eval_base_300.yaml")
    data_path, expected_samples = validate_dataset(config, args.data)
    records = load_json_records(data_path)
    if len(records) != expected_samples:
        raise ValueError(f"expected {expected_samples} official records in {data_path}, got {len(records)}")

    # Rank 1 encountered index 85. Match its seed without launching DDP.
    seed = int(config["project"]["seed"]) + 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("tf32", True))

    processor = load_processor(config)
    scanned = 0
    def length_of(raw: dict[str, Any]) -> int:
        nonlocal scanned
        length = processor_token_length(processor, training_record(raw), data_path)
        scanned += 1
        if args.select_longest and scanned % 100 == 0:
            print(f"[SCAN] processor_lengths_completed={scanned}/{len(records)}", flush=True)
        return length

    sample_index, raw_sample, actual_length = select_sample(
        records, sample_index=args.sample_index,
        select_longest=args.select_longest, token_length=length_of)
    sample = training_record(raw_sample)
    identity = {key: sample[key] for key in (
        "_sample_id", "_source", "_source_index", "_original_index",
        "source", "original_index") if key in sample}
    print(f"[SAMPLE] index={sample_index} path={data_path} identity={identity} "
          f"actual_token_length={actual_length} image_count={len(sample.get('images', []))}",
          flush=True)
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
    activate_sft_training_mode(model)
    print_memory(torch, device, "model_on_cuda")
    audit = parameter_audit(model)
    print(f"[MODEL] parameter_summary={parameter_summary(model)}", flush=True)
    print(f"[MODEL] parameter_audit_total={audit['total_parameters']} "
          f"trainable={audit['trainable_parameters']}", flush=True)
    decoder_path, language_model, layers = find_language_decoder_layers(model)
    print(f"[MODEL] decoder_path={decoder_path} decoder_layers={len(layers)}", flush=True)
    print(f"[MODEL] decoder_layers_training={sum(layer.training for layer in layers)}/{len(layers)} "
          f"decoder_layers_gradient_checkpointing="
          f"{sum(bool(getattr(layer, 'gradient_checkpointing', False)) for layer in layers)}/{len(layers)}",
          flush=True)
    requested_attention = config["model"].get("attn_implementation", "sdpa")
    flags = resolved_attention_flags(model, language_model, layers, requested_attention)
    print(f"[MODEL] flags={flags}", flush=True)
    if not flags["attention_matches_request"]:
        raise RuntimeError("resolved attention implementation does not match requested config")

    collator = OpenSearchVLCollator(processor, data_path,
                                   int(config["data"]["max_length"]))
    batch = collator([sample])
    print_memory(torch, device, "collated_on_cpu")
    valid_length = int(batch.get("attention_mask", torch.ones_like(batch["input_ids"]))[0].sum())
    shapes = {name: list(value.shape) for name, value in batch.items()
              if hasattr(value, "shape")}
    print(f"[BATCH] sample_index={sample_index} actual_token_length={actual_length} "
          f"training_input_length={valid_length} "
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
        torch.cuda.synchronize(device)
        forward_memory = print_memory(torch, device, "forward_complete")
        loss = getattr(result, "loss", None)
        loss_value = float(loss.detach().float().item()) if loss is not None else None
        if loss_value is None or not math.isfinite(loss_value):
            raise RuntimeError(f"forward did not produce a finite loss: {loss_value}")
        print(f"[RESULT] forward_passed=True loss={loss_value} "
              f"forward_peak_allocated_gib={forward_memory['max_allocated_gib']}", flush=True)
    except torch.OutOfMemoryError:
        print(f"[OOM] stage=forward sample_index={sample_index} "
              f"last_entered={hook_state['last_entered']} "
              f"last_exited={hook_state['last_exited']}", flush=True)
        print_memory(torch, device, "forward_oom")
        raise
    finally:
        for handle in handles:
            handle.remove()

    if should_run_backward(args.with_backward, loss_value):
        torch.cuda.reset_peak_memory_stats(device)
        print_memory(torch, device, "before_backward")
        try:
            # One unscaled micro-batch loss, as requested; no DDP or optimizer.
            loss.backward()
            torch.cuda.synchronize(device)
            backward_memory = print_memory(torch, device, "backward_complete")
            print(f"[RESULT] backward_passed=True "
                  f"backward_peak_allocated_gib={backward_memory['max_allocated_gib']}", flush=True)
        except torch.OutOfMemoryError:
            print(f"[OOM] stage=backward sample_index={sample_index}", flush=True)
            print_memory(torch, device, "backward_oom")
            raise
    else:
        print("[RESULT] backward_skipped=True", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
