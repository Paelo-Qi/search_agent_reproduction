from __future__ import annotations

from typing import Any


VISION_NAME_FRAGMENTS = (
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "multimodal_projector",
    "mm_projector",
    "merger",
)


def torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def load_processor(config: dict[str, Any], *, local_files_only: bool = False) -> Any:
    from transformers import AutoProcessor

    model_cfg = config["model"]
    return AutoProcessor.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg.get("revision"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        max_pixels=int(model_cfg.get("image_max_pixels", 262144)),
        local_files_only=local_files_only,
    )


def load_base_model(config: dict[str, Any], *, for_training: bool) -> Any:
    from transformers import Qwen3VLForConditionalGeneration

    model_cfg = config["model"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg.get("revision"),
        dtype=torch_dtype(model_cfg["dtype"]),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        low_cpu_mem_usage=True,
    )
    if for_training:
        model.config.use_cache = False
    return model


def freeze_vision_components(model: Any) -> list[str]:
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        lower = name.lower()
        if any(fragment in lower for fragment in VISION_NAME_FRAGMENTS):
            parameter.requires_grad = False
            frozen.append(name)
    if not frozen:
        raise RuntimeError("no vision/projector parameters matched the freeze audit")
    return frozen


def add_lora(model: Any, config: dict[str, Any]) -> tuple[Any, list[str]]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = config["lora"]
    requested_suffixes = set(lora_cfg["target_modules"])
    resolved_targets = [
        name
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in requested_suffixes
        and isinstance(module, torch.nn.Linear)
        and not any(fragment in name.lower() for fragment in VISION_NAME_FRAGMENTS)
    ]
    if not resolved_targets:
        raise RuntimeError(
            f"no language-model linear layers matched LoRA suffixes: {sorted(requested_suffixes)}"
        )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        # Full module paths make exclusion of the vision tower explicit instead
        # of relying on architecture-specific name coincidences.
        target_modules=resolved_targets,
        bias="none",
    )
    return get_peft_model(model, peft_config), resolved_targets


def parameter_audit(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
            lower = name.lower()
            if "lora_" not in lower:
                unexpected.append(name)
            if any(fragment in lower for fragment in VISION_NAME_FRAGMENTS):
                unexpected.append(name)
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {sorted(set(unexpected))[:20]}")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total,
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_names": trainable_names,
        "unexpected_trainable": [],
    }


def select_probe_parameter(model: Any) -> tuple[str, Any]:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_B" in name
    ]
    if not candidates:
        candidates = [
            (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
    if not candidates:
        raise RuntimeError("no trainable parameter available for update proof")
    return candidates[0]


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
