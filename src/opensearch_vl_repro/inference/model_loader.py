from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import InferenceConfig


@dataclass
class InferenceBundle:
    config: InferenceConfig
    model: Any
    processor: Any
    environment: dict[str, Any]


def _torch_dtype(torch_module: Any, name: str) -> Any:
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported inference dtype: {name}") from exc


def validate_device(config: InferenceConfig, torch_module: Any) -> None:
    if config.device.startswith("cuda"):
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is required by the inference config but is not available")
        if config.dtype.lower() in {"bfloat16", "bf16"} and not torch_module.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support BF16")


def inference_environment(config: InferenceConfig, torch_module: Any, transformers_module: Any) -> dict[str, Any]:
    gpu_name = None
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if config.device.startswith("cuda") and torch_module.cuda.is_available():
        gpu_name = torch_module.cuda.get_device_name(config.device)
    return {
        "model": config.model_name_or_path,
        "model_revision": config.revision,
        "torch_version": torch_module.__version__,
        "transformers_version": transformers_module.__version__,
        "dtype": config.dtype,
        "device": config.device,
        "gpu": gpu_name,
        "cuda_version": cuda_version,
    }


def load_inference_bundle(
    config: InferenceConfig,
    *,
    model_class: Any | None = None,
    processor_class: Any | None = None,
    torch_module: Any | None = None,
    transformers_module: Any | None = None,
) -> InferenceBundle:
    if torch_module is None:
        import torch as torch_module
    if transformers_module is None:
        import transformers as transformers_module
    if model_class is None:
        from transformers import Qwen3VLForConditionalGeneration

        model_class = Qwen3VLForConditionalGeneration
    if processor_class is None:
        from transformers import AutoProcessor

        processor_class = AutoProcessor

    validate_device(config, torch_module)
    processor = processor_class.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
        max_pixels=config.image_max_pixels,
    )
    model = model_class.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        dtype=_torch_dtype(torch_module, config.dtype),
        device_map=config.device,
        attn_implementation=config.attn_implementation,
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    return InferenceBundle(
        config=config,
        model=model,
        processor=processor,
        environment=inference_environment(config, torch_module, transformers_module),
    )

