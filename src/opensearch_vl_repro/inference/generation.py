from __future__ import annotations

from typing import Any

from .model_loader import InferenceBundle
from .smoke_support import StagedInferenceError


def generate_chat(
    bundle: InferenceBundle,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    import torch

    torch.manual_seed(bundle.config.seed)
    if bundle.config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(bundle.config.seed)
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if tools:
        template_kwargs["tools"] = tools
    try:
        inputs = bundle.processor.apply_chat_template(messages, **template_kwargs)
        if hasattr(inputs, "to"):
            inputs = inputs.to(bundle.config.device)
        else:
            inputs = {
                key: value.to(bundle.config.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        input_length = inputs["input_ids"].shape[-1]
    except Exception as exc:
        raise StagedInferenceError("preprocessing", exc) from exc
    try:
        with torch.inference_mode():
            output_ids = bundle.model.generate(
                **inputs, **bundle.config.generation_kwargs()
            )
    except Exception as exc:
        raise StagedInferenceError("generation", exc) from exc
    generated_ids = output_ids[:, input_length:]
    try:
        return bundle.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    except Exception as exc:
        raise StagedInferenceError("decode", exc) from exc


class QwenAgentModel:
    """Adapter exposing a real Qwen bundle through AgentRuntime's model protocol."""

    def __init__(self, bundle: InferenceBundle) -> None:
        self.bundle = bundle

    def generate(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str:
        return generate_chat(self.bundle, messages, tools=tools)
