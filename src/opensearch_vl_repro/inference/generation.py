from __future__ import annotations

from typing import Any

from .model_loader import InferenceBundle


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
    inputs = bundle.processor.apply_chat_template(messages, **template_kwargs)
    if hasattr(inputs, "to"):
        inputs = inputs.to(bundle.config.device)
    else:
        inputs = {
            key: value.to(bundle.config.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = bundle.model.generate(
            **inputs, **bundle.config.generation_kwargs()
        )
    generated_ids = output_ids[:, input_length:]
    return bundle.processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


class QwenAgentModel:
    """Adapter exposing a real Qwen bundle through AgentRuntime's model protocol."""

    def __init__(self, bundle: InferenceBundle) -> None:
        self.bundle = bundle

    def generate(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str:
        return generate_chat(self.bundle, messages, tools=tools)

