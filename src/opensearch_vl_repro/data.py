from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


ROLE_MAP = {"human": "user", "gpt": "assistant", "observation": "tool"}
IMAGE_MARKER = "<image>"


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Smoke dataset must be a top-level JSON array")
    return records


def parse_tools(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    tools = json.loads(value) if isinstance(value, str) else value
    if not isinstance(tools, list):
        raise ValueError("tools must decode to a list")
    return tools


def validate_raw_sample(sample: dict[str, Any], require_images: bool = True) -> None:
    conversations = sample.get("conversations")
    images = sample.get("images") or []
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations must be a non-empty list")
    if not isinstance(images, list):
        raise ValueError("images must be a list")
    if require_images and not images:
        raise ValueError("sample has no image")

    expected_groups = ({"human", "observation"}, {"gpt"})
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict) or not isinstance(turn.get("value"), str):
            raise ValueError(f"invalid conversation turn at index {index}")
        role = turn.get("from")
        if role not in ROLE_MAP:
            raise ValueError(f"unsupported role {role!r} at index {index}")
        if role not in expected_groups[index % 2]:
            raise ValueError(f"unexpected role order at index {index}: {role}")
    if conversations[-1].get("from") != "gpt":
        raise ValueError("sample must end with a gpt turn")

    marker_count = sum(turn["value"].count(IMAGE_MARKER) for turn in conversations)
    if marker_count != len(images):
        raise ValueError(
            f"image marker/path mismatch: {marker_count} markers vs {len(images)} paths"
        )
    parse_tools(sample.get("tools", ""))


def _content_parts(text: str, image_iter: Iterable[Image.Image]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    chunks = text.split(IMAGE_MARKER)
    for index, chunk in enumerate(chunks):
        if chunk:
            parts.append({"type": "text", "text": chunk})
        if index < len(chunks) - 1:
            parts.append({"type": "image", "image": next(image_iter)})
    return parts or [{"type": "text", "text": ""}]


def build_messages(
    sample: dict[str, Any], dataset_path: str | Path
) -> tuple[list[dict[str, Any]], list[Image.Image], list[dict[str, Any]]]:
    validate_raw_sample(sample)
    dataset_dir = Path(dataset_path).resolve().parent
    opened_images: list[Image.Image] = []
    for image_path in sample["images"]:
        resolved = Path(image_path)
        if not resolved.is_absolute():
            resolved = dataset_dir / resolved
        if not resolved.is_file():
            raise FileNotFoundError(f"image not found: {resolved}")
        with Image.open(resolved) as image:
            opened_images.append(image.convert("RGB"))

    image_iter = iter(opened_images)
    messages: list[dict[str, Any]] = []
    system = sample.get("system") or ""
    if system:
        messages.append({"role": "system", "content": system})

    for turn in sample["conversations"]:
        role = ROLE_MAP[turn["from"]]
        content = turn["value"]
        if IMAGE_MARKER in content:
            content = _content_parts(content, image_iter)
        messages.append({"role": role, "content": content})

    # Exhaustion here proves every path was attached exactly once.
    try:
        next(image_iter)
    except StopIteration:
        pass
    else:
        raise AssertionError("unused image after message construction")
    return messages, opened_images, parse_tools(sample.get("tools", ""))


def render_prompt(processor: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if tools:
        kwargs["tools"] = tools
    return processor.apply_chat_template(messages, **kwargs)


def find_subsequence(sequence: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        raise ValueError("needle must not be empty")
    stop = len(sequence) - len(needle) + 1
    for index in range(start, max(start, stop)):
        if sequence[index : index + len(needle)] == needle:
            return index
    return -1


def assistant_token_spans(
    token_ids: Sequence[int],
    assistant_start_ids: Sequence[int],
    message_end_id: int,
) -> list[tuple[int, int]]:
    """Return half-open spans for assistant bodies, including ``message_end_id``.

    This pure-Python helper deliberately knows nothing about torch. It makes the
    loss-mask boundary behavior testable on hosts that do not have the GPU
    training stack installed. User and tool/observation regions are excluded
    because only explicit assistant boundary sequences open a target span.
    """

    values = list(token_ids)
    start_ids = list(assistant_start_ids)
    if not start_ids:
        raise ValueError("assistant_start_ids must not be empty")

    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        begin = find_subsequence(values, start_ids, cursor)
        if begin < 0:
            break
        body_start = begin + len(start_ids)
        try:
            body_end = values.index(message_end_id, body_start)
        except ValueError:
            # Right truncation may cut an assistant turn. Its surviving body is
            # still a legitimate target, but an empty truncated body is not.
            if body_start < len(values):
                spans.append((body_start, len(values)))
                break
            raise RuntimeError("assistant block has no content or end token") from None
        spans.append((body_start, body_end + 1))
        cursor = body_end + 1

    if not spans:
        raise RuntimeError("no assistant tokens found after tokenization")
    return spans


def assistant_token_mask(input_ids: Any, tokenizer: Any, attention_mask: Any | None = None) -> Any:
    """Mask only Qwen assistant message bodies, including their end token.

    Qwen3-VL's published chat template does not expose Jinja ``generation``
    blocks, so Transformers cannot return an assistant mask. The template does
    provide stable ``<|im_start|>assistant\n ... <|im_end|>`` boundaries; this
    scanner uses those token boundaries and fails closed if a block is malformed.
    """

    import torch

    squeeze = input_ids.ndim == 1
    rows = input_ids.unsqueeze(0) if squeeze else input_ids
    attn_rows = None
    if attention_mask is not None:
        attn_rows = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

    start_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not start_ids or end_id is None or end_id == tokenizer.unk_token_id:
        raise RuntimeError("Qwen assistant boundary tokens are unavailable")

    result = torch.zeros_like(rows, dtype=torch.bool)
    for row_index, row in enumerate(rows):
        valid_len = int(attn_rows[row_index].sum().item()) if attn_rows is not None else row.numel()
        values = row[:valid_len].tolist()
        for body_start, body_end in assistant_token_spans(values, start_ids, end_id):
            result[row_index, body_start:body_end] = True
    return result[0] if squeeze else result


@dataclass
class OpenSearchVLCollator:
    processor: Any
    dataset_path: Path
    max_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        prompts: list[str] = []
        image_batches: list[list[Image.Image]] = []
        for feature in features:
            messages, images, tools = build_messages(feature, self.dataset_path)
            prompts.append(render_prompt(self.processor, messages, tools))
            image_batches.append(images)

        batch = self.processor(
            text=prompts,
            images=image_batches,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        mask = assistant_token_mask(
            batch["input_ids"], self.processor.tokenizer, batch.get("attention_mask")
        )
        labels = batch["input_ids"].clone()
        labels[~mask] = -100
        labels[batch.get("attention_mask", torch.ones_like(labels)) == 0] = -100
        if not torch.all((labels != -100).sum(dim=1) > 0):
            raise RuntimeError("at least one sample has no supervised assistant tokens")
        batch["labels"] = labels
        return dict(batch)


def tensor_shapes(batch: dict[str, Any]) -> dict[str, list[int]]:
    return {
        key: list(value.shape)
        for key, value in batch.items()
        if hasattr(value, "shape")
    }
