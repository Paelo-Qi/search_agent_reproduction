from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image


ROLE_MAP = {"human": "user", "gpt": "assistant", "observation": "tool"}
IMAGE_MARKER = "<image>"
SFT_MASK_VERSION = "structured-message-prefix-v1"
SFT_INPUT_MESSAGE_VERSION = "runtime-image-id-grounding-v1"
SFT_RUNTIME_IMAGE_RULES = (
    "Use image tools only with registered runtime image IDs such as img_1, img_2, "
    "and later IDs listed in observations. For image_search, pass a registered "
    "img_n in its url argument. Never use a dataset filename, filesystem path, "
    "or HTTP URL as an image ID."
)


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
    # The first human turn is the runtime's initial user input. Later image
    # markers belong to tool observations and must be registered there, not
    # advertised as initial images before their producing tool has run.
    initial_count = sample["conversations"][0]["value"].count(IMAGE_MARKER)
    if initial_count:
        registered = "\n".join(
            f"- img_{index}: width={image.width}, height={image.height}"
            for index, image in enumerate(opened_images[:initial_count], 1)
        )
        grounding = f"Registered input images:\n{registered}\n\nRuntime rules:\n{SFT_RUNTIME_IMAGE_RULES}"
        system = f"{system}\n\n{grounding}" if system else grounding
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


def messages_to_json_safe(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy processor messages, replacing PIL payloads with report metadata.

    The runtime messages must retain their real images for multimodal processor
    input. This function is intentionally limited to the inspection/reporting
    boundary and never mutates the supplied message list or its content parts.
    """

    safe_messages: list[dict[str, Any]] = []
    for message in messages:
        safe_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            safe_content: list[Any] = []
            for part in content:
                if isinstance(part, dict):
                    safe_part = dict(part)
                    image = part.get("image")
                    if isinstance(image, Image.Image):
                        safe_part["image"] = {
                            "type": f"{type(image).__module__}.{type(image).__name__}",
                            "width": image.width,
                            "height": image.height,
                            "mode": image.mode,
                        }
                    safe_content.append(safe_part)
                else:
                    safe_content.append(part)
            safe_message["content"] = safe_content
        safe_messages.append(safe_message)
    return safe_messages


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

    Legacy token-only helper, retained for existing diagnostics. Training and
    formal SFT preflight use ``message_role_spans`` instead: token scanning
    cannot distinguish a template boundary from the same literal in content.
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


@dataclass(frozen=True)
class RoleTokenSpan:
    message_index: int
    body_start: int
    body_end: int  # Includes the real template end token, not trailing whitespace.


def _message_image_count(messages: Sequence[dict[str, Any]]) -> int:
    return sum(part.get("type") == "image" for message in messages
               if isinstance(message.get("content"), list)
               for part in message["content"] if isinstance(part, dict))


def _processor_ids(processor: Any, messages: list[dict[str, Any]],
                   images: list[Image.Image], tools: list[dict[str, Any]]) -> list[int]:
    prompt = render_prompt(processor, messages, tools)
    batch = processor(text=[prompt], images=[images] if images else None,
                      padding=False, truncation=False, return_tensors="pt")
    return batch["input_ids"][0].tolist()


def message_role_spans(processor: Any, messages: list[dict[str, Any]],
                       images: list[Image.Image], tools: list[dict[str, Any]],
                       full_input_ids: Sequence[int]) -> list[RoleTokenSpan]:
    """Locate assistant bodies from message roles and verified template prefixes.

    For each assistant turn, rendering an empty assistant gives its header end;
    rendering the actual turn gives its true final template end. The last
    ``<|im_end|>`` in each *message prefix* is structural, even if preceding
    content contains identical literal tokens. Every prefix is checked against
    the real multimodal processor sequence; incompatible templates fail closed.
    """
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id is None or end_id == processor.tokenizer.unk_token_id:
        raise RuntimeError("Qwen message end token is unavailable")
    full = list(full_input_ids)
    spans: list[RoleTokenSpan] = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prior = messages[:index]
        prior_images = images[:_message_image_count(prior)]
        empty = prior + [{**message, "content": ""}]
        empty_ids = _processor_ids(processor, empty, prior_images, tools)
        if end_id not in empty_ids:
            raise RuntimeError(f"assistant template end missing at message {index}")
        body_start = len(empty_ids) - 1 - empty_ids[::-1].index(end_id)
        if full[:body_start] != empty_ids[:body_start]:
            raise RuntimeError(f"assistant header is not a multimodal token prefix at message {index}")

        through = messages[:index + 1]
        actual_images = images[:_message_image_count(through)]
        actual_ids = _processor_ids(processor, through, actual_images, tools)
        if full[:len(actual_ids)] != actual_ids:
            raise RuntimeError(f"message tokens are not a multimodal prefix at message {index}")
        if end_id not in actual_ids:
            raise RuntimeError(f"assistant template end missing at message {index}")
        body_end = len(actual_ids) - actual_ids[::-1].index(end_id)
        if actual_ids[body_end:] != empty_ids[body_start + 1:]:
            raise RuntimeError(f"assistant template suffix changed at message {index}")
        if not body_start < body_end:
            raise RuntimeError(f"empty assistant target at message {index}")
        spans.append(RoleTokenSpan(index, body_start, body_end))
    if not spans:
        raise RuntimeError("structured trajectory has no assistant turn")
    return spans


def supervised_positions(spans: Sequence[RoleTokenSpan], cutoff: int) -> set[int]:
    """Pure-Python source of truth for right-truncated assistant supervision."""
    return {position for span in spans
            for position in range(span.body_start, min(span.body_end, cutoff))}


def assistant_token_mask(input_ids: Any, spans_by_row: Sequence[Sequence[RoleTokenSpan]],
                         attention_mask: Any | None = None) -> Any:
    """Apply structured assistant spans to padded processor output."""

    import torch

    squeeze = input_ids.ndim == 1
    rows = input_ids.unsqueeze(0) if squeeze else input_ids
    attn_rows = None
    if attention_mask is not None:
        attn_rows = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

    result = torch.zeros_like(rows, dtype=torch.bool)
    for row_index, row in enumerate(rows):
        valid = (torch.nonzero(attn_rows[row_index], as_tuple=True)[0]
                 if attn_rows is not None else torch.arange(row.numel(), device=row.device))
        for span in spans_by_row[row_index]:
            stop = min(span.body_end, len(valid))
            if span.body_start < stop:
                result[row_index, valid[span.body_start:stop]] = True
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
        message_batches: list[list[dict[str, Any]]] = []
        tool_batches: list[list[dict[str, Any]]] = []
        for feature in features:
            messages, images, tools = build_messages(feature, self.dataset_path)
            prompts.append(render_prompt(self.processor, messages, tools))
            image_batches.append(images)
            message_batches.append(messages)
            tool_batches.append(tools)

        batch = self.processor(
            text=prompts,
            images=image_batches,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)
        full_batch = self.processor(text=prompts, images=image_batches, padding=True,
                                    truncation=False, return_tensors="pt")
        spans_by_row = []
        for index, messages in enumerate(message_batches):
            full_valid = full_batch["input_ids"][index]
            if "attention_mask" in full_batch:
                full_valid = full_valid[full_batch["attention_mask"][index].bool()]
            full_ids = full_valid.tolist()
            truncated_valid = batch["input_ids"][index]
            if "attention_mask" in batch:
                truncated_valid = truncated_valid[batch["attention_mask"][index].bool()]
            truncated_ids = truncated_valid.tolist()
            if truncated_ids != full_ids[:len(truncated_ids)]:
                raise RuntimeError("multimodal processor truncation is not a right-hand prefix")
            spans_by_row.append(message_role_spans(
                self.processor, messages, image_batches[index], tool_batches[index], full_ids))
        mask = assistant_token_mask(
            batch["input_ids"], spans_by_row, batch.get("attention_mask")
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
