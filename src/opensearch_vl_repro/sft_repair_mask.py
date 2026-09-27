"""Processor-verified, fail-closed targeted SFT repair masks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import (OpenSearchVLCollator, _message_image_count, _processor_ids,
                   build_messages, message_role_spans, render_prompt)


MASK_VERSIONS = {"argument_only": "targeted-repair-v1-argument-only",
                 "full_tool_call": "targeted-repair-v1-full-tool-call"}
TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.S)
ARGUMENT_FRAGMENT = re.compile(
    r'"arguments"\s*:\s*\{\s*"url"\s*:\s*"img_[1-9][0-9]*"\s*\}')
RESPONSE_BLOCK = re.compile(r"<response>.*?</response>", re.S)


def target_character_spans(record: dict[str, Any], mode: str) -> list[tuple[int, int, int, str]]:
    """Return (raw conversation turn, start, end, exact target text)."""
    if mode not in MASK_VERSIONS:
        raise ValueError(f"unknown repair mask mode: {mode}")
    target_turn = record["_repair_target_turn_index"]
    target_tool = record["_repair_target_tool"]
    found = []
    for turn_index, turn in enumerate(record["conversations"]):
        if turn["from"] != "gpt":
            continue
        content = turn["value"]
        for block in TOOL_BLOCK.finditer(content):
            try:
                parsed = json.loads(block.group()[len("<tool_call>"):-len("</tool_call>")])
            except (TypeError, ValueError):
                continue
            if not isinstance(parsed, dict):
                continue
            if mode == "argument_only":
                if parsed.get("name") != "image_search":
                    continue
                arguments = parsed.get("arguments")
                if (not isinstance(arguments, dict) or set(arguments) != {"url"}
                        or re.fullmatch(r"img_[1-9][0-9]*", str(arguments["url"])) is None):
                    raise ValueError("R1 image_search arguments are not a single registered img_n")
                fragments = list(ARGUMENT_FRAGMENT.finditer(block.group()))
                if len(fragments) != 1:
                    raise ValueError("R1 argument fragment cannot be isolated exactly")
                fragment = fragments[0]
                found.append((turn_index, block.start() + fragment.start(),
                              block.start() + fragment.end(), fragment.group()))
            elif turn_index == target_turn and parsed.get("name") == target_tool:
                found.append((turn_index, block.start(), block.end(), block.group()))
                break
        if mode == "full_tool_call" and found:
            break
    if mode == "full_tool_call" and target_tool is None:
        content = record["conversations"][target_turn]["value"]
        matches = list(RESPONSE_BLOCK.finditer(content))
        if len(matches) != 1 or TOOL_BLOCK.search(content):
            raise ValueError("R2 direct-answer target is not one isolated response block")
        match = matches[0]
        found = [(target_turn, match.start(), match.end(), match.group())]
    if not found or (mode == "full_tool_call" and len(found) != 1):
        raise ValueError("repair target missing or ambiguous")
    return found


def _prefix_boundary(processor: Any, messages: list[dict[str, Any]], images: list[Any],
                     tools: list[dict[str, Any]], message_index: int, char_cut: int,
                     full_ids: list[int]) -> int:
    prefix = messages[:message_index] + [{**messages[message_index],
                                          "content": messages[message_index]["content"][:char_cut]}]
    prefix_images = images[:_message_image_count(prefix)]
    ids = _processor_ids(processor, prefix, prefix_images, tools)
    end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_id not in ids:
        raise ValueError("assistant prefix has no template end")
    boundary = len(ids) - 1 - ids[::-1].index(end_id)
    if full_ids[:boundary] != ids[:boundary]:
        raise ValueError("repair span is not an exact processor token prefix")
    return boundary


def target_token_spans(processor: Any, record: dict[str, Any], dataset_path: Path,
                       mode: str) -> tuple[list[int], list[dict[str, Any]]]:
    """Verify every character boundary against full multimodal processor tokens."""
    messages, images, tools = build_messages(record, dataset_path)
    prompt = render_prompt(processor, messages, tools)
    full = processor(text=[prompt], images=[images], padding=False,
                     truncation=False, return_tensors="pt")["input_ids"][0].tolist()
    assistant_spans = {span.message_index: span for span in message_role_spans(
        processor, messages, images, tools, full)}
    raw_targets = target_character_spans(record, mode)
    spans = []
    positions: list[int] = []
    for turn_index, begin_char, end_char, expected in raw_targets:
        message_index = turn_index + (1 if messages[0]["role"] == "system" else 0)
        message = messages[message_index]
        if message["role"] != "assistant" or not isinstance(message["content"], str):
            raise ValueError("repair target is not assistant text")
        role_span = assistant_spans[message_index]
        begin = _prefix_boundary(processor, messages, images, tools,
                                 message_index, begin_char, full)
        end = _prefix_boundary(processor, messages, images, tools,
                               message_index, end_char, full)
        if not role_span.body_start <= begin < end < role_span.body_end:
            raise ValueError("repair target crosses assistant message boundary")
        decoded = processor.tokenizer.decode(full[begin:end], skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
        if decoded != expected:
            raise ValueError("repair span tokenizer decode differs from exact source fragment")
        positions.extend(range(begin, end))
        spans.append({"turn_index": turn_index, "start": begin, "end_exclusive": end,
                      "text": expected, "decoded": decoded, "alignment_exact": True})
    if len(positions) != len(set(positions)):
        raise ValueError("overlapping repair targets")
    return full, spans


@dataclass
class RepairCollator(OpenSearchVLCollator):
    repair_mode: str

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = super().__call__(features)  # Formal processor/collator validation remains in force.
        labels = torch.full_like(batch["input_ids"], -100)
        attention = batch.get("attention_mask")
        for row, record in enumerate(features):
            full, spans = target_token_spans(self.processor, record, self.dataset_path,
                                             self.repair_mode)
            valid = (torch.nonzero(attention[row], as_tuple=True)[0] if attention is not None
                     else torch.arange(batch["input_ids"].shape[1]))
            actual = batch["input_ids"][row, valid].tolist()
            if actual != full[:len(actual)]:
                raise ValueError("repair processor truncation is not a right-hand prefix")
            for span in spans:
                if span["end_exclusive"] > len(valid):
                    raise ValueError("repair target truncated; refusing partial supervision")
                indices = valid[span["start"]:span["end_exclusive"]]
                labels[row, indices] = batch["input_ids"][row, indices]
            if not bool((labels[row] != -100).any()):
                raise ValueError("repair sample has zero supervised tokens")
        batch["labels"] = labels
        return batch
