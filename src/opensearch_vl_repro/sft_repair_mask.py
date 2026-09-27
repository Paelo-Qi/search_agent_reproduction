"""Processor-verified, fail-closed targeted SFT repair masks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import (OpenSearchVLCollator, build_messages, message_role_spans,
                   render_prompt)


MASK_VERSIONS = {"argument_only": "targeted-repair-v1-argument-only",
                 "full_tool_call": "targeted-repair-v1-full-tool-call"}
TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.S)
ARGUMENT_FRAGMENT = re.compile(
    r'"arguments"\s*:\s*\{\s*"url"\s*:\s*"img_[1-9][0-9]*"\s*\}')


def target_character_spans(record: dict[str, Any], mode: str) -> list[tuple[int, int, int, str]]:
    """Return (raw conversation turn, start, end, exact target text)."""
    if mode not in MASK_VERSIONS:
        raise ValueError(f"unknown repair mask mode: {mode}")
    target_turn = record["_repair_target_turn_index"]
    target_tool = record["_repair_target_tool"]
    if mode == "full_tool_call" and target_tool is None:
        raise ValueError("R2 requires a selected tool call")
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
    if not found or (mode == "full_tool_call" and len(found) != 1):
        raise ValueError("repair target missing or ambiguous")
    return found


def _unique_subsequence_start(haystack: list[int], needle: list[int]) -> int:
    """Find one exact token run in linear time; duplicated runs are ambiguous."""
    if not needle:
        raise ValueError("empty assistant token body")
    prefix = [0] * len(needle)
    matched = 0
    for index in range(1, len(needle)):
        while matched and needle[index] != needle[matched]:
            matched = prefix[matched - 1]
        if needle[index] == needle[matched]:
            matched += 1
        prefix[index] = matched
    found = None
    matched = 0
    for index, value in enumerate(haystack):
        while matched and value != needle[matched]:
            matched = prefix[matched - 1]
        if value == needle[matched]:
            matched += 1
        if matched == len(needle):
            start = index + 1 - len(needle)
            if found is not None:
                raise ValueError("assistant body token alignment is ambiguous")
            found = start
            matched = prefix[matched - 1]
    if found is None:
        raise ValueError("assistant body tokens are absent from complete rendered prompt")
    return found


def _complete_prompt_offsets(tokenizer: Any, prompt: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Offsets come from tokenizing the complete rendered prompt, never a prefix."""
    try:
        encoded = tokenizer(prompt, add_special_tokens=False,
                            return_offsets_mapping=True)
    except (TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError("repair alignment requires complete-prompt tokenizer offsets") from exc
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if (not isinstance(ids, list) or not ids or isinstance(ids[0], list)
            or not isinstance(offsets, list) or len(ids) != len(offsets)):
        raise ValueError("invalid complete-prompt tokenizer offsets")
    pairs = []
    for pair in offsets:
        if (len(pair) != 2 or not all(isinstance(value, int) for value in pair)
                or pair[0] < 0 or pair[1] < pair[0] or pair[1] > len(prompt)):
            raise ValueError("invalid tokenizer character offset")
        pairs.append((pair[0], pair[1]))
    return ids, pairs


def _assistant_body_alignment(prompt: str, processor_ids: list[int],
                              tokenizer_ids: list[int], offsets: list[tuple[int, int]],
                              role_span: Any, content: str) -> tuple[int, int]:
    """Map a verified processor assistant body to one complete-prompt text-token run."""
    body_ids = processor_ids[role_span.body_start:role_span.body_end]
    if len(body_ids) < 2:
        raise ValueError("invalid assistant body token span")
    start = _unique_subsequence_start(tokenizer_ids, body_ids)
    end_marker_index = start + len(body_ids) - 1
    end_marker_start = offsets[end_marker_index][0]
    if end_marker_start <= 0:
        raise ValueError("assistant structural end has no character offset")
    direct_start = end_marker_start - len(content)
    if direct_start >= 0 and prompt[direct_start:end_marker_start] == content:
        return start, direct_start
    lower = offsets[start - 1][1] if start else 0
    if lower > end_marker_start:
        raise ValueError("assistant body offset window is inverted")
    occurrences = []
    cursor = lower
    while True:
        position = prompt.find(content, cursor, end_marker_start + 1)
        if position < 0:
            break
        if position + len(content) <= end_marker_start:
            occurrences.append(position)
        cursor = position + 1
    if len(occurrences) != 1:
        raise ValueError("assistant text is not unique inside its token offset window")
    return start, occurrences[0]


def _target_token_bounds(offsets: list[tuple[int, int]], *, body_text_start: int,
                         body_token_start: int, body_token_end: int,
                         begin_char: int, end_char: int) -> tuple[int, int]:
    """Require exact token boundaries; a BPE token crossing either edge fails."""
    absolute_begin = body_text_start + begin_char
    absolute_end = body_text_start + end_char
    selected = []
    for token_index in range(body_token_start, body_token_end):
        start, end = offsets[token_index]
        if start < absolute_end and end > absolute_begin:
            if start < absolute_begin or end > absolute_end:
                raise ValueError("repair target boundary falls inside a contextual BPE token")
            selected.append(token_index)
    if (not selected or selected != list(range(selected[0], selected[-1] + 1))
            or offsets[selected[0]][0] != absolute_begin
            or offsets[selected[-1]][1] != absolute_end):
        raise ValueError("repair target has no exact contiguous token/character alignment")
    return selected[0], selected[-1] + 1


def target_token_spans(processor: Any, record: dict[str, Any], dataset_path: Path,
                       mode: str) -> tuple[list[int], list[dict[str, Any]]]:
    """Locate targets via full-prompt offsets and verified multimodal body IDs."""
    messages, images, tools = build_messages(record, dataset_path)
    prompt = render_prompt(processor, messages, tools)
    full = processor(text=[prompt], images=[images], padding=False,
                     truncation=False, return_tensors="pt")["input_ids"][0].tolist()
    text_ids, offsets = _complete_prompt_offsets(processor.tokenizer, prompt)
    assistant_spans = {span.message_index: span for span in message_role_spans(
        processor, messages, images, tools, full)}
    raw_targets = target_character_spans(record, mode)
    spans = []
    positions: list[int] = []
    aligned_bodies: dict[int, tuple[int, int]] = {}
    for turn_index, begin_char, end_char, expected in raw_targets:
        message_index = turn_index + (1 if messages[0]["role"] == "system" else 0)
        message = messages[message_index]
        if message["role"] != "assistant" or not isinstance(message["content"], str):
            raise ValueError("repair target is not assistant text")
        role_span = assistant_spans[message_index]
        if message_index not in aligned_bodies:
            aligned_bodies[message_index] = _assistant_body_alignment(
                prompt, full, text_ids, offsets, role_span, message["content"])
        text_body_start, body_char_start = aligned_bodies[message_index]
        text_begin, text_end = _target_token_bounds(
            offsets, body_text_start=body_char_start, body_token_start=text_body_start,
            body_token_end=text_body_start + role_span.body_end - role_span.body_start - 1,
            begin_char=begin_char, end_char=end_char)
        begin = role_span.body_start + text_begin - text_body_start
        end = role_span.body_start + text_end - text_body_start
        if not role_span.body_start <= begin < end < role_span.body_end:
            raise ValueError("repair target crosses assistant message boundary")
        if prompt[body_char_start + begin_char:body_char_start + end_char] != expected:
            raise ValueError("repair target does not match complete rendered prompt")
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
