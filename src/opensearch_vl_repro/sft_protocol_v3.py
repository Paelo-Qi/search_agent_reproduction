"""Fail-closed, source-only image_search.url -> runtime image_id adaptation.

This module never participates in inference. It edits only JSON keys inside
parsed source image_search calls and preserves surrounding assistant text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from .agent.tool_parser import FUNCTION_CALL, TOOL_CALL_BLOCK, ToolCallParser


ASSISTANT_CALL_TRANSFORM_VERSION = "source-image-search-url-to-image-id-v1"
REGISTERED_ID = re.compile(r"img_[1-9][0-9]*\Z")
_PARSER = ToolCallParser(TOOL_DECLARATIONS_BY_NAME)
_DECODER = json.JSONDecoder()
_IMAGE_CALL_LIKE = re.compile(r"^\s*(?:```(?:json)?\s*)?image_search\s*\(|<tool_call>.*?\"name\"\s*:\s*\"image_search\"",
                              re.DOTALL)


def _skip_space(value: str, index: int) -> int:
    while index < len(value) and value[index].isspace():
        index += 1
    return index


def _object_fields(value: str, start: int = 0) -> list[tuple[str, int, int, int, int, Any]]:
    """Return lexical spans of direct JSON object members, rejecting duplicates."""
    index = _skip_space(value, start)
    if index >= len(value) or value[index] != "{":
        raise ValueError("image_search call must contain a JSON object")
    index += 1
    fields = []
    seen = set()
    while True:
        index = _skip_space(value, index)
        if index >= len(value):
            raise ValueError("unterminated image_search JSON object")
        if value[index] == "}":
            return fields
        key_start = index
        key, index = _DECODER.raw_decode(value, index)
        key_end = index
        if not isinstance(key, str) or key in seen:
            raise ValueError("non-string or duplicate image_search JSON key")
        seen.add(key)
        index = _skip_space(value, index)
        if index >= len(value) or value[index] != ":":
            raise ValueError("image_search JSON member lacks colon")
        value_start = _skip_space(value, index + 1)
        decoded, value_end = _DECODER.raw_decode(value, value_start)
        fields.append((key, key_start, key_end, value_start, value_end, decoded))
        index = _skip_space(value, value_end)
        if index < len(value) and value[index] == ",":
            index += 1
            continue
        if index < len(value) and value[index] == "}":
            return fields
        raise ValueError("invalid image_search JSON object separator")


def _field(value: str, key: str, start: int = 0) -> tuple[str, int, int, int, int, Any]:
    matching = [field for field in _object_fields(value, start) if field[0] == key]
    if len(matching) != 1:
        raise ValueError(f"image_search JSON requires one {key!r} field")
    return matching[0]


def _rewrite_arguments(value: str, start: int) -> tuple[int, int, str]:
    fields = _object_fields(value, start)
    if {field[0] for field in fields} != {"url"}:
        raise ValueError("legacy image_search arguments must contain only url")
    url = fields[0]
    if not isinstance(url[5], str) or REGISTERED_ID.fullmatch(url[5]) is None:
        raise ValueError("legacy image_search.url must be a registered img_n target")
    return url[1], url[2], '"image_id"'


def _rewrite_call_payload(payload: str) -> str:
    outer = _object_fields(payload)
    function = next((field for field in outer if field[0] == "function"), None)
    call_start = function[3] if function is not None else 0
    arguments = _field(payload, "arguments", call_start)
    if isinstance(arguments[5], dict):
        begin, end, replacement = _rewrite_arguments(payload, arguments[3])
        return payload[:begin] + replacement + payload[end:]
    if isinstance(arguments[5], str):
        inner = arguments[5]
        begin, end, replacement = _rewrite_arguments(inner, 0)
        changed = inner[:begin] + replacement + inner[end:]
        return (payload[:arguments[3]] + json.dumps(changed, ensure_ascii=False)
                + payload[arguments[4]:])
    raise ValueError("legacy image_search arguments must be an object or JSON object string")


def canonicalize_source_assistant_text(text: str) -> str:
    """Transform only verified source image_search calls; never use global replace."""
    parsed = _PARSER.parse(text)
    if parsed.kind != "valid_tool_call" and _IMAGE_CALL_LIKE.search(text):
        raise ValueError(f"cannot canonicalize malformed source image_search: {parsed.error}")
    if not parsed.tool_calls or not any(call.name == "image_search" for call in parsed.tool_calls):
        return text
    blocks = list(TOOL_CALL_BLOCK.finditer(text))
    if blocks:
        if len(blocks) != len(parsed.tool_calls):
            raise ValueError("source tool-call block count changed")
        changed = text
        for block, call in reversed(list(zip(blocks, parsed.tool_calls, strict=True))):
            if call.name != "image_search":
                continue
            payload = _rewrite_call_payload(block.group(1))
            changed = changed[:block.start(1)] + payload + changed[block.end(1):]
    else:
        functional = FUNCTION_CALL.match(text)
        if functional is None or len(parsed.tool_calls) != 1:
            raise ValueError("unsupported source image_search call representation")
        payload = functional.group(2)
        begin, end, replacement = _rewrite_arguments(payload, 0)
        changed_payload = payload[:begin] + replacement + payload[end:]
        changed = text[:functional.start(2)] + changed_payload + text[functional.end(2):]
    reparsed = _PARSER.parse(changed)
    if reparsed.kind != "valid_tool_call" or len(reparsed.tool_calls) != len(parsed.tool_calls):
        raise ValueError("canonicalized image_search call no longer parses")
    for before, after in zip(parsed.tool_calls, reparsed.tool_calls, strict=True):
        if before.name != after.name:
            raise ValueError("canonicalization changed tool selection")
        expected = ({"image_id": before.arguments["url"]}
                    if before.name == "image_search" else before.arguments)
        if after.arguments != expected:
            raise ValueError("canonicalization changed non-image-search arguments")
    return changed


def effective_image_search_call_counts(record: dict[str, Any]) -> dict[str, int]:
    """Count effective calls and reject any legacy/invalid model-facing target."""
    counts = {"image_search_image_id": 0, "image_search_legacy_url": 0,
              "image_search_http_target": 0, "image_search_non_img_n": 0,
              "image_search_extra_arguments": 0}
    for turn in record["conversations"]:
        if turn["from"] != "gpt":
            continue
        parsed = _PARSER.parse(turn["value"])
        if parsed.kind != "valid_tool_call" and _IMAGE_CALL_LIKE.search(turn["value"]):
            raise ValueError("effective image_search call is malformed")
        for call in parsed.tool_calls:
            if call.name != "image_search":
                continue
            arguments = call.arguments
            if "url" in arguments:
                counts["image_search_legacy_url"] += 1
            if set(arguments) != {"image_id"}:
                counts["image_search_extra_arguments"] += 1
            reference = arguments.get("image_id")
            if isinstance(reference, str) and re.search(r"https?://", reference, re.I):
                counts["image_search_http_target"] += 1
            if not isinstance(reference, str) or REGISTERED_ID.fullmatch(reference) is None:
                counts["image_search_non_img_n"] += 1
            if "image_id" in arguments:
                counts["image_search_image_id"] += 1
    return counts
