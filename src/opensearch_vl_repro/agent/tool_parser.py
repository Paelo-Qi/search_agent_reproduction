from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Collection


TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_CALL = re.compile(
    r"^\s*(?:```(?:json)?\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\((\{.*\})\)\s*(?:```)?\s*$",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParsedAssistantOutput:
    kind: str
    raw_text: str
    tool_calls: tuple[ParsedToolCall, ...] = ()
    final_answer: str | None = None
    error: str | None = None


class ToolCallParser:
    def __init__(self, known_tools: Collection[str]) -> None:
        self._known_tools = frozenset(known_tools)

    @staticmethod
    def _decode_call(payload: str) -> ParsedToolCall:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tool-call JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError("tool call must be a JSON object")
        function = value.get("function") if isinstance(value.get("function"), dict) else value
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool call is missing a non-empty name")
        if "arguments" not in function:
            raise ValueError("tool call is missing arguments")
        arguments = function["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"arguments contain invalid JSON: {exc.msg}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool-call arguments must be an object")
        return ParsedToolCall(name=name.strip(), arguments=arguments)

    def parse(self, text: str) -> ParsedAssistantOutput:
        if not isinstance(text, str) or not text.strip():
            return ParsedAssistantOutput(
                kind="malformed_tool_call",
                raw_text=text if isinstance(text, str) else repr(text),
                error="assistant output is empty or not text",
            )

        matches = list(TOOL_CALL_BLOCK.finditer(text))
        has_tool_syntax = "<tool_call>" in text or "</tool_call>" in text
        calls: list[ParsedToolCall] = []
        try:
            if matches:
                if text.count("<tool_call>") != len(matches) or text.count("</tool_call>") != len(matches):
                    raise ValueError("unbalanced tool_call tags")
                calls = [self._decode_call(match.group(1)) for match in matches]
            elif has_tool_syntax:
                raise ValueError("incomplete tool_call block")
            else:
                functional = FUNCTION_CALL.match(text)
                if functional:
                    payload = json.dumps(
                        {
                            "name": functional.group(1),
                            "arguments": json.loads(functional.group(2)),
                        }
                    )
                    calls = [self._decode_call(payload)]
        except (ValueError, json.JSONDecodeError) as exc:
            return ParsedAssistantOutput(
                kind="malformed_tool_call", raw_text=text, error=str(exc)
            )

        if not calls:
            return ParsedAssistantOutput(
                kind="final_answer", raw_text=text, final_answer=text.strip()
            )
        unknown = sorted({call.name for call in calls if call.name not in self._known_tools})
        if unknown:
            return ParsedAssistantOutput(
                kind="unknown_tool",
                raw_text=text,
                tool_calls=tuple(calls),
                error=f"unknown tools: {unknown}",
            )
        return ParsedAssistantOutput(
            kind="valid_tool_call", raw_text=text, tool_calls=tuple(calls)
        )

