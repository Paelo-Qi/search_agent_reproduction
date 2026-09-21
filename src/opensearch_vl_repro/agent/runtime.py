from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .image_registry import ImageRegistry
from .tool_parser import ParsedAssistantOutput, ParsedToolCall, ToolCallParser
from .tool_registry import ToolContext, ToolRegistry, ToolResult


class AgentModel(Protocol):
    def generate(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str: ...


@dataclass
class AgentTurn:
    assistant_output: str
    tool_call: dict[str, Any] | None
    observation: str | None
    status: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrajectory:
    sample_id: str
    benchmark: str
    turns: list[AgentTurn]
    final_answer: str | None
    status: str
    image_ids: list[str]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentRuntime:
    def __init__(
        self,
        *,
        model: AgentModel,
        tool_registry: ToolRegistry,
        max_agent_turns: int,
        image_registry_factory: Callable[[], ImageRegistry] = ImageRegistry,
    ) -> None:
        if max_agent_turns < 1:
            raise ValueError("max_agent_turns must be positive")
        self.model = model
        self.tool_registry = tool_registry
        self.max_agent_turns = max_agent_turns
        self.image_registry_factory = image_registry_factory
        self.parser = ToolCallParser(tool_registry.list_tools())

    @staticmethod
    def _initial_messages(question: str, images: Sequence[Any]) -> list[dict[str, Any]]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": question})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _structured_assistant_message(
        parsed: ParsedAssistantOutput,
    ) -> dict[str, Any]:
        content = re.sub(
            r"<tool_call>.*?</tool_call>", "", parsed.raw_text, flags=re.DOTALL
        ).strip()
        if not content and len(parsed.tool_calls) == 1:
            functional = re.fullmatch(
                r"\s*[A-Za-z_][A-Za-z0-9_]*\s*\(\{.*\}\)\s*",
                parsed.raw_text,
                flags=re.DOTALL,
            )
            if not functional:
                content = parsed.raw_text
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in parsed.tool_calls
            ],
        }

    @staticmethod
    def _error_result(error_type: str, detail: str) -> ToolResult:
        return ToolResult(
            status="error",
            error_type=error_type,
            observation=(
                "<observation>\nTool execution failed.\n"
                f"Error type: {error_type}\nDetail: {detail}\n</observation>"
            ),
        )

    def _execute_call(
        self, call: ParsedToolCall, context: ToolContext
    ) -> ToolResult:
        try:
            return self.tool_registry.execute(call.name, call.arguments, context)
        except Exception as exc:
            return self._error_result(type(exc).__name__, str(exc))

    def run(
        self,
        *,
        question: str,
        images: Sequence[Any],
        sample_id: str = "mock-sample",
        benchmark: str = "synthetic",
    ) -> AgentTrajectory:
        if not question.strip():
            raise ValueError("question must not be empty")
        if not images:
            raise ValueError("at least one initial image is required")
        image_registry = self.image_registry_factory()
        for image in images:
            image_registry.register_initial_image(image)
        context = ToolContext(
            image_registry=image_registry, sample_id=sample_id, benchmark=benchmark
        )
        messages = self._initial_messages(question, images)
        declarations = self.tool_registry.declarations_for_model()
        turns: list[AgentTurn] = []

        for _ in range(self.max_agent_turns):
            try:
                assistant_output = self.model.generate(
                    messages=messages, tools=declarations
                )
            except Exception as exc:
                return AgentTrajectory(
                    sample_id=sample_id,
                    benchmark=benchmark,
                    turns=turns,
                    final_answer=None,
                    status="model_error",
                    image_ids=[entry.image_id for entry in image_registry.list_images()],
                    error=f"{type(exc).__name__}: {exc}",
                )

            parsed = self.parser.parse(assistant_output)
            if parsed.kind == "final_answer":
                return AgentTrajectory(
                    sample_id=sample_id,
                    benchmark=benchmark,
                    turns=turns,
                    final_answer=parsed.final_answer,
                    status="success",
                    image_ids=[entry.image_id for entry in image_registry.list_images()],
                )

            if parsed.kind == "malformed_tool_call":
                result = self._error_result("invalid_tool_call", parsed.error or "unknown")
                turns.append(
                    AgentTurn(
                        assistant_output=assistant_output,
                        tool_call=None,
                        observation=result.observation,
                        status="error",
                        error=result.error_type,
                    )
                )
                messages.append({"role": "assistant", "content": assistant_output})
                messages.append({"role": "tool", "content": result.observation})
                continue

            messages.append(self._structured_assistant_message(parsed))
            if parsed.kind == "unknown_tool":
                for call in parsed.tool_calls:
                    if self.tool_registry.has(call.name):
                        result = self._execute_call(call, context)
                    else:
                        result = self._error_result(
                            "unknown_tool", f"unknown tool: {call.name}"
                        )
                    turns.append(
                        AgentTurn(
                            assistant_output=assistant_output,
                            tool_call={"name": call.name, "arguments": call.arguments},
                            observation=result.observation,
                            status=result.status,
                            error=result.error_type,
                            metadata=result.metadata,
                        )
                    )
                    messages.append({"role": "tool", "content": result.observation})
                continue

            for call in parsed.tool_calls:
                result = self._execute_call(call, context)
                turns.append(
                    AgentTurn(
                        assistant_output=assistant_output,
                        tool_call={"name": call.name, "arguments": call.arguments},
                        observation=result.observation,
                        status=result.status,
                        error=result.error_type,
                        metadata=result.metadata,
                    )
                )
                messages.append({"role": "tool", "content": result.observation})

        return AgentTrajectory(
            sample_id=sample_id,
            benchmark=benchmark,
            turns=turns,
            final_answer=None,
            status="max_agent_turns_exceeded",
            image_ids=[entry.image_id for entry in image_registry.list_images()],
            error=f"no final answer after {self.max_agent_turns} model turns",
        )
