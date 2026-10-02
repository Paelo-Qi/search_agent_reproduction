from __future__ import annotations

import time
from typing import Any, Callable, Protocol, Sequence

from .image_registry import ImageRegistry
from .question_normalization import normalize_model_question
from .tool_parser import ToolCallParser
from .tool_registry import ToolContext, ToolRegistry
from .interaction import (
    AGENT_SYSTEM_GUIDANCE, IMAGE_REFERENCE_ARGUMENTS, AgentInteraction,
    AgentTrajectory, AgentTurn,
)


class AgentModel(Protocol):
    def generate(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str: ...


class AgentRuntime(AgentInteraction):
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
        image_ids = [image_registry.register_initial_image(image) for image in images]
        context = ToolContext(
            image_registry=image_registry, sample_id=sample_id, benchmark=benchmark
        )
        messages = self._initial_messages(normalize_model_question(question), images, image_ids)
        declarations = self.tool_registry.declarations_for_model()
        turns: list[AgentTurn] = []
        seen_tool_calls: dict[str, int] = {}

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
                    images=self._image_summaries(image_registry),
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
                    images=self._image_summaries(image_registry),
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
            for call in parsed.tool_calls:
                tool_started = time.perf_counter()
                result = self._execute_once(call, context, seen_tool_calls, len(turns))
                tool_latency_seconds = time.perf_counter() - tool_started
                turn = self._commit_safely(
                    result=result, call=call, context=context,
                    messages=messages, assistant_output=assistant_output,
                    tool_latency_seconds=tool_latency_seconds,
                )
                turns.append(turn)

        return AgentTrajectory(
            sample_id=sample_id,
            benchmark=benchmark,
            turns=turns,
            final_answer=None,
            status="max_agent_turns_exceeded",
            image_ids=[entry.image_id for entry in image_registry.list_images()],
            error=f"no final answer after {self.max_agent_turns} model turns",
            images=self._image_summaries(image_registry),
        )
