from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence

from .image_registry import ImageRegistry
from .reliability import canonical_json, image_sha256
from .tool_parser import ParsedAssistantOutput, ParsedToolCall, ToolCallParser
from .tool_registry import ToolContext, ToolRegistry, ToolResult


class AgentModel(Protocol):
    def generate(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str: ...


AGENT_SYSTEM_GUIDANCE = """You are a Visual Investigation Agent. Answer the user's question accurately from the image and, when needed, evidence obtained with the available tools.

Core policy — Verify, Don't Guess:
Before each action, assess what the user asks, image quality, what is directly visible, what information is missing, and the next useful action. Prefer a tool when it can clarify a small or unclear visual detail or verify a needed fact. A clear image with a directly answerable question needs no tool call. For complex questions, chain useful tools and inspect each observation before deciding whether another step is needed; do not stop after the first tool call while a relevant gap remains.

Tool selection:
- crop: Isolate a relevant object, text region, or chart section that is small relative to the full image or surrounded by distracting content.
- layout_parsing: Extract text and structure from document-like images, receipts, labels, tables, or charts when accurate reading or layout matters. Check its observation rather than inventing text it did not return.
- perspective_correct: Straighten a document or text region photographed at an angle or visibly skewed.
- super_resolution: Enlarge a genuinely low-resolution or pixelated image or relevant region.
- sharpen: Improve blurred text or soft edges when that could make details readable.
- image_search: Use visual matching to identify an unknown landmark, object, artwork, product, or scene. It provides candidate identities and source links, not a complete factual answer.
- text_search: Look up a known entity, specific fact, or context not directly visible in pixels. It returns search results and page passages when available; verify claims against the returned evidence.
- web_search: Get concise web result titles, URLs, and snippets for a text query. Use text_search when page-level passages are needed.

Retrieval workflow:
If the answer depends on a specific entity, fact, history, or other context not directly visible in the image, use retrieval to validate it. After image_search suggests an identity, normally follow with text_search when the question asks for detailed facts about that entity. Useful sequences include perspective_correct -> sharpen -> layout_parsing for skewed blurry text, crop -> layout_parsing for a small document region, and image_search -> text_search for visual identification followed by factual lookup. Use the newly registered image ID from a visual tool's observation for the next image step.

Runtime rules:
Use image tools only with registered runtime image IDs such as img_1, img_2, and later IDs listed in observations. For image_search, pass a registered img_n in its url argument. Never use a dataset filename, filesystem path, or HTTP URL as an image ID. Do not repeat an identical tool call that has already been executed. If a tool call fails, do not retry the same invalid arguments; use the available image IDs and previous observations to change arguments or strategy. Use additional tools only when they are likely to add evidence. When the available evidence is sufficient, stop using tools and provide the final answer."""

IMAGE_REFERENCE_ARGUMENTS = {
    "image_search": "url", "crop": "image", "layout_parsing": "image",
    "super_resolution": "image", "sharpen": "image", "perspective_correct": "image",
}


@dataclass
class AgentTurn:
    assistant_output: str
    tool_call: dict[str, Any] | None
    observation: str | None
    status: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    derived_images: list[dict[str, Any]] = field(default_factory=list)
    tool_latency_seconds: float | None = None


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
    images: list[dict[str, Any]] = field(default_factory=list)

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
    def _initial_messages(
        question: str, images: Sequence[Any], image_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": question})
        registered = "\n".join(f"- {image_id}" for image_id in image_ids)
        system = f"{AGENT_SYSTEM_GUIDANCE}\n\nRegistered input images:\n{registered}"
        return [{"role": "system", "content": system},
                {"role": "user", "content": content}]

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
        reference_argument = IMAGE_REFERENCE_ARGUMENTS.get(call.name)
        reference = call.arguments.get(reference_argument) if reference_argument else None
        if isinstance(reference, str) and not context.image_registry.exists(reference):
            available = ", ".join(
                entry.image_id for entry in context.image_registry.list_images()
            ) or "none"
            return ToolResult(
                status="error", error_type="unknown_image_id",
                observation=(
                    "<observation>\n"
                    f"Unknown registered image ID: {reference[:240]}\n"
                    f"Available registered image IDs: {available}\n"
                    "Use one of the registered IDs above. Do not use filenames, "
                    "filesystem paths, or HTTP URLs.\n</observation>"
                ),
                metadata={"provider_called": False, "available_image_ids": available.split(", ")},
            )
        try:
            return self.tool_registry.execute(call.name, call.arguments, context)
        except Exception as exc:
            return self._error_result(type(exc).__name__, str(exc))

    @staticmethod
    def _image_size(value: Any) -> tuple[int, int]:
        from PIL import Image
        from pathlib import Path

        if isinstance(value, Image.Image):
            return value.size
        with Image.open(Path(value)) as image:
            return image.size

    @classmethod
    def _image_summaries(cls, registry: ImageRegistry) -> list[dict[str, Any]]:
        return [
            {
                "image_id": entry.image_id,
                "parent_id": entry.parent_id,
                "kind": entry.kind,
                "size": list(cls._image_size(entry.value)),
                "sha256": image_sha256(entry.value),
                "metadata": entry.metadata,
            }
            for entry in registry.list_images()
        ]

    def _commit_result(
        self,
        *,
        result: ToolResult,
        call: ParsedToolCall,
        context: ToolContext,
        messages: list[dict[str, Any]],
        assistant_output: str,
        tool_latency_seconds: float | None = None,
    ) -> AgentTurn:
        derived: list[dict[str, Any]] = []
        content: list[dict[str, Any]] = []
        # Validate the whole batch before assigning any IDs, so a malformed
        # second image cannot leave a half-committed tool result behind.
        for item in result.derived_images:
            if not isinstance(item.metadata, dict):
                raise TypeError("derived image metadata must be a dict")
            context.image_registry.get(item.parent_id)
            context.image_registry._validate_value(item.value)
            self._image_size(item.value)
        for item in result.derived_images:
            parent_size = self._image_size(context.image_registry.get(item.parent_id))
            image_id = context.image_registry.register_derived_image(
                item.value,
                parent_id=item.parent_id,
                metadata={**item.metadata, "producing_tool": call.name},
            )
            entry = context.image_registry.get_entry(image_id)
            entry.metadata["result_image_id"] = image_id
            summary = {
                "parent_image_id": item.parent_id,
                "image_id": image_id,
                "producing_tool": call.name,
                "source_size": list(parent_size),
                "result_size": list(self._image_size(entry.value)),
                "metadata": entry.metadata,
            }
            derived.append(summary)
            # The PIL object, not just its img_n identifier, must reach
            # processor.apply_chat_template on the next model turn.
            from PIL import Image

            if isinstance(entry.value, Image.Image):
                visual = entry.value
            else:
                with Image.open(entry.value) as loaded:
                    visual = loaded.convert("RGB").copy()
            content.append({"type": "image", "image": visual})
        observation = result.observation
        if derived:
            ids_text = "\n".join(
                f"New image ID: {item['image_id']}." for item in derived
            )
            if "</observation>" in observation:
                observation = observation.replace(
                    "</observation>", f"{ids_text}\n</observation>", 1
                )
            else:
                observation = f"{observation.rstrip()}\n{ids_text}"
        if content:
            content.append({"type": "text", "text": observation})
            messages.append({"role": "tool", "content": content})
        else:
            messages.append({"role": "tool", "content": observation})
        return AgentTurn(
            assistant_output=assistant_output,
            tool_call={"name": call.name, "arguments": call.arguments},
            observation=observation,
            status=result.status,
            error=result.error_type,
            metadata={**result.metadata, "derived_image_ids": [d["image_id"] for d in derived]},
            derived_images=derived,
            tool_latency_seconds=tool_latency_seconds,
        )

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
        messages = self._initial_messages(question, images, image_ids)
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
                signature = f"{call.name}\0{canonical_json(call.arguments)}"
                original_turn = seen_tool_calls.get(signature)
                if original_turn is not None:
                    result = ToolResult(
                        status="error", error_type="duplicate_tool_call",
                        observation=(
                            "<observation>\nThis exact tool call has already been executed "
                            "in this trajectory. Do not repeat the same call. Use the previous "
                            "evidence, change the query/tool/strategy, or provide a final answer "
                            "if the evidence is sufficient.\n</observation>"
                        ),
                        metadata={"duplicate_tool_call": True,
                                  "original_tool_call_turn": original_turn,
                                  "provider_called": False},
                    )
                else:
                    seen_tool_calls[signature] = len(turns) + 1
                    result = (
                        self._execute_call(call, context)
                        if self.tool_registry.has(call.name)
                        else self._error_result("unknown_tool", f"unknown tool: {call.name}")
                    )
                tool_latency_seconds = time.perf_counter() - tool_started
                try:
                    turn = self._commit_result(
                        result=result, call=call, context=context,
                        messages=messages, assistant_output=assistant_output,
                        tool_latency_seconds=tool_latency_seconds,
                    )
                except Exception as exc:
                    turn = self._commit_result(
                        result=self._error_result(type(exc).__name__, str(exc)),
                        call=call, context=context, messages=messages,
                        assistant_output=assistant_output,
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
