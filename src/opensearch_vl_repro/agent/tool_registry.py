from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .image_registry import ImageRegistry
from .tool_contracts import ToolDeclaration


@dataclass(frozen=True)
class DerivedImage:
    value: Any
    parent_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    status: str
    observation: str
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    derived_images: tuple[DerivedImage, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"success", "error"}:
            raise ValueError(f"unsupported ToolResult status: {self.status}")
        if not isinstance(self.observation, str):
            raise TypeError("model-facing observation must be plain text")
        if self.status != "success" and self.derived_images:
            raise ValueError("failed tool results cannot contain derived images")


@dataclass
class ToolContext:
    image_registry: ImageRegistry
    sample_id: str | None = None
    benchmark: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


ToolBackend = Callable[[dict[str, Any], ToolContext], ToolResult]


@dataclass(frozen=True)
class RegisteredTool:
    declaration: ToolDeclaration
    backend: ToolBackend


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        name = tool.declaration.name
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def declarations_for_model(self) -> list[dict[str, Any]]:
        return [tool.declaration.as_chat_template_tool() for tool in self._tools.values()]

    def execute(
        self, name: str, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        tool = self.get(name)
        validated = tool.declaration.validate_arguments(arguments)
        result = tool.backend(validated, context)
        if not isinstance(result, ToolResult):
            raise TypeError(f"{name}: backend must return ToolResult")
        return result
