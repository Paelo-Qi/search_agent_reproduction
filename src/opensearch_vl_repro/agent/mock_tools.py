from __future__ import annotations

from typing import Any

from .tool_contracts import TOOL_DECLARATIONS
from .tool_registry import RegisteredTool, ToolContext, ToolRegistry, ToolResult


def _require_image(context: ToolContext, image_id: str) -> Any:
    return context.image_registry.get(image_id)


def _text_search(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    del context
    return ToolResult(
        status="success",
        observation=(
            "<observation>\nTool execution result:\n\n"
            "============================================================[Passage 1]\n"
            "Title: Example\nURL: https://example.com\nSummary:\n"
            f"Example summary for query: {arguments['q']}.\n</observation>"
        ),
        metadata={"mock": True},
    )


def _image_search(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    reference = arguments["url"]
    if reference.startswith("img_"):
        _require_image(context, reference)
    return ToolResult(
        status="success",
        observation=(
            "<observation>\nTool execution result:\n\nSearch results:\n"
            "Title: Example visual match\nURL: https://example.com/a\n"
            "Source: Example Source\n</observation>"
        ),
        metadata={"mock": True, "input_reference": reference},
    )


def _layout_parsing(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    _require_image(context, arguments["image"])
    return ToolResult(
        status="success",
        observation=(
            "<observation>\n✅ Layout Parsing SUCCESS: Text detected successfully!\n\n"
            "[Text Block 1]\n  Content: \"Example\"\n\n"
            "Combined text:\nExample\n</observation>"
        ),
        metadata={"mock": True},
    )


def _web_search(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    del context
    return ToolResult(
        status="success",
        observation=(
            "<observation>\nTool execution result:\n\n"
            f"Title: Example web result for {arguments['q']}\n"
            "URL: https://example.com/web\nSnippet: Example snippet.\n</observation>"
        ),
        metadata={"mock": True},
    )


def _derived_image_backend(action: str):
    def backend(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        source_id = arguments["image"]
        value = _require_image(context, source_id)
        new_id = context.image_registry.register_derived_image(
            value,
            parent_id=source_id,
            metadata={"mock": True, "operation": action},
        )
        return ToolResult(
            status="success",
            observation=(
                f"<image><observation>\nImage {action} completed successfully.\n"
                f"New image ID: {new_id}.\n</observation>"
            ),
            metadata={"mock": True, "image_id": new_id, "source_image_id": source_id},
        )

    return backend


MOCK_BACKENDS = {
    "text_search": _text_search,
    "image_search": _image_search,
    "crop": _derived_image_backend("cropping"),
    "layout_parsing": _layout_parsing,
    "super_resolution": _derived_image_backend("super resolution"),
    "sharpen": _derived_image_backend("sharpening"),
    "web_search": _web_search,
    "perspective_correct": _derived_image_backend("perspective correction"),
}


def create_mock_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for declaration in TOOL_DECLARATIONS:
        registry.register(RegisteredTool(declaration, MOCK_BACKENDS[declaration.name]))
    return registry


class ScriptedAgentModel:
    """CPU-only model substitute used solely to exercise the Agent protocol."""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> str:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self._outputs:
            raise RuntimeError("scripted model has no remaining output")
        return self._outputs.pop(0)

