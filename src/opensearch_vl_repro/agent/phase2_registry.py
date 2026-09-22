"""Registry with real local visual tools and intentionally non-real search tools."""

from __future__ import annotations

from pathlib import Path

from .layout_parsing import BaiduLayoutParsingBackend, layout_tool, load_layout_api_config
from .local_visual_tools import LOCAL_VISUAL_BACKENDS
from .mock_tools import MOCK_BACKENDS
from .tool_contracts import TOOL_DECLARATIONS
from .tool_registry import RegisteredTool, ToolRegistry


def create_phase2_tool_registry(*, layout_config: str | Path | None = None) -> ToolRegistry:
    layout_backend = (
        BaiduLayoutParsingBackend(load_layout_api_config(layout_config))
        if layout_config is not None else None
    )
    registry = ToolRegistry()
    for declaration in TOOL_DECLARATIONS:
        if declaration.name in LOCAL_VISUAL_BACKENDS:
            backend = LOCAL_VISUAL_BACKENDS[declaration.name]
        elif declaration.name == "layout_parsing":
            backend = layout_tool(layout_backend)
        else:
            # Search remains explicitly mock-only in Phase 2.
            backend = MOCK_BACKENDS[declaration.name]
        registry.register(RegisteredTool(declaration, backend))
    return registry
