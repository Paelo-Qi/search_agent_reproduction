"""Eight-tool registry with real Phase 3 search backends."""

from __future__ import annotations

from pathlib import Path

from .layout_parsing import PaddleOCRAiStudioBackend, layout_tool, load_layout_api_config
from .local_visual_tools import LOCAL_VISUAL_BACKENDS
from .search_providers import load_search_config
from .search_tools import SearchTools
from .tool_contracts import TOOL_DECLARATIONS
from .tool_registry import RegisteredTool, ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def create_phase3_tool_registry(
    *,
    search_config: str | Path = PROJECT_ROOT / "configs" / "search_backends.example.yaml",
    layout_config: str | Path = PROJECT_ROOT / "configs" / "layout_parsing.example.yaml",
    search_tools: SearchTools | None = None,
) -> ToolRegistry:
    tools = search_tools or SearchTools(load_search_config(search_config))
    layout_backend = PaddleOCRAiStudioBackend(load_layout_api_config(layout_config))
    registry = ToolRegistry()
    backends = {
        "web_search": tools.web_search,
        "text_search": tools.text_search,
        "image_search": tools.image_search,
        "layout_parsing": layout_tool(layout_backend),
        **LOCAL_VISUAL_BACKENDS,
    }
    for declaration in TOOL_DECLARATIONS:
        registry.register(RegisteredTool(declaration, backends[declaration.name]))
    return registry
