"""Eight-tool registry with real Phase 3 search backends."""

from __future__ import annotations

from pathlib import Path

from .layout_parsing import PaddleOCRAiStudioBackend, layout_tool, load_layout_api_config
from .local_visual_tools import LOCAL_VISUAL_BACKENDS
from .reliability import (
    FileSystemToolCache, LAYOUT_BEHAVIOR_VERSION, SEARCH_BEHAVIOR_VERSION,
    behavior_namespace, cached_tool_backend,
)
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
    cache_dir: str | Path | None = None,
) -> ToolRegistry:
    search_settings = load_search_config(search_config)
    layout_settings = load_layout_api_config(layout_config)
    tools = search_tools or SearchTools(search_settings)
    layout_backend = PaddleOCRAiStudioBackend(layout_settings)
    registry = ToolRegistry()
    backends = {
        "web_search": tools.web_search,
        "text_search": tools.text_search,
        "image_search": tools.image_search,
        "layout_parsing": layout_tool(layout_backend),
        **LOCAL_VISUAL_BACKENDS,
    }
    if cache_dir is not None:
        cache = FileSystemToolCache(cache_dir)
        behavior = {
            "web_search": behavior_namespace(
                f"search-{SEARCH_BEHAVIOR_VERSION}",
                {"provider": "serper", **search_settings.web_search},
            ),
            "text_search": behavior_namespace(
                f"search-{SEARCH_BEHAVIOR_VERSION}",
                {"providers": ["serper", "jina_reader"], **search_settings.text_search},
            ),
            "image_search": behavior_namespace(
                f"search-{SEARCH_BEHAVIOR_VERSION}",
                {"provider": "serpapi_google_lens", "encoder": 1,
                 **search_settings.image_search},
            ),
            "layout_parsing": behavior_namespace(
                f"layout-{LAYOUT_BEHAVIOR_VERSION}",
                {"provider": layout_settings.provider, "model": layout_settings.model,
                 "formatter": LAYOUT_BEHAVIOR_VERSION},
            ),
        }
        defaults = {
            "text_search": {"top_k": search_settings.text_search["default_top_k"]},
        }
        for name in ("web_search", "text_search", "image_search", "layout_parsing"):
            backends[name] = cached_tool_backend(
                tool=name, backend=backends[name], cache=cache,
                behavior_version=behavior[name], argument_defaults=defaults.get(name),
            )
    for declaration in TOOL_DECLARATIONS:
        registry.register(RegisteredTool(declaration, backends[declaration.name]))
    return registry
