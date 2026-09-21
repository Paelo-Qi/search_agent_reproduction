"""SFT-compatible agent protocol with replaceable tool backends."""

from .image_registry import ImageRegistry
from .runtime import AgentRuntime, AgentTrajectory
from .tool_contracts import TOOL_DECLARATIONS, ToolDeclaration
from .tool_parser import ParsedToolCall, ToolCallParser
from .tool_registry import ToolContext, ToolRegistry, ToolResult

__all__ = [
    "AgentRuntime",
    "AgentTrajectory",
    "ImageRegistry",
    "ParsedToolCall",
    "TOOL_DECLARATIONS",
    "ToolCallParser",
    "ToolContext",
    "ToolDeclaration",
    "ToolRegistry",
    "ToolResult",
]

