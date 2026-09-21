from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDeclaration:
    name: str
    description: str
    parameters: dict[str, Any]

    def as_chat_template_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_arguments(self, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError(f"{self.name}: arguments must be an object")
        properties = self.parameters["properties"]
        required = self.parameters.get("required", [])
        missing = [key for key in required if key not in arguments]
        if missing:
            raise ValueError(f"{self.name}: missing required arguments: {missing}")
        if self.parameters.get("additionalProperties") is False:
            unexpected = sorted(set(arguments) - set(properties))
            if unexpected:
                raise ValueError(f"{self.name}: unexpected arguments: {unexpected}")
        for key, value in arguments.items():
            expected = properties[key]["type"]
            valid = {
                "string": isinstance(value, str),
                "boolean": isinstance(value, bool),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            }.get(expected, False)
            if not valid:
                raise ValueError(
                    f"{self.name}: argument {key!r} must be {expected}, "
                    f"got {type(value).__name__}"
                )
        return dict(arguments)


def _object_schema(
    properties: dict[str, str], required: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {"type": value_type} for name, value_type in properties.items()
        },
        "required": list(required),
        "additionalProperties": False,
    }


# These declarations intentionally mirror the schemas observed in the pinned
# SearchVL-SFT-36K audit. Do not rename image_search.url to image.
TOOL_DECLARATIONS = (
    ToolDeclaration(
        "text_search",
        "Search text passages for a query.",
        _object_schema(
            {"q": "string", "hl": "string", "top_k": "number"}, ("q",)
        ),
    ),
    ToolDeclaration(
        "image_search",
        "Search using an image URL or an ImageRegistry reference such as img_1.",
        _object_schema({"url": "string"}, ("url",)),
    ),
    ToolDeclaration(
        "crop",
        "Crop a registered image and create a derived image.",
        _object_schema(
            {
                "image": "string",
                "x": "number",
                "y": "number",
                "width": "number",
                "height": "number",
            },
            ("image", "x", "y", "width", "height"),
        ),
    ),
    ToolDeclaration(
        "layout_parsing",
        "Parse text and layout from a registered image.",
        _object_schema(
            {
                "image": "string",
                "use_chart_recognition": "boolean",
                "use_doc_orientation_classify": "boolean",
            },
            ("image",),
        ),
    ),
    ToolDeclaration(
        "super_resolution",
        "Create a super-resolution derived image.",
        _object_schema({"image": "string", "scale": "number"}, ("image", "scale")),
    ),
    ToolDeclaration(
        "sharpen",
        "Create a sharpened derived image.",
        _object_schema({"image": "string", "amount": "number"}, ("image", "amount")),
    ),
    ToolDeclaration(
        "web_search",
        "Search the web for a text query.",
        _object_schema({"q": "string", "hl": "string"}, ("q",)),
    ),
    ToolDeclaration(
        "perspective_correct",
        "Create a perspective-corrected derived image.",
        _object_schema({"image": "string"}, ("image",)),
    ),
)


TOOL_DECLARATIONS_BY_NAME = {tool.name: tool for tool in TOOL_DECLARATIONS}

