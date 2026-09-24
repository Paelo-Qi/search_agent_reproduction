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
        "Look up a known entity, specific fact, or context not visible in the image. Returns web result titles, URLs, snippets, and page passages when available; use after image_search when detailed facts about a likely match are needed.",
        _object_schema(
            {"q": "string", "hl": "string", "top_k": "number"}, ("q",)
        ),
    ),
    ToolDeclaration(
        "image_search",
        "Visually identify an unknown landmark, object, artwork, product, or scene using reverse-image-style matches and source links. Pass a registered runtime image ID such as img_1 in the url argument, not a filename, filesystem path, or HTTP URL. Follow with text_search if the question needs detailed facts about a likely match.",
        _object_schema({"url": "string"}, ("url",)),
    ),
    ToolDeclaration(
        "crop",
        "Isolate a relevant object, small text region, or chart section when the full image contains distracting detail. Creates a new registered image for closer inspection or layout_parsing. Pass a registered runtime image ID such as img_1 in image, not a filename, filesystem path, or HTTP URL.",
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
        "Extract readable text and layout from document-like images, receipts, labels, tables, or charts when accurate wording or structure matters. A prior crop or image enhancement may help. Pass a registered runtime image ID such as img_1 in image, not a filename, filesystem path, or HTTP URL.",
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
        "Enlarge a genuinely low-resolution or pixelated image or region when small details need inspection. Creates a new registered image. Pass a registered runtime image ID such as img_1 in image, not a filename, filesystem path, or HTTP URL.",
        _object_schema({"image": "string", "scale": "number"}, ("image", "scale")),
    ),
    ToolDeclaration(
        "sharpen",
        "Improve blurred text or soft edges when sharper detail may make evidence readable. Creates a new registered image. Pass a registered runtime image ID such as img_1 in image, not a filename, filesystem path, or HTTP URL.",
        _object_schema({"image": "string", "amount": "number"}, ("image", "amount")),
    ),
    ToolDeclaration(
        "web_search",
        "Find concise web result titles, URLs, and snippets for a text query. Useful for locating a source or quick context; use text_search when the question needs fuller page passages.",
        _object_schema({"q": "string", "hl": "string"}, ("q",)),
    ),
    ToolDeclaration(
        "perspective_correct",
        "Straighten a document or text region photographed at an angle or visibly skewed before reading it. Creates a new registered image. Pass a registered runtime image ID such as img_1 in image, not a filename, filesystem path, or HTTP URL.",
        _object_schema({"image": "string"}, ("image",)),
    ),
)


TOOL_DECLARATIONS_BY_NAME = {tool.name: tool for tool in TOOL_DECLARATIONS}
