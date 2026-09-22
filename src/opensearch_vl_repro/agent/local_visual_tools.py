"""Small local visual backends; image IDs are assigned by AgentRuntime."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

from .tool_registry import DerivedImage, ToolContext, ToolResult


def _image(context: ToolContext, image_id: str) -> Image.Image:
    value = context.image_registry.get(image_id)
    if isinstance(value, Image.Image):
        return value
    with Image.open(Path(value)) as loaded:
        return loaded.convert("RGB").copy()


def _number(value: Any, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return float(value)


def crop(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    source_id = arguments["image"]
    source = _image(context, source_id)
    x = _number(arguments["x"], "x")
    y = _number(arguments["y"], "y")
    width = _number(arguments["width"], "width", minimum=0)
    height = _number(arguments["height"], "height", minimum=0)
    if width == 0 or height == 0:
        raise ValueError("crop width and height must be positive")
    left = max(0, min(source.width, math.floor(x)))
    top = max(0, min(source.height, math.floor(y)))
    right = max(0, min(source.width, math.ceil(x + width)))
    bottom = max(0, min(source.height, math.ceil(y + height)))
    if right <= left or bottom <= top:
        raise ValueError("crop region is empty after clipping to image bounds")
    box = (left, top, right, bottom)
    result = source.crop(box).copy()
    metadata = {"backend": "pillow", "source_image_id": source_id,
                "source_size": list(source.size), "crop_box": list(box),
                "result_size": list(result.size)}
    return ToolResult(
        status="success",
        observation="<image><observation>\nImage cropped and registered.\n</observation>",
        metadata=metadata,
        derived_images=(DerivedImage(result, source_id, metadata),),
    )


def sharpen(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    source_id = arguments["image"]
    source = _image(context, source_id)
    amount = _number(arguments["amount"], "amount", minimum=0, maximum=4)
    result = ImageEnhance.Sharpness(source).enhance(amount)
    metadata = {"backend": "pillow_sharpness", "amount": amount,
                "source_image_id": source_id, "source_size": list(source.size),
                "result_size": list(result.size)}
    return ToolResult(
        status="success",
        observation="<image><observation>\nImage sharpness adjusted and registered.\n</observation>",
        metadata=metadata,
        derived_images=(DerivedImage(result, source_id, metadata),),
    )


class LanczosSuperResolutionBackend:
    """Replaceable lightweight resize, not learned super-resolution."""

    def __call__(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        source_id = arguments["image"]
        source = _image(context, source_id)
        scale = _number(arguments["scale"], "scale", minimum=1, maximum=4)
        size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        if size[0] * size[1] > 16_000_000:
            raise ValueError("upscaled image exceeds the 16 megapixel local safety limit")
        result = source.resize(size, Image.Resampling.LANCZOS)
        metadata = {"backend": "lanczos", "scale": scale,
                    "source_image_id": source_id, "source_size": list(source.size),
                    "result_size": list(result.size), "learned_sr": False}
        return ToolResult(
            status="success",
            observation=("<image><observation>\nImage enlarged with lightweight "
                         "Lanczos interpolation; no lost detail was recovered.\n</observation>"),
            metadata=metadata,
            derived_images=(DerivedImage(result, source_id, metadata),),
        )


def perspective_correct(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    source_id = arguments["image"]
    source = _image(context, source_id)
    result = source.copy()
    metadata = {"backend_mode": "identity_fallback", "changed": False,
                "source_image_id": source_id, "source_size": list(source.size),
                "result_size": list(result.size)}
    return ToolResult(
        status="success",
        observation=("<image><observation>\nPerspective correction is in identity "
                     "fallback mode. Returned an unchanged derived image.\n</observation>"),
        metadata=metadata,
        derived_images=(DerivedImage(result, source_id, metadata),),
    )


LOCAL_VISUAL_BACKENDS = {
    "crop": crop,
    "sharpen": sharpen,
    "super_resolution": LanczosSuperResolutionBackend(),
    "perspective_correct": perspective_correct,
}
