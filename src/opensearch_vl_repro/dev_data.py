from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .data import validate_raw_sample


DEV_SEED = 20260506

DEV_TOOL = {
    "type": "function",
    "function": {
        "name": "inspect_color",
        "description": "Return the dominant color and visible geometric shape.",
        "parameters": {
            "type": "object",
            "properties": {"image": {"type": "string"}},
            "required": ["image"],
        },
    },
}


def _draw_image(path: Path, background: tuple[int, int, int], shape: str) -> None:
    image = Image.new("RGB", (128, 128), background)
    draw = ImageDraw.Draw(image)
    if shape == "square":
        draw.rectangle((38, 38, 90, 90), fill=(255, 255, 255), outline=(20, 20, 20), width=3)
    elif shape == "circle":
        draw.ellipse((34, 34, 94, 94), fill=(255, 255, 255), outline=(20, 20, 20), width=3)
    elif shape == "triangle":
        draw.polygon(((64, 27), (27, 99), (101, 99)), fill=(255, 255, 255), outline=(20, 20, 20))
    elif shape == "cross":
        draw.rectangle((54, 25, 74, 103), fill=(255, 255, 255))
        draw.rectangle((25, 54, 103, 74), fill=(255, 255, 255))
    else:
        raise ValueError(f"unsupported shape: {shape}")
    image.save(path, format="PNG", optimize=False)


def build_synthetic_records(media_dir_name: str = "dev_media") -> list[dict[str, Any]]:
    tools = json.dumps([DEV_TOOL], ensure_ascii=False, sort_keys=True)
    system = (
        "You are a tiny visual test agent. Describe only the visible color and shape, "
        "and use tool observations when they are provided."
    )
    specs = [
        ("red", "square", "The image has a red background and a white square."),
        ("green", "circle", "The image has a green background and a white circle."),
        ("blue", "triangle", "The image has a blue background and a white triangle."),
        ("yellow", "cross", "The image has a yellow background and a white cross."),
    ]
    records: list[dict[str, Any]] = []
    for index, (color, shape, answer) in enumerate(specs):
        if index in (1, 3):
            conversations = [
                {"from": "human", "value": "<image>What color and shape are visible?"},
                {
                    "from": "gpt",
                    "value": (
                        "<think>I should verify the simple visual.</think>"
                        f'<tool_call>{{"name":"inspect_color","arguments":{{"image":"img_{index}"}}}}</tool_call>'
                    ),
                },
                {
                    "from": "observation",
                    "value": f"Tool observation: dominant color={color}; shape={shape}.",
                },
                {"from": "gpt", "value": answer},
            ]
        else:
            conversations = [
                {"from": "human", "value": "<image>Describe the background color and central shape."},
                {"from": "gpt", "value": answer},
            ]
        record = {
            "conversations": conversations,
            "images": [f"{media_dir_name}/dev_{index}.png"],
            "system": system,
            "tools": tools,
            "_source": "synthetic_dev",
            "_source_index": index,
        }
        validate_raw_sample(record)
        records.append(record)
    return records


def write_dev_dataset(
    output_path: str | Path,
    media_dir: str | Path,
    seed: int = DEV_SEED,
) -> tuple[Path, Path]:
    output = Path(output_path).resolve()
    media = Path(media_dir).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)

    palette = {
        "red": (210, 45, 45),
        "green": (35, 155, 75),
        "blue": (45, 95, 205),
        "yellow": (225, 190, 35),
    }
    shapes = ["square", "circle", "triangle", "cross"]
    colors = ["red", "green", "blue", "yellow"]
    rng = random.Random(seed)
    # A deterministic no-op shuffle exercises the seed without changing the
    # semantic mapping expected by the records.
    draw_order = list(range(4))
    rng.shuffle(draw_order)
    for index in draw_order:
        _draw_image(media / f"dev_{index}.png", palette[colors[index]], shapes[index])

    records = build_synthetic_records(media.name)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "kind": "fully_local_synthetic_multimodal",
        "seed": seed,
        "sample_count": len(records),
        "image_count": 4,
        "contains_observation_trajectories": True,
        "downloads_required": False,
        "output_sha256": digest,
    }
    metadata_path = output.with_suffix(".meta.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return output, metadata_path

