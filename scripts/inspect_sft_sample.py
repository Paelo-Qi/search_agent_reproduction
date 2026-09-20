#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.config import load_config  # noqa: E402
from opensearch_vl_repro.data import (  # noqa: E402
    OpenSearchVLCollator,
    build_messages,
    load_json_records,
    messages_to_json_safe,
    render_prompt,
    tensor_shapes,
)
from opensearch_vl_repro.model import load_processor  # noqa: E402
from opensearch_vl_repro.reporting import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the complete SFT preprocessing chain.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sft_smoke.yaml")
    parser.add_argument("--index", type=int)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "sample_inspection.json")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = (PROJECT_ROOT / config["data"]["path"]).resolve()
    records = load_json_records(dataset_path)
    index = args.index if args.index is not None else random.Random(args.seed).randrange(len(records))
    sample = records[index]
    processor = load_processor(config)
    messages, images, tools = build_messages(sample, dataset_path)
    prompt = render_prompt(processor, messages, tools)
    collator = OpenSearchVLCollator(processor, dataset_path, int(config["data"]["max_length"]))
    batch = collator([sample])

    labels = batch["labels"]
    input_ids = batch["input_ids"]
    attention = batch.get("attention_mask")
    active = int(attention.sum().item()) if attention is not None else input_ids.numel()
    supervised = int((labels != -100).sum().item())
    image_info = [
        {"path": path, "width": image.width, "height": image.height, "mode": image.mode}
        for path, image in zip(sample["images"], images)
    ]
    report = {
        "sample_index": index,
        "source": sample.get("_source"),
        "dataset_size": len(records),
        "raw_sample": sample,
        "messages": messages_to_json_safe(messages),
        "formatted_prompt": prompt,
        "images": image_info,
        "tensor_shapes": tensor_shapes(batch),
        "vision_fields": {
            key: list(value.shape)
            for key, value in batch.items()
            if key not in {"input_ids", "labels", "attention_mask"} and hasattr(value, "shape")
        },
        "label_masking": {
            "active_tokens": active,
            "supervised_assistant_tokens": supervised,
            "masked_tokens": active - supervised,
            "supervised_fraction": supervised / active,
            "input_ids_labels_same_shape": list(input_ids.shape) == list(labels.shape),
            "has_assistant_targets": supervised > 0,
        },
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
