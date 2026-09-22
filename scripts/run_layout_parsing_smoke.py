#!/usr/bin/env python3
"""Opt-in real AI Studio API smoke on a deterministic local document."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.agent.image_registry import ImageRegistry  # noqa: E402
from opensearch_vl_repro.agent.layout_parsing import load_layout_api_config  # noqa: E402
from opensearch_vl_repro.agent.phase2_registry import create_phase2_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.tool_registry import ToolContext  # noqa: E402
from opensearch_vl_repro.inference.eval_reader import read_eval_sample  # noqa: E402


def synthetic_document() -> Image.Image:
    image = Image.new("RGB", (920, 510), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    lines = (
        "OpenSearch-VL Evaluation Report",
        "This is a document parsing test.",
        "Model: Qwen3-VL-4B-Instruct",
        "Accuracy: 78.5%",
        "The experiment completed successfully.",
    )
    for index, line in enumerate(lines):
        draw.text((50, 40 + index * 85), line, fill="black", font=font)
    return image


def _redact(value: Any, token: str) -> Any:
    if isinstance(value, str):
        return value.replace(token, "[REDACTED]") if token else value
    if isinstance(value, dict):
        return {key: _redact(item, token) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, token) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real PaddleOCR AI Studio layout API smoke")
    parser.add_argument("--use-eval-sample", action="store_true",
                        help="Explicitly use the first image of a frozen eval sample instead of the default synthetic document")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--layout-config", type=Path,
                        default=PROJECT_ROOT / "configs" / "layout_parsing.example.yaml")
    parser.add_argument("--report", type=Path,
                        default=PROJECT_ROOT / "reports" / "layout_parsing_smoke.json")
    args = parser.parse_args(argv)
    report: dict[str, Any] = {
        "passed": False, "provider": None, "sample_id": None,
        "status": "error", "error_type": None, "metadata": {},
        "observation_preview": "", "elapsed_seconds": 0.0, "error": None,
    }
    started = time.perf_counter()
    token = ""
    try:
        config = load_layout_api_config(args.layout_config)
        report["provider"] = config.provider
        token = os.environ.get(config.access_token_env, "")
        if args.use_eval_sample:
            sample = read_eval_sample(PROJECT_ROOT / "data" / "eval" / "combined_eval_300.parquet", args.index)
            image = sample.images[0]
            report["sample_id"] = sample.sample_id
        else:
            image = synthetic_document()
            report["sample_id"] = "synthetic-layout-document"
        images = ImageRegistry()
        images.register_initial_image(image)
        registry = create_phase2_tool_registry(layout_config=args.layout_config)
        result = registry.execute("layout_parsing", {"image": "img_1"}, ToolContext(images))
        report["status"] = result.status
        report["error_type"] = result.error_type
        report["metadata"] = result.metadata
        report["observation_preview"] = result.observation[:1000]
        report["passed"] = (
            result.status == "success" and result.error_type is None
            and "Content:" in result.observation
            and re.search(r"(?m)^\[[^\]]+\]$", result.observation) is not None
            and bool(result.observation.strip())
        )
        if not report["passed"]:
            report["error"] = result.error_type or "observation_validation_failed"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = f"smoke failed at {type(exc).__name__}"
    report["elapsed_seconds"] = time.perf_counter() - started
    report = _redact(report, token)
    path = args.report.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
