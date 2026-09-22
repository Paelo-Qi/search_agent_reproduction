#!/usr/bin/env python3
"""Explicit real-API Phase 3 smoke; never run from pytest by default."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.agent.image_registry import ImageRegistry  # noqa: E402
from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.tool_registry import ToolContext  # noqa: E402


def _image() -> Image.Image:
    image = Image.new("RGB", (256, 192), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((55, 35, 190, 150), fill="navy")
    draw.ellipse((90, 65, 155, 130), fill="gold")
    return image


def _redact(value: Any) -> Any:
    secrets = [os.environ.get(name) for name in ("SERPER_API_KEY", "JINA_API_KEY", "SERPAPI_API_KEY")]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {_redact(key): _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in real search API smoke")
    parser.add_argument("--tool", required=True,
                        choices=("web_search", "text_search", "image_search"))
    parser.add_argument("--search-config", type=Path,
                        default=PROJECT_ROOT / "configs" / "search_backends.example.yaml")
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional shared Phase 4 filesystem cache directory.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report: dict[str, Any] = {
        "passed": False, "tool": args.tool, "status": "error", "error_type": None,
        "metadata": {}, "observation_preview": "", "elapsed_seconds": 0.0,
        "error": None, "semantic_matches_count": None,
    }
    started = time.perf_counter()
    try:
        registry = create_phase3_tool_registry(
            search_config=args.search_config, cache_dir=args.cache_dir,
        )
        images = ImageRegistry()
        images.register_initial_image(_image())
        arguments = {
            "web_search": {"q": "OpenAI official website", "hl": "en"},
            "text_search": {"q": "Qwen3-VL model architecture", "hl": "en", "top_k": 5},
            "image_search": {"url": "img_1"},
        }[args.tool]
        result = registry.execute(args.tool, arguments, ToolContext(images))
        report.update(status=result.status, error_type=result.error_type,
                      metadata=result.metadata, observation_preview=result.observation[:1200])
        if args.tool == "web_search":
            report["passed"] = (result.status == "success"
                                and result.metadata.get("result_count", 0) >= 1
                                and "Title:" in result.observation and "URL:" in result.observation)
        elif args.tool == "text_search":
            report["passed"] = (result.status == "success"
                                and result.metadata.get("reader_success_count", 0) >= 1
                                and all(field in result.observation for field in ("Title:", "URL:", "Passage:")))
        else:
            report["semantic_matches_count"] = result.metadata.get("result_count")
            report["passed"] = (result.status == "success"
                                and isinstance(result.metadata.get("provider_image_id"), str)
                                and "Image Search Results:" in result.observation)
        if not report["passed"]:
            report["error"] = result.error_type or "smoke_validation_failed"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = f"smoke failed at {type(exc).__name__}"
    report["elapsed_seconds"] = time.perf_counter() - started
    report = _redact(report)
    path = (args.report or PROJECT_ROOT / "reports" / f"{args.tool}_smoke.json").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
