#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.inference import (  # noqa: E402
    generate_chat,
    load_inference_bundle,
    load_inference_config,
    read_eval_sample,
)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B single-GPU generation smoke.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "eval_4b.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=int)
    group.add_argument("--indices", type=int, nargs="+")
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Load the pinned model/processor and report environment/VRAM without generation.",
    )
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "4b_smoke.json")
    args = parser.parse_args()
    indices = args.indices if args.indices is not None else [args.index if args.index is not None else 0]
    report: dict[str, Any] = {
        "passed": False,
        "config": str(args.config.resolve()),
        "indices": indices,
        "load_only": args.load_only,
        "environment": {},
        "samples": [],
        "error": None,
    }
    report_path = args.report.resolve()
    try:
        config = load_inference_config(args.config)
        report["environment"] = {
            "model": config.model_name_or_path,
            "model_revision": config.revision,
            "dtype": config.dtype,
            "device": config.device,
        }
        if not config.device.startswith("cuda"):
            raise RuntimeError("4B smoke requires a CUDA device in the inference config")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed; 4B CUDA smoke was not run") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; 4B smoke was not run")
        torch.cuda.reset_peak_memory_stats(config.device)
        load_started = time.perf_counter()
        bundle = load_inference_bundle(config)
        load_seconds = time.perf_counter() - load_started
        load_peak_vram_mb = torch.cuda.max_memory_allocated(config.device) / (1024**2)
        report["environment"] = {
            **bundle.environment,
            "load_seconds": load_seconds,
            "load_peak_vram_mb": load_peak_vram_mb,
            "model_eval_mode": not bundle.model.training,
        }
        for index in ([] if args.load_only else indices):
            sample = read_eval_sample(config.data_path, index)
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in sample.images],
                        {"type": "text", "text": sample.question},
                    ],
                }
            ]
            torch.cuda.reset_peak_memory_stats(config.device)
            started = time.perf_counter()
            generated_text = generate_chat(bundle, messages)
            elapsed = time.perf_counter() - started
            report["samples"].append(
                {
                    "index": index,
                    "sample_id": sample.sample_id,
                    "benchmark": sample.benchmark,
                    "question": sample.question,
                    "model": config.model_name_or_path,
                    "model_revision": config.revision,
                    "generation": generated_text,
                    "generated_text": generated_text,
                    "elapsed_seconds": elapsed,
                    "peak_vram_mb": torch.cuda.max_memory_allocated(config.device) / (1024**2),
                    "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(config.device) / (1024**2),
                    "passed": bool(generated_text),
                    "error": None,
                }
            )
        report["passed"] = args.load_only or (
            bool(report["samples"])
            and all(sample["passed"] for sample in report["samples"])
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_report(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
