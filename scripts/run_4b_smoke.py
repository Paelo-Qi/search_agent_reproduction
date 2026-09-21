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
from opensearch_vl_repro.inference.smoke_support import (  # noqa: E402
    CudaSmokeContext,
    exception_report,
)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
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
    args = parser.parse_args(argv)
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
    stage = "config_load"
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
        stage = "cuda_init"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed; 4B CUDA smoke was not run") from exc
        cuda_memory = CudaSmokeContext.initialize(torch, config.device)
        stage = "cuda_memory_reset"
        cuda_memory.reset_peak_memory_stats()
        stage = "model_load"
        load_started = time.perf_counter()
        bundle = load_inference_bundle(config)
        load_seconds = time.perf_counter() - load_started
        stage = "cuda_memory_read"
        load_peak_vram_mb = cuda_memory.peak_memory_mb()
        report["environment"] = {
            **bundle.environment,
            "load_seconds": load_seconds,
            "load_peak_vram_mb": load_peak_vram_mb,
            "model_eval_mode": not bundle.model.training,
        }
        for index in ([] if args.load_only else indices):
            stage = "sample_load"
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
            stage = "cuda_memory_reset"
            cuda_memory.reset_peak_memory_stats()
            stage = "generation"
            started = time.perf_counter()
            generated_text = generate_chat(bundle, messages)
            elapsed = time.perf_counter() - started
            stage = "cuda_memory_read"
            peak_memory_mb = cuda_memory.peak_memory_mb()
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
                    "peak_vram_mb": peak_memory_mb,
                    "peak_cuda_memory_mb": peak_memory_mb,
                    "passed": bool(generated_text),
                    "error": None,
                }
            )
        report["passed"] = args.load_only or (
            bool(report["samples"])
            and all(sample["passed"] for sample in report["samples"])
        )
    except Exception as exc:
        report["error"] = exception_report(exc, stage)
    try:
        write_report(report_path, report)
    except Exception as exc:
        report["passed"] = False
        report["error"] = exception_report(exc, "report_write")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
