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

from opensearch_vl_repro.agent.mock_tools import create_mock_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.runtime import AgentRuntime  # noqa: E402
from opensearch_vl_repro.inference import (  # noqa: E402
    QwenAgentModel,
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
    parser = argparse.ArgumentParser(
        description="Real Qwen3-VL-4B Agent protocol smoke using mock tool backends."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "eval_4b.yaml")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--synthetic-tool-prompt",
        action="store_true",
        help="Explicitly request a tool call to test mechanics; this is not benchmark evaluation.",
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports" / "4b_agent_smoke.json"
    )
    args = parser.parse_args()
    report: dict[str, Any] = {
        "passed": False,
        "tool_call_chain_passed": False,
        "synthetic_tool_call_smoke": args.synthetic_tool_prompt,
        "config": str(args.config.resolve()),
        "index": args.index,
        "environment": {},
        "trajectory": None,
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
            raise RuntimeError("4B Agent smoke requires a CUDA device in the inference config")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed; 4B CUDA Agent smoke was not run") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; 4B Agent smoke was not run")
        torch.cuda.reset_peak_memory_stats(config.device)
        load_started = time.perf_counter()
        bundle = load_inference_bundle(config)
        report["environment"] = {
            **bundle.environment,
            "load_seconds": time.perf_counter() - load_started,
            "load_peak_vram_mb": torch.cuda.max_memory_allocated(config.device) / (1024**2),
            "model_eval_mode": not bundle.model.training,
        }
        sample = read_eval_sample(config.data_path, args.index)
        question = sample.question
        if args.synthetic_tool_prompt:
            question = (
                "Synthetic tool-call protocol smoke only; this is not a benchmark result. "
                "First call image_search with the exact argument {\"url\": \"img_1\"}. "
                "After receiving its observation, provide a short final answer.\n\n"
                f"Original sample question: {sample.question}"
            )
        runtime = AgentRuntime(
            model=QwenAgentModel(bundle),
            tool_registry=create_mock_tool_registry(),
            max_agent_turns=config.max_agent_turns,
        )
        torch.cuda.reset_peak_memory_stats(config.device)
        started = time.perf_counter()
        trajectory = runtime.run(
            question=question,
            images=sample.images,
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
        )
        report["trajectory"] = trajectory.to_dict()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_vram_mb"] = torch.cuda.max_memory_allocated(config.device) / (1024**2)
        report["tool_call_chain_passed"] = bool(trajectory.turns) and any(
            turn.status == "success" and turn.tool_call is not None
            for turn in trajectory.turns
        )
        report["passed"] = trajectory.status == "success"
        if args.synthetic_tool_prompt and not report["tool_call_chain_passed"]:
            report["passed"] = False
            report["error"] = "synthetic prompt did not produce a valid executable tool call"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    write_report(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
