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
from opensearch_vl_repro.agent.phase2_registry import create_phase2_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.runtime import AgentRuntime  # noqa: E402
from opensearch_vl_repro.inference import (  # noqa: E402
    QwenAgentModel,
    load_inference_bundle,
    load_inference_config,
    read_eval_sample,
)
from opensearch_vl_repro.inference.smoke_support import (  # noqa: E402
    CudaSmokeContext,
    error_report,
    exception_report,
)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
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
        "--local-visual-tools",
        action="store_true",
        help="Opt in to Phase 2 backends and ask for a crop of img_1; not benchmark evaluation.",
    )
    parser.add_argument(
        "--layout-config", type=Path,
        help="Optional layout API config for --local-visual-tools; no API is called unless requested by the model.",
    )
    parser.add_argument(
        "--report", type=Path, default=PROJECT_ROOT / "reports" / "4b_agent_smoke.json"
    )
    args = parser.parse_args(argv)
    report: dict[str, Any] = {
        "passed": False,
        "tool_call_chain_passed": False,
        "synthetic_tool_call_smoke": args.synthetic_tool_prompt,
        "local_visual_tools": args.local_visual_tools,
        "visual_tool_chain_passed": False,
        "config": str(args.config.resolve()),
        "index": args.index,
        "environment": {},
        "trajectory": None,
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
            raise RuntimeError("4B Agent smoke requires a CUDA device in the inference config")
        stage = "cuda_init"
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed; 4B CUDA Agent smoke was not run") from exc
        cuda_memory = CudaSmokeContext.initialize(torch, config.device)
        stage = "cuda_memory_reset"
        cuda_memory.reset_peak_memory_stats()
        stage = "model_load"
        load_started = time.perf_counter()
        bundle = load_inference_bundle(config)
        stage = "cuda_memory_read"
        report["environment"] = {
            **bundle.environment,
            "load_seconds": time.perf_counter() - load_started,
            "load_peak_vram_mb": cuda_memory.peak_memory_mb(),
            "model_eval_mode": not bundle.model.training,
        }
        stage = "sample_load"
        sample = read_eval_sample(config.data_path, args.index)
        question = sample.question
        if args.local_visual_tools:
            question = (
                "Synthetic Phase 2 visual-tool smoke only; this is not a benchmark result. "
                "First call crop with exactly "
                '{"image":"img_1","x":0,"y":0,"width":64,"height":64}. '
                "After receiving the derived image, look at it and give a short final answer. "
                "Do not call another tool."
            )
        elif args.synthetic_tool_prompt:
            question = (
                "Synthetic tool-call protocol smoke only; this is not a benchmark result. "
                "First call image_search with the exact argument {\"url\": \"img_1\"}. "
                "After receiving its observation, provide a short final answer.\n\n"
                f"Original sample question: {sample.question}"
            )
        runtime = AgentRuntime(
            model=QwenAgentModel(bundle),
            tool_registry=(
                create_phase2_tool_registry(layout_config=args.layout_config)
                if args.local_visual_tools else create_mock_tool_registry()
            ),
            max_agent_turns=config.max_agent_turns,
        )
        stage = "cuda_memory_reset"
        cuda_memory.reset_peak_memory_stats()
        stage = "agent_runtime"
        started = time.perf_counter()
        trajectory = runtime.run(
            question=question,
            images=sample.images,
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
        )
        report["trajectory"] = trajectory.to_dict()
        report["elapsed_seconds"] = time.perf_counter() - started
        stage = "cuda_memory_read"
        report["peak_vram_mb"] = cuda_memory.peak_memory_mb()
        report["tool_call_chain_passed"] = bool(trajectory.turns) and any(
            turn.status == "success" and turn.tool_call is not None
            for turn in trajectory.turns
        )
        report["visual_tool_chain_passed"] = any(
            turn.status == "success" and turn.tool_call is not None
            and turn.tool_call["name"] == "crop" and bool(turn.derived_images)
            for turn in trajectory.turns
        ) and trajectory.status == "success"
        report["passed"] = trajectory.status == "success"
        if trajectory.status != "success":
            report["error"] = error_report(
                type_name="AgentRuntimeError",
                message=trajectory.error or f"agent status: {trajectory.status}",
                stage="agent_runtime",
            )
        if (
            (args.synthetic_tool_prompt and not report["tool_call_chain_passed"])
            or (args.local_visual_tools and not report["visual_tool_chain_passed"])
        ) and report["error"] is None:
            report["passed"] = False
            report["error"] = error_report(
                type_name="AgentProtocolError",
                message=("visual prompt did not produce a successful crop and final answer"
                         if args.local_visual_tools else
                         "synthetic prompt did not produce a valid executable tool call"),
                stage="agent_runtime",
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
