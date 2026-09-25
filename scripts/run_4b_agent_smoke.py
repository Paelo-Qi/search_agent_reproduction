#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.agent.mock_tools import create_mock_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.phase2_registry import create_phase2_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry  # noqa: E402
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


def redact_report(value: Any) -> Any:
    secrets = [os.environ.get(name) for name in (
        "SERPER_API_KEY", "JINA_API_KEY", "SERPAPI_API_KEY", "PADDLEOCR_ACCESS_TOKEN",
    )]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact_report(key): redact_report(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_report(item) for item in value]
    return value


class ObservedAgentModel:
    """Record only in-memory evidence that a tool message reached generate()."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.generate_call_count = 0
        self.tool_observations_seen: list[str] = []

    def generate(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        self.generate_call_count += 1
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if isinstance(content, str):
                self.tool_observations_seen.append(content)
            elif isinstance(content, list):
                self.tool_observations_seen.extend(
                    part["text"] for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
        return self.model.generate(messages=messages, tools=tools)


def phase3_evidence(trajectory: Any, requested_tool: str, model: ObservedAgentModel) -> dict[str, Any]:
    called = [turn for turn in trajectory.turns if turn.tool_call is not None]
    matching = [turn for turn in called if turn.tool_call["name"] == requested_tool]
    successful = next((turn for turn in matching if turn.status == "success"), None)
    metadata = successful.metadata if successful is not None else {}
    if requested_tool == "text_search":
        backend = ["serper", "jina_reader"]
        provider_ok = (
            all(name in metadata.get("providers", []) for name in backend)
            and metadata.get("reader_success_count", 0) >= 1
        )
    else:
        backend = "serpapi_google_lens"
        provider_ok = (metadata.get("provider") == backend
                       and bool(metadata.get("provider_image_id")))
    observation_seen = (
        successful is not None and bool(successful.observation)
        and successful.observation in model.tool_observations_seen
        and model.generate_call_count >= 2
    )
    final_present = isinstance(trajectory.final_answer, str) and bool(trajectory.final_answer.strip())
    return {
        "requested_phase3_tool": requested_tool,
        "real_search_backend": backend,
        "tool_call_count": len(called),
        "tool_call_name": successful.tool_call["name"] if successful else (called[0].tool_call["name"] if called else None),
        "tool_call_status": successful.status if successful else (matching[0].status if matching else None),
        "final_answer_present": final_present,
        "provider_metadata": metadata,
        "observation_fed_to_next_qwen_turn": bool(observation_seen),
        "generate_call_count": model.generate_call_count,
        "phase3_tool_chain_passed": (
            trajectory.status == "success" and successful is not None
            and provider_ok and observation_seen and final_present
        ),
    }


def phase3_image() -> Image.Image:
    image = Image.new("RGB", (256, 192), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((55, 35, 190, 150), fill="navy")
    draw.ellipse((90, 65, 155, 130), fill="gold")
    return image


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(redact_report(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Real Qwen3-VL-4B Agent smoke: mock, Phase 2 visual, or opt-in Phase 3 search."
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
    parser.add_argument("--phase3-search-tools", action="store_true",
                        help="Opt in to the Phase 3 real search registry and a synthetic tool protocol prompt.")
    parser.add_argument("--phase3-tool", choices=("text_search", "image_search"),
                        help="Required with --phase3-search-tools; selects the exact integration tool.")
    parser.add_argument("--search-config", type=Path,
                        default=PROJECT_ROOT / "configs" / "search_backends.example.yaml")
    parser.add_argument("--cache-dir", type=Path,
                        help="Optional shared Phase 4 cache for external Phase 3 tools.")
    parser.add_argument(
        "--layout-config", type=Path,
        help="Layout API config for Phase 2/3; Phase 3 defaults to configs/layout_parsing.example.yaml.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--adapter", type=Path,
                        help="Validated formal SFT checkpoint adapter; Base-only when omitted")
    args = parser.parse_args(argv)
    if args.phase3_search_tools != bool(args.phase3_tool):
        parser.error("--phase3-search-tools and --phase3-tool must be used together")
    if args.phase3_search_tools and (args.local_visual_tools or args.synthetic_tool_prompt):
        parser.error("Phase 3 search mode is exclusive with existing synthetic/Phase 2 modes")
    report: dict[str, Any] = {
        "passed": False,
        "tool_call_chain_passed": False,
        "synthetic_tool_call_smoke": args.synthetic_tool_prompt,
        "local_visual_tools": args.local_visual_tools,
        "phase3_search_tools": args.phase3_search_tools,
        "visual_tool_chain_passed": False,
        "config": str(args.config.resolve()),
        "index": args.index,
        "environment": {},
        "trajectory": None,
        "error": None,
    }
    if args.phase3_search_tools:
        report.update({
            "requested_phase3_tool": args.phase3_tool,
            "real_search_backend": (["serper", "jina_reader"] if args.phase3_tool == "text_search"
                                    else "serpapi_google_lens"),
            "tool_call_count": 0,
            "tool_call_name": None,
            "tool_call_status": None,
            "provider_metadata": {},
            "observation_fed_to_next_qwen_turn": False,
            "generate_call_count": 0,
            "final_answer_present": False,
            "phase3_tool_chain_passed": False,
            "elapsed_seconds": None,
            "peak_vram_mb": None,
        })
    report_path = (args.report or (
        PROJECT_ROOT / "reports" / f"4b_phase3_{args.phase3_tool}_smoke.json"
        if args.phase3_search_tools else PROJECT_ROOT / "reports" / "4b_agent_smoke.json"
    )).resolve()
    stage = "config_load"
    try:
        config = load_inference_config(args.config)
        if args.adapter is not None:
            config = replace(config, adapter_path=args.adapter.expanduser().resolve())
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
        if args.phase3_search_tools:
            sample = None
            images = [phase3_image()]
            sample_id = f"phase3-{args.phase3_tool}-synthetic"
            benchmark = "synthetic"
            if args.phase3_tool == "text_search":
                question = (
                    "Synthetic Phase 3 text-search integration smoke only; this is not a benchmark result. "
                    "First call text_search exactly once with "
                    '{"q":"Qwen3-VL technical report","hl":"en","top_k":3}. '
                    "After receiving the tool observation, do not call another tool. "
                    "Use the returned search evidence and provide a short final answer."
                )
            else:
                question = (
                    "Synthetic Phase 3 image-search integration smoke only; this is not a benchmark result. "
                    'First call image_search exactly once with {"url":"img_1"}. '
                    "After receiving the image-search observation, do not call another tool. "
                    "Give a short final answer based on the returned observation."
                )
        else:
            sample = read_eval_sample(config.data_path, args.index)
            images = sample.images
            sample_id, benchmark = sample.sample_id, sample.benchmark
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
        observed_model = ObservedAgentModel(QwenAgentModel(bundle)) if args.phase3_search_tools else None
        runtime = AgentRuntime(
            model=observed_model if observed_model is not None else QwenAgentModel(bundle),
            tool_registry=(
                create_phase3_tool_registry(
                    search_config=args.search_config,
                    layout_config=args.layout_config or PROJECT_ROOT / "configs" / "layout_parsing.example.yaml",
                    cache_dir=args.cache_dir,
                ) if args.phase3_search_tools else
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
            images=images,
            sample_id=sample_id,
            benchmark=benchmark,
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
        if args.phase3_search_tools:
            evidence = phase3_evidence(trajectory, args.phase3_tool, observed_model)
            report.update(evidence)
            report["passed"] = evidence["phase3_tool_chain_passed"]
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
        if args.phase3_search_tools and not report["passed"] and report["error"] is None:
            report["error"] = error_report(
                type_name="AgentProtocolError",
                message=f"Phase 3 {args.phase3_tool} call, provider metadata, observation re-entry, or final answer missing",
                stage="agent_runtime",
            )
    except Exception as exc:
        report["error"] = exception_report(exc, stage)
    try:
        write_report(report_path, report)
    except Exception as exc:
        report["passed"] = False
        report["error"] = exception_report(exc, "report_write")
        print(json.dumps(redact_report(report), ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(redact_report(report), ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
