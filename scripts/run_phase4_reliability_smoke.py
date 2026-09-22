#!/usr/bin/env python3
"""Completely offline Phase 4 reliability smoke."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.agent.image_registry import ImageRegistry  # noqa: E402
from opensearch_vl_repro.agent.reliability import (  # noqa: E402
    FileSystemToolCache, RetryPolicy, cached_tool_backend,
)
from opensearch_vl_repro.agent.runtime import AgentTrajectory, AgentTurn  # noqa: E402
from opensearch_vl_repro.agent.search_providers import SearchBackendError  # noqa: E402
from opensearch_vl_repro.agent.tool_registry import ToolContext, ToolResult  # noqa: E402
from opensearch_vl_repro.evaluation import (  # noqa: E402
    BatchRunner, BatchSample, create_run_manifest,
)


class OfflineBatchRuntime:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def run(self, *, question, images, sample_id, benchmark):
        self.calls[sample_id] = self.calls.get(sample_id, 0) + 1
        failed = sample_id == "B" and self.calls[sample_id] == 1
        return AgentTrajectory(
            sample_id=sample_id, benchmark=benchmark,
            turns=[AgentTurn(
                assistant_output='text_search({"q":"shared"})',
                tool_call={"name": "text_search", "arguments": {"q": "shared"}},
                observation="<observation>offline evidence</observation>",
                status="error" if failed else "success",
                error="synthetic_failure" if failed else None,
                metadata={"cache_hit": sample_id != "A", "attempt_count": 1},
                tool_latency_seconds=0.001,
            )],
            final_answer=None if failed else "offline final answer",
            status="tool_error" if failed else "success",
            image_ids=["img_1"], error="synthetic failure" if failed else None,
            images=[{"image_id": "img_1", "kind": "initial", "parent_id": None,
                     "size": [8, 8], "sha256": "0" * 64, "metadata": {}}],
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Phase 4 reliability smoke")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    run_name = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    output = (args.output_dir or PROJECT_ROOT / "reports" / "phase4_smoke" / run_name).resolve()

    provider_calls = 0
    retry_attempts = 0
    retry = RetryPolicy(max_attempts=3, backoff_seconds=(0, 0), sleeper=lambda _: None)

    def real_backend(arguments, context):
        nonlocal provider_calls, retry_attempts
        provider_calls += 1

        def transient_then_success():
            nonlocal retry_attempts
            retry_attempts += 1
            if retry_attempts == 1:
                raise SearchBackendError("network_error", "synthetic", retryable=True)
            return "<observation>cached offline evidence</observation>"

        observation, attempts = retry.run(transient_then_success)
        return ToolResult(status="success", observation=observation,
                          metadata={"provider": "fake", "attempt_count": attempts})

    cached = cached_tool_backend(
        tool="text_search", backend=real_backend,
        cache=FileSystemToolCache(output / "cache"), behavior_version="offline-smoke-v1",
        argument_defaults={"top_k": 5},
    )
    registry = ImageRegistry()
    registry.register_initial_image(Image.new("RGB", (8, 8), "navy"))
    first = cached({"q": "shared"}, ToolContext(registry, sample_id="first"))
    second = cached({"q": "shared"}, ToolContext(registry, sample_id="different-history"))

    samples = [BatchSample(name, "synthetic", f"question {name}",
                           [Image.new("RGB", (8, 8), "white")]) for name in "ABCD"]
    runtime = OfflineBatchRuntime()
    manifest = create_run_manifest(
        run_id="phase4-offline-smoke", model_name_or_path="offline-fake-model",
        model_revision="v1", inference_config_fingerprint="offline-inference-v1",
        dataset_path=output / "synthetic-dataset",
        dataset_identity={"kind": "synthetic", "version": 1},
        start=0, limit=4, max_agent_turns=2,
        search_config_fingerprint="offline-search-v1",
        layout_config_fingerprint="offline-layout-v1",
    )
    runner = BatchRunner(runtime, output / "batch", run_manifest=manifest)
    first_batch = runner.run(samples, max_samples=2)
    resumed = runner.run(samples)
    calls_after_resume = dict(runtime.calls)
    retried = runner.run(samples, retry_failed=True)
    checks = {
        "first_call_cache_miss": first.metadata["cache_hit"] is False,
        "second_call_cache_hit": second.metadata["cache_hit"] is True,
        "cache_hit_skipped_provider": provider_calls == 1,
        "transient_retry_attempts": first.metadata["attempt_count"] == 2,
        "first_batch_has_failure_and_pending": first_batch["failed"] == 1 and first_batch["pending"] == 2,
        "resume_skips_success_and_failed": calls_after_resume == {"A": 1, "B": 1, "C": 1, "D": 1},
        "retry_failed_exactly_once": retried["failed"] == 0 and runtime.calls["B"] == 2,
        "trajectory_jsonl_exists": (output / "batch" / "trajectories.jsonl").is_file(),
        "run_manifest_created": (output / "batch" / "run_manifest.json").is_file(),
        "status_and_summary_exist": all((output / "batch" / name).is_file()
                                        for name in ("status.json", "summary.json")),
        "resume_left_two_success_before_retry": resumed["success"] == 3 and resumed["failed"] == 1,
    }
    report = {"passed": all(checks.values()), "checks": checks,
              "output_dir": str(output), "provider_calls": provider_calls,
              "retry_attempts": retry_attempts}
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
