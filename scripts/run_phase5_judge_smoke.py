#!/usr/bin/env python3
"""Fully offline Phase 5 judge persistence/retry/resume smoke."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.evaluation import (  # noqa: E402
    JudgeConfig, JudgeResult, JudgeRunner, JudgeSample, build_judge_manifest,
)
from opensearch_vl_repro.agent.reliability import RetryPolicy  # noqa: E402


class TransientFakeError(RuntimeError):
    retryable = True


class FakeProvider:
    def __init__(self, systemic=False):
        self.calls, self.transport_attempts, self.systemic = {}, {}, systemic
    def judge(self, sample):
        self.calls[sample.sample_id] = self.calls.get(sample.sample_id, 0) + 1
        if self.systemic and sample.sample_id == "B":
            return JudgeResult("error", error_type="quota_error", reason="fake quota",
                               metadata={"attempt_count": 3})
        if sample.sample_id == "B":
            return JudgeResult("success", "incorrect", "fake incorrect",
                               metadata={"attempt_count": 1})
        if sample.sample_id == "C" and self.calls[sample.sample_id] == 1:
            return JudgeResult("error", error_type="network_error", reason="fake exhausted retry",
                               metadata={"attempt_count": 3})
        if sample.sample_id == "A":
            def operation():
                self.transport_attempts["A"] = self.transport_attempts.get("A", 0) + 1
                if self.transport_attempts["A"] == 1:
                    raise TransientFakeError("fake transient network error")
                return "ok"
            _, attempts = RetryPolicy(max_attempts=3, backoff_seconds=(0, 0),
                                      sleeper=lambda _: None).run(operation)
        else:
            attempts = 1
        return JudgeResult("success", "correct", "fake correct",
                           metadata={"attempt_count": attempts})


def _samples():
    return [JudgeSample(name, "simplevqa" if name < "C" else "mmsearch",
                        f"question {name}", f"reference {name}", f"answer {name}")
            for name in "ABCD"]


def _manifest(samples):
    return build_judge_manifest(
        parent_manifest={"run_id": "offline-agent", "run_config_fingerprint": "offline-parent"},
        config=JudgeConfig("deepseek", "https://offline.invalid", "fake-deepseek"),
        samples=samples,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output_dir or ROOT / "reports/phase5_judge_smoke" / stamp).resolve()
    samples, provider = _samples(), FakeProvider()
    runner = JudgeRunner(provider, output / "resume", judge_manifest=_manifest(samples))
    first = runner.run(samples, max_samples=3)
    resumed = runner.run(samples)
    retried = runner.run(samples, retry_failed=True)
    systemic_provider = FakeProvider(systemic=True)
    systemic = JudgeRunner(systemic_provider, output / "systemic",
                           judge_manifest=_manifest(samples)).run(samples)
    passed = (
        first["success"] == 2 and first["failed"] == 1 and first["pending"] == 1
        and resumed["pending"] == 0 and resumed["failed"] == 1
        and retried["success"] == 4 and provider.calls["C"] == 2
        and provider.transport_attempts["A"] == 2
        and systemic["success"] == 1 and systemic["failed"] == 1
        and systemic["pending"] == 2
        and (output / "resume/judge_manifest.json").is_file()
        and (output / "resume/judge_summary.json").is_file()
    )
    report = {"passed": passed, "offline": True, "provider_calls": provider.calls,
              "transient_transport_attempts": provider.transport_attempts,
              "first": first, "resumed": resumed, "retry_failed": retried,
              "systemic_fail_fast": systemic, "output_dir": str(output)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
