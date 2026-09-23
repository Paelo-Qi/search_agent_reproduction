"""Crash-safe, resumable persistence for correctness judging."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from opensearch_vl_repro.agent.reliability import canonical_json, redact_secrets
from .judge import (JUDGE_PROMPT_VERSION, SYSTEMIC_ERROR_TYPES, JudgeConfig,
                    JudgeResult, JudgeSample)
from .run_manifest import RunManifestMismatchError


JUDGE_MANIFEST_VERSION = 1


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(redact_secrets(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_judge_manifest(*, parent_manifest: dict[str, Any], config: JudgeConfig,
                         samples: Sequence[JudgeSample],
                         created_at: str | None = None) -> dict[str, Any]:
    public_config = {
        "provider": config.provider, "base_url": config.base_url, "model": config.model,
        "api_key_env": config.api_key_env, "timeout_seconds": config.timeout_seconds,
        "max_attempts": config.max_attempts, "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    inputs = [{"sample_id": item.sample_id, "benchmark": item.benchmark,
               "question": item.question, "reference_answer": item.reference_answer,
               "model_answer": item.model_answer, "upstream_status": item.upstream_status}
              for item in samples]
    identity = {
        "parent_run_id": parent_manifest.get("run_id"),
        "parent_run_config_fingerprint": parent_manifest.get("run_config_fingerprint"),
        "judge_provider": config.provider, "judge_model": config.model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_config_fingerprint": _hash(public_config),
        "sample_count": len(samples), "judge_inputs_fingerprint": _hash(inputs),
    }
    return {"judge_manifest_version": JUDGE_MANIFEST_VERSION, **identity,
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "judge_run_fingerprint": _hash(identity)}


class JudgeRunner:
    def __init__(self, provider: Any, output_dir: str | Path, *,
                 judge_manifest: dict[str, Any],
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.provider, self.clock = provider, clock
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.manifest_path = self.output_dir / "judge_manifest.json"
        self.status_path = self.output_dir / "judge_status.json"
        self.results_path = self.output_dir / "judge_results.jsonl"
        self.summary_path = self.output_dir / "judge_summary.json"
        self.manifest = redact_secrets(judge_manifest)

    def _prepare_manifest(self) -> None:
        if self.manifest_path.is_file():
            try:
                persisted = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RunManifestMismatchError("invalid existing judge manifest") from exc
            fields = ("judge_manifest_version", "parent_run_id",
                      "parent_run_config_fingerprint", "judge_provider", "judge_model",
                      "judge_prompt_version", "judge_config_fingerprint", "sample_count",
                      "judge_inputs_fingerprint", "judge_run_fingerprint")
            mismatches = [name for name in fields if persisted.get(name) != self.manifest.get(name)]
            if mismatches:
                raise RunManifestMismatchError(
                    "Judge configuration mismatch; refusing to resume.\n" +
                    "\n".join(f"- {name}" for name in mismatches)
                )
            return
        if self.output_dir.exists():
            raise RunManifestMismatchError(
                "Judge configuration mismatch; refusing to adopt directory without judge_manifest.json"
            )
        _atomic_json(self.manifest_path, self.manifest)

    def _load_status(self, samples: Sequence[JudgeSample]) -> dict[str, Any]:
        if self.status_path.is_file():
            state = json.loads(self.status_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or not isinstance(state.get("samples"), dict):
                raise ValueError("judge_status.json has an invalid structure")
        else:
            state = {"version": 1, "samples": {}}
        known = state["samples"]
        expected_ids = {sample.sample_id for sample in samples}
        if set(known) - expected_ids:
            raise ValueError("judge status contains IDs outside this run")
        for sample in samples:
            known.setdefault(sample.sample_id, {
                "benchmark": sample.benchmark, "status": "pending", "attempts": 0,
                "error_type": None, "error": None,
            })
        for item in known.values():
            if item.get("status") not in {"pending", "running", "success", "failed"}:
                raise ValueError("unsupported judge sample status")
            if item["status"] == "running":
                item.update(status="failed", error_type="interrupted",
                            error="previous judge invocation ended while running")
        _atomic_json(self.status_path, state)
        return state

    def _load_records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if self.results_path.is_file():
            for line in self.results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    records[str(record["sample_id"])] = record
        return records

    def _persist_records(self, records: dict[str, dict[str, Any]]) -> None:
        temporary = self.results_path.with_name(f".{self.results_path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records.values():
                handle.write(json.dumps(redact_secrets(record), ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, self.results_path)

    @staticmethod
    def _summary(state: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
        def counts(ids: list[str]) -> dict[str, Any]:
            statuses = {name: 0 for name in ("pending", "running", "success", "failed")}
            correct = incorrect = upstream = 0
            for sample_id in ids:
                statuses[state["samples"][sample_id]["status"]] += 1
                record = records.get(sample_id, {})
                correct += record.get("verdict") == "correct"
                incorrect += record.get("verdict") == "incorrect"
                upstream += record.get("error_type") == "upstream_agent_failure"
            denominator = correct + incorrect
            return {"total": len(ids), **statuses, "correct": correct,
                    "incorrect": incorrect, "upstream_failed": upstream,
                    "accuracy_among_successful_judges":
                        (correct / denominator if denominator else None)}
        all_ids = list(state["samples"])
        per_benchmark: dict[str, Any] = {}
        benchmarks = sorted({item["benchmark"] for item in state["samples"].values()})
        for benchmark in benchmarks:
            ids = [sample_id for sample_id, item in state["samples"].items()
                   if item["benchmark"] == benchmark]
            per_benchmark[benchmark] = counts(ids)
        accuracies = [item["accuracy_among_successful_judges"]
                      for item in per_benchmark.values()
                      if item["accuracy_among_successful_judges"] is not None]
        return {**counts(all_ids), "per_benchmark": per_benchmark,
                "macro_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
                "accuracy_denominator": "correct + incorrect (successful judges only)"}

    def run(self, samples: Sequence[JudgeSample], *, retry_failed: bool = False,
            max_samples: int | None = None) -> dict[str, Any]:
        if len({item.sample_id for item in samples}) != len(samples):
            raise ValueError("judge sample IDs must be globally unique")
        self._prepare_manifest()
        state = self._load_status(samples)
        records = self._load_records()
        by_id = {item.sample_id: item for item in samples}
        eligible = [item.sample_id for item in samples
                    if state["samples"][item.sample_id]["status"] == "pending"
                    or (retry_failed and state["samples"][item.sample_id]["status"] == "failed")]
        if max_samples is not None:
            if max_samples < 0:
                raise ValueError("max_samples must not be negative")
            eligible = eligible[:max_samples]
        for sample_id in eligible:
            sample, item = by_id[sample_id], state["samples"][sample_id]
            item.update(status="running", error_type=None, error=None,
                        attempts=int(item.get("attempts", 0)) + 1)
            _atomic_json(self.status_path, state)
            started = self.clock()
            if sample.upstream_status != "success" or not sample.model_answer:
                result = JudgeResult("error", reason="Agent rollout did not produce a final answer",
                                     error_type="upstream_agent_failure", metadata={
                                         "provider": self.manifest["judge_provider"],
                                         "model": self.manifest["judge_model"],
                                         "prompt_version": JUDGE_PROMPT_VERSION,
                                         "attempt_count": 0,
                                     })
            else:
                try:
                    result = self.provider.judge(sample)
                except Exception as exc:
                    result = JudgeResult("error", reason=str(exc),
                                         error_type="provider_exception", metadata={})
            metadata = dict(result.metadata)
            metadata.setdefault("latency_seconds", self.clock() - started)
            record = redact_secrets({
                "sample_id": sample.sample_id, "benchmark": sample.benchmark,
                "question": sample.question, "reference_answer": sample.reference_answer,
                "model_answer": sample.model_answer, "status": result.status,
                "verdict": result.verdict, "reason": result.reason,
                "raw_response": result.raw_response, "error_type": result.error_type,
                "attempt_count": metadata.get("attempt_count", 0),
                "latency_seconds": metadata.get("latency_seconds"),
                "provider": metadata.get("provider", self.manifest["judge_provider"]),
                "model": metadata.get("model", self.manifest["judge_model"]),
                "prompt_version": metadata.get("prompt_version", JUDGE_PROMPT_VERSION),
            })
            records[sample_id] = record
            self._persist_records(records)
            final_status = "success" if result.status == "success" else "failed"
            item.update(status=final_status, error_type=result.error_type,
                        error=redact_secrets(result.reason))
            _atomic_json(self.status_path, state)
            _atomic_json(self.summary_path, self._summary(state, records))
            if result.error_type in SYSTEMIC_ERROR_TYPES:
                break
        summary = self._summary(state, records)
        _atomic_json(self.summary_path, summary)
        return summary
