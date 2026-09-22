"""Sequential sample-level execution, resume, and trajectory persistence."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from opensearch_vl_repro.agent.reliability import redact_secrets
from opensearch_vl_repro.agent.runtime import AgentRuntime, AgentTrajectory
from .run_manifest import RunManifestMismatchError, manifest_mismatches


@dataclass(frozen=True)
class BatchSample:
    sample_id: str
    benchmark: str
    question: str
    images: Sequence[Any]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(redact_secrets(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class BatchRunner:
    """Run each eligible sample at most once per invocation."""

    def __init__(self, runtime: AgentRuntime, output_dir: str | Path,
                 *, run_manifest: dict[str, Any],
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.runtime = runtime
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.status_path = self.output_dir / "status.json"
        self.trajectory_path = self.output_dir / "trajectories.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.run_manifest = redact_secrets(run_manifest)
        self.clock = clock

    def _prepare_manifest(self) -> None:
        if self.manifest_path.is_file():
            try:
                persisted = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RunManifestMismatchError(
                    "Run configuration mismatch; refusing to resume existing run.\n"
                    f"- run_manifest ({type(exc).__name__})"
                ) from None
            if not isinstance(persisted, dict):
                raise RunManifestMismatchError(
                    "Run configuration mismatch; refusing to resume existing run.\n"
                    "- run_manifest"
                )
            mismatches = manifest_mismatches(persisted, self.run_manifest)
            if mismatches:
                fields = "\n".join(f"- {field}" for field in mismatches)
                raise RunManifestMismatchError(
                    "Run configuration mismatch; refusing to resume existing run.\n" + fields
                )
            return
        if self.output_dir.exists():
            raise RunManifestMismatchError(
                "Run configuration mismatch; refusing to resume existing run.\n"
                "- run_manifest_missing"
            )
        _atomic_json(self.manifest_path, self.run_manifest)

    def _load_status(self, samples: Sequence[BatchSample]) -> dict[str, Any]:
        if self.status_path.is_file():
            raw = json.loads(self.status_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("samples"), dict):
                raise ValueError("status.json has an invalid structure")
            state = raw
        else:
            state = {"version": 1, "samples": {}}
        known = state["samples"]
        for sample in samples:
            known.setdefault(sample.sample_id, {
                "benchmark": sample.benchmark, "status": "pending", "attempts": 0,
                "error_type": None, "error": None,
            })
        valid = {"pending", "running", "success", "failed"}
        if any(item.get("status") not in valid for item in known.values()):
            raise ValueError("status.json contains an unsupported sample status")
        for item in known.values():
            if item["status"] == "running":
                item.update(status="failed", error_type="interrupted",
                            error="previous batch invocation ended while this sample was running")
        _atomic_json(self.status_path, state)
        return state

    def _load_records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.trajectory_path.is_file():
            return records
        for line in self.trajectory_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["sample_id"])] = record
        return records

    def _persist_records(self, records: dict[str, dict[str, Any]]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.trajectory_path.with_name(
            f".{self.trajectory_path.name}.{os.getpid()}.tmp"
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records.values():
                handle.write(json.dumps(redact_secrets(record), ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, self.trajectory_path)

    @staticmethod
    def _record(sample: BatchSample, trajectory: AgentTrajectory,
                elapsed_seconds: float) -> dict[str, Any]:
        payload = trajectory.to_dict()
        # AgentTrajectory image summaries contain IDs, hashes, dimensions and
        # lineage only. The original Sample.images are intentionally omitted.
        return {
            "sample_id": sample.sample_id,
            "benchmark": sample.benchmark,
            "question": sample.question,
            "status": "success" if trajectory.status == "success" else "failed",
            "trajectory_status": trajectory.status,
            "final_answer": trajectory.final_answer,
            "error": trajectory.error,
            "elapsed_seconds": elapsed_seconds,
            "tool_call_count": sum(turn.tool_call is not None for turn in trajectory.turns),
            "trajectory": payload,
        }

    @staticmethod
    def _summary(state: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
        counts = {name: 0 for name in ("pending", "running", "success", "failed")}
        for item in state["samples"].values():
            counts[item["status"]] += 1
        cache_hits = cache_misses = 0
        for record in records.values():
            for turn in record.get("trajectory", {}).get("turns", []):
                metadata = turn.get("metadata", {})
                if metadata.get("cache_hit") is True:
                    cache_hits += 1
                elif metadata.get("cache_hit") is False:
                    cache_misses += 1
        return {
            "total": len(state["samples"]), **counts,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "real_tool_executions": cache_misses,
        }

    def run(self, samples: Sequence[BatchSample], *, retry_failed: bool = False,
            max_samples: int | None = None) -> dict[str, Any]:
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("sample IDs must be unique")
        self._prepare_manifest()
        state = self._load_status(samples)
        records = self._load_records()
        by_id = {sample.sample_id: sample for sample in samples}
        eligible = [sample.sample_id for sample in samples if (
            state["samples"][sample.sample_id]["status"] == "pending"
            or (retry_failed and state["samples"][sample.sample_id]["status"] == "failed")
        )]
        if max_samples is not None:
            if max_samples < 0:
                raise ValueError("max_samples must not be negative")
            eligible = eligible[:max_samples]

        for sample_id in eligible:
            sample = by_id[sample_id]
            item = state["samples"][sample_id]
            item.update(status="running", error_type=None, error=None,
                        attempts=int(item.get("attempts", 0)) + 1)
            _atomic_json(self.status_path, state)
            started = self.clock()
            try:
                trajectory = self.runtime.run(
                    question=sample.question, images=sample.images,
                    sample_id=sample.sample_id, benchmark=sample.benchmark,
                )
                elapsed = self.clock() - started
                record = self._record(sample, trajectory, elapsed)
                status = record["status"]
                error_type = None if status == "success" else trajectory.status
                error = trajectory.error
            except Exception as exc:
                elapsed = self.clock() - started
                status, error_type = "failed", type(exc).__name__
                error = str(exc)
                record = {
                    "sample_id": sample.sample_id, "benchmark": sample.benchmark,
                    "question": sample.question, "status": status,
                    "trajectory_status": "exception", "final_answer": None,
                    "error": error, "elapsed_seconds": elapsed,
                    "tool_call_count": 0, "trajectory": None,
                }
            records[sample_id] = redact_secrets(record)
            self._persist_records(records)
            item.update(status=status, error_type=error_type, error=redact_secrets(error))
            _atomic_json(self.status_path, state)
            _atomic_json(self.summary_path, self._summary(state, records))

        summary = self._summary(state, records)
        _atomic_json(self.summary_path, summary)
        return summary
