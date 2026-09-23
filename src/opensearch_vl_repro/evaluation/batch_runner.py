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


TOOL_NAMES = (
    "web_search", "text_search", "image_search", "layout_parsing",
    "crop", "sharpen", "super_resolution", "perspective_correct",
)
PROVIDER_ERROR_TYPES = {
    "authentication_error", "configuration_error", "invalid_response",
    "network_error", "provider_error", "quota_error", "timeout",
}


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
        def counts_for(sample_ids: list[str]) -> dict[str, Any]:
            counts = {name: 0 for name in ("pending", "running", "success", "failed")}
            tool_calls = {name: 0 for name in TOOL_NAMES}
            tool_stats = {name: {"calls": 0, "cache_hits": 0, "cache_misses": 0,
                                 "real_tool_executions": 0, "errors": 0}
                          for name in TOOL_NAMES}
            cache_hits = cache_misses = total_tool_calls = 0
            duplicate_calls = unknown_image_ids = provider_errors = 0
            for sample_id in sample_ids:
                counts[state["samples"][sample_id]["status"]] += 1
                record = records.get(sample_id, {})
                trajectory = record.get("trajectory") or {}
                for turn in trajectory.get("turns", []):
                    call = turn.get("tool_call")
                    if isinstance(call, dict) and isinstance(call.get("name"), str):
                        name = call["name"]
                        total_tool_calls += 1
                        if name in tool_calls:
                            tool_calls[name] += 1
                            tool_stats[name]["calls"] += 1
                    error_type = turn.get("error")
                    if (isinstance(call, dict) and call.get("name") in tool_stats
                            and error_type is not None):
                        tool_stats[call["name"]]["errors"] += 1
                    duplicate_calls += error_type == "duplicate_tool_call"
                    unknown_image_ids += error_type == "unknown_image_id"
                    provider_errors += error_type in PROVIDER_ERROR_TYPES
                    metadata = turn.get("metadata", {})
                    if metadata.get("cache_hit") is True:
                        cache_hits += 1
                        if isinstance(call, dict) and call.get("name") in tool_stats:
                            tool_stats[call["name"]]["cache_hits"] += 1
                    elif metadata.get("cache_hit") is False:
                        cache_misses += 1
                        if isinstance(call, dict) and call.get("name") in tool_stats:
                            tool_stats[call["name"]]["cache_misses"] += 1
                            tool_stats[call["name"]]["real_tool_executions"] += 1
            total = len(sample_ids)
            processed = counts["success"] + counts["failed"]
            return {
                "total": total, **counts,
                "processed": processed,
                "completion_rate": (processed / total if total else None),
                "total_tool_calls": total_tool_calls,
                "tool_calls": tool_calls,
                "tool_stats": tool_stats,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                # Cache wrappers cover the real external tools. A miss means
                # their backend executed; it is not necessarily one HTTP call.
                "real_tool_executions": cache_misses,
                "duplicate_tool_call": duplicate_calls,
                "unknown_image_id": unknown_image_ids,
                "provider_errors": provider_errors,
            }

        sample_ids = list(state["samples"])
        overall = counts_for(sample_ids)
        benchmarks = sorted({
            state["samples"][sample_id].get("benchmark")
            or records.get(sample_id, {}).get("benchmark") or "unknown"
            for sample_id in sample_ids
        })
        per_benchmark = {}
        for benchmark in benchmarks:
            ids = [sample_id for sample_id in sample_ids if (
                state["samples"][sample_id].get("benchmark")
                or records.get(sample_id, {}).get("benchmark") or "unknown"
            ) == benchmark]
            per_benchmark[benchmark] = counts_for(ids)
        return {**overall, "per_benchmark": per_benchmark}

    def run(self, samples: Sequence[BatchSample], *, retry_failed: bool = False,
            max_samples: int | None = None) -> dict[str, Any]:
        if len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError("sample IDs must be unique")
        self._prepare_manifest()
        state = self._load_status(samples)
        records = self._load_records()
        by_id = {sample.sample_id: sample for sample in samples}
        invocation_samples = samples
        if max_samples is not None:
            if max_samples < 0:
                raise ValueError("max_samples must not be negative")
            # A cap identifies a stable prefix of the full run universe. On
            # restart, do not fill the quota from later samples and silently
            # cross a planned invocation boundary.
            invocation_samples = samples[:max_samples]
        eligible = [sample.sample_id for sample in invocation_samples if (
            state["samples"][sample.sample_id]["status"] == "pending"
            or (retry_failed and state["samples"][sample.sample_id]["status"] == "failed")
        )]

        already_successful = sum(
            item["status"] == "success" for item in state["samples"].values()
        )
        already_failed = sum(
            item["status"] == "failed" for item in state["samples"].values()
        )
        pending = sum(item["status"] == "pending" for item in state["samples"].values())
        print(f"Run ID: {self.run_manifest.get('run_id', 'unknown')}", flush=True)
        print(f"Total samples: {len(state['samples'])}", flush=True)
        print(f"Eligible this invocation: {len(eligible)}", flush=True)
        print(f"Already successful: {already_successful}", flush=True)
        print(f"Already failed: {already_failed}", flush=True)
        print(f"Pending: {pending}", flush=True)
        print(f"Retry failed: {str(retry_failed).lower()}", flush=True)

        for progress_index, sample_id in enumerate(eligible, 1):
            sample = by_id[sample_id]
            item = state["samples"][sample_id]
            item.update(status="running", error_type=None, error=None,
                        attempts=int(item.get("attempts", 0)) + 1)
            _atomic_json(self.status_path, state)
            print(
                f"[{progress_index}/{len(eligible)}] START "
                f"{sample.benchmark} / {sample.sample_id}",
                flush=True,
            )
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
            done = (
                f"[{progress_index}/{len(eligible)}] DONE  "
                f"{sample.benchmark} / {sample.sample_id} status={status} "
                f"tools={record['tool_call_count']} elapsed={elapsed:.1f}s"
            )
            if status == "failed":
                done += f" error_type={error_type}"
            print(done, flush=True)

        summary = self._summary(state, records)
        _atomic_json(self.summary_path, summary)
        return summary
