from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator


DATASET_ID = "OpenSearch-VL/Search-VL-SFT-36K"
DATASET_REVISION = "2c1c460af4fa15bd63210cbf426a96664b959944"
SOURCE_FILES = {
    "fvqa": "fvqa/fvqa_llama_factory_clean.json",
    "livevqa": "livevqa/livevqa_llama_factory_filtered.json",
    "palace": "palace/palace_llama_factory_filtered.json",
    "webqa": "webqa/webqa_llama_factory_filtered.json",
    "wiki_art": "wiki_art/wikiart_llama_factory_filtered.json",
    "wiki_en": "wiki_en/wiki_en_llama_factory_filtered.json",
    "wiki_zh": "wiki_zh/wiki_zh_llama_factory_filtered.json",
}
REFERENCE_TOOLS = {
    "crop",
    "layout_parsing",
    "text_search",
    "image_search",
    "web_search",
    "visit",
    "perspective_correct",
    "super_resolution",
    "sharpen",
    "python_interpreter",
}
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FAILURE_PATTERN = re.compile(
    r"\b(error|failed|failure|timeout|timed out|exception|not found|invalid)\b",
    re.IGNORECASE,
)
TEXT_STRUCTURE_MARKERS = (
    "Title:",
    "URL:",
    "Snippet:",
    "OCR Result:",
    "Image saved as",
    "Search results",
    "Content:",
)
MAX_EXAMPLES = 5
MAX_PREVIEW_CHARS = 500
IMPLEMENTATION_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: str | Path, text: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, output)
    return output


def write_json_atomic(path: str | Path, value: Any) -> Path:
    return write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def current_git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def download_source_file(
    dataset: str,
    revision: str,
    remote_path: str,
    destination: str | Path,
) -> Path:
    output = Path(destination)
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = urllib.parse.quote(dataset, safe="/")
    filename = urllib.parse.quote(remote_path, safe="/")
    url = (
        f"https://huggingface.co/datasets/{dataset_path}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/{filename}?download=true"
    )
    temporary = output.with_name(f".{output.name}.download")
    request = urllib.request.Request(url, headers={"User-Agent": "OpenSearch-VL-Reproduction/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def iter_json_array(path: str | Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Incrementally parse a top-level JSON array with bounded memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with Path(path).open("r", encoding="utf-8") as handle:
        eof = False

        def read_more() -> None:
            nonlocal buffer, eof
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

        while True:
            if not buffer and not eof:
                read_more()

            index = 0
            while index < len(buffer) and buffer[index].isspace():
                index += 1
            if not started:
                if index >= len(buffer):
                    if eof:
                        raise ValueError(f"{path}: empty JSON file")
                    continue
                if buffer[index] != "[":
                    raise ValueError(f"{path}: expected a top-level JSON array")
                started = True
                buffer = buffer[index + 1 :]
                continue

            index = 0
            while index < len(buffer) and (buffer[index].isspace() or buffer[index] == ","):
                index += 1
            if index < len(buffer) and buffer[index] == "]":
                trailing = buffer[index + 1 :] + handle.read()
                if trailing:
                    if trailing.strip():
                        raise ValueError(f"{path}: unexpected content after JSON array")
                return
            if index >= len(buffer):
                if eof:
                    raise ValueError(f"{path}: unterminated JSON array")
                buffer = ""
                read_more()
                continue

            try:
                value, end = decoder.raw_decode(buffer, index)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer = buffer[index:]
                read_more()
                continue
            yield value
            buffer = buffer[end:]


def _python_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _bounded_value(value: Any, limit: int = 1000) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        encoded = repr(value)
    if len(encoded) <= limit:
        return value
    return encoded[:limit] + "…"


def _argument_schema(arguments: Any) -> tuple[str, dict[str, Any], Any]:
    actual_type = _python_type(arguments)
    normalized = arguments
    encoding = "native"
    if isinstance(arguments, str):
        try:
            normalized = json.loads(arguments)
            encoding = "json_string"
        except json.JSONDecodeError:
            normalized = arguments
    normalized_type = _python_type(normalized)
    schema: dict[str, Any] = {"encoding": encoding, "type": normalized_type}
    if isinstance(normalized, dict):
        schema["keys"] = {
            str(key): _python_type(value) for key, value in sorted(normalized.items())
        }
    signature = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    return actual_type, schema, normalized


def _observation_details(value: Any) -> dict[str, Any]:
    raw_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = raw_text.strip()
    parsed = value
    is_json = not isinstance(value, str)
    if isinstance(value, str) and text:
        try:
            parsed = json.loads(text)
            is_json = True
        except json.JSONDecodeError:
            parsed = value

    top_level_keys = sorted(str(key) for key in parsed) if isinstance(parsed, dict) else []
    explicit_status: str | None = None
    if isinstance(parsed, dict):
        status_value = parsed.get("status")
        success_value = parsed.get("success")
        error_value = parsed.get("error")
        if success_value is False or (error_value not in (None, "", False, [])):
            explicit_status = "failure_like"
        elif success_value is True:
            explicit_status = "success_like"
        elif isinstance(status_value, str):
            lowered = status_value.lower()
            if lowered in {"error", "failed", "failure", "timeout", "invalid"}:
                explicit_status = "failure_like"
            elif lowered in {"ok", "success", "succeeded", "complete", "completed"}:
                explicit_status = "success_like"

    if not text:
        status = "unknown"
    elif explicit_status:
        status = explicit_status
    elif FAILURE_PATTERN.search(text):
        status = "failure_like"
    else:
        status = "success_like"

    return {
        "raw_text": raw_text,
        "preview": raw_text[:MAX_PREVIEW_CHARS] + ("…" if len(raw_text) > MAX_PREVIEW_CHARS else ""),
        "python_type": _python_type(parsed) if is_json else "string",
        "format": "json" if is_json else "text",
        "top_level_keys": top_level_keys,
        "length": len(raw_text),
        "empty": not text,
        "status": status,
        "text_markers": [marker for marker in TEXT_STRUCTURE_MARKERS if marker.lower() in text.lower()],
    }


def _parse_call_payload(payload: Any) -> tuple[str | None, Any, str | None]:
    value = payload
    if isinstance(payload, str):
        try:
            value = json.loads(payload.strip())
        except json.JSONDecodeError as exc:
            name_match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)', payload)
            name = name_match.group(1) if name_match else None
            return name, None, f"invalid tool-call JSON: {exc.msg}"
    if not isinstance(value, dict):
        return None, None, f"tool call must decode to an object, got {_python_type(value)}"
    function = value.get("function") if isinstance(value.get("function"), dict) else value
    name = function.get("name")
    arguments = function.get("arguments", function.get("parameters", {}))
    if not isinstance(name, str) or not name.strip():
        return None, arguments, "tool call has no non-empty name"
    return name.strip(), arguments, None


def extract_tool_calls(turn: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    calls: list[dict[str, Any]] = []
    errors: list[str] = []
    value = turn.get("value", "")
    if isinstance(value, str):
        matches = list(TOOL_CALL_PATTERN.finditer(value))
        opening_count = value.count("<tool_call>")
        if opening_count != len(matches):
            errors.append(
                f"found {opening_count} opening tool_call tags but {len(matches)} complete blocks"
            )
        for match in matches:
            name, arguments, error = _parse_call_payload(match.group(1))
            if error:
                errors.append(error)
            if name:
                calls.append({"name": name, "arguments": arguments, "parse_error": error})

    structured = turn.get("tool_calls")
    if structured is not None:
        if not isinstance(structured, list):
            errors.append("turn.tool_calls is not a list")
        else:
            for payload in structured:
                name, arguments, error = _parse_call_payload(payload)
                if error:
                    errors.append(error)
                if name:
                    calls.append({"name": name, "arguments": arguments, "parse_error": error})
    return calls, errors


@dataclass
class ToolAccumulator:
    total_call_count: int = 0
    trajectory_count: int = 0
    max_calls_in_single_trajectory: int = 0
    argument_types: Counter[str] = field(default_factory=Counter)
    argument_keys: Counter[str] = field(default_factory=Counter)
    argument_schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    observation_types: Counter[str] = field(default_factory=Counter)
    observation_formats: Counter[str] = field(default_factory=Counter)
    observation_lengths: list[int] = field(default_factory=list)
    observation_keys: Counter[str] = field(default_factory=Counter)
    text_structure_markers: Counter[str] = field(default_factory=Counter)
    empty_observation_count: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    missing_observation_count: int = 0
    success_examples: list[dict[str, Any]] = field(default_factory=list)
    failure_examples: list[dict[str, Any]] = field(default_factory=list)
    unknown_examples: list[dict[str, Any]] = field(default_factory=list)

    def add_call(
        self,
        arguments: Any,
        observation: Any,
        sample_ref: str,
        source: str,
    ) -> None:
        self.total_call_count += 1
        actual_type, schema, normalized_arguments = _argument_schema(arguments)
        self.argument_types[actual_type] += 1
        if isinstance(normalized_arguments, dict):
            self.argument_keys.update(str(key) for key in normalized_arguments)
        signature = json.dumps(schema, ensure_ascii=False, sort_keys=True)
        entry = self.argument_schemas.setdefault(
            signature,
            {"schema": schema, "count": 0, "example_arguments": []},
        )
        entry["count"] += 1
        if len(entry["example_arguments"]) < 3:
            entry["example_arguments"].append(_bounded_value(arguments))

        if observation is None:
            self.missing_observation_count += 1
            details = {
                "preview": "",
                "python_type": "missing",
                "format": "missing",
                "top_level_keys": [],
                "length": 0,
                "empty": True,
                "status": "unknown",
                "text_markers": [],
            }
        else:
            details = _observation_details(observation)
        self.observation_types[details["python_type"]] += 1
        self.observation_formats[details["format"]] += 1
        self.observation_lengths.append(details["length"])
        self.observation_keys.update(details["top_level_keys"])
        self.text_structure_markers.update(details["text_markers"])
        self.empty_observation_count += int(details["empty"])
        self.status_counts[details["status"]] += 1

        example = {
            "source": source,
            "sample_ref": sample_ref,
            "arguments": _bounded_value(arguments),
            "observation_preview": details["preview"],
            "observation_type": details["python_type"],
            "status": details["status"],
        }
        target = {
            "success_like": self.success_examples,
            "failure_like": self.failure_examples,
            "unknown": self.unknown_examples,
        }[details["status"]]
        if len(target) < MAX_EXAMPLES:
            target.append(example)

    def note_trajectory(self, calls: int) -> None:
        self.trajectory_count += 1
        self.max_calls_in_single_trajectory = max(self.max_calls_in_single_trajectory, calls)

    def finalize(self, total_trajectories: int) -> dict[str, Any]:
        lengths = self.observation_lengths or [0]
        schemas = sorted(
            self.argument_schemas.values(), key=lambda value: (-value["count"], json.dumps(value["schema"], sort_keys=True))
        )
        for schema in schemas:
            schema["frequency"] = schema["count"] / self.total_call_count
        object_call_count = sum(
            entry["count"]
            for entry in schemas
            if entry["schema"].get("type") == "object"
        )
        return {
            "total_call_count": self.total_call_count,
            "trajectory_count": self.trajectory_count,
            "trajectory_usage_rate": self.trajectory_count / total_trajectories if total_trajectories else 0.0,
            "average_calls_per_used_trajectory": (
                self.total_call_count / self.trajectory_count if self.trajectory_count else 0.0
            ),
            "max_calls_in_single_trajectory": self.max_calls_in_single_trajectory,
            "argument_types": dict(self.argument_types.most_common()),
            "argument_keys": {
                key: {
                    "count": count,
                    "frequency_among_object_calls": count / object_call_count if object_call_count else 0.0,
                }
                for key, count in self.argument_keys.most_common()
            },
            "argument_schemas": schemas,
            "has_multiple_argument_schemas": len(schemas) > 1,
            "observation_python_types": dict(self.observation_types.most_common()),
            "observation_formats": dict(self.observation_formats.most_common()),
            "text_vs_json_ratio": {
                "text": self.observation_formats["text"] / self.total_call_count,
                "json": self.observation_formats["json"] / self.total_call_count,
            },
            "observation_length": {
                "min": min(lengths),
                "max": max(lengths),
                "mean": statistics.fmean(lengths),
                "median": statistics.median(lengths),
            },
            "common_top_level_keys": dict(self.observation_keys.most_common()),
            "common_text_structure_markers": dict(self.text_structure_markers.most_common()),
            "empty_observation_count": self.empty_observation_count,
            "missing_observation_count": self.missing_observation_count,
            "success_like_count": self.status_counts["success_like"],
            "failure_like_count": self.status_counts["failure_like"],
            "unknown_count": self.status_counts["unknown"],
            "examples": {
                "success_like": self.success_examples,
                "failure_like": self.failure_examples,
                "unknown": self.unknown_examples,
            },
        }


@dataclass
class SFTToolAuditor:
    total_trajectories: int = 0
    trajectories_with_any_tool: int = 0
    malformed_trajectory_count: int = 0
    total_tool_calls: int = 0
    tool_call_counts_per_trajectory: list[int] = field(default_factory=list)
    tools: dict[str, ToolAccumulator] = field(default_factory=lambda: defaultdict(ToolAccumulator))
    combinations: Counter[tuple[str, ...]] = field(default_factory=Counter)
    cooccurrence: Counter[tuple[str, str]] = field(default_factory=Counter)
    transitions: Counter[tuple[str, str]] = field(default_factory=Counter)
    malformed_reasons: Counter[str] = field(default_factory=Counter)
    malformed_examples: list[dict[str, Any]] = field(default_factory=list)
    source_counts: Counter[str] = field(default_factory=Counter)
    declared_tool_counts: Counter[str] = field(default_factory=Counter)

    def _record_malformed(self, reason: str, source: str, sample_ref: str) -> None:
        self.malformed_reasons[reason] += 1
        if len(self.malformed_examples) < 20:
            self.malformed_examples.append(
                {"source": source, "sample_ref": sample_ref, "reason": reason}
            )

    def add_trajectory(self, sample: Any, source: str, source_index: int) -> None:
        self.total_trajectories += 1
        self.source_counts[source] += 1
        sample_ref = f"{source}:{source_index}"
        malformed = False
        if not isinstance(sample, dict):
            self.malformed_trajectory_count += 1
            self._record_malformed("trajectory is not an object", source, sample_ref)
            self.tool_call_counts_per_trajectory.append(0)
            return

        tools_value = sample.get("tools")
        if isinstance(tools_value, str):
            try:
                tools_value = json.loads(tools_value)
            except json.JSONDecodeError:
                tools_value = None
        if isinstance(tools_value, list):
            for item in tools_value:
                if isinstance(item, dict):
                    function = item.get("function", item)
                    if isinstance(function, dict) and isinstance(function.get("name"), str):
                        self.declared_tool_counts[function["name"]] += 1

        conversations = sample.get("conversations")
        if not isinstance(conversations, list):
            self.malformed_trajectory_count += 1
            self._record_malformed("conversations is not a list", source, sample_ref)
            self.tool_call_counts_per_trajectory.append(0)
            return

        call_sequence: list[str] = []
        per_tool_calls: Counter[str] = Counter()
        for index, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                malformed = True
                self._record_malformed("conversation turn is not an object", source, sample_ref)
                continue
            if turn.get("from") != "gpt":
                continue
            calls, errors = extract_tool_calls(turn)
            if errors:
                malformed = True
                for error in errors:
                    self._record_malformed(error, source, sample_ref)
            observation = None
            if calls:
                if index + 1 < len(conversations):
                    next_turn = conversations[index + 1]
                    if isinstance(next_turn, dict) and next_turn.get("from") == "observation":
                        observation = next_turn.get("value")
                    else:
                        malformed = True
                        self._record_malformed(
                            "tool call is not immediately followed by observation", source, sample_ref
                        )
                else:
                    malformed = True
                    self._record_malformed("tool call is the final turn", source, sample_ref)
            for call in calls:
                name = call["name"]
                self.tools[name].add_call(call["arguments"], observation, sample_ref, source)
                call_sequence.append(name)
                per_tool_calls[name] += 1

        if malformed:
            self.malformed_trajectory_count += 1
        call_count = len(call_sequence)
        self.total_tool_calls += call_count
        self.tool_call_counts_per_trajectory.append(call_count)
        if call_sequence:
            self.trajectories_with_any_tool += 1
            combination = tuple(sorted(set(call_sequence)))
            self.combinations[combination] += 1
            for pair in combinations(combination, 2):
                self.cooccurrence[pair] += 1
            for previous, following in zip(call_sequence, call_sequence[1:]):
                self.transitions[(previous, following)] += 1
            for name, count in per_tool_calls.items():
                self.tools[name].note_trajectory(count)

    def finalize(self) -> dict[str, Any]:
        counts = self.tool_call_counts_per_trajectory or [0]
        tools = {
            name: accumulator.finalize(self.total_trajectories)
            for name, accumulator in sorted(self.tools.items())
        }
        for name, stats in tools.items():
            usage = stats["trajectory_usage_rate"]
            incoming_transitions = sum(
                count for (_, following), count in self.transitions.items() if following == name
            )
            outgoing_transitions = sum(
                count for (previous, _), count in self.transitions.items() if previous == name
            )
            cooccurring_trajectories = sum(
                count for pair, count in self.cooccurrence.items() if name in pair
            )
            if usage >= 0.10:
                priority = "high priority to preserve"
            elif usage >= 0.01:
                priority = "medium priority"
            else:
                priority = "low-frequency / optional candidate"
            stats["compatibility_implication"] = {
                "priority": priority,
                "basis": {
                    "trajectory_usage_rate": usage,
                    "total_call_count": stats["total_call_count"],
                    "incoming_transition_count": incoming_transitions,
                    "outgoing_transition_count": outgoing_transitions,
                    "pairwise_cooccurrence_trajectory_sum": cooccurring_trajectories,
                    "argument_schema_count": len(stats["argument_schemas"]),
                    "observation_format_count": len(stats["observation_formats"]),
                    "mean_observation_length": stats["observation_length"]["mean"],
                    "max_observation_length": stats["observation_length"]["max"],
                },
                "runtime_backend_may_be_replaceable_if_interface_preserved": True,
            }

        combination_rows = [
            {"tools": list(names), "trajectory_count": count}
            for names, count in self.combinations.most_common(20)
        ]
        cooccurrence_rows = [
            {"tool_a": pair[0], "tool_b": pair[1], "trajectory_count": count}
            for pair, count in self.cooccurrence.most_common()
        ]
        transition_rows = [
            {"from": pair[0], "to": pair[1], "count": count}
            for pair, count in self.transitions.most_common()
        ]
        discovered = set(tools)
        focus_tool_contracts: dict[str, Any] = {}
        if "image_search" in tools:
            image_search = tools["image_search"]
            focus_tool_contracts["image_search"] = {
                "observed_argument_schemas": image_search["argument_schemas"],
                "observation_formats": image_search["observation_formats"],
                "common_text_structure_markers": image_search["common_text_structure_markers"],
                "multiple_argument_schemas_observed": image_search["has_multiple_argument_schemas"],
                "compatibility_guidance": (
                    "Preserve the observed `url` string argument (whose values may be dataset image "
                    "references such as img_1) and a plain-text search-results observation. Do not "
                    "silently replace the observation with a JSON-only contract."
                ),
            }
        if "layout_parsing" in tools:
            layout_parsing = tools["layout_parsing"]
            focus_tool_contracts["layout_parsing"] = {
                "observed_argument_schemas": layout_parsing["argument_schemas"],
                "observation_formats": layout_parsing["observation_formats"],
                "common_text_structure_markers": layout_parsing["common_text_structure_markers"],
                "multiple_argument_schemas_observed": layout_parsing["has_multiple_argument_schemas"],
                "compatibility_guidance": (
                    "Preserve the required `image` string argument, accept calls with no optional "
                    "flags, and support the observed boolean `use_chart_recognition` and "
                    "`use_doc_orientation_classify` variants. Preserve a plain-text parsing result; "
                    "`Content:` is the most common detected structural marker."
                ),
            }
        return {
            "total_trajectories": self.total_trajectories,
            "total_tool_calls": self.total_tool_calls,
            "trajectories_with_any_tool": self.trajectories_with_any_tool,
            "trajectories_without_tool": self.total_trajectories - self.trajectories_with_any_tool,
            "average_tool_calls_per_trajectory": self.total_tool_calls / self.total_trajectories
            if self.total_trajectories
            else 0.0,
            "median_tool_calls_per_trajectory": statistics.median(counts),
            "max_tool_calls_per_trajectory": max(counts),
            "malformed_trajectory_count": self.malformed_trajectory_count,
            "malformed_reasons": dict(self.malformed_reasons.most_common()),
            "malformed_examples": self.malformed_examples,
            "source_trajectory_counts": dict(self.source_counts),
            "declared_tool_trajectory_counts": dict(self.declared_tool_counts.most_common()),
            "discovered_tool_count": len(discovered),
            "discovered_tools": sorted(discovered),
            "unexpected_tools_not_in_reference_set": sorted(discovered - REFERENCE_TOOLS),
            "reference_tools_not_observed": sorted(REFERENCE_TOOLS - discovered),
            "runtime_implemented_tools_in_this_reproduction": [],
            "observed_but_not_implemented_in_this_reproduction": sorted(discovered),
            "tools": tools,
            "focus_tool_contracts": focus_tool_contracts,
            "tool_combinations": combination_rows,
            "tool_cooccurrence": cooccurrence_rows,
            "tool_transitions": transition_rows,
        }


def audit_records(records: Iterable[Any], source: str = "fixture") -> dict[str, Any]:
    auditor = SFTToolAuditor()
    for index, sample in enumerate(records):
        auditor.add_trajectory(sample, source, index)
    return auditor.finalize()


def resolve_input_files(
    input_value: str | Path,
    raw_dir: str | Path,
    dataset: str,
    revision: str,
    download_missing: bool,
) -> list[tuple[str, Path]]:
    input_path = Path(input_value)
    if input_path.exists():
        if input_path.is_file():
            return [(input_path.stem, input_path.resolve())]
        resolved: list[tuple[str, Path]] = []
        for source, remote_path in SOURCE_FILES.items():
            candidate = input_path / remote_path
            if candidate.is_file():
                resolved.append((source, candidate.resolve()))
        if resolved:
            return resolved
        json_files = sorted(input_path.rglob("*.json"))
        if json_files:
            return [(path.stem, path.resolve()) for path in json_files]
        raise FileNotFoundError(f"no JSON source files found under {input_path}")

    if str(input_value) != dataset:
        raise FileNotFoundError(f"input is neither a local path nor the configured dataset: {input_value}")
    destination_root = Path(raw_dir).resolve()
    files: list[tuple[str, Path]] = []
    for source, remote_path in SOURCE_FILES.items():
        destination = destination_root / remote_path
        if not destination.is_file():
            if not download_missing:
                raise FileNotFoundError(f"source JSON is missing: {destination}")
            download_source_file(dataset, revision, remote_path, destination)
        files.append((source, destination))
    return files


def run_audit(
    *,
    project_root: str | Path,
    input_value: str | Path = DATASET_ID,
    raw_dir: str | Path,
    dataset: str = DATASET_ID,
    revision: str = DATASET_REVISION,
    download_missing: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    input_files = resolve_input_files(
        input_value, raw_dir, dataset, revision, download_missing
    )
    auditor = SFTToolAuditor()
    sources: dict[str, Any] = {}
    source_errors: list[dict[str, str]] = []
    for source, path in input_files:
        before = auditor.total_trajectories
        try:
            for index, sample in enumerate(iter_json_array(path)):
                auditor.add_trajectory(sample, source, index)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            source_errors.append(
                {"source": source, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
        sources[source] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "trajectory_count": auditor.total_trajectories - before,
        }

    core = auditor.finalize()
    generated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "dataset": dataset if str(input_value) == dataset else "local SearchVL-SFT-compatible input",
        "dataset_revision": revision if str(input_value) == dataset else None,
        "dataset_source": str(input_value),
        "sources": sources,
        "source_errors": source_errors,
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "script_git_commit": current_git_commit(root),
        "generated_at_utc": generated_at,
        "classification_rules": {
            "observation_status": (
                "Explicit status/success/error fields take precedence; otherwise empty observations are unknown, "
                "failure keywords are failure_like, and other non-empty observations are success_like."
            ),
            "failure_keywords": FAILURE_PATTERN.pattern,
            "priority": (
                "high when trajectory usage >=10%; medium when >=1%; otherwise low-frequency/optional."
            ),
        },
        **core,
    }
    return report


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SearchVL-SFT-36K tool contract audit",
        "",
        "This is a read-only statistical audit. It does not modify trajectories or implement tools.",
        "",
        "## 1. Dataset summary",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Revision: `{report.get('dataset_revision')}`",
        f"- Trajectories: {report['total_trajectories']:,}",
        f"- Tool calls: {report['total_tool_calls']:,}",
        f"- With tools: {report['trajectories_with_any_tool']:,}",
        f"- Without tools: {report['trajectories_without_tool']:,}",
        f"- Malformed trajectories: {report['malformed_trajectory_count']:,}",
        f"- Average / median / max calls per trajectory: "
        f"{report['average_tool_calls_per_trajectory']:.3f} / "
        f"{report['median_tool_calls_per_trajectory']} / {report['max_tool_calls_per_trajectory']}",
        "",
        "## 2. Tool frequency and trajectory usage",
        "",
        "| Tool | Calls | Trajectories | Usage | Avg calls when used | Max in one trajectory |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ordered_tools = sorted(
        report["tools"].items(), key=lambda item: (-item[1]["total_call_count"], item[0])
    )
    for name, stats in ordered_tools:
        lines.append(
            f"| `{name}` | {stats['total_call_count']:,} | {stats['trajectory_count']:,} | "
            f"{_percent(stats['trajectory_usage_rate'])} | "
            f"{stats['average_calls_per_used_trajectory']:.3f} | "
            f"{stats['max_calls_in_single_trajectory']} |"
        )

    lines += ["", "## 3. Per-tool argument and observation contracts", ""]
    for name, stats in ordered_tools:
        lines += [f"### `{name}`", ""]
        lines.append(
            f"Priority: **{stats['compatibility_implication']['priority']}**. "
            "Runtime backend may be replaceable if the interface is preserved."
        )
        lines += ["", "Argument schemas:", ""]
        for schema in stats["argument_schemas"][:10]:
            lines.append(
                f"- {schema['count']:,} ({_percent(schema['frequency'])}): "
                f"`{json.dumps(schema['schema'], ensure_ascii=False, sort_keys=True)}`"
            )
            if schema["example_arguments"]:
                preview = json.dumps(schema["example_arguments"][0], ensure_ascii=False)
                lines.append(f"  - Example: `{preview[:500]}`")
        lines += ["", "Observation format:", ""]
        lines.append(
            f"- Types: `{json.dumps(stats['observation_python_types'], ensure_ascii=False)}`"
        )
        lines.append(
            f"- Text/JSON: {_percent(stats['text_vs_json_ratio']['text'])} / "
            f"{_percent(stats['text_vs_json_ratio']['json'])}"
        )
        lines.append(
            f"- Length min/mean/median/max: {stats['observation_length']['min']} / "
            f"{stats['observation_length']['mean']:.1f} / "
            f"{stats['observation_length']['median']} / {stats['observation_length']['max']}"
        )
        lines.append(
            f"- Top-level JSON keys: `{json.dumps(stats['common_top_level_keys'], ensure_ascii=False)}`"
        )
        lines.append(
            f"- Text markers: `{json.dumps(stats['common_text_structure_markers'], ensure_ascii=False)}`"
        )
        lines.append(
            f"- success_like / failure_like / unknown: {stats['success_like_count']} / "
            f"{stats['failure_like_count']} / {stats['unknown_count']}"
        )
        for status in ("success_like", "failure_like", "unknown"):
            examples = stats["examples"][status]
            if examples:
                lines += ["", f"{status} examples:", ""]
                for example in examples[:3]:
                    lines.append(
                        f"- `{example['sample_ref']}` args="
                        f"`{json.dumps(example['arguments'], ensure_ascii=False)[:300]}`; "
                        f"observation: {example['observation_preview'][:500]!r}"
                    )
        lines.append("")

    lines += ["## 4. Most common tool combinations", ""]
    for row in report["tool_combinations"]:
        lines.append(f"- {' + '.join(row['tools'])}: {row['trajectory_count']:,}")
    lines += ["", "## 5. Pairwise co-occurrence", ""]
    for row in report["tool_cooccurrence"][:20]:
        lines.append(f"- {row['tool_a']} + {row['tool_b']}: {row['trajectory_count']:,}")
    lines += ["", "## 6. Adjacent tool transitions", ""]
    for row in report["tool_transitions"][:30]:
        lines.append(f"- {row['from']} → {row['to']}: {row['count']:,}")

    lines += [
        "",
        "## 7. Failure and malformed patterns",
        "",
        f"Malformed trajectories: {report['malformed_trajectory_count']:,}",
        "",
    ]
    for reason, count in report["malformed_reasons"].items():
        lines.append(f"- {reason}: {count:,}")

    lines += [
        "",
        "## 8. Compatibility implications",
        "",
        "Priorities below are derived only from trajectory usage frequency, call count, and observed format complexity.",
        "They are not implementation decisions.",
        "",
    ]
    for name, stats in ordered_tools:
        implication = stats["compatibility_implication"]
        lines.append(
            f"- `{name}`: **{implication['priority']}**; "
            f"{len(stats['argument_schemas'])} argument schema(s), "
            f"{len(stats['observation_formats'])} observation format(s). "
            "A replacement backend should preserve the observed call and observation contract."
        )
    lines += ["", "### Focused contract conclusions", ""]
    for name in ("image_search", "layout_parsing"):
        contract = report.get("focus_tool_contracts", {}).get(name)
        if not contract:
            lines.append(f"- `{name}` was not observed in the scanned data.")
            continue
        lines.append(f"#### `{name}`")
        lines.append("")
        for schema in contract["observed_argument_schemas"]:
            lines.append(
                f"- Arguments ({schema['count']:,} calls): "
                f"`{json.dumps(schema['schema'], ensure_ascii=False, sort_keys=True)}`"
            )
        lines.append(
            f"- Observation formats: "
            f"`{json.dumps(contract['observation_formats'], ensure_ascii=False)}`"
        )
        lines.append(
            f"- Detected text markers: "
            f"`{json.dumps(contract['common_text_structure_markers'], ensure_ascii=False)}`"
        )
        lines.append(
            f"- Multiple argument schemas observed: "
            f"`{str(contract['multiple_argument_schemas_observed']).lower()}`"
        )
        lines.append(f"- Backend compatibility: {contract['compatibility_guidance']}")
        lines.append("")
    lines += [
        "The current reproduction repository implements no runtime tool backends yet. Therefore every observed tool is",
        "reported as observed-but-not-implemented; this audit does not add any backend.",
        "",
        "## 9. Reproducibility",
        "",
        f"- Implementation SHA-256: `{report['implementation_sha256']}`",
        f"- Git commit at generation: `{report.get('script_git_commit')}`",
        f"- Generated at: `{report['generated_at_utc']}`",
        "",
    ]
    for source, details in report["sources"].items():
        lines.append(
            f"- `{source}`: {details['trajectory_count']:,} trajectories, SHA-256 `{details['sha256']}`"
        )
    return "\n".join(lines) + "\n"
