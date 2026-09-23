"""Phase 4 cache, retry, hashing, and secret-safe persistence primitives."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .tool_registry import ToolBackend, ToolContext, ToolResult


CACHE_SCHEMA_VERSION = 1
SEARCH_BEHAVIOR_VERSION = 2
LAYOUT_BEHAVIOR_VERSION = 1
AGENT_BEHAVIOR_VERSION = 2
RUNTIME_METADATA_FIELDS = {
    "cache_hit", "cache_key", "cache_version", "attempt_count",
    "latency_seconds", "cache_warning",
}
SECRET_ENV_NAMES = (
    "SERPER_API_KEY", "JINA_API_KEY", "SERPAPI_API_KEY", "PADDLEOCR_ACCESS_TOKEN",
    "DEEPSEEK_API_KEY",
)


def redact_secrets(value: Any) -> Any:
    secrets = [os.environ.get(name) for name in SECRET_ENV_NAMES]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact_secrets(key): redact_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def image_sha256(value: Any) -> str:
    """Hash decoded pixels, not paths or compressed file bytes."""
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        with Image.open(Path(value)) as loaded:
            image = loaded.convert("RGB").copy()
    digest = hashlib.sha256()
    digest.update(canonical_json({"mode": image.mode, "size": list(image.size)}).encode("utf-8"))
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = (1.0, 2.0, 4.0)
    sleeper: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if any(delay < 0 for delay in self.backoff_seconds):
            raise ValueError("retry backoff must be non-negative")

    def delay_after(self, attempt: int) -> float:
        if not self.backoff_seconds:
            return 0.0
        return self.backoff_seconds[min(attempt - 1, len(self.backoff_seconds) - 1)]

    def run(self, operation: Callable[[], Any]) -> tuple[Any, int]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation(), attempt
            except Exception as exc:
                setattr(exc, "attempt_count", attempt)
                if not getattr(exc, "retryable", False) or attempt >= self.max_attempts:
                    raise
                self.sleeper(self.delay_after(attempt))
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class CacheLookup:
    result: ToolResult | None
    warning: str | None = None


class FileSystemToolCache:
    """Direct SHA256-keyed JSON cache; successful ToolResults only."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve() / f"v{CACHE_SCHEMA_VERSION}"

    def path_for(self, tool: str, key: str) -> Path:
        return self.root / tool / key[:2] / f"{key}.json"

    def get(self, tool: str, key: str, *, behavior_version: str) -> CacheLookup:
        path = self.path_for(tool, key)
        if not path.is_file():
            return CacheLookup(None)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("entry is not an object")
            required = {"tool", "cache_schema_version", "behavior_version", "observation", "metadata"}
            if not required.issubset(raw):
                raise ValueError("entry is missing required fields")
            if (raw["tool"] != tool or raw["cache_schema_version"] != CACHE_SCHEMA_VERSION
                    or raw["behavior_version"] != behavior_version
                    or not isinstance(raw["observation"], str)
                    or not isinstance(raw["metadata"], dict)):
                raise ValueError("entry is incompatible")
            return CacheLookup(ToolResult(
                status="success", observation=raw["observation"], metadata=raw["metadata"],
            ))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return CacheLookup(None, f"corrupt_cache_entry:{type(exc).__name__}")

    def put(self, tool: str, key: str, *, behavior_version: str,
            safe_input: dict[str, Any], result: ToolResult) -> None:
        if result.status != "success":
            return
        metadata = {
            name: value for name, value in result.metadata.items()
            if name not in RUNTIME_METADATA_FIELDS
        }
        entry = redact_secrets({
            "tool": tool,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "behavior_version": behavior_version,
            "canonical_input": safe_input,
            "observation": result.observation,
            "metadata": metadata,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        path = self.path_for(tool, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(entry, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def _normalized_args(tool: str, arguments: dict[str, Any], context: ToolContext,
                     argument_defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = {**(argument_defaults or {}), **arguments}
    if tool in {"text_search", "web_search"}:
        normalized: dict[str, Any] = {"q": arguments["q"].strip()}
        if arguments.get("hl") is not None:
            normalized["hl"] = arguments["hl"].strip().lower()
        if tool == "text_search" and arguments.get("top_k") is not None:
            normalized["top_k"] = int(arguments["top_k"])
        return normalized
    reference_name = "url" if tool == "image_search" else "image"
    image = context.image_registry.get(arguments[reference_name])
    normalized = {"image_sha256": image_sha256(image)}
    for name, value in arguments.items():
        if name != reference_name:
            normalized[name] = value
    return normalized


def cache_identity(tool: str, arguments: dict[str, Any], context: ToolContext,
                   *, behavior_version: str,
                   argument_defaults: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    safe_input = redact_secrets(
        _normalized_args(tool, arguments, context, argument_defaults)
    )
    payload = {
        "tool": tool,
        "args": safe_input,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "behavior_version": behavior_version,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(), safe_input


def cached_tool_backend(*, tool: str, backend: ToolBackend, cache: FileSystemToolCache,
                        behavior_version: str,
                        argument_defaults: dict[str, Any] | None = None,
                        clock: Callable[[], float] = time.perf_counter) -> ToolBackend:
    def execute(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        key, safe_input = cache_identity(
            tool, arguments, context, behavior_version=behavior_version,
            argument_defaults=argument_defaults,
        )
        started = clock()
        lookup = cache.get(tool, key, behavior_version=behavior_version)
        if lookup.result is not None:
            metadata = {
                **lookup.result.metadata,
                "cache_hit": True,
                "cache_key": key,
                "cache_version": CACHE_SCHEMA_VERSION,
                "attempt_count": 0,
                "latency_seconds": clock() - started,
            }
            return ToolResult(status="success", observation=lookup.result.observation, metadata=metadata)
        result = backend(arguments, context)
        metadata = {
            **result.metadata,
            "cache_hit": False,
            "cache_key": key,
            "cache_version": CACHE_SCHEMA_VERSION,
            "attempt_count": int(result.metadata.get("attempt_count", 1)),
            "latency_seconds": clock() - started,
        }
        if lookup.warning:
            metadata["cache_warning"] = lookup.warning
        if result.status == "success":
            try:
                cache.put(tool, key, behavior_version=behavior_version,
                          safe_input=safe_input, result=ToolResult(
                              status="success", observation=result.observation,
                              metadata=metadata, derived_images=result.derived_images,
                          ))
            except (OSError, TypeError, ValueError) as exc:
                metadata["cache_warning"] = f"cache_write_failed:{type(exc).__name__}"
        return ToolResult(
            status=result.status, observation=result.observation, error_type=result.error_type,
            metadata=metadata, derived_images=result.derived_images,
        )

    return execute


def behavior_namespace(prefix: str, configuration: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"
