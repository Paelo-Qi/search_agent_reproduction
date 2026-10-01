"""Stable tool-cache identity; no cache runtime or network requests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION

TOOL_CACHE_SCHEMA_VERSION = 1


def tool_cache_key(*, tool_name: str, arguments: dict[str, Any], provider: str,
                   tool_behavior_version: str,
                   protocol_version: str = RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION) -> str:
    if not all(isinstance(value, str) and value for value in (
            tool_name, provider, tool_behavior_version, protocol_version)) or not isinstance(arguments, dict):
        raise ValueError("tool cache identity fields are invalid")
    payload = {"schema_version": TOOL_CACHE_SCHEMA_VERSION, "tool_name": tool_name,
               "normalized_arguments": arguments, "provider": provider,
               "tool_behavior_version": tool_behavior_version,
               "protocol_version": protocol_version}
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("tool cache arguments must be finite JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()
