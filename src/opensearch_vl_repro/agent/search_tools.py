"""Model-facing Phase 3 search tools; providers stay below this boundary."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .search_providers import (
    JinaReaderBackend, SearchBackendError, SearchConfig, SerpApiLensBackend,
    SerperSearchBackend,
)
from .tool_registry import ToolContext, ToolResult


def _redact(value: Any) -> Any:
    secrets = [os.environ.get(name) for name in ("SERPER_API_KEY", "JINA_API_KEY", "SERPAPI_API_KEY")]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _error(tool: str, exc: SearchBackendError) -> ToolResult:
    return ToolResult(
        status="error", error_type=exc.error_type,
        observation=_redact(f"<observation>\n{tool} failed ({exc.error_type}): {str(exc)[:180]}.\n</observation>"),
        metadata={"error_type": exc.error_type},
    )


def _query(arguments: dict[str, Any]) -> str:
    query = arguments["q"].strip()
    if not query:
        raise SearchBackendError("invalid_argument", "query must not be empty")
    return query


def _top_k(value: Any, *, default: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SearchBackendError("invalid_argument", "top_k must be a finite integer")
    if not float(value).is_integer() or not 1 <= value <= maximum:
        raise SearchBackendError("invalid_argument", f"top_k must be an integer from 1 to {maximum}")
    return int(value)


def _observation(results: list[str], heading: str, *, max_chars: int | None = None) -> tuple[str, set[int]]:
    prefix = f"<observation>\n{heading}:\n\n"
    suffix = "</observation>"
    budget = max_chars - len(prefix) - len(suffix) if max_chars is not None else None
    parts: list[str] = []
    truncated: set[int] = set()
    for position, chunk in enumerate(results):
        if budget is not None:
            if budget <= 0:
                truncated.update(range(position + 1, len(results) + 1))
                break
            if len(chunk) > budget:
                parts.append(chunk[:budget])
                truncated.update(range(position + 1, len(results) + 1))
                break
            budget -= len(chunk)
        parts.append(chunk)
    return prefix + "".join(parts) + suffix, truncated


class SearchTools:
    def __init__(self, config: SearchConfig, *, serper: SerperSearchBackend | None = None,
                 reader: JinaReaderBackend | None = None,
                 lens: SerpApiLensBackend | None = None) -> None:
        self.config = config
        self.serper = serper or SerperSearchBackend(config.serper)
        self.reader = reader or JinaReaderBackend(config.jina_reader)
        self.lens = lens or SerpApiLensBackend(config.serpapi)

    def web_search(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        try:
            results = self.serper.search(
                _query(arguments), hl=arguments.get("hl"),
                limit=self.config.web_search["max_results"],
            )
            if not results:
                raise SearchBackendError("no_results", "search returned no usable results")
            chunks = [
                f"[{index}]\nTitle: {item.title[:300]}\nURL: {item.url[:1000]}\n"
                f"Snippet: {item.snippet[:1000]}\n\n"
                for index, item in enumerate(results, 1)
            ]
            observation, _ = _observation(chunks, "Search Results")
            return ToolResult(status="success", observation=_redact(observation),
                              metadata={"provider": "serper", "result_count": len(results)})
        except SearchBackendError as exc:
            return _error("web_search", exc)
        except Exception:
            return _error("web_search", SearchBackendError("provider_error", "unexpected provider failure"))

    def text_search(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        try:
            settings = self.config.text_search
            k = _top_k(arguments.get("top_k"), default=settings["default_top_k"],
                       maximum=settings["max_top_k"])
            results = self.serper.search(_query(arguments), hl=arguments.get("hl"), limit=k)
            if not results:
                raise SearchBackendError("no_results", "search returned no usable results")
            chunks = []
            reader_success = reader_failure = 0
            truncated_indices: set[int] = set()
            for index, item in enumerate(results, 1):
                try:
                    passage = self.reader.read(item.url)
                    reader_success += 1
                except SearchBackendError:
                    passage = ""
                    reader_failure += 1
                if len(passage) > settings["max_chars_per_page"]:
                    passage = passage[:settings["max_chars_per_page"]]
                    truncated_indices.add(index)
                chunk = (f"[{index}]\nTitle: {item.title[:300]}\nURL: {item.url[:1000]}\n"
                         f"Snippet: {item.snippet[:1000]}\n")
                if passage:
                    chunk += f"Passage:\n{passage}\n"
                chunks.append(chunk + "\n")
            observation, total_truncated = _observation(
                chunks, "Search Results", max_chars=settings["max_total_chars"])
            if not reader_success and not any(item.snippet for item in results):
                raise SearchBackendError("no_results", "readers failed and no snippets are available")
            return ToolResult(
                status="success", observation=_redact(observation),
                metadata={"providers": ["serper", "jina_reader"],
                          "result_count": len(results),
                          "reader_success_count": reader_success,
                          "reader_failure_count": reader_failure,
                          "reader_fallback_used": reader_failure > 0,
                          "truncated_result_count": len(truncated_indices | total_truncated)},
            )
        except SearchBackendError as exc:
            return _error("text_search", exc)
        except Exception:
            return _error("text_search", SearchBackendError("provider_error", "unexpected provider failure"))

    def image_search(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            reference = arguments["url"]
            if re.fullmatch(r"img_[1-9][0-9]*", reference) is None or not context.image_registry.exists(reference):
                raise SearchBackendError("invalid_argument", "url must reference a registered img_n")
            value = context.image_registry.get(reference)
            if isinstance(value, Image.Image):
                image = value
            else:
                with Image.open(Path(value)) as loaded:
                    image = loaded.convert("RGB").copy()
            search = self.lens.search(image, limit=self.config.image_search["max_results"])
            image_id, matches = search.image_id, search.matches
            chunks = [
                f"[{index}]\nTitle: {item.title[:300]}\nSource: {item.source[:300]}\n"
                f"URL: {item.link[:1000]}\n\n"
                for index, item in enumerate(matches, 1)
            ]
            if not chunks:
                chunks = ["No visual matches found.\n"]
            observation, _ = _observation(chunks, "Image Search Results")
            return ToolResult(
                status="success", observation=_redact(observation),
                metadata=_redact({"provider": "serpapi_google_lens", "source_image_id": reference,
                                  "provider_image_id": image_id, "result_count": len(matches),
                                  **search.upload_metadata,
                                  "thumbnails": [match.thumbnail for match in matches if match.thumbnail]}),
            )
        except SearchBackendError as exc:
            return _error("image_search", exc)
        except (OSError, ValueError):
            return _error("image_search", SearchBackendError("invalid_argument", "registered image is unreadable"))
        except Exception:
            return _error("image_search", SearchBackendError("provider_error", "unexpected provider failure"))
