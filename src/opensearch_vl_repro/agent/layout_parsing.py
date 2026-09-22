"""Provider-neutral layout output with a PaddleOCR AI Studio job adapter."""

from __future__ import annotations

import io
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import quote

import requests
import yaml
from PIL import Image

from .reliability import RetryPolicy
from .tool_registry import ToolContext, ToolResult


@dataclass(frozen=True)
class LayoutBlock:
    kind: str
    content: str
    bbox: tuple[float, ...] | None = None


@dataclass(frozen=True)
class LayoutDocument:
    blocks: tuple[LayoutBlock, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class LayoutParsingBackend(Protocol):
    def parse(self, image: Image.Image, *, use_chart_recognition: bool | None,
              use_doc_orientation_classify: bool | None) -> LayoutDocument: ...


class LayoutBackendError(RuntimeError):
    def __init__(self, error_type: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


def format_layout_observation(document: LayoutDocument) -> str:
    if not document.blocks:
        raise LayoutBackendError("invalid_response", "layout result contains no readable content")
    lines = ["<observation>", "Content:", ""]
    for block in document.blocks:
        label = block.kind.strip().replace("\n", " ").title() or "Text"
        lines.extend((f"[{label}]", block.content.strip(), ""))
    lines.append("</observation>")
    return "\n".join(lines)


def layout_tool(backend: LayoutParsingBackend | None) -> Callable[[dict[str, Any], ToolContext], ToolResult]:
    def execute(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if backend is None:
            return _failure("configuration_error", "layout parsing provider is not configured")
        try:
            value = context.image_registry.get(arguments["image"])
            if isinstance(value, Image.Image):
                image = value
            else:
                with Image.open(Path(value)) as loaded:
                    image = loaded.convert("RGB").copy()
            document = backend.parse(
                image,
                use_chart_recognition=arguments.get("use_chart_recognition"),
                use_doc_orientation_classify=arguments.get("use_doc_orientation_classify"),
            )
            return ToolResult(
                status="success",
                observation=format_layout_observation(document),
                metadata={"backend": type(backend).__name__, **document.metadata,
                          "attempt_count": getattr(backend, "last_attempt_count", 1)},
            )
        except LayoutBackendError as exc:
            return _failure(exc.error_type, str(exc),
                            attempt_count=getattr(exc, "attempt_count", 1))
        except (KeyError, FileNotFoundError, OSError):
            return _failure("invalid_image", "registered image is unavailable or unreadable")
        except Exception as exc:
            # Never place arbitrary provider exception text in a model message.
            return _failure("provider_error", f"unexpected {type(exc).__name__}")

    return execute


def _failure(error_type: str, message: str, *, attempt_count: int = 1) -> ToolResult:
    safe = message.replace("\n", " ").replace("\r", " ")[:240]
    return ToolResult(
        status="error",
        error_type=error_type,
        observation=f"<observation>\nLayout parsing failed ({error_type}): {safe}.\n</observation>",
        metadata={"error_type": error_type, "attempt_count": attempt_count},
    )


@dataclass(frozen=True)
class LayoutApiConfig:
    provider: str
    job_url: str
    model: str
    access_token_env: str
    request_timeout_seconds: float
    poll_interval_seconds: float
    max_poll_seconds: float


def load_layout_api_config(path: str | Path) -> LayoutApiConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    section = raw.get("layout_parsing") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise ValueError("layout_parsing config section is required")
    required = ("provider", "job_url", "model", "access_token_env",
                "request_timeout_seconds", "poll_interval_seconds", "max_poll_seconds")
    for key in required:
        if section.get(key) is None or section.get(key) == "":
            raise ValueError(f"layout_parsing.{key} is required")
    numbers = {}
    for key in ("request_timeout_seconds", "poll_interval_seconds", "max_poll_seconds"):
        number = float(section[key])
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"layout_parsing.{key} must be finite and positive")
        numbers[key] = number
    if str(section["provider"]) != "paddleocr_aistudio":
        raise ValueError(f"unsupported layout provider: {section['provider']}")
    if not str(section["job_url"]).startswith("https://"):
        raise ValueError("layout_parsing.job_url must be HTTPS")
    return LayoutApiConfig(
        provider=str(section["provider"]), job_url=str(section["job_url"]).rstrip("/"),
        model=str(section["model"]), access_token_env=str(section["access_token_env"]),
        **numbers,
    )


class PaddleOCRAiStudioBackend:
    """Submit a PNG, poll the asynchronous job, then parse its JSONL result."""

    def __init__(self, config: LayoutApiConfig, *, session: Any | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 poll_retry: RetryPolicy | None = None) -> None:
        if config.provider != "paddleocr_aistudio":
            raise ValueError(f"unsupported layout provider: {config.provider}")
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.clock = clock
        self.sleep = sleep
        self.poll_retry = poll_retry or RetryPolicy(sleeper=sleep)
        self.last_attempt_count = 1

    @staticmethod
    def _checked_response(response: Any) -> Any:
        code = response.status_code
        if code in {401, 403}:
            raise LayoutBackendError("authentication_error", f"provider HTTP status {code}")
        if code == 429:
            raise LayoutBackendError("quota_error", "provider HTTP status 429", retryable=True)
        if 500 <= code < 600:
            raise LayoutBackendError("provider_error", f"provider HTTP status {code}", retryable=True)
        if not 200 <= code < 300:
            raise LayoutBackendError("provider_error", f"provider HTTP status {code}")
        return response

    @staticmethod
    def _json(response: Any) -> dict[str, Any]:
        try:
            raw = response.json()
        except ValueError:
            raise LayoutBackendError("invalid_response", "provider returned invalid JSON") from None
        if not isinstance(raw, dict):
            raise LayoutBackendError("invalid_response", "provider returned a non-object response")
        if raw.get("code") not in (None, 0):
            raise LayoutBackendError("provider_error", "provider returned a nonzero API code")
        return raw

    @staticmethod
    def _data(raw: dict[str, Any]) -> dict[str, Any]:
        data = raw.get("data")
        if not isinstance(data, dict):
            raise LayoutBackendError("invalid_response", "provider response is missing data")
        return data

    @staticmethod
    def _request_error(exc: requests.RequestException) -> LayoutBackendError:
        if isinstance(exc, requests.Timeout):
            return LayoutBackendError("timeout", "provider request timed out", retryable=True)
        if isinstance(exc, requests.ConnectionError):
            return LayoutBackendError("network_error", "provider network request failed", retryable=True)
        return LayoutBackendError("provider_error", "provider request failed")

    def _submit(self, image: Image.Image, headers: dict[str, str],
                optional_payload: dict[str, bool]) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        try:
            response = self.session.post(
                self.config.job_url,
                headers=headers,
                data={"model": self.config.model,
                      "optionalPayload": json.dumps(optional_payload)},
                files={"file": ("image.png", buffer.getvalue(), "image/png")},
                timeout=self.config.request_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise self._request_error(exc) from None
        data = self._data(self._json(self._checked_response(response)))
        job_id = data.get("jobId")
        if not isinstance(job_id, str) or not job_id.strip():
            raise LayoutBackendError("invalid_response", "submit response is missing data.jobId")
        return job_id

    def _poll(self, job_id: str, headers: dict[str, str]) -> str:
        deadline = self.clock() + self.config.max_poll_seconds
        url = f"{self.config.job_url}/{quote(job_id, safe='')}"
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise LayoutBackendError("timeout", "provider job polling exceeded its deadline")
            def request_status() -> Any:
                remaining_now = deadline - self.clock()
                if remaining_now <= 0:
                    raise LayoutBackendError("timeout", "provider job polling exceeded its deadline")
                try:
                    return self._checked_response(self.session.get(
                        url, headers=headers,
                        timeout=min(self.config.request_timeout_seconds, remaining_now),
                    ))
                except requests.RequestException as exc:
                    raise self._request_error(exc) from None

            try:
                response, attempts = self.poll_retry.run(request_status)
                self.last_attempt_count = max(self.last_attempt_count, attempts)
            except LayoutBackendError as exc:
                self.last_attempt_count = max(
                    self.last_attempt_count, getattr(exc, "attempt_count", 1),
                )
                raise
            data = self._data(self._json(response))
            state = data.get("state")
            if state == "done":
                result_url = data.get("resultUrl")
                json_url = result_url.get("jsonUrl") if isinstance(result_url, dict) else None
                if not isinstance(json_url, str) or not json_url.startswith("https://"):
                    raise LayoutBackendError("invalid_response", "done response is missing resultUrl.jsonUrl")
                return json_url
            if state == "failed":
                reason = data.get("errorMsg")
                token = os.environ.get(self.config.access_token_env, "")
                safe_reason = str(reason or "unspecified error")
                if token:
                    safe_reason = safe_reason.replace(token, "[REDACTED]")
                safe_reason = safe_reason.replace("\n", " ").replace("\r", " ")[:160]
                raise LayoutBackendError("provider_error", f"provider job failed: {safe_reason}")
            if state not in {"pending", "running"}:
                raise LayoutBackendError("invalid_response", "provider returned an unknown job state")
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise LayoutBackendError("timeout", "provider job polling exceeded its deadline")
            self.sleep(min(self.config.poll_interval_seconds, remaining))

    @staticmethod
    def _adapt_jsonl(body: str, job_id: str) -> LayoutDocument:
        blocks: list[LayoutBlock] = []
        pages_seen = 0
        lines = [line for line in body.splitlines() if line.strip()]
        if not lines:
            raise LayoutBackendError("invalid_response", "provider returned empty JSONL")
        for line in lines:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                raise LayoutBackendError("invalid_response", "provider returned malformed JSONL") from None
            result = raw.get("result") if isinstance(raw, dict) else None
            pages = result.get("layoutParsingResults") if isinstance(result, dict) else None
            if not isinstance(pages, list) or not pages:
                raise LayoutBackendError("invalid_response", "JSONL line is missing layoutParsingResults")
            for page in pages:
                if not isinstance(page, dict):
                    raise LayoutBackendError("invalid_response", "invalid layout page")
                pages_seen += 1
                page_blocks: list[LayoutBlock] = []
                pruned = page.get("prunedResult")
                items = pruned.get("parsing_res_list") if isinstance(pruned, dict) else None
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        content = item.get("block_content")
                        if not isinstance(content, str) or not content.strip():
                            continue
                        label = item.get("block_label")
                        bbox = item.get("block_bbox")
                        page_blocks.append(LayoutBlock(
                            kind=label if isinstance(label, str) else "text",
                            content=content,
                            bbox=tuple(bbox) if isinstance(bbox, list) else None,
                        ))
                if not page_blocks:
                    markdown = page.get("markdown")
                    markdown_text = markdown.get("text") if isinstance(markdown, dict) else None
                    if isinstance(markdown_text, str) and markdown_text.strip():
                        page_blocks.append(LayoutBlock(kind="text", content=markdown_text))
                blocks.extend(page_blocks)
        if not blocks:
            raise LayoutBackendError("invalid_response", "provider returned no readable layout content")
        return LayoutDocument(
            blocks=tuple(blocks),
            metadata={"provider": "paddleocr_aistudio", "job_id": job_id,
                      "page_count": pages_seen, "block_count": len(blocks),
                      "bounding_boxes": [list(block.bbox) for block in blocks if block.bbox]},
        )

    def _download(self, json_url: str) -> str:
        try:
            response = self.session.get(json_url, timeout=self.config.request_timeout_seconds)
        except requests.RequestException as exc:
            raise self._request_error(exc) from None
        return self._checked_response(response).text

    def parse(self, image: Image.Image, *, use_chart_recognition: bool | None,
              use_doc_orientation_classify: bool | None) -> LayoutDocument:
        self.last_attempt_count = 1
        token = os.environ.get(self.config.access_token_env)
        if not token:
            raise LayoutBackendError(
                "configuration_error", f"missing access token environment variable {self.config.access_token_env}"
            )
        headers = {"Authorization": f"bearer {token}"}
        optional_payload = {"useDocUnwarping": False}
        if use_chart_recognition is not None:
            optional_payload["useChartRecognition"] = use_chart_recognition
        if use_doc_orientation_classify is not None:
            optional_payload["useDocOrientationClassify"] = use_doc_orientation_classify
        job_id = self._submit(image, headers, optional_payload)
        json_url = self._poll(job_id, headers)
        document = self._adapt_jsonl(self._download(json_url), job_id)
        # A provider must never be able to echo our credential into observations
        # or reports, including through OCR text or a malformed job ID.
        return LayoutDocument(
            blocks=tuple(LayoutBlock(
                kind=block.kind.replace(token, "[REDACTED]"),
                content=block.content.replace(token, "[REDACTED]"),
                bbox=block.bbox,
            ) for block in document.blocks),
            metadata={**document.metadata,
                      "job_id": document.metadata["job_id"].replace(token, "[REDACTED]")},
        )
