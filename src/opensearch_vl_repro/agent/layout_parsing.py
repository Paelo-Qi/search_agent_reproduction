"""Provider-independent layout adapter with an optional Baidu Qianfan transport."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml
from PIL import Image

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
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


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
                metadata={"backend": type(backend).__name__, **document.metadata},
            )
        except LayoutBackendError as exc:
            return _failure(exc.error_type, str(exc))
        except (KeyError, FileNotFoundError, OSError):
            return _failure("invalid_image", "registered image is unavailable or unreadable")
        except Exception as exc:
            return _failure("provider_error", f"unexpected {type(exc).__name__}")

    return execute


def _failure(error_type: str, message: str) -> ToolResult:
    # Keep provider credentials and raw responses out of model-visible text.
    safe = message.replace("\n", " ")[:240]
    return ToolResult(
        status="error",
        error_type=error_type,
        observation=f"<observation>\nLayout parsing failed ({error_type}): {safe}.\n</observation>",
        metadata={"error_type": error_type},
    )


@dataclass(frozen=True)
class LayoutApiConfig:
    provider: str
    endpoint: str
    model: str
    api_key_env: str
    timeout_seconds: float


def load_layout_api_config(path: str | Path) -> LayoutApiConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    section = raw.get("layout_parsing") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise ValueError("layout_parsing config section is required")
    for key in ("provider", "endpoint", "model", "api_key_env", "timeout_seconds"):
        if not section.get(key):
            raise ValueError(f"layout_parsing.{key} is required")
    timeout = float(section["timeout_seconds"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("layout_parsing.timeout_seconds must be finite and positive")
    return LayoutApiConfig(
        provider=str(section["provider"]), endpoint=str(section["endpoint"]),
        model=str(section["model"]), api_key_env=str(section["api_key_env"]),
        timeout_seconds=timeout,
    )


Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


def urllib_json_transport(endpoint: str, headers: dict[str, str],
                          body: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class BaiduLayoutParsingBackend:
    def __init__(self, config: LayoutApiConfig,
                 transport: Transport = urllib_json_transport) -> None:
        if config.provider != "baidu":
            raise ValueError(f"unsupported layout provider: {config.provider}")
        self.config = config
        self.transport = transport

    def _request_body(self, image: Image.Image, *,
                      use_chart_recognition: bool | None,
                      use_doc_orientation_classify: bool | None) -> dict[str, Any]:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        body: dict[str, Any] = {
            "model": self.config.model,
            "file": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "fileType": 1,
            "visualize": False,
        }
        if use_chart_recognition is not None:
            body["useChartRecognition"] = use_chart_recognition
        if use_doc_orientation_classify is not None:
            body["useDocOrientationClassify"] = use_doc_orientation_classify
        return body

    @staticmethod
    def _adapt_response(raw: Any) -> LayoutDocument:
        if not isinstance(raw, dict):
            raise LayoutBackendError("invalid_response", "provider returned a non-object response")
        if raw.get("error"):
            error = raw["error"]
            code = str(error.get("code", "")) if isinstance(error, dict) else ""
            kind = "authentication_error" if code in {"invalid_model", "invalid_api_key"} else "provider_error"
            raise LayoutBackendError(kind, f"provider returned error code {code or 'unknown'}")
        result = raw.get("result")
        pages = result.get("layoutParsingResults") if isinstance(result, dict) else None
        if not isinstance(pages, list) or not pages:
            raise LayoutBackendError("invalid_response", "missing layoutParsingResults")
        blocks: list[LayoutBlock] = []
        for page in pages:
            if not isinstance(page, dict):
                raise LayoutBackendError("invalid_response", "invalid page result")
            pruned = page.get("prunedResult")
            block_items = pruned.get("parsing_res_list") if isinstance(pruned, dict) else None
            if isinstance(block_items, list):
                for item in block_items:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("block_content")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    label = item.get("block_label")
                    bbox = item.get("block_bbox")
                    blocks.append(LayoutBlock(
                        kind=label if isinstance(label, str) else "text",
                        content=content,
                        bbox=tuple(bbox) if isinstance(bbox, list) else None,
                    ))
            if not block_items:
                markdown = page.get("markdown")
                text = markdown.get("text") if isinstance(markdown, dict) else None
                if isinstance(text, str) and text.strip():
                    blocks.append(LayoutBlock(kind="text", content=text))
        if not blocks:
            raise LayoutBackendError("invalid_response", "provider returned no readable blocks")
        request_id = raw.get("id")
        return LayoutDocument(
            blocks=tuple(blocks),
            metadata={"provider": "baidu", "request_id": request_id,
                      "block_count": len(blocks),
                      "bounding_boxes": [list(block.bbox) for block in blocks if block.bbox]},
        )

    def parse(self, image: Image.Image, *, use_chart_recognition: bool | None,
              use_doc_orientation_classify: bool | None) -> LayoutDocument:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise LayoutBackendError(
                "configuration_error", f"missing API key environment variable {self.config.api_key_env}"
            )
        body = self._request_body(
            image, use_chart_recognition=use_chart_recognition,
            use_doc_orientation_classify=use_doc_orientation_classify,
        )
        try:
            raw = self.transport(
                self.config.endpoint,
                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                body,
                self.config.timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            kind = ("authentication_error" if exc.code in {401, 403}
                    else "quota_error" if exc.code == 429 else "provider_error")
            raise LayoutBackendError(kind, f"provider HTTP status {exc.code}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise LayoutBackendError("timeout", "provider request timed out") from exc
        except urllib.error.URLError as exc:
            raise LayoutBackendError("network_error", "provider network request failed") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LayoutBackendError("invalid_response", "provider response is not valid JSON") from exc
        return self._adapt_response(raw)
