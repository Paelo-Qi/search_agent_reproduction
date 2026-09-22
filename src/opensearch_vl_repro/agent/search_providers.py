"""Small HTTP adapters and normalized search results for Phase 3."""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image

from .reliability import RetryPolicy


class SearchBackendError(RuntimeError):
    def __init__(self, error_type: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class ImageSearchResult:
    title: str
    source: str
    link: str
    thumbnail: str | None = None


@dataclass(frozen=True)
class LensSearchResponse:
    image_id: str
    matches: tuple[ImageSearchResult, ...]
    upload_metadata: dict[str, int | bool]
    attempt_count: int = 1


@dataclass(frozen=True)
class EncodedUpload:
    filename: str
    data: bytes
    mime_type: str
    metadata: dict[str, int | bool]

    def multipart_file(self) -> tuple[str, bytes, str]:
        return self.filename, self.data, self.mime_type


@dataclass(frozen=True)
class SearchConfig:
    serper: dict[str, Any]
    jina_reader: dict[str, Any]
    serpapi: dict[str, Any]
    text_search: dict[str, int]
    web_search: dict[str, int]
    image_search: dict[str, int]


def load_search_config(path: str | Path) -> SearchConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    search = raw.get("search") if isinstance(raw, dict) else None
    if not isinstance(search, dict):
        raise ValueError("search config section is required")
    for section, fields in {
        "serper": ("endpoint", "api_key_env", "timeout_seconds"),
        "jina_reader": ("endpoint_prefix", "api_key_env", "timeout_seconds"),
        "serpapi": ("upload_endpoint", "lens_endpoint", "api_key_env", "timeout_seconds"),
        "text_search": ("default_top_k", "max_top_k", "max_chars_per_page", "max_total_chars"),
        "web_search": ("max_results",),
        "image_search": ("max_results",),
    }.items():
        block = search.get(section)
        if not isinstance(block, dict) or any(key not in block for key in fields):
            raise ValueError(f"search.{section} is missing required fields")
        for key in fields:
            value = block[key]
            if key.endswith("seconds"):
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"search.{section}.{key} must be positive and finite")
            elif key.startswith("max_") or key == "default_top_k":
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"search.{section}.{key} must be a positive integer")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"search.{section}.{key} must be a nonempty string")
        for key in ("endpoint", "endpoint_prefix", "upload_endpoint", "lens_endpoint"):
            if key in block and not block[key].startswith("https://"):
                raise ValueError(f"search.{section}.{key} must use HTTPS")
    if search["text_search"]["default_top_k"] > search["text_search"]["max_top_k"]:
        raise ValueError("default_top_k exceeds max_top_k")
    if search["text_search"]["max_total_chars"] < 100:
        raise ValueError("max_total_chars must be at least 100")
    return SearchConfig(**{key: dict(search[key]) for key in SearchConfig.__dataclass_fields__})


def _credential(env_name: str, *, required: bool = True) -> str | None:
    value = os.environ.get(env_name)
    if required and not value:
        raise SearchBackendError("configuration_error", f"missing {env_name}")
    return value


def _response(response: Any) -> Any:
    code = response.status_code
    if code in (401, 403):
        raise SearchBackendError("authentication_error", f"provider HTTP {code}")
    if code == 429:
        raise SearchBackendError("quota_error", "provider HTTP 429", retryable=True)
    if 500 <= code < 600:
        raise SearchBackendError("provider_error", f"provider HTTP {code}", retryable=True)
    if not 200 <= code < 300:
        raise SearchBackendError("provider_error", f"provider HTTP {code}")
    return response


def _json(response: Any) -> dict[str, Any]:
    try:
        raw = response.json()
    except ValueError:
        raise SearchBackendError("invalid_response", "provider returned invalid JSON") from None
    if not isinstance(raw, dict):
        raise SearchBackendError("invalid_response", "provider returned a non-object response")
    if raw.get("error"):
        raise SearchBackendError("provider_error", "provider reported an error")
    return raw


def _request(method: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return _response(method(*args, **kwargs))
    except requests.Timeout:
        raise SearchBackendError("timeout", "provider request timed out", retryable=True) from None
    except requests.ConnectionError:
        raise SearchBackendError("network_error", "provider network request failed", retryable=True) from None
    except requests.RequestException:
        raise SearchBackendError("provider_error", "provider request failed") from None


def _field(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


class SerperSearchBackend:
    def __init__(self, config: dict[str, Any], *, session: Any | None = None,
                 retry: RetryPolicy | None = None) -> None:
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.retry = retry or RetryPolicy()
        self.last_attempt_count = 1

    def search(self, query: str, *, hl: str | None, limit: int) -> tuple[SearchResult, ...]:
        self.last_attempt_count = 1
        key = _credential(self.config["api_key_env"])
        body: dict[str, Any] = {"q": query, "num": limit}
        if hl is not None:
            body["hl"] = hl
        try:
            response, self.last_attempt_count = self.retry.run(lambda: _request(
                self.session.post, self.config["endpoint"],
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json=body, timeout=self.config["timeout_seconds"],
            ))
        except SearchBackendError as exc:
            self.last_attempt_count = getattr(exc, "attempt_count", 1)
            raise
        raw = _json(response)
        organic = raw.get("organic")
        if organic is None and "searchParameters" in raw:
            organic = []
        if not isinstance(organic, list):
            raise SearchBackendError("invalid_response", "provider response is missing organic results")
        results = []
        for item in organic:
            if not isinstance(item, dict):
                continue
            title, url = _field(item.get("title")), _field(item.get("link"))
            if title and url:
                results.append(SearchResult(title, url, _field(item.get("snippet"))))
        return tuple(results[:limit])


class JinaReaderBackend:
    def __init__(self, config: dict[str, Any], *, session: Any | None = None,
                 retry: RetryPolicy | None = None) -> None:
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.retry = retry or RetryPolicy()
        self.last_attempt_count = 1

    def read(self, url: str) -> str:
        self.last_attempt_count = 1
        if not url.startswith(("https://", "http://")):
            raise SearchBackendError("invalid_argument", "reader URL must be HTTP(S)")
        key = _credential(self.config["api_key_env"], required=False)
        headers = {"Accept": "text/plain"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            response, self.last_attempt_count = self.retry.run(lambda: _request(
                self.session.get, self.config["endpoint_prefix"].rstrip("/") + "/" + url,
                headers=headers, timeout=self.config["timeout_seconds"],
            ))
        except SearchBackendError as exc:
            self.last_attempt_count = getattr(exc, "attempt_count", 1)
            raise
        text = response.text.strip()
        if not text:
            raise SearchBackendError("invalid_response", "reader returned empty text")
        return text


class SerpApiLensBackend:
    """Upload image bytes first; query Google Lens with the short-lived image_id."""

    def __init__(self, config: dict[str, Any], *, session: Any | None = None,
                 retry: RetryPolicy | None = None) -> None:
        self.config = config
        self.session = session if session is not None else requests.Session()
        self.retry = retry or RetryPolicy()
        self.last_attempt_count = 1

    @staticmethod
    def _encoded_image(image: Image.Image) -> EncodedUpload:
        rgb = image.convert("RGB")
        original_size = rgb.size

        def encode(candidate: Image.Image) -> tuple[str, bytes, str] | None:
            output = io.BytesIO()
            candidate.save(output, format="PNG")
            if len(output.getvalue()) <= 500_000:
                return "image.png", output.getvalue(), "image/png"
            for quality in (85, 70, 55):
                output = io.BytesIO()
                candidate.save(output, format="JPEG", quality=quality, optimize=True)
                if len(output.getvalue()) <= 500_000:
                    return "image.jpg", output.getvalue(), "image/jpeg"
            return None

        candidate = rgb
        scale = 1.0
        for attempt in range(13):
            if attempt:
                scale *= 0.8
                size = (round(original_size[0] * scale), round(original_size[1] * scale))
                if min(size) < 32 or size == candidate.size:
                    break
                candidate = rgb.resize(size, Image.Resampling.LANCZOS)
            encoded = encode(candidate)
            if encoded is not None:
                filename, data, mime_type = encoded
                return EncodedUpload(
                    filename, data, mime_type,
                    {"original_width": original_size[0], "original_height": original_size[1],
                     "uploaded_width": candidate.width, "uploaded_height": candidate.height,
                     "upload_bytes": len(data),
                     "resized_for_upload": candidate.size != original_size},
                )
        raise SearchBackendError("invalid_argument", "image exceeds SerpApi 500 KB upload limit after bounded resize")

    def search(self, image: Image.Image, *, limit: int) -> LensSearchResponse:
        self.last_attempt_count = 1
        key = _credential(self.config["api_key_env"])
        encoded = self._encoded_image(image)
        try:
            upload, upload_attempts = self.retry.run(lambda: _request(
                self.session.post, self.config["upload_endpoint"],
                data={"api_key": key}, files={"image": encoded.multipart_file()},
                timeout=self.config["timeout_seconds"],
            ))
            self.last_attempt_count = upload_attempts
        except SearchBackendError as exc:
            self.last_attempt_count = getattr(exc, "attempt_count", 1)
            raise
        upload_raw = _json(upload)
        image_id = upload_raw.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise SearchBackendError("invalid_response", "upload response is missing image_id")
        try:
            lens, lens_attempts = self.retry.run(lambda: _request(
                self.session.get, self.config["lens_endpoint"],
                params={"engine": "google_lens", "type": "visual_matches",
                        "image_id": image_id, "api_key": key},
                timeout=self.config["timeout_seconds"],
            ))
            self.last_attempt_count = max(upload_attempts, lens_attempts)
        except SearchBackendError as exc:
            self.last_attempt_count = max(upload_attempts, getattr(exc, "attempt_count", 1))
            raise
        raw = _json(lens)
        matches = raw.get("visual_matches")
        metadata = raw.get("search_metadata")
        if matches is None and isinstance(metadata, dict) and metadata.get("status") == "Success":
            matches = []
        if not isinstance(matches, list):
            raise SearchBackendError("invalid_response", "Lens response is missing visual_matches")
        results = []
        for item in matches:
            if not isinstance(item, dict):
                continue
            title, link = _field(item.get("title")), _field(item.get("link"))
            if title and link:
                results.append(ImageSearchResult(title, _field(item.get("source")),
                                                 link, _field(item.get("thumbnail")) or None))
        return LensSearchResponse(
            image_id, tuple(results[:limit]), encoded.metadata, self.last_attempt_count,
        )
