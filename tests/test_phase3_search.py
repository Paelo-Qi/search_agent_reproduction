from __future__ import annotations

import importlib.util
import io
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from opensearch_vl_repro.agent.image_registry import ImageRegistry
from opensearch_vl_repro.agent.phase2_registry import create_phase2_tool_registry
from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry
from opensearch_vl_repro.agent.reliability import RetryPolicy
from opensearch_vl_repro.agent.search_providers import (
    ImageSearchResult, JinaReaderBackend, LensSearchResponse, SearchBackendError, SearchResult,
    SerpApiLensBackend, SerperSearchBackend, load_search_config,
)
from opensearch_vl_repro.agent.search_tools import SearchTools
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS
from opensearch_vl_repro.agent.tool_registry import ToolContext, ToolResult


ROOT = Path(__file__).resolve().parents[1]


class Response:
    def __init__(self, data=None, *, text="", status=200):
        self.data, self.text, self.status_code = data, text, status

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


class Session:
    def __init__(self, *, posts=(), gets=()):
        self.posts, self.gets = list(posts), list(gets)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        item = self.posts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        item = self.gets.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def config():
    return load_search_config(ROOT / "configs" / "search_backends.example.yaml")


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "serper-secret")
    monkeypatch.setenv("JINA_API_KEY", "jina-secret")
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-secret")


def _context():
    registry = ImageRegistry()
    registry.register_initial_image(Image.new("RGB", (40, 30), "navy"))
    return ToolContext(registry)


def _organic(count=2):
    return {"organic": [
        {"title": f"Title {i}", "link": f"https://example.com/{i}",
         "snippet": f"Snippet {i}"} for i in range(count)
    ], "provider_internal": "do-not-show"}


def test_phase3_registry_keeps_eight_contracts_and_phase2_mock(config):
    phase3 = create_phase3_tool_registry()
    assert phase3.declarations_for_model() == [item.as_chat_template_tool() for item in TOOL_DECLARATIONS]
    assert len(phase3.list_tools()) == 8
    phase2 = create_phase2_tool_registry()
    assert phase2.get("web_search").backend is not phase3.get("web_search").backend
    assert config.text_search["default_top_k"] == 5


def test_serper_request_optional_hl_normalization_and_empty(config, keys):
    session = Session(posts=[Response(_organic()), Response({"organic": []})])
    backend = SerperSearchBackend(config.serper, session=session)
    results = backend.search("query", hl="en", limit=5)
    assert results == (SearchResult("Title 0", "https://example.com/0", "Snippet 0"),
                       SearchResult("Title 1", "https://example.com/1", "Snippet 1"))
    url, kwargs = session.calls[0][1:]
    assert url == "https://google.serper.dev/search"
    assert kwargs["headers"]["X-API-KEY"] == "serper-secret"
    assert kwargs["json"] == {"q": "query", "num": 5, "hl": "en"}
    assert backend.search("query", hl=None, limit=2) == ()
    assert "hl" not in session.calls[1][2]["json"]


def test_serper_valid_empty_page_without_organic(config, keys):
    backend = SerperSearchBackend(config.serper, session=Session(
        posts=[Response({"searchParameters": {"q": "nothing"}})]))
    assert backend.search("nothing", hl=None, limit=5) == ()


def test_missing_serper_key_fails_before_network(config, monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    session = Session()
    backend = SerperSearchBackend(config.serper, session=session)
    with pytest.raises(SearchBackendError) as caught:
        backend.search("q", hl=None, limit=5)
    assert caught.value.error_type == "configuration_error"
    assert not session.calls


@pytest.mark.parametrize("failure,expected", [
    (Response(status=401), "authentication_error"),
    (Response(status=429), "quota_error"),
    (Response(status=500), "provider_error"),
    (requests.Timeout("serper-secret"), "timeout"),
    (requests.ConnectionError("serper-secret"), "network_error"),
    (Response(ValueError("bad JSON")), "invalid_response"),
    (Response({"wrong": []}), "invalid_response"),
])
def test_serper_failure_mapping(config, keys, failure, expected):
    backend = SerperSearchBackend(
        config.serper, session=Session(posts=[failure]), retry=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(SearchBackendError) as caught:
        backend.search("q", hl=None, limit=1)
    assert caught.value.error_type == expected
    assert "serper-secret" not in str(caught.value)


def test_reader_text_auth_and_optional_key(config, keys, monkeypatch):
    session = Session(gets=[Response(text="  Page body  "), Response(text="No key")])
    reader = JinaReaderBackend(config.jina_reader, session=session)
    assert reader.read("https://example.com/page") == "Page body"
    assert session.calls[0][1] == "https://r.jina.ai/https://example.com/page"
    assert session.calls[0][2]["headers"]["Authorization"] == "Bearer jina-secret"
    monkeypatch.delenv("JINA_API_KEY")
    assert reader.read("https://example.com/other") == "No key"
    assert "Authorization" not in session.calls[1][2]["headers"]


@pytest.mark.parametrize("failure,expected", [
    (Response(status=401), "authentication_error"),
    (Response(status=429), "quota_error"),
    (Response(text=""), "invalid_response"),
    (requests.Timeout("jina-secret"), "timeout"),
    (requests.ConnectionError("jina-secret"), "network_error"),
])
def test_reader_failure_mapping(config, keys, failure, expected):
    reader = JinaReaderBackend(
        config.jina_reader, session=Session(gets=[failure]), retry=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(SearchBackendError) as caught:
        reader.read("https://example.com")
    assert caught.value.error_type == expected
    assert "jina-secret" not in str(caught.value)


class FakeSerper:
    def __init__(self, results):
        self.results = tuple(results)
        self.calls = []

    def search(self, query, *, hl, limit):
        self.calls.append((query, hl, limit))
        return self.results[:limit]


class FakeReader:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def read(self, url):
        self.calls.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_web_tool_normalizes_and_no_results(config, keys):
    serper = FakeSerper([SearchResult("Example", "https://example.com", "Snippet")])
    tool = SearchTools(config, serper=serper)
    result = tool.web_search({"q": "  query ", "hl": "en"}, _context())
    assert result.status == "success"
    assert serper.calls == [("query", "en", 5)]
    assert "Title: Example" in result.observation
    assert "URL: https://example.com" in result.observation
    assert "Snippet: Snippet" in result.observation
    serper.results = ()
    assert tool.web_search({"q": "q"}, _context()).error_type == "no_results"


def test_text_tool_args_order_reader_partial_fallback(config, keys):
    results = [SearchResult(f"Title {i}", f"https://example.com/{i}", f"Snippet {i}")
               for i in range(3)]
    serper = FakeSerper(results)
    reader = FakeReader(["Page zero", SearchBackendError("timeout", "timeout"), "Page two"])
    tool = SearchTools(config, serper=serper, reader=reader)
    result = tool.text_search({"q": " q ", "hl": "en", "top_k": 3.0}, _context())
    assert result.status == "success"
    assert serper.calls == [("q", "en", 3)]
    assert reader.calls == [item.url for item in results]
    assert result.metadata["reader_success_count"] == 2
    assert result.metadata["reader_failure_count"] == 1
    assert result.metadata["reader_fallback_used"] is True
    assert result.observation.index("Title 0") < result.observation.index("Title 1") < result.observation.index("Title 2")
    assert "Passage:\nPage zero" in result.observation
    assert "Snippet: Snippet 1" in result.observation
    assert "Summary:" not in result.observation


@pytest.mark.parametrize("top_k", [0, -1, 3.7, 11, float("inf"), float("nan")])
def test_text_top_k_validation(config, top_k):
    serper = FakeSerper([SearchResult("T", "https://x.test", "S")])
    tool = SearchTools(config, serper=serper)
    result = tool.text_search({"q": "x", "top_k": top_k}, _context())
    assert result.error_type == "invalid_argument"
    assert not serper.calls


def test_text_q_only_defaults_and_all_reader_fail_snippet_fallback(config):
    serper = FakeSerper([SearchResult("T", "https://x.test", "S")])
    reader = FakeReader([SearchBackendError("timeout", "timeout")])
    result = SearchTools(config, serper=serper, reader=reader).text_search({"q": "x"}, _context())
    assert serper.calls == [("x", None, 5)]
    assert result.status == "success"
    assert result.metadata["reader_failure_count"] == 1
    assert result.metadata["reader_fallback_used"] is True
    assert "Snippet: S" in result.observation and "Passage:" not in result.observation


def test_unexpected_reader_bug_is_not_silent_fallback(config):
    serper = FakeSerper([SearchResult("T", "https://x.test", "S")])
    reader = FakeReader([TypeError("programming bug")])
    result = SearchTools(config, serper=serper, reader=reader).text_search({"q": "x"}, _context())
    assert result.status == "error"
    assert result.error_type == "provider_error"
    assert "reader_failure_count" not in result.metadata


def test_text_length_limits_and_no_raw_json(config):
    settings = {**config.text_search, "max_chars_per_page": 30, "max_total_chars": 150}
    limited = replace(config, text_search=settings)
    results = [SearchResult("Title", "https://x.test", "Snippet") for _ in range(3)]
    reader = FakeReader(["A" * 200 for _ in results])
    result = SearchTools(limited, serper=FakeSerper(results), reader=reader).text_search({"q": "x"}, _context())
    assert result.status == "success"
    assert len(result.observation) <= 150
    assert result.metadata["truncated_result_count"] >= 1
    assert "provider_internal" not in result.observation


def test_serpapi_upload_lens_and_result_normalization(config, keys):
    session = Session(
        posts=[Response({"image_id": "provider-img-1"})],
        gets=[Response({"visual_matches": [{"title": "Match", "link": "https://x.test",
                                           "source": "Source", "thumbnail": "https://thumb.test"}]})],
    )
    backend = SerpApiLensBackend(
        config.serpapi, session=session, retry=RetryPolicy(max_attempts=1),
    )
    response = backend.search(Image.new("RGB", (40, 30)), limit=10)
    assert response.image_id == "provider-img-1"
    assert response.matches == (ImageSearchResult("Match", "Source", "https://x.test", "https://thumb.test"),)
    assert response.upload_metadata["resized_for_upload"] is False
    assert response.upload_metadata["original_width"] == 40
    assert response.upload_metadata["uploaded_width"] == 40
    assert response.upload_metadata["upload_bytes"] <= 500_000
    upload = session.calls[0]
    assert upload[1] == "https://serpapi.com/image"
    assert upload[2]["data"] == {"api_key": "serpapi-secret"}
    filename, data, mime = upload[2]["files"]["image"]
    assert filename == "image.png" and mime == "image/png"
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (40, 30)
    lens = session.calls[1]
    assert lens[1] == "https://serpapi.com/search"
    assert lens[2]["params"] == {"engine": "google_lens", "type": "visual_matches",
                                   "image_id": "provider-img-1", "api_key": "serpapi-secret"}


def test_serpapi_success_without_visual_matches(config, keys):
    session = Session(posts=[Response({"image_id": "provider-img-2"})],
                      gets=[Response({"search_metadata": {"status": "Success"}})])
    response = SerpApiLensBackend(config.serpapi, session=session).search(
        Image.new("RGB", (4, 4)), limit=10)
    assert response.image_id == "provider-img-2" and response.matches == ()


def test_serpapi_large_noisy_image_uses_bounded_resize(config, keys):
    size = (1600, 1200)
    pixels = random.Random(123).randbytes(size[0] * size[1] * 3)
    image = Image.frombytes("RGB", size, pixels)
    session = Session(posts=[Response({"image_id": "resized-id"})],
                      gets=[Response({"visual_matches": []})])
    response = SerpApiLensBackend(config.serpapi, session=session).search(image, limit=10)
    metadata = response.upload_metadata
    assert metadata["resized_for_upload"] is True
    assert metadata["upload_bytes"] <= 500_000
    assert metadata["uploaded_width"] < size[0]
    assert metadata["uploaded_height"] < size[1]
    assert abs(metadata["uploaded_width"] / metadata["uploaded_height"] - size[0] / size[1]) < 0.01
    _, uploaded_bytes, _ = session.calls[0][2]["files"]["image"]
    assert len(uploaded_bytes) == metadata["upload_bytes"]


def test_image_tool_exposes_upload_metadata(config, keys):
    session = Session(posts=[Response({"image_id": "metadata-id"})],
                      gets=[Response({"visual_matches": []})])
    lens = SerpApiLensBackend(config.serpapi, session=session)
    result = SearchTools(config, lens=lens).image_search({"url": "img_1"}, _context())
    assert result.status == "success"
    assert result.metadata["original_width"] == 40
    assert result.metadata["uploaded_width"] == 40
    assert result.metadata["upload_bytes"] <= 500_000
    assert result.metadata["resized_for_upload"] is False


@pytest.mark.parametrize("upload,lens,expected", [
    (Response(status=401), None, "authentication_error"),
    (Response(status=429), None, "quota_error"),
    (requests.ConnectionError("serpapi-secret"), None, "network_error"),
    (Response({}), None, "invalid_response"),
    (None, Response({}), "invalid_response"),
    (None, Response({"error": "bad"}), "provider_error"),
])
def test_serpapi_failure_mapping(config, keys, upload, lens, expected):
    session = Session(posts=[upload or Response({"image_id": "id"})],
                      gets=[lens or Response({"visual_matches": []})])
    backend = SerpApiLensBackend(
        config.serpapi, session=session, retry=RetryPolicy(max_attempts=1),
    )
    with pytest.raises(SearchBackendError) as caught:
        backend.search(Image.new("RGB", (4, 4)), limit=10)
    assert caught.value.error_type == expected
    assert "serpapi-secret" not in str(caught.value)


def test_image_tool_img_reference_unknown_and_zero_results(config, keys):
    class Lens:
        def __init__(self):
            self.calls = []

        def search(self, image, *, limit):
            self.calls.append((image.size, limit))
            return LensSearchResponse("provider-id", (), {})

    lens = Lens()
    tool = SearchTools(config, lens=lens)
    context = _context()
    assert tool.image_search({"url": "img_99"}, context).error_type == "invalid_argument"
    assert not lens.calls
    result = tool.image_search({"url": "img_1"}, context)
    assert result.status == "success"
    assert result.metadata["result_count"] == 0
    assert result.derived_images == ()
    assert "No visual matches found" in result.observation
    assert lens.calls == [((40, 30), 10)]


def test_image_tool_accepts_registered_local_path(config, keys, tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (12, 9), "white").save(path)
    images = ImageRegistry()
    images.register_initial_image(path)
    class Lens:
        def search(self, image, *, limit):
            assert image.size == (12, 9)
            return LensSearchResponse("provider-id", (), {})
    result = SearchTools(config, lens=Lens()).image_search({"url": "img_1"}, ToolContext(images))
    assert result.status == "success"


def test_credential_redaction_in_observation_metadata_and_report(config, keys, tmp_path, monkeypatch):
    serper = FakeSerper([SearchResult("serper-secret", "https://x.test", "jina-secret")])
    web = SearchTools(config, serper=serper).web_search({"q": "x"}, _context())
    assert "serper-secret" not in web.observation + json.dumps(web.metadata)
    class Lens:
        def search(self, image, *, limit):
            return LensSearchResponse(
                "serpapi-secret", (ImageSearchResult("serpapi-secret", "S", "https://x.test"),), {})
    image = SearchTools(config, lens=Lens()).image_search({"url": "img_1"}, _context())
    assert "serpapi-secret" not in image.observation + json.dumps(image.metadata)
    path = ROOT / "scripts" / "run_search_backends_smoke.py"
    spec = importlib.util.spec_from_file_location("search_smoke_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_registry = SimpleNamespace(execute=lambda *args: ToolResult(
        status="success", observation="<observation>\nSearch Results:\n[1]\nTitle: serper-secret\nURL: https://x.test\n</observation>",
        metadata={"result_count": 1, "echo": "jina-secret"}))
    monkeypatch.setattr(module, "create_phase3_tool_registry", lambda **kwargs: fake_registry)
    report_path = tmp_path / "smoke.json"
    assert module.main(["--tool", "web_search", "--report", str(report_path)]) == 0
    report = report_path.read_text(encoding="utf-8")
    assert all(secret not in report for secret in ("serper-secret", "jina-secret", "serpapi-secret"))
