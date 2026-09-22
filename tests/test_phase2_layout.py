from __future__ import annotations

import io
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from opensearch_vl_repro.agent.image_registry import ImageRegistry
from opensearch_vl_repro.agent.layout_parsing import (
    LayoutApiConfig, LayoutBackendError, PaddleOCRAiStudioBackend,
    layout_tool, load_layout_api_config,
)
from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel
from opensearch_vl_repro.agent.phase2_registry import create_phase2_tool_registry
from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_registry import ToolContext, ToolResult


def _config(max_poll=6.0):
    return LayoutApiConfig(
        "paddleocr_aistudio", "https://example.test/api/v2/ocr/jobs",
        "PaddleOCR-VL-1.6", "PADDLEOCR_ACCESS_TOKEN", 7.0, 2.0, max_poll,
    )


def _context():
    images = ImageRegistry()
    images.register_initial_image(Image.new("RGB", (5, 4), "white"))
    return ToolContext(images)


def _line(*pages):
    return json.dumps({"result": {"layoutParsingResults": list(pages)}})


def _structured():
    return {"prunedResult": {"parsing_res_list": [
        {"block_label": "title", "block_content": "Report", "block_bbox": [0, 0, 10, 2]},
        {"block_label": "paragraph", "block_content": "A short paragraph"},
        {"block_label": "table", "block_content": "| A | B |"},
        {"block_label": "figure", "block_content": "Chart 1"},
    ]}, "markdown": {"text": "should not replace structured data", "images": {"x": "url"}},
            "outputImages": {"x": "url"}}


class FakeResponse:
    def __init__(self, payload=None, *, status=200, text=""):
        self.payload = payload
        self.status_code = status
        self.text = text

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, *, submit=None, polls=None, download=None):
        self.submit = submit or FakeResponse({"code": 0, "data": {"jobId": "job-123"}})
        self.polls = list(polls or [FakeResponse({"code": 0, "data": {
            "state": "done", "resultUrl": {"jsonUrl": "https://result.test/out.jsonl"}}})])
        self.download = download or FakeResponse(text=_line(_structured()))
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        if isinstance(self.submit, Exception):
            raise self.submit
        return self.submit

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        result = self.polls.pop(0) if "/jobs/" in url else self.download
        if isinstance(result, Exception):
            raise result
        return result


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _tool(monkeypatch, session, *, clock=None, max_poll=6.0):
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "fake-token")
    clock = clock or FakeClock()
    backend = PaddleOCRAiStudioBackend(
        _config(max_poll), session=session, clock=clock.monotonic, sleep=clock.sleep,
    )
    return layout_tool(backend), clock


def test_submit_multipart_poll_download_and_observation(monkeypatch):
    session = FakeSession(polls=[
        FakeResponse({"data": {"state": "pending"}}),
        FakeResponse({"data": {"state": "running"}}),
        FakeResponse({"data": {"state": "done", "resultUrl": {
            "jsonUrl": "https://result.test/out.jsonl"}}}),
    ])
    tool, clock = _tool(monkeypatch, session)
    result = tool({"image": "img_1", "use_chart_recognition": True,
                   "use_doc_orientation_classify": False}, _context())
    assert result.status == "success" and result.error_type is None
    assert "[Title]\nReport" in result.observation
    assert "[Paragraph]\nA short paragraph" in result.observation
    assert "[Table]\n| A | B |" in result.observation
    assert "[Figure]\nChart 1" in result.observation
    assert "should not replace" not in result.observation
    assert "Content:" in result.observation and "prunedResult" not in result.observation
    assert "job-123" not in result.observation
    assert result.metadata["provider"] == "paddleocr_aistudio"
    assert result.metadata["job_id"] == "job-123"
    assert result.metadata["page_count"] == 1
    assert result.metadata["bounding_boxes"] == [[0, 0, 10, 2]]
    assert clock.sleeps == [2.0, 2.0]
    method, url, kwargs = session.calls[0]
    assert method == "post" and url == _config().job_url
    assert kwargs["headers"] == {"Authorization": "bearer fake-token"}
    assert kwargs["data"]["model"] == "PaddleOCR-VL-1.6"
    assert json.loads(kwargs["data"]["optionalPayload"]) == {
        "useDocUnwarping": False, "useChartRecognition": True,
        "useDocOrientationClassify": False,
    }
    filename, payload, mime = kwargs["files"]["file"]
    assert filename == "image.png" and mime == "image/png"
    with Image.open(io.BytesIO(payload)) as image:
        assert image.size == (5, 4)
    assert kwargs["timeout"] == 7.0
    assert [call[1] for call in session.calls[1:]] == [
        _config().job_url + "/job-123"] * 3 + ["https://result.test/out.jsonl"]
    assert "headers" not in session.calls[-1][2]  # signed result URL: no bearer token
    assert "fake-token" not in result.observation + json.dumps(result.metadata)


def test_optional_args_omitted_by_default(monkeypatch):
    session = FakeSession()
    tool, _ = _tool(monkeypatch, session)
    assert tool({"image": "img_1"}, _context()).status == "success"
    payload = json.loads(session.calls[0][2]["data"]["optionalPayload"])
    assert payload == {"useDocUnwarping": False}


@pytest.mark.parametrize("submit,expected", [
    (FakeResponse(status=401), "authentication_error"),
    (FakeResponse(status=403), "authentication_error"),
    (FakeResponse(status=429), "quota_error"),
    (FakeResponse(status=500), "provider_error"),
    (requests.ConnectionError("fake-token secret"), "network_error"),
    (requests.Timeout("fake-token secret"), "timeout"),
])
def test_http_and_transport_failure_categories(monkeypatch, submit, expected):
    tool, _ = _tool(monkeypatch, FakeSession(submit=submit))
    result = tool({"image": "img_1"}, _context())
    assert result.status == "error" and result.error_type == expected
    assert "fake-token" not in result.observation + json.dumps(result.metadata)


def test_direct_backend_exception_does_not_echo_token(monkeypatch):
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "fake-token")
    backend = PaddleOCRAiStudioBackend(
        _config(), session=FakeSession(submit=requests.ConnectionError("fake-token secret")))
    with pytest.raises(LayoutBackendError) as caught:
        backend.parse(Image.new("RGB", (5, 4)), use_chart_recognition=None,
                      use_doc_orientation_classify=None)
    assert caught.value.error_type == "network_error"
    assert "fake-token" not in str(caught.value)
    assert caught.value.__suppress_context__ is True


@pytest.mark.parametrize("submit,polls,expected", [
    (FakeResponse({"data": {}}), None, "invalid_response"),
    (FakeResponse(ValueError("bad JSON")), None, "invalid_response"),
    (None, [FakeResponse({"data": {"state": "done"}})], "invalid_response"),
    (None, [FakeResponse({"data": {"state": "unknown"}})], "invalid_response"),
    (None, [FakeResponse({"data": {"state": "failed", "errorMsg": "fake-token bad input"}})], "provider_error"),
])
def test_submit_and_poll_response_validation(monkeypatch, submit, polls, expected):
    tool, _ = _tool(monkeypatch, FakeSession(submit=submit, polls=polls))
    result = tool({"image": "img_1"}, _context())
    assert result.error_type == expected
    assert "fake-token" not in result.observation + json.dumps(result.metadata)


def test_polling_deadline_uses_monotonic_clock(monkeypatch):
    session = FakeSession(polls=[FakeResponse({"data": {"state": "pending"}}) for _ in range(4)])
    clock = FakeClock()
    tool, _ = _tool(monkeypatch, session, clock=clock, max_poll=3.0)
    result = tool({"image": "img_1"}, _context())
    assert result.error_type == "timeout"
    assert clock.sleeps == [2.0, 1.0]
    assert len(session.polls) == 2  # no query after the deadline


def test_jsonl_multipage_structured_then_markdown_fallback(monkeypatch):
    second_page = {"prunedResult": {"parsing_res_list": [
        {"block_label": "text", "block_content": " "}]},
        "markdown": {"text": "Fallback page", "images": {"x": "should-not-download"}},
        "outputImages": {"x": "should-not-download"}}
    body = _line(_structured(), second_page) + "\n\n" + _line({"markdown": {"text": "Third page"}})
    session = FakeSession(download=FakeResponse(text=body))
    tool, _ = _tool(monkeypatch, session)
    result = tool({"image": "img_1"}, _context())
    assert result.status == "success"
    assert result.metadata["page_count"] == 3
    assert result.metadata["block_count"] == 6
    assert result.observation.index("Report") < result.observation.index("Fallback page")
    assert result.observation.index("Fallback page") < result.observation.index("Third page")
    assert "should-not-download" not in result.observation
    assert len(session.calls) == 3  # submit, one poll, one JSONL GET


@pytest.mark.parametrize("body", [
    "", "not json", json.dumps({"result": {}}), _line({}),
    _line({"markdown": {"text": ""}}),
])
def test_invalid_or_empty_jsonl(monkeypatch, body):
    tool, _ = _tool(monkeypatch, FakeSession(download=FakeResponse(text=body)))
    result = tool({"image": "img_1"}, _context())
    assert result.error_type == "invalid_response"


def test_missing_token_and_agent_failure_are_safe(monkeypatch):
    monkeypatch.delenv("PADDLEOCR_ACCESS_TOKEN", raising=False)
    session = FakeSession()
    backend = PaddleOCRAiStudioBackend(_config(), session=session)
    result = layout_tool(backend)({"image": "img_1"}, _context())
    assert result.error_type == "configuration_error" and not session.calls
    unconfigured = create_phase2_tool_registry().execute("layout_parsing", {"image": "img_1"}, _context())
    assert unconfigured.error_type == "configuration_error"
    model = ScriptedAgentModel(['layout_parsing({"image":"img_1"})', "Recovered"])
    trajectory = AgentRuntime(model=model, tool_registry=create_phase2_tool_registry(),
                              max_agent_turns=2).run(question="read", images=[Image.new("RGB", (5, 4))])
    assert trajectory.status == "success"
    assert trajectory.turns[0].error == "configuration_error"


def test_provider_echoed_token_is_redacted_from_output(monkeypatch):
    session = FakeSession(
        submit=FakeResponse({"data": {"jobId": "fake-token"}}),
        download=FakeResponse(text=_line({"markdown": {"text": "OCR fake-token text"}})),
    )
    tool, _ = _tool(monkeypatch, session)
    result = tool({"image": "img_1"}, _context())
    assert result.status == "success"
    assert "fake-token" not in result.observation + json.dumps(result.metadata)
    assert "[REDACTED]" in result.observation


def test_smoke_report_redacts_token_without_network(monkeypatch, tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_layout_parsing_smoke.py"
    spec = importlib.util.spec_from_file_location("layout_smoke_test_module", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "fake-token")
    monkeypatch.setattr(module, "read_eval_sample", lambda path, index: SimpleNamespace(
        sample_id="sample-1", images=[Image.new("RGB", (5, 4))]))
    fake_registry = SimpleNamespace(execute=lambda *args: ToolResult(
        status="success", observation="<observation>\nContent:\n\n[Text]\nfake-token\n</observation>",
        metadata={"job_id": "fake-token"}))
    monkeypatch.setattr(module, "create_phase2_tool_registry", lambda **kwargs: fake_registry)
    report_path = tmp_path / "layout-report.json"
    assert module.main(["--report", str(report_path)]) == 0
    report_text = report_path.read_text(encoding="utf-8")
    assert "fake-token" not in report_text
    assert "[REDACTED]" in report_text


def test_config_and_registry_select_ai_studio():
    path = Path(__file__).resolve().parents[1] / "configs" / "layout_parsing.example.yaml"
    config = load_layout_api_config(path)
    assert config.provider == "paddleocr_aistudio"
    assert config.model == "PaddleOCR-VL-1.6"
    assert config.access_token_env == "PADDLEOCR_ACCESS_TOKEN"
    assert config.max_poll_seconds == 120
    registry = create_phase2_tool_registry(layout_config=path)
    assert len(registry.list_tools()) == 8
    assert registry.get("layout_parsing").backend is not None
