from __future__ import annotations

import base64
import io
import socket
import urllib.error

import pytest
from PIL import Image

from opensearch_vl_repro.agent.image_registry import ImageRegistry
from opensearch_vl_repro.agent.layout_parsing import (
    BaiduLayoutParsingBackend, LayoutApiConfig, layout_tool, load_layout_api_config,
)
from opensearch_vl_repro.agent.phase2_registry import create_phase2_tool_registry
from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel
from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_registry import ToolContext


def _config():
    return LayoutApiConfig("baidu", "https://example.test/ocr", "test-model", "TEST_LAYOUT_API_KEY", 7.0)


def _context():
    images = ImageRegistry()
    images.register_initial_image(Image.new("RGB", (5, 4), "white"))
    return ToolContext(images)


def _response():
    return {"id": "request-1", "result": {"layoutParsingResults": [
        {"prunedResult": {"parsing_res_list": [
            {"block_label": "title", "block_content": "Report", "block_bbox": [0, 0, 10, 2]},
            {"block_label": "paragraph", "block_content": "A short paragraph"},
            {"block_label": "table", "block_content": "| A | B |"},
            {"block_label": "figure", "block_content": "Chart 1"},
        ]}}
    ]}}


def test_baidu_request_response_and_optional_flags(monkeypatch):
    monkeypatch.setenv("TEST_LAYOUT_API_KEY", "test-token")
    captured = []

    def transport(endpoint, headers, body, timeout):
        captured.append((endpoint, headers, body, timeout))
        return _response()

    tool = layout_tool(BaiduLayoutParsingBackend(_config(), transport=transport))
    result = tool({"image": "img_1", "use_chart_recognition": True,
                   "use_doc_orientation_classify": False}, _context())
    assert result.status == "success"
    assert "[Title]\nReport" in result.observation
    assert "[Paragraph]\nA short paragraph" in result.observation
    assert "[Table]\n| A | B |" in result.observation
    assert "[Figure]\nChart 1" in result.observation
    assert "prunedResult" not in result.observation
    assert result.metadata["bounding_boxes"] == [[0, 0, 10, 2]]
    endpoint, headers, body, timeout = captured[0]
    assert endpoint == _config().endpoint and timeout == 7.0
    assert headers["Authorization"] == "Bearer test-token"
    assert body["model"] == "test-model" and body["fileType"] == 1
    assert body["useChartRecognition"] is True
    assert body["useDocOrientationClassify"] is False
    with Image.open(io.BytesIO(base64.b64decode(body["file"]))) as image:
        assert image.size == (5, 4)
    assert "test-token" not in result.observation
    tool({"image": "img_1"}, _context())
    assert "useChartRecognition" not in captured[1][2]


@pytest.mark.parametrize("failure, expected", [
    (urllib.error.HTTPError("https://example.test", 401, "unauthorized", {}, None), "authentication_error"),
    (urllib.error.HTTPError("https://example.test", 429, "quota", {}, None), "quota_error"),
    (socket.timeout(), "timeout"),
    (urllib.error.URLError("offline"), "network_error"),
])
def test_layout_transport_failure_mapping(monkeypatch, failure, expected):
    monkeypatch.setenv("TEST_LAYOUT_API_KEY", "test-token")

    def transport(*args):
        raise failure

    result = layout_tool(BaiduLayoutParsingBackend(_config(), transport=transport))(
        {"image": "img_1"}, _context())
    assert result.status == "error" and result.error_type == expected
    assert expected in result.observation
    assert "test-token" not in result.observation


def test_layout_configuration_and_invalid_response(monkeypatch):
    monkeypatch.delenv("TEST_LAYOUT_API_KEY", raising=False)
    backend = BaiduLayoutParsingBackend(_config(), transport=lambda *args: _response())
    missing = layout_tool(backend)({"image": "img_1"}, _context())
    assert missing.error_type == "configuration_error"
    unconfigured = create_phase2_tool_registry().execute("layout_parsing", {"image": "img_1"}, _context())
    assert unconfigured.error_type == "configuration_error"
    monkeypatch.setenv("TEST_LAYOUT_API_KEY", "test-token")
    bad = layout_tool(BaiduLayoutParsingBackend(_config(), transport=lambda *args: {"result": {}}))(
        {"image": "img_1"}, _context())
    assert bad.error_type == "invalid_response"

    model = ScriptedAgentModel(['layout_parsing({"image":"img_1"})', "Recovered"])
    trajectory = AgentRuntime(model=model, tool_registry=create_phase2_tool_registry(),
                              max_agent_turns=2).run(question="read", images=[Image.new("RGB", (5, 4))])
    assert trajectory.status == "success"
    assert trajectory.turns[0].status == "error"
    assert trajectory.turns[0].error == "configuration_error"
    assert "configuration_error" in model.calls[1]["messages"][-1]["content"]


def test_layout_example_config_loads():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = load_layout_api_config(root / "configs" / "layout_parsing.example.yaml")
    assert config.provider == "baidu"
    assert config.timeout_seconds > 0
