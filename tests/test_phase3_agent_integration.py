from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel
from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry
from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_registry import ToolResult


ROOT = Path(__file__).resolve().parents[1]


def _script():
    path = ROOT / "scripts" / "run_4b_agent_smoke.py"
    spec = importlib.util.spec_from_file_location("phase3_agent_smoke_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSearchTools:
    def text_search(self, arguments, context):
        assert arguments == {"q": "Qwen3-VL technical report", "hl": "en", "top_k": 3}
        return ToolResult(
            status="success",
            observation="<observation>\nSearch Results:\n[1]\nTitle: Evidence\nPassage: page text\n</observation>",
            metadata={"providers": ["serper", "jina_reader"], "reader_success_count": 1},
        )

    def image_search(self, arguments, context):
        assert arguments == {"url": "img_1"}
        assert isinstance(context.image_registry.get("img_1"), Image.Image)
        return ToolResult(
            status="success",
            observation="<observation>\nImage Search Results:\nNo visual matches found.\n</observation>",
            metadata={"provider": "serpapi_google_lens", "provider_image_id": "provider-123",
                      "result_count": 0},
        )

    def web_search(self, arguments, context):
        raise AssertionError("unexpected tool")


@pytest.mark.parametrize("tool,first_output", [
    ("text_search", 'text_search({"q":"Qwen3-VL technical report","hl":"en","top_k":3})'),
    ("image_search", 'image_search({"url":"img_1"})'),
])
def test_offline_agent_tool_observation_reenters_next_model_turn(tool, first_output):
    module = _script()
    base = ScriptedAgentModel([first_output, "Final answer grounded in the tool observation."])
    observed = module.ObservedAgentModel(base)
    trajectory = AgentRuntime(
        model=observed,
        tool_registry=create_phase3_tool_registry(search_tools=FakeSearchTools()),
        max_agent_turns=2,
    ).run(question="synthetic protocol test", images=[Image.new("RGB", (10, 10))])
    assert trajectory.status == "success"
    assert trajectory.turns[0].tool_call["name"] == tool
    assert trajectory.turns[0].status == "success"
    tool_message = base.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert trajectory.turns[0].observation == tool_message["content"]
    evidence = module.phase3_evidence(trajectory, tool, observed)
    assert evidence["phase3_tool_chain_passed"] is True
    assert evidence["observation_fed_to_next_qwen_turn"] is True
    assert evidence["generate_call_count"] == 2
    assert evidence["final_answer_present"] is True
    assert evidence["tool_call_count"] == 1


def test_text_integration_requires_successful_reader_passage():
    module = _script()

    class SnippetOnlySearchTools(FakeSearchTools):
        def text_search(self, arguments, context):
            result = super().text_search(arguments, context)
            result.metadata["reader_success_count"] = 0
            result.metadata["reader_fallback_used"] = True
            return result

    observed = module.ObservedAgentModel(ScriptedAgentModel([
        'text_search({"q":"Qwen3-VL technical report","hl":"en","top_k":3})',
        "Final answer from snippets.",
    ]))
    trajectory = AgentRuntime(
        model=observed,
        tool_registry=create_phase3_tool_registry(search_tools=SnippetOnlySearchTools()),
        max_agent_turns=2,
    ).run(question="synthetic protocol test", images=[Image.new("RGB", (10, 10))])
    assert trajectory.status == "success"
    assert trajectory.turns[0].status == "success"
    assert module.phase3_evidence(trajectory, "text_search", observed)["phase3_tool_chain_passed"] is False


class FakeMemory:
    def reset_peak_memory_stats(self):
        return None

    def peak_memory_mb(self):
        return 64


@pytest.mark.parametrize("tool,first_output", [
    ("text_search", 'text_search({"q":"Qwen3-VL technical report","hl":"en","top_k":3})'),
    ("image_search", 'image_search({"url":"img_1"})'),
])
def test_phase3_cli_report_requires_real_tool_evidence_without_cuda(
    tool, first_output, tmp_path, monkeypatch,
):
    module = _script()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(module.CudaSmokeContext, "initialize", lambda torch, device: FakeMemory())
    monkeypatch.setattr(module, "load_inference_bundle", lambda config: SimpleNamespace(
        environment={"device": "cuda:0", "dtype": "bfloat16"},
        model=SimpleNamespace(training=False),
    ))
    base = ScriptedAgentModel([first_output, "Final answer."])
    monkeypatch.setattr(module, "QwenAgentModel", lambda bundle: base)
    calls = []

    def registry(**kwargs):
        calls.append(kwargs)
        return create_phase3_tool_registry(search_tools=FakeSearchTools())

    monkeypatch.setattr(module, "create_phase3_tool_registry", registry)
    monkeypatch.setattr(module, "read_eval_sample", lambda *args: pytest.fail("Phase 3 should use a synthetic image"))
    monkeypatch.setenv("SERPER_API_KEY", "secret-value")
    report_path = tmp_path / "phase3.json"
    result = module.main([
        "--phase3-search-tools", "--phase3-tool", tool,
        "--report", str(report_path),
    ])
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert result == 0 and report["passed"] is True
    assert report["requested_phase3_tool"] == tool
    assert report["tool_call_name"] == tool and report["tool_call_status"] == "success"
    assert report["observation_fed_to_next_qwen_turn"] is True
    assert report["final_answer_present"] is True
    assert report["provider_metadata"]
    assert report["real_search_backend"] == (
        ["serper", "jina_reader"] if tool == "text_search" else "serpapi_google_lens"
    )
    assert report["peak_vram_mb"] == 64
    assert report["trajectory"]["status"] == "success"
    assert "secret-value" not in report_text
    assert calls and calls[0]["search_config"].name == "search_backends.example.yaml"
    assert calls[0]["layout_config"].name == "layout_parsing.example.yaml"


def test_phase3_cli_does_not_pass_on_direct_final_answer(tmp_path, monkeypatch):
    module = _script()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(module.CudaSmokeContext, "initialize", lambda torch, device: FakeMemory())
    monkeypatch.setattr(module, "load_inference_bundle", lambda config: SimpleNamespace(
        environment={}, model=SimpleNamespace(training=False)))
    monkeypatch.setattr(module, "QwenAgentModel", lambda bundle: ScriptedAgentModel(["Final without tool call."]))
    monkeypatch.setattr(module, "create_phase3_tool_registry", lambda **kwargs:
                        create_phase3_tool_registry(search_tools=FakeSearchTools()))
    report_path = tmp_path / "failure.json"
    assert module.main(["--phase3-search-tools", "--phase3-tool", "text_search",
                        "--report", str(report_path)]) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["trajectory"]["status"] == "success"
    assert report["passed"] is False
    assert report["phase3_tool_chain_passed"] is False
    assert report["tool_call_count"] == 0


def test_phase3_cli_mode_flags_are_explicit_and_exclusive():
    module = _script()
    for args in (
        ["--phase3-search-tools"],
        ["--phase3-tool", "text_search"],
        ["--phase3-search-tools", "--phase3-tool", "image_search", "--local-visual-tools"],
        ["--phase3-search-tools", "--phase3-tool", "text_search", "--synthetic-tool-prompt"],
    ):
        with pytest.raises(SystemExit) as caught:
            module.main(args)
        assert caught.value.code == 2


def test_agent_report_redacts_all_api_credentials(tmp_path, monkeypatch):
    module = _script()
    values = {
        "SERPER_API_KEY": "serper-secret",
        "JINA_API_KEY": "jina-secret",
        "SERPAPI_API_KEY": "serpapi-secret",
        "PADDLEOCR_ACCESS_TOKEN": "layout-secret",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    path = tmp_path / "secret-report.json"
    module.write_report(path, {
        "trajectory": {"observation": " ".join(values.values())},
        "provider_metadata": {"echo": list(values.values()), "serper-secret": "key echo"},
        "error": {"traceback": "Bearer layout-secret and serpapi-secret"},
    })
    output = path.read_text(encoding="utf-8")
    assert all(secret not in output for secret in values.values())
    assert "[REDACTED]" in output
