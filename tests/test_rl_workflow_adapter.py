"""Shared step semantics; real Pillow backend, no model/API in CPU tests."""

import inspect
import json

import pytest
from PIL import Image

from opensearch_vl_repro.agent.interaction import AgentInteraction
from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel
from opensearch_vl_repro.agent.runtime import AgentRuntime, AGENT_SYSTEM_GUIDANCE
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from opensearch_vl_repro.agent.tool_registry import RegisteredTool, ToolRegistry, ToolResult
from opensearch_vl_repro.rl.rollout_gate import create_gate_tool_registry
from opensearch_vl_repro.rl.workflow_adapter import RLWorkflowAdapter, build_rllm_workflow


def tool(name="crop", **arguments):
    if name == "crop" and not arguments:
        arguments = {"image": "img_1", "x": 1, "y": 1, "width": 4, "height": 3}
    return '<tool_call>' + json.dumps({"name": name, "arguments": arguments}) + '</tool_call>'


@pytest.fixture
def adapter():
    value = RLWorkflowAdapter(create_gate_tool_registry())
    value.initialize_episode(question="CPU fixture", images=[Image.new("RGB", (8, 6), "red")], sample_id="rl_000001")
    return value


def apply(adapter, text):
    parsed = adapter.handle_model_output(text)
    adapter.apply_tool_calls(parsed)
    return adapter.state()


def test_real_crop_registration_image_before_text_and_json_safe(adapter):
    state = apply(adapter, tool())
    entry = state.context.image_registry.get_entry("img_2")
    assert entry.parent_id == "img_1" and entry.metadata["producing_tool"] == "crop"
    assert entry.value.size == (4, 3) and isinstance(entry.value, Image.Image)
    observation = adapter.build_next_messages()[-1]
    assert observation["role"] == "tool"
    assert [part["type"] for part in observation["content"]] == ["image", "text"]
    assert observation["content"][0]["image"] is entry.value
    assert "New image ID: img_2" in observation["content"][1]["text"]
    apply(adapter, "A red detail is visible.")
    trajectory = adapter.finalize_episode(termination="env_done")
    json.dumps(trajectory.to_dict())
    assert trajectory.status == "success" and trajectory.image_ids == ["img_1", "img_2"]


def test_multiple_initial_images_use_next_available_id():
    adapter = RLWorkflowAdapter(create_gate_tool_registry())
    images = [Image.new("RGB", (8, 6)), Image.new("RGB", (10, 10))]
    adapter.initialize_episode(question="two", images=images, sample_id="rl_000001")
    assert "img_2: width=10, height=10" in adapter.build_next_messages()[0]["content"]
    apply(adapter, tool())
    assert adapter.state().context.image_registry.get_entry("img_3").parent_id == "img_1"


@pytest.mark.parametrize("reference", ["img_99", "https://example.com/a.png", "/tmp/a.png", "filename.jpg"])
def test_image_search_v3_invalid_reference_never_calls_backend(reference):
    calls = []
    registry = ToolRegistry()
    registry.register(RegisteredTool(TOOL_DECLARATIONS_BY_NAME["image_search"], lambda a, c: calls.append(a)))
    adapter = RLWorkflowAdapter(registry)
    adapter.initialize_episode(question="test", images=[Image.new("RGB", (4, 4))], sample_id="s")
    apply(adapter, tool("image_search", image_id=reference))
    assert not calls
    assert adapter.state().turns[0].error == "unknown_image_id"
    assert adapter.state().turns[0].metadata["provider_called"] is False


def test_image_search_v3_valid_id_and_legacy_url_rejected():
    calls = []
    def backend(arguments, context):
        calls.append(arguments)
        return ToolResult("success", "<observation>CPU schema fixture</observation>")
    registry = ToolRegistry()
    registry.register(RegisteredTool(TOOL_DECLARATIONS_BY_NAME["image_search"], backend))
    adapter = RLWorkflowAdapter(registry)
    adapter.initialize_episode(question="test", images=[Image.new("RGB", (4, 4))], sample_id="s")
    apply(adapter, tool("image_search", image_id="img_1"))
    assert calls == [{"image_id": "img_1"}]
    apply(adapter, tool("image_search", url="img_1"))
    assert len(calls) == 1 and adapter.state().turns[-1].status == "error"


def test_duplicate_blocked_without_new_image(adapter):
    apply(adapter, tool())
    apply(adapter, tool())
    assert adapter.state().turns[-1].error == "duplicate_tool_call"
    assert adapter.state().turns[-1].metadata["provider_called"] is False
    assert len(adapter.state().context.image_registry.list_images()) == 2


def test_direct_answer_zero_tool_still_legal(adapter):
    apply(adapter, "This is red.")
    trajectory = adapter.finalize_episode(termination="env_done")
    assert trajectory.status == "success" and trajectory.final_answer == "This is red." and not trajectory.turns
    with pytest.raises(RuntimeError, match="terminated"):
        apply(adapter, "Another answer")


def test_malformed_unknown_max_turns_and_model_failure_distinct(adapter):
    apply(adapter, '<tool_call>{"name":"crop",}</tool_call>')
    apply(adapter, tool("unknown_tool", a=1))
    assert [turn.error for turn in adapter.state().turns] == ["invalid_tool_call", "unknown_tool"]
    assert adapter.finalize_episode(termination="max_turns_exceeded").status == "max_agent_turns_exceeded"
    adapter.initialize_episode(question="again", images=[Image.new("RGB", (4, 4))], sample_id="s")
    adapter.record_failure(origin="model_generation_error", message="CPU simulated model exception")
    trajectory = adapter.finalize_episode(termination="error")
    assert trajectory.status == "model_error" and trajectory.metadata["error_origin"] == "model_generation_error"


def test_backend_exception_has_separate_error_origin(adapter):
    apply(adapter, tool("crop", image="img_1", x=1000, y=0, width=2, height=2))
    assert adapter.state().turns[0].status == "error"
    assert adapter.state().turns[0].metadata["error_origin"] == "tool_backend_error"
    assert adapter.state().status == "running"


def test_eval_and_adapter_share_helpers_and_crop_semantics(adapter):
    for name in ("_initial_messages", "_structured_assistant_message", "_execute_once", "_execute_call", "_commit_safely", "_commit_result", "_image_summaries"):
        assert name not in AgentRuntime.__dict__ and name not in RLWorkflowAdapter.__dict__
        assert name in AgentInteraction.__dict__
    original_image = adapter.state().context.image_registry.get("img_1")
    runtime_model = ScriptedAgentModel([tool(), "A red detail."])
    eval_trajectory = AgentRuntime(model=runtime_model, tool_registry=create_gate_tool_registry(), max_agent_turns=4).run(
        question="CPU fixture", images=[original_image], sample_id="rl_000001", benchmark="rl-gate")
    assert adapter.build_next_messages() == runtime_model.calls[0]["messages"]
    apply(adapter, tool()); apply(adapter, "A red detail.")
    ours = adapter.finalize_episode(termination="env_done")
    assert ours.images == eval_trajectory.images
    for a, b in zip(ours.turns, eval_trajectory.turns):
        assert a.tool_call == b.tool_call and a.observation == b.observation and a.metadata == b.metadata
    assert AGENT_SYSTEM_GUIDANCE in adapter.build_next_messages()[0]["content"]


def test_framework_binding_inherits_actual_upstream_loop_no_agent_runtime_run():
    source = inspect.getsource(build_rllm_workflow)
    assert "from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow" in source
    assert "class NoRewardWorkflow(MultiTurnWorkflow)" in source
    assert "workflow.run.__func__ is not MultiTurnWorkflow.run" in source
    assert "def run(" not in source and "AgentRuntime" not in source
    assert "super().postprocess_episode" not in source
    assert "backend.generate(" in source and "RolloutEngine" in source


def test_installed_real_rllm_loop_if_available(monkeypatch):
    """On AutoDL this exercises real rLLM; model outputs alone are CPU fixtures."""
    pytest.importorskip("rllm.workflows.multi_turn_workflow", reason="real rLLM is not installed locally")
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from rllm.workflows.multi_turn_workflow import MultiTurnWorkflow
    monkeypatch.setattr(AgentRuntime, "run", lambda *a, **k: pytest.fail("Eval loop must not be called"))
    for responses, expected in (([tool(), "A red crop."], "env_done"),
                                (["Direct answer"], "env_done"),
                                (['<tool_call>{broken}</tool_call>'] * 2, "max_turns_exceeded"),
                                ([RuntimeError("CPU fixture generation failed")], "error")):
        adapter = RLWorkflowAdapter(create_gate_tool_registry())
        class Backend:
            def __init__(self):
                self.outputs = iter(responses)
                self.calls = []
            def generate(self, *, messages, tools):
                self.calls.append(list(messages))
                response = next(self.outputs)
                if isinstance(response, Exception):
                    raise response
                return {"text": response, "prompt_ids": [1], "completion_ids": [2], "finish_reason": "stop"}
        backend = Backend()
        with ThreadPoolExecutor(max_workers=1) as executor:
            workflow, provenance = build_rllm_workflow(adapter=adapter, backend=backend, executor=executor, max_turns=2)
            assert workflow.run.__func__ is MultiTurnWorkflow.run
            episode = asyncio.run(workflow.run_with_termination_handling(
                task={"sample_id": "s", "question": "CPU fixture", "images": [Image.new("RGB", (8, 6), "red")]}, uid="cpu-unit"))
        assert episode.termination_reason.value == expected
        assert episode.info["reward_computed"] is False
        json.dumps(episode.to_dict())
        if responses[0] == tool():
            assert isinstance(backend.calls[1][-1]["content"][0]["image"], Image.Image)
            assert adapter.state().context.image_registry.exists("img_2")
        if expected == "error":
            assert adapter.finalize_episode(termination="error").status == "model_error"
        assert provenance["generation_loop"].endswith("MultiTurnWorkflow.run")
