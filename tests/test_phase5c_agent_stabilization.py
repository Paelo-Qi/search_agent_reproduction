from __future__ import annotations

import json

from PIL import Image

from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel, create_mock_tool_registry
from opensearch_vl_repro.agent.reliability import FileSystemToolCache, cached_tool_backend
from opensearch_vl_repro.agent.runtime import AgentRuntime, AgentTrajectory, AgentTurn
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from opensearch_vl_repro.agent.tool_registry import RegisteredTool, ToolRegistry, ToolResult
from opensearch_vl_repro.evaluation import BatchRunner, BatchSample, create_run_manifest
from opensearch_vl_repro.evaluation.run_manifest import manifest_mismatches


def _registry(backends):
    registry = ToolRegistry()
    for name, backend in backends.items():
        registry.register(RegisteredTool(TOOL_DECLARATIONS_BY_NAME[name], backend))
    return registry


def test_registered_image_ids_and_stop_guidance_are_agent_visible_without_answer():
    model = ScriptedAgentModel(["final"])
    AgentRuntime(model=model, tool_registry=create_mock_tool_registry(), max_agent_turns=8).run(
        question="question only", images=[Image.new("RGB", (2, 2))])
    serialized = json.dumps(model.calls[0]["messages"], default=str)
    assert "Registered input images" in serialized and "img_1" in serialized
    assert "Do not repeat an identical tool call" in serialized
    assert "provide the final answer" in serialized
    assert "SECRET_REFERENCE_ANSWER" not in serialized


def test_all_image_tool_descriptions_require_registered_ids_not_external_references():
    for name in ("image_search", "crop", "layout_parsing", "super_resolution",
                 "sharpen", "perspective_correct"):
        description = TOOL_DECLARATIONS_BY_NAME[name].description.lower()
        assert "registered runtime image id" in description and "img_1" in description
        assert "filename" in description and "filesystem path" in description
        assert "http url" in description


def test_unknown_image_reference_returns_actionable_recovery_without_backend_call():
    calls = 0
    def backend(arguments, context):
        nonlocal calls
        calls += 1
        raise AssertionError("invalid image reference reached backend")
    model = ScriptedAgentModel([
        'sharpen({"image":"dataset_file.jpg","amount":1})', "recovered final",
    ])
    trajectory = AgentRuntime(model=model, tool_registry=_registry({"sharpen": backend}),
                              max_agent_turns=8).run(
        question="q", images=[Image.new("RGB", (2, 2))])
    turn = trajectory.turns[0]
    assert calls == 0 and turn.error == "unknown_image_id"
    assert "Unknown registered image ID" in turn.observation
    assert "Available registered image IDs: img_1" in turn.observation
    assert "filenames" in turn.observation and "HTTP URLs" in turn.observation
    assert turn.metadata["provider_called"] is False


def test_exact_duplicate_is_blocked_episode_wide_and_key_order_independent():
    calls = 0
    def backend(arguments, context):
        nonlocal calls
        calls += 1
        return ToolResult("success", "<observation>evidence</observation>")
    model = ScriptedAgentModel([
        'web_search({"q":"same query","hl":"en"})',
        'web_search({"hl":"en","q":"same query"})',
        "final",
    ])
    trajectory = AgentRuntime(model=model, tool_registry=_registry({"web_search": backend}),
                              max_agent_turns=8).run(
        question="q", images=[Image.new("RGB", (2, 2))])
    assert calls == 1 and len(trajectory.turns) == 2
    assert trajectory.turns[1].error == "duplicate_tool_call"
    assert trajectory.turns[1].metadata == {
        "duplicate_tool_call": True, "original_tool_call_turn": 1,
        "provider_called": False, "derived_image_ids": [],
    }


def test_a_b_a_loop_is_blocked_but_different_arguments_are_allowed():
    calls = {"image_search": 0, "layout_parsing": 0}
    def image_backend(arguments, context):
        calls["image_search"] += 1
        return ToolResult("success", "<observation>image evidence</observation>")
    def layout_backend(arguments, context):
        calls["layout_parsing"] += 1
        return ToolResult("success", "<observation>layout evidence</observation>")
    model = ScriptedAgentModel([
        'image_search({"url":"img_1"})',
        'layout_parsing({"image":"img_1"})',
        'image_search({"url":"img_1"})',
        "final",
    ])
    trajectory = AgentRuntime(model=model, tool_registry=_registry({
        "image_search": image_backend, "layout_parsing": layout_backend,
    }), max_agent_turns=8).run(question="q", images=[Image.new("RGB", (2, 2))])
    assert calls == {"image_search": 1, "layout_parsing": 1}
    assert trajectory.turns[2].error == "duplicate_tool_call"

    queries = []
    def web_backend(arguments, context):
        queries.append(arguments["q"])
        return ToolResult("success", "<observation>ok</observation>")
    model = ScriptedAgentModel([
        'web_search({"q":"foo"})', 'web_search({"q":"foo official"})', "final",
    ])
    AgentRuntime(model=model, tool_registry=_registry({"web_search": web_backend}),
                 max_agent_turns=8).run(question="q", images=[Image.new("RGB", (2, 2))])
    assert queries == ["foo", "foo official"]


def test_duplicate_bypasses_shared_cache_and_real_execution_count(tmp_path):
    provider_calls = 0
    def provider(arguments, context):
        nonlocal provider_calls
        provider_calls += 1
        return ToolResult("success", "<observation>provider evidence</observation>")
    cached = cached_tool_backend(
        tool="web_search", backend=provider, cache=FileSystemToolCache(tmp_path / "cache"),
        behavior_version="test-v1",
    )
    model = ScriptedAgentModel([
        'web_search({"q":"same"})', 'web_search({"q":"same"})', "final",
    ])
    trajectory = AgentRuntime(model=model, tool_registry=_registry({"web_search": cached}),
                              max_agent_turns=8).run(
        question="q", images=[Image.new("RGB", (2, 2))])
    record = BatchRunner._record(
        BatchSample("S", "synthetic", "q", []), trajectory, 1.0)
    summary = BatchRunner._summary(
        {"samples": {"S": {"status": "success"}}}, {"S": record})
    assert provider_calls == 1
    assert trajectory.turns[1].metadata["provider_called"] is False
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 1
    assert summary["cache_misses"] == summary["real_tool_executions"] == 1


class ProgressRuntime:
    def __init__(self, failed=()): self.calls, self.failed = [], set(failed)
    def run(self, *, question, images, sample_id, benchmark):
        self.calls.append(sample_id)
        failed = sample_id in self.failed
        turn = AgentTurn("web_search({})", {"name": "web_search", "arguments": {}},
                         "obs", "error" if failed else "success",
                         error="synthetic_error" if failed else None)
        return AgentTrajectory(sample_id, benchmark, [turn], None if failed else "answer",
            "max_agent_turns_exceeded" if failed else "success", ["img_1"],
            error="failed" if failed else None)


def _manifest(count):
    return create_run_manifest(
        run_id="progress-test", model_name_or_path="fake", model_revision="r",
        inference_config_fingerprint="i", dataset_path="fake.parquet",
        dataset_identity={"sha256": "x"}, start=0, limit=count, max_agent_turns=8,
        search_config_fingerprint="s", layout_config_fingerprint="l",
        checkpoint={"kind": "fake"}, tool_fingerprint="tools",
    )


def test_agent_behavior_version_prevents_resume_of_old_manifest():
    current = _manifest(1)
    old = dict(current)
    old.pop("agent_behavior_version")
    assert "agent_behavior_version" in manifest_mismatches(old, current)


def test_batch_progress_start_done_and_failure_fields(tmp_path, capsys):
    samples = [BatchSample(name, "bench", f"q{name}", []) for name in "AB"]
    BatchRunner(ProgressRuntime(failed={"B"}), tmp_path / "run",
                run_manifest=_manifest(2)).run(samples)
    output = capsys.readouterr().out
    for marker in ("[1/2] START", "[1/2] DONE", "[2/2] START", "[2/2] DONE"):
        assert marker in output
    assert "status=success tools=1 elapsed=" in output
    assert "status=failed tools=1 elapsed=" in output
    assert "error_type=max_agent_turns_exceeded" in output


def test_resume_progress_uses_invocation_eligible_denominator(tmp_path, capsys):
    samples = [BatchSample(name, "bench", f"q{name}", []) for name in "ABC"]
    runtime = ProgressRuntime()
    runner = BatchRunner(runtime, tmp_path / "run", run_manifest=_manifest(3))
    runner.run(samples, max_samples=2)
    capsys.readouterr()
    runner.run(samples)
    output = capsys.readouterr().out
    assert "Total samples: 3" in output
    assert "Eligible this invocation: 1" in output
    assert "Already successful: 2" in output
    assert "[1/1] START" in output and "[1/1] DONE" in output
    assert "[1/3]" not in output
