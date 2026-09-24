from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from opensearch_vl_repro.agent.image_registry import ImageRegistry
from opensearch_vl_repro.agent.mock_tools import (
    ScriptedAgentModel,
    create_mock_tool_registry,
)
from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_contracts import (
    TOOL_DECLARATIONS,
    TOOL_DECLARATIONS_BY_NAME,
)
from opensearch_vl_repro.agent.tool_parser import ToolCallParser
from opensearch_vl_repro.agent.tool_registry import ToolContext
from opensearch_vl_repro.agent.tool_registry import RegisteredTool, ToolRegistry


EXPECTED_TOOLS = {
    "text_search",
    "image_search",
    "crop",
    "layout_parsing",
    "super_resolution",
    "sharpen",
    "web_search",
    "perspective_correct",
}


def test_tool_declarations_match_the_sft_audit() -> None:
    assert {tool.name for tool in TOOL_DECLARATIONS} == EXPECTED_TOOLS
    assert "visit" not in TOOL_DECLARATIONS_BY_NAME
    assert "python_interpreter" not in TOOL_DECLARATIONS_BY_NAME
    assert TOOL_DECLARATIONS_BY_NAME["image_search"].parameters == {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    }
    assert set(TOOL_DECLARATIONS_BY_NAME["crop"].parameters["required"]) == {
        "image",
        "x",
        "y",
        "width",
        "height",
    }
    layout = TOOL_DECLARATIONS_BY_NAME["layout_parsing"].parameters
    assert layout["required"] == ["image"]
    assert layout["properties"]["use_chart_recognition"]["type"] == "boolean"
    assert layout["properties"]["use_doc_orientation_classify"]["type"] == "boolean"


def test_tool_schema_variants_and_validation() -> None:
    text_search = TOOL_DECLARATIONS_BY_NAME["text_search"]
    for arguments in (
        {"q": "entity", "hl": "en", "top_k": 5},
        {"q": "entity", "top_k": 5},
        {"q": "entity", "hl": "en"},
    ):
        assert text_search.validate_arguments(arguments) == arguments
    layout = TOOL_DECLARATIONS_BY_NAME["layout_parsing"]
    assert layout.validate_arguments({"image": "img_1"})
    assert layout.validate_arguments({"image": "img_1", "use_chart_recognition": True})
    assert layout.validate_arguments(
        {"image": "img_1", "use_doc_orientation_classify": False}
    )
    with pytest.raises(ValueError, match="missing required"):
        text_search.validate_arguments({"hl": "en"})
    with pytest.raises(ValueError, match="must be number"):
        text_search.validate_arguments({"q": "entity", "top_k": True})
    with pytest.raises(ValueError, match="unexpected arguments"):
        TOOL_DECLARATIONS_BY_NAME["image_search"].validate_arguments(
            {"image": "img_1", "url": "img_1"}
        )


def test_image_registry_monotonic_lookup_duplicate_and_isolation(tmp_path: Path) -> None:
    image = Image.new("RGB", (4, 3), color=(10, 20, 30))
    image_path = tmp_path / "local.png"
    image.save(image_path)
    registry = ImageRegistry()
    assert registry.register_initial_image(image) == "img_1"
    assert registry.register_derived_image(image_path, parent_id="img_1") == "img_2"
    assert registry.exists("img_1")
    assert registry.get("img_1") is image
    assert registry.get("img_2") == image_path.resolve()
    assert [entry.image_id for entry in registry.list_images()] == ["img_1", "img_2"]
    with pytest.raises(ValueError, match="already exists"):
        registry.register_derived_image(image, parent_id="img_1", image_id="img_2")
    with pytest.raises(KeyError, match="unknown image ID"):
        registry.get("img_99")
    with pytest.raises(KeyError, match="unknown parent"):
        registry.register_derived_image(image, parent_id="img_99")

    independent = ImageRegistry()
    assert independent.register_initial_image(image) == "img_1"
    assert not independent.exists("img_2")


def test_tool_call_parser_variants_and_failure_categories() -> None:
    parser = ToolCallParser(EXPECTED_TOOLS)
    xml = parser.parse(
        '<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}</tool_call>'
    )
    assert xml.kind == "valid_tool_call"
    assert xml.tool_calls[0].name == "image_search"
    assert xml.tool_calls[0].arguments == {"url": "img_1"}

    function_style = parser.parse('text_search({"q":"entity","top_k":5})')
    assert function_style.kind == "valid_tool_call"
    assert function_style.tool_calls[0].name == "text_search"

    nested = parser.parse(
        '<tool_call>{"function":{"name":"web_search",'
        '"arguments":"{\\"q\\":\\"entity\\"}"}}</tool_call>'
    )
    assert nested.tool_calls[0].arguments == {"q": "entity"}
    assert parser.parse("This is the final answer.").kind == "final_answer"
    assert parser.parse('<tool_call>{"name":"crop",}</tool_call>').kind == "malformed_tool_call"
    assert parser.parse('<tool_call>{"name":"crop"}</tool_call>').kind == "malformed_tool_call"
    unknown = parser.parse('<tool_call>{"name":"visit","arguments":{}}</tool_call>')
    assert unknown.kind == "unknown_tool"
    assert unknown.tool_calls[0].name == "visit"


def test_all_mock_tools_execute_through_registry_with_plain_text_observations() -> None:
    registry = create_mock_tool_registry()
    assert set(registry.list_tools()) == EXPECTED_TOOLS
    images = ImageRegistry()
    images.register_initial_image(Image.new("RGB", (8, 8), color="navy"))
    context = ToolContext(images, sample_id="fixture", benchmark="synthetic")
    arguments = {
        "text_search": {"q": "entity", "hl": "en", "top_k": 5},
        "image_search": {"url": "img_1"},
        "crop": {"image": "img_1", "x": 0, "y": 0, "width": 4, "height": 4},
        "layout_parsing": {"image": "img_1"},
        "super_resolution": {"image": "img_1", "scale": 2},
        "sharpen": {"image": "img_1", "amount": 1.5},
        "web_search": {"q": "entity"},
        "perspective_correct": {"image": "img_1"},
    }
    for name in registry.list_tools():
        result = registry.execute(name, arguments[name], context)
        assert result.status == "success"
        assert isinstance(result.observation, str)
        assert "<observation>" in result.observation
        if name in {"crop", "super_resolution", "sharpen", "perspective_correct"}:
            assert len(result.derived_images) == 1
            assert result.derived_images[0].parent_id == "img_1"
        else:
            assert not result.derived_images
    # Backend execution produces an image; only AgentRuntime assigns img_n IDs.
    assert [entry.image_id for entry in images.list_images()] == ["img_1"]


def test_mock_agent_loop_runs_model_tool_observation_and_final_answer() -> None:
    model = ScriptedAgentModel(
        [
            'image_search({"url":"img_1"})',
            '<tool_call>{"name":"text_search","arguments":'
            '{"q":"example entity","hl":"en","top_k":5}}</tool_call>',
            "Final answer from the scripted model.",
        ]
    )
    runtime = AgentRuntime(
        model=model, tool_registry=create_mock_tool_registry(), max_agent_turns=4
    )
    trajectory = runtime.run(
        question="What is shown?",
        images=[Image.new("RGB", (8, 8), color="green")],
        sample_id="sample-1",
        benchmark="synthetic",
    )
    assert trajectory.status == "success"
    assert trajectory.final_answer == "Final answer from the scripted model."
    assert [turn.tool_call["name"] for turn in trajectory.turns] == [
        "image_search",
        "text_search",
    ]
    assert all(turn.status == "success" for turn in trajectory.turns)
    assert any(
        message["role"] == "tool" and "Search results:" in message["content"]
        for message in model.calls[1]["messages"]
    )
    assert trajectory.image_ids == ["img_1"]


def test_agent_runtime_handles_malformed_unknown_and_max_turns() -> None:
    image = Image.new("RGB", (2, 2))
    model = ScriptedAgentModel(
        [
            '<tool_call>{"name":"crop",}</tool_call>',
            '<tool_call>{"name":"visit","arguments":{}}</tool_call>',
        ]
    )
    trajectory = AgentRuntime(
        model=model, tool_registry=create_mock_tool_registry(), max_agent_turns=2
    ).run(question="test", images=[image])
    assert trajectory.status == "max_agent_turns_exceeded"
    assert [turn.error for turn in trajectory.turns] == [
        "invalid_tool_call",
        "unknown_tool",
    ]


def test_truncated_tool_call_remains_invalid() -> None:
    trajectory = AgentRuntime(
        model=ScriptedAgentModel(['<tool_call>\n{"name": "crop", "arguments": {']),
        tool_registry=create_mock_tool_registry(), max_agent_turns=1,
    ).run(question="test", images=[Image.new("RGB", (2, 2))])
    assert trajectory.turns[0].error == "invalid_tool_call"
    assert "incomplete tool_call block" in trajectory.turns[0].observation


def test_agent_runtime_converts_backend_exception_to_observation() -> None:
    def broken_backend(arguments, context):
        del arguments, context
        raise RuntimeError("fixture backend failure")

    registry = ToolRegistry()
    registry.register(
        RegisteredTool(TOOL_DECLARATIONS_BY_NAME["image_search"], broken_backend)
    )
    model = ScriptedAgentModel(
        [
            '<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}</tool_call>',
            "Recovered final answer.",
        ]
    )
    trajectory = AgentRuntime(
        model=model, tool_registry=registry, max_agent_turns=2
    ).run(question="test", images=[Image.new("RGB", (2, 2))])
    assert trajectory.status == "success"
    assert trajectory.turns[0].status == "error"
    assert trajectory.turns[0].error == "RuntimeError"
    assert "fixture backend failure" in trajectory.turns[0].observation
