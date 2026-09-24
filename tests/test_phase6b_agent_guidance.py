from __future__ import annotations

from PIL import Image

from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel, create_mock_tool_registry
from opensearch_vl_repro.agent.reliability import AGENT_BEHAVIOR_VERSION
from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS
from opensearch_vl_repro.evaluation.run_manifest import create_run_manifest, manifest_mismatches


def test_model_receives_visual_investigation_policy_and_registered_ids():
    model = ScriptedAgentModel(["Visible answer."])
    trajectory = AgentRuntime(
        model=model, tool_registry=create_mock_tool_registry(), max_agent_turns=8,
    ).run(question="What is visible?", images=[Image.new("RGB", (3, 3))])

    assert trajectory.status == "success" and not trajectory.turns
    messages = model.calls[0]["messages"]
    guidance = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "Verify, Don't Guess" in guidance
    assert "what the user asks" in guidance
    assert "image quality" in guidance and "what information is missing" in guidance
    assert "next useful action" in guidance
    assert "clear image with a directly answerable question needs no tool call" in guidance
    for recipe in (
        "perspective_correct -> sharpen -> layout_parsing",
        "crop -> layout_parsing", "image_search -> text_search",
    ):
        assert recipe in guidance
    assert "After image_search suggests an identity, normally follow with text_search" in guidance
    assert "Registered input images:\n- img_1" in guidance
    assert "registered runtime image IDs" in guidance
    assert "image_search, pass a registered img_n in its url argument" in guidance
    for restriction in ("dataset filename", "filesystem path", "HTTP URL",
                        "Do not repeat an identical tool call", "same invalid arguments",
                        "previous observations", "stop using tools and provide the final answer"):
        assert restriction in guidance
    assert "Qwen3-32B" not in guidance and "<think>" not in guidance


def test_tool_descriptions_add_triggers_without_changing_argument_contracts():
    expected = {
        "text_search": ({"q": "string", "hl": "string", "top_k": "number"}, ("q",)),
        "image_search": ({"url": "string"}, ("url",)),
        "crop": ({"image": "string", "x": "number", "y": "number",
                  "width": "number", "height": "number"},
                 ("image", "x", "y", "width", "height")),
        "layout_parsing": ({"image": "string", "use_chart_recognition": "boolean",
                            "use_doc_orientation_classify": "boolean"}, ("image",)),
        "super_resolution": ({"image": "string", "scale": "number"}, ("image", "scale")),
        "sharpen": ({"image": "string", "amount": "number"}, ("image", "amount")),
        "web_search": ({"q": "string", "hl": "string"}, ("q",)),
        "perspective_correct": ({"image": "string"}, ("image",)),
    }
    assert {tool.name for tool in TOOL_DECLARATIONS} == set(expected)
    for tool in TOOL_DECLARATIONS:
        properties, required = expected[tool.name]
        assert tool.parameters == {
            "type": "object",
            "properties": {name: {"type": kind} for name, kind in properties.items()},
            "required": list(required),
            "additionalProperties": False,
        }
        assert len(tool.description) > 100
    descriptions = {tool.name: tool.description.lower() for tool in TOOL_DECLARATIONS}
    assert "snippets" in descriptions["web_search"]
    assert "page passages" in descriptions["text_search"]
    assert "reverse-image-style" in descriptions["image_search"]
    for name in ("crop", "layout_parsing", "perspective_correct",
                 "super_resolution", "sharpen", "image_search"):
        assert "registered runtime image id" in descriptions[name]
        assert "filename" in descriptions[name]
        assert "filesystem path" in descriptions[name]
        assert "http url" in descriptions[name]


def test_agent_behavior_version_3_rejects_version_2_run_identity():
    current = create_run_manifest(
        run_id="base-dev30-v3", model_name_or_path="Qwen/model", model_revision="r",
        inference_config_fingerprint="i", dataset_path="eval.parquet",
        dataset_identity={"sha256": "frozen"}, start=0, limit=30,
        max_agent_turns=8, search_config_fingerprint="s",
        layout_config_fingerprint="l", checkpoint={"kind": "remote"},
        tool_fingerprint="tool-contract", created_at="fixed",
    )
    previous = dict(current, agent_behavior_version=2)
    assert AGENT_BEHAVIOR_VERSION == current["agent_behavior_version"] == 3
    assert "agent_behavior_version" in manifest_mismatches(previous, current)
