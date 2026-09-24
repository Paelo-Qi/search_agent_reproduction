from __future__ import annotations

import json

import pytest
from PIL import Image

from opensearch_vl_repro.agent.local_visual_tools import crop
from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel, create_mock_tool_registry
from opensearch_vl_repro.agent.question_normalization import normalize_model_question
from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from opensearch_vl_repro.agent.tool_registry import RegisteredTool, ToolRegistry
from opensearch_vl_repro.evaluation import BatchRunner, BatchSample, create_run_manifest


@pytest.mark.parametrize("raw,expected", [
    (
        "image_id: 79.jpg Question: What kind of people can go to this building without buying ticket?",
        "What kind of people can go to this building without buying ticket?",
    ),
    (
        "image_id: base_Art&Music_deepeyesv2_550f6b42cdee36d9_001.png "
        "Question: What is the name of the musical group?",
        "What is the name of the musical group?",
    ),
    (
        "  image_id : c1_Architecture_oven_04_oven_04953817.jpg\n "
        "Question :  What changed during the 17th century?  ",
        "What changed during the 17th century?",
    ),
    ("What is in this image?", "What is in this image?"),
    ("What does image_id: 79.jpg Question: mean?", "What does image_id: 79.jpg Question: mean?"),
    ("image_id: /tmp/79.jpg Question: Keep this unchanged?",
     "image_id: /tmp/79.jpg Question: Keep this unchanged?"),
])
def test_only_known_dataset_filename_wrapper_is_removed(raw, expected):
    assert normalize_model_question(raw) == expected


def test_initial_model_message_lists_actual_dimensions_for_each_registered_image(tmp_path):
    second = tmp_path / "second.png"
    Image.new("RGB", (11, 7)).save(second)
    model = ScriptedAgentModel(["Visible answer."])
    AgentRuntime(model=model, tool_registry=create_mock_tool_registry(),
                 max_agent_turns=8).run(
        question="What is visible?",
        images=[Image.new("RGB", (704, 466)), second],
    )
    system = model.calls[0]["messages"][0]["content"]
    assert "Registered input images:\n- img_1: width=704, height=466\n- img_2: width=11, height=7" in system
    assert "pixel coordinates of the referenced img_n" in system
    assert "consult its listed width and height" in system
    assert "x and y within those image bounds" in system
    assert "meaningful non-empty region" in system


def test_derived_image_observation_reports_real_clipped_size():
    registry = ToolRegistry()
    registry.register(RegisteredTool(TOOL_DECLARATIONS_BY_NAME["crop"], crop))
    model = ScriptedAgentModel([
        'crop({"image":"img_1","x":450,"y":450,"width":300,"height":100})',
        "Final answer.",
    ])
    trajectory = AgentRuntime(model=model, tool_registry=registry,
                              max_agent_turns=8).run(
        question="Inspect the corner.", images=[Image.new("RGB", (704, 466))],
    )
    assert trajectory.status == "success"
    assert trajectory.turns[0].derived_images[0]["result_size"] == [254, 16]
    assert "New image ID: img_2. Image size: width=254, height=16." in trajectory.turns[0].observation
    assert "New image ID: img_2. Image size: width=254, height=16." in str(
        model.calls[1]["messages"][-1]["content"]
    )
    assert "width=300, height=100" not in trajectory.turns[0].observation


def test_batch_record_keeps_raw_question_while_model_sees_normalized_text(tmp_path):
    raw = "image_id: 79.jpg Question: What kind of people can go to this building without buying ticket?"
    expected = "What kind of people can go to this building without buying ticket?"
    model = ScriptedAgentModel(["A final answer."])
    runtime = AgentRuntime(model=model, tool_registry=create_mock_tool_registry(),
                           max_agent_turns=8)
    manifest = create_run_manifest(
        run_id="question-audit", model_name_or_path="fixture", model_revision="fixture",
        inference_config_fingerprint="i", dataset_path="fixture.parquet",
        dataset_identity={"sha256": "fixture"}, start=0, limit=1,
        max_agent_turns=8, search_config_fingerprint="s",
        layout_config_fingerprint="l", checkpoint={"kind": "fixture"},
        tool_fingerprint="fixture", created_at="fixed",
    )
    run_dir = tmp_path / "question-audit"
    summary = BatchRunner(runtime, run_dir, run_manifest=manifest).run([
        BatchSample("sample-1", "mmsearch", raw, [Image.new("RGB", (4, 3))]),
    ])
    record = json.loads((run_dir / "trajectories.jsonl").read_text(encoding="utf-8"))
    assert summary["success"] == 1
    assert record["question"] == raw
    assert record["sample_id"] == "sample-1" and record["benchmark"] == "mmsearch"
    assert model.calls[0]["messages"][1]["content"][-1]["text"] == expected
