from __future__ import annotations

import contextlib
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from opensearch_vl_repro.agent.image_registry import ImageRegistry
from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel
from opensearch_vl_repro.agent.phase2_registry import create_phase2_tool_registry
from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_registry import DerivedImage, RegisteredTool, ToolContext, ToolRegistry, ToolResult
from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from opensearch_vl_repro.inference.generation import QwenAgentModel


def _context(size=(10, 8)):
    images = ImageRegistry()
    images.register_initial_image(Image.new("RGB", size, "navy"))
    return ToolContext(images)


def test_local_visual_backends_and_validation():
    registry = create_phase2_tool_registry()
    context = _context()
    assert len(registry.list_tools()) == 8
    cropped = registry.execute("crop", {"image": "img_1", "x": 8, "y": 6,
                                        "width": 9, "height": 9}, context)
    assert cropped.derived_images[0].value.size == (2, 2)
    assert cropped.metadata["crop_box"] == [8, 6, 10, 8]
    boundary = registry.execute("crop", {"image": "img_1", "x": -2, "y": -2,
                                         "width": 4, "height": 4}, context)
    assert boundary.derived_images[0].value.size == (2, 2)
    with pytest.raises(ValueError, match="empty"):
        registry.execute("crop", {"image": "img_1", "x": 20, "y": 0,
                                   "width": 2, "height": 2}, context)
    with pytest.raises(ValueError, match="positive"):
        registry.execute("crop", {"image": "img_1", "x": 0, "y": 0,
                                   "width": 0, "height": 2}, context)
    with pytest.raises(KeyError, match="unknown image ID"):
        registry.execute("crop", {"image": "img_99", "x": 0, "y": 0,
                                   "width": 2, "height": 2}, context)
    sharpened = registry.execute("sharpen", {"image": "img_1", "amount": 2}, context)
    assert sharpened.derived_images[0].value.size == (10, 8)
    with pytest.raises(ValueError, match="amount"):
        registry.execute("sharpen", {"image": "img_1", "amount": 5}, context)
    enlarged = registry.execute("super_resolution", {"image": "img_1", "scale": 2}, context)
    assert enlarged.derived_images[0].value.size == (20, 16)
    assert enlarged.metadata["backend"] == "lanczos"
    assert enlarged.metadata["learned_sr"] is False
    assert "no lost detail was recovered" in enlarged.observation
    with pytest.raises(ValueError, match="scale"):
        registry.execute("super_resolution", {"image": "img_1", "scale": 5}, context)
    unchanged = registry.execute("perspective_correct", {"image": "img_1"}, context)
    assert unchanged.derived_images[0].value is not context.image_registry.get("img_1")
    assert list(unchanged.derived_images[0].value.getdata()) == list(context.image_registry.get("img_1").getdata())
    assert unchanged.metadata["backend_mode"] == "identity_fallback"
    assert unchanged.metadata["changed"] is False
    assert [entry.image_id for entry in context.image_registry.list_images()] == ["img_1"]


def test_derived_image_chain_reaches_next_model_turn():
    model = ScriptedAgentModel([
        'crop({"image":"img_1","x":2,"y":1,"width":4,"height":3})',
        'sharpen({"image":"img_2","amount":2})',
        'super_resolution({"image":"img_3","scale":2})',
        "The image was inspected.",
    ])
    trajectory = AgentRuntime(model=model, tool_registry=create_phase2_tool_registry(),
                              max_agent_turns=4).run(question="Inspect this image", images=[Image.new("RGB", (10, 8))])
    assert trajectory.status == "success"
    assert trajectory.image_ids == ["img_1", "img_2", "img_3", "img_4"]
    assert [item["parent_id"] for item in trajectory.images] == [None, "img_1", "img_2", "img_3"]
    assert [item["size"] for item in trajectory.images] == [[10, 8], [4, 3], [4, 3], [8, 6]]
    assert [item["metadata"]["result_image_id"] for item in trajectory.images[1:]] == ["img_2", "img_3", "img_4"]
    tool_message = model.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert isinstance(tool_message["content"][0]["image"], Image.Image)
    assert tool_message["content"][0]["image"].size == (4, 3)
    assert "img_2" in tool_message["content"][1]["text"]
    assert "PIL" not in json.dumps(trajectory.to_dict())


def test_multiple_derived_images_and_tool_error_recovery():
    def two_images(arguments, context):
        return ToolResult(status="success", observation="<observation>Two images.</observation>",
                          derived_images=(DerivedImage(Image.new("RGB", (2, 2)), "img_1"),
                                          DerivedImage(Image.new("RGB", (3, 3)), "img_1")))

    registry = ToolRegistry()
    registry.register(RegisteredTool(TOOL_DECLARATIONS_BY_NAME["crop"], two_images))
    model = ScriptedAgentModel(['crop({"image":"img_1","x":0,"y":0,"width":2,"height":2})', "Done"])
    trajectory = AgentRuntime(model=model, tool_registry=registry, max_agent_turns=2).run(
        question="test", images=[Image.new("RGB", (5, 5))])
    assert trajectory.image_ids == ["img_1", "img_2", "img_3"]
    assert [part["image"].size for part in model.calls[1]["messages"][-1]["content"][:2]] == [(2, 2), (3, 3)]
    model = ScriptedAgentModel(['crop({"image":"img_1","x":99,"y":0,"width":1,"height":1})', "Recovered"])
    failed = AgentRuntime(model=model, tool_registry=create_phase2_tool_registry(), max_agent_turns=2).run(
        question="test", images=[Image.new("RGB", (5, 5))])
    assert failed.status == "success"
    assert failed.turns[0].status == "error"
    assert failed.image_ids == ["img_1"]

    def malformed_batch(arguments, context):
        return ToolResult(status="success", observation="<observation>bad batch</observation>",
                          derived_images=(DerivedImage(Image.new("RGB", (2, 2)), "img_1"),
                                          DerivedImage("missing-image.png", "img_1")))

    registry = ToolRegistry()
    registry.register(RegisteredTool(TOOL_DECLARATIONS_BY_NAME["crop"], malformed_batch))
    model = ScriptedAgentModel(['crop({"image":"img_1","x":0,"y":0,"width":2,"height":2})', "Recovered"])
    rejected = AgentRuntime(model=model, tool_registry=registry, max_agent_turns=2).run(
        question="test", images=[Image.new("RGB", (5, 5))])
    assert rejected.turns[0].status == "error"
    assert rejected.image_ids == ["img_1"]


def test_qwen_adapter_passes_derived_pil_to_processor(monkeypatch):
    # No torch/model download: exercise the real generate_chat -> processor call.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        manual_seed=lambda seed: None, inference_mode=contextlib.nullcontext))

    class Tokens:
        shape = (1, 2)

    class Outputs:
        def __getitem__(self, key):
            return [[1]]

    class Processor:
        def __init__(self):
            self.calls = []
            self.outputs = ['crop({"image":"img_1","x":1,"y":1,"width":3,"height":2})', "Done"]

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((list(messages), kwargs))
            return {"input_ids": Tokens()}

        def batch_decode(self, ids, **kwargs):
            return [self.outputs.pop(0)]

    processor = Processor()
    bundle = SimpleNamespace(processor=processor, model=SimpleNamespace(generate=lambda **kwargs: Outputs()),
                             config=SimpleNamespace(seed=1, device="cpu", generation_kwargs=lambda: {}))
    result = AgentRuntime(model=QwenAgentModel(bundle), tool_registry=create_phase2_tool_registry(),
                          max_agent_turns=2).run(question="crop", images=[Image.new("RGB", (6, 6))])
    assert result.status == "success"
    messages, kwargs = processor.calls[1]
    tool_content = messages[-1]["content"]
    assert tool_content[0]["type"] == "image"
    assert isinstance(tool_content[0]["image"], Image.Image)
    assert tool_content[0]["image"].size == (3, 2)
    assert kwargs["tokenize"] and kwargs["return_tensors"] == "pt"
