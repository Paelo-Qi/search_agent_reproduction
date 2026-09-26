from __future__ import annotations

import copy

import pytest
from PIL import Image

from opensearch_vl_repro.agent.runtime import AGENT_SYSTEM_GUIDANCE
from opensearch_vl_repro.data import (SFT_RUNTIME_IMAGE_RULES, build_messages, render_prompt,
                                      validate_raw_sample)
from opensearch_vl_repro.sft_image_grounding import (audit_sample_image_grounding,
                                                     summarize_image_grounding)
from opensearch_vl_repro.sft_main_data import runtime_tools


class _Template:
    def apply_chat_template(self, messages, **kwargs):
        def body(content):
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "<|image_pad|>") for part in content)
        return "".join(f"<|im_start|>{row['role']}\n{body(row['content'])}<|im_end|>"
                       for row in messages)


def _sample(tmp_path, initial=1, derived=0):
    paths = []
    for index in range(initial + derived):
        name = f"image_{index + 1}.png"
        Image.new("RGB", (11 + index, 7 + index), "navy").save(tmp_path / name)
        paths.append(name)
    call = '<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}</tool_call>'
    turns = [{"from": "human", "value": "<image>" * initial + "Question?"},
             {"from": "gpt", "value": call}]
    for index in range(derived):
        turns.extend([
            {"from": "observation", "value":
             f"<image><observation>New image ID: img_{initial + index + 1}. "
             "Image size: width=12, height=8.</observation>"},
            {"from": "gpt", "value":
             f'<tool_call>{{"name":"image_search","arguments":{{"url":"img_{initial + index + 1}"}}}}</tool_call>'},
        ])
    return {"_sample_id": "fvqa:42", "images": paths, "tools": runtime_tools(),
            "conversations": turns}


def test_first_turn_image_id_is_grounded_in_final_context_without_target_or_tool_change(tmp_path):
    assert SFT_RUNTIME_IMAGE_RULES in AGENT_SYSTEM_GUIDANCE
    sample = _sample(tmp_path)
    original = copy.deepcopy(sample)
    messages, images, tools = build_messages(sample, tmp_path / "main_a_1k.json")
    prefix = render_prompt(_Template(), messages[:2], tools)
    assert "Registered input images:\n- img_1: width=11, height=7" in prefix
    assert "Never use a dataset filename, filesystem path, or HTTP URL as an image ID." in prefix
    assert prefix.index("- img_1:") < len(prefix)
    assert messages[2]["content"] == original["conversations"][1]["value"]
    assert tools == original["tools"] and sample == original
    assert len(images) == 1
    assert sum(part["type"] == "image" for row in messages
               if isinstance(row["content"], list) for part in row["content"]) == 1
    assert audit_sample_image_grounding(sample, messages)["passed"]


def test_legacy_system_is_preserved_but_runtime_id_rule_is_last(tmp_path):
    sample = _sample(tmp_path)
    sample["system"] = 'Legacy example: image_search.url can be a direct URL.'
    messages, _, _ = build_messages(sample, tmp_path / "shard.json")
    system = messages[0]["content"]
    assert system.startswith(sample["system"])
    assert system.index("Registered input images:") > system.index("Legacy example:")
    assert system.endswith(SFT_RUNTIME_IMAGE_RULES)


@pytest.mark.parametrize("count", [2, 3])
def test_initial_multi_image_order_is_stable(tmp_path, count):
    sample = _sample(tmp_path, initial=count)
    messages, images, _ = build_messages(sample, tmp_path / "shard.json")
    lines = messages[0]["content"].split("Registered input images:\n", 1)[1].split("\n\nRuntime rules:", 1)[0].splitlines()
    assert lines == [f"- img_{index}: width={10 + index}, height={6 + index}"
                     for index in range(1, count + 1)]
    assert [part["image"] for part in messages[1]["content"]
            if part["type"] == "image"] == images
    assert audit_sample_image_grounding(sample, messages)["passed"]


def test_derived_image_is_not_claimed_as_initial_and_is_grounded_by_observation(tmp_path):
    sample = _sample(tmp_path, derived=1)
    messages, images, tools = build_messages(sample, tmp_path / "shard.json")
    assert "- img_2:" not in messages[0]["content"]
    assert len(images) == 2 and messages[3]["role"] == "tool"
    assert "New image ID: img_2" in render_prompt(_Template(), messages[:4], tools)
    report = audit_sample_image_grounding(sample, messages)
    assert report["passed"] and report["calls"]["grounded_image_search_img_n"] == 2


def test_derived_id_gap_fails_closed_and_reports_sample_id(tmp_path):
    sample = _sample(tmp_path, derived=1)
    sample["conversations"][2]["value"] = sample["conversations"][2]["value"].replace("img_2", "img_3")
    messages, _, _ = build_messages(sample, tmp_path / "shard.json")
    row = audit_sample_image_grounding(sample, messages)
    summary = summarize_image_grounding([row])
    assert not row["passed"] and not summary["passed"]
    assert summary["failures"][0]["sample_id"] == "fvqa:42"
    assert summary["failures"][0]["errors"][0]["kind"] == "derived_image_registration_gap"


def test_marker_path_mismatch_remains_rejected(tmp_path):
    sample = _sample(tmp_path)
    sample["images"].append("missing.png")
    with pytest.raises(ValueError, match="marker/path mismatch"):
        validate_raw_sample(sample)
