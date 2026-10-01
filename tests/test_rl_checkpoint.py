import json

import pytest

from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.data import SFT_INPUT_MESSAGE_VERSION
from opensearch_vl_repro.rl.checkpoint import RLLineage, build_rl_lineage, validate_sft_overlap_scope
from opensearch_vl_repro.sft_tool_audit import sha256_file


@pytest.fixture
def checkpoint(tmp_path):
    adapter = tmp_path / "checkpoint-3k" / "adapter"
    adapter.mkdir(parents=True)
    sft = {"model": {"name_or_path": "base", "revision": "rev",
                     "freeze_vision_tower": True, "freeze_multimodal_projector": True},
           "lora": {"rank": 16, "alpha": 32, "dropout": .05,
                    "target_modules": ["q_proj", "v_proj"]}}
    config = {"model": {"continue_from_sft_adapter": True, "sft_stage": "main_b_2k"}}
    adapter_config = {"base_model_name_or_path": "base", "peft_type": "LORA", "r": 16,
                      "lora_alpha": 32, "lora_dropout": .05,
                      "target_modules": ["q_proj", "v_proj"]}
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fake local weights")
    metadata = {"checkpoint_complete": True, "stage_complete": True,
                "model": "base", "model_revision": "rev", "stage": "main_b_2k",
                "lineage": ["main_a_1k", "main_b_2k"],
                "sft_input_message_version": SFT_INPUT_MESSAGE_VERSION,
                "runtime_tool_protocol_version": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
                "file_sha256": {f"adapter/{p.name}": sha256_file(p) for p in adapter.iterdir()}}
    (adapter.parent / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return adapter, config, sft


def build(checkpoint):
    adapter, config, sft = checkpoint
    return build_rl_lineage(config=config, sft_config=sft, adapter_path=adapter, run_id="test")


def test_valid_lineage_round_trip(checkpoint):
    value = build(checkpoint)
    assert value.sft_stage == "main_b_2k"
    assert RLLineage.from_dict(value.to_dict()) == value
    assert value.rl_adapter_fingerprint is None
    validate_sft_overlap_scope(value, ["main_a_1k", "main_b_2k"])


def test_overlap_scope_must_match_adapter_lineage(checkpoint):
    value = build(checkpoint)
    with pytest.raises(ValueError, match="overlap shard scope"):
        validate_sft_overlap_scope(value, ["main_a_1k", "main_b_2k", "extra_1k"])


@pytest.mark.parametrize("field,value", [("name_or_path", "wrong"), ("revision", "wrong")])
def test_base_mismatch_rejected(checkpoint, field, value):
    checkpoint[2]["model"][field] = value
    with pytest.raises(ValueError):
        build(checkpoint)


def test_adapter_fingerprint_mismatch_rejected(checkpoint):
    (checkpoint[0] / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        build(checkpoint)


def test_lora_mismatch_rejected(checkpoint):
    checkpoint[2]["lora"]["target_modules"] = ["q_proj"]
    with pytest.raises(ValueError, match="LoRA"):
        build(checkpoint)


def test_image_protocol_mismatch_rejected(checkpoint):
    metadata_path = checkpoint[0].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["runtime_tool_protocol_version"] = "legacy-v2"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        build(checkpoint)


def test_incomplete_stage_rejected(checkpoint):
    metadata_path = checkpoint[0].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["stage_complete"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        build(checkpoint)
