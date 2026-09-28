"""CPU-only regression tests for the source-to-runtime image_search v3 boundary."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from opensearch_vl_repro.agent.mock_tools import ScriptedAgentModel
from opensearch_vl_repro.agent.runtime import AgentRuntime
from opensearch_vl_repro.agent.tool_contracts import (RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
                                                     TOOL_DECLARATIONS_BY_NAME)
from opensearch_vl_repro.agent.tool_registry import RegisteredTool, ToolRegistry, ToolResult
from opensearch_vl_repro.data import SFT_INPUT_MESSAGE_VERSION
from opensearch_vl_repro.inference.adapter import adapter_identity
from opensearch_vl_repro.sft_main_data import (canonicalize_sft_training_record,
                                               load_sft_manifest, prepare_sft_pool)
from opensearch_vl_repro.sft_protocol_v3 import (canonicalize_source_assistant_text,
                                                 effective_image_search_call_counts)
from opensearch_vl_repro.sft_repair_training import run_repair
from opensearch_vl_repro.tool_protocol_dev import (require_unchanged_dev_ids,
                                                   validate_dev_manifest)


def _block(name: str, arguments: dict) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def test_only_verified_image_search_argument_keys_are_transformed():
    url = "https://example.com/real-page"
    source = (f"I found {url}; the word url is ordinary prose.\n"
              + _block("web_search", {"q": url})
              + _block("crop", {"image": "img_1", "x": 1, "y": 2, "width": 3, "height": 4})
              + _block("layout_parsing", {"image": "img_2"})
              + _block("image_search", {"url": "img_2"})
              + _block("image_search", {"url": "img_3"}))
    changed = canonicalize_source_assistant_text(source)
    assert changed == source.replace('"arguments": {"url": "img_2"}',
                                     '"arguments": {"image_id": "img_2"}').replace(
                                         '"arguments": {"url": "img_3"}',
                                         '"arguments": {"image_id": "img_3"}')
    assert url in changed and _block("web_search", {"q": url}) in changed
    assert _block("crop", {"image": "img_1", "x": 1, "y": 2,
                           "width": 3, "height": 4}) in changed
    assert _block("layout_parsing", {"image": "img_2"}) in changed


def test_functional_and_stringified_arguments_reparse_without_prose_changes():
    prose = 'An example is image_search({"url":"img_1"}), but I will answer directly.'
    assert canonicalize_source_assistant_text(prose) == prose
    assert canonicalize_source_assistant_text('image_search({"url":"img_1"})') == (
        'image_search({"image_id":"img_1"})')
    source = '<tool_call>{"name":"image_search","arguments":"{\\"url\\":\\"img_2\\"}"}</tool_call>'
    changed = canonicalize_source_assistant_text(source)
    assert json.loads(changed.removeprefix("<tool_call>").removesuffix("</tool_call>"))[
        "arguments"] == '{"image_id":"img_2"}'
    nested = ('<tool_call>{"function":{"name":"image_search",'
              '"arguments":{"url":"img_3"}}}</tool_call>')
    assert canonicalize_source_assistant_text(nested) == nested.replace(
        '"arguments":{"url":', '"arguments":{"image_id":')
    with pytest.raises(ValueError, match="malformed source"):
        canonicalize_source_assistant_text('image_search({"url":})')


@pytest.mark.parametrize("argument", [{"url": "https://example.com/x.jpg"},
                                      {"url": "local.jpg"}, {"url": "img_0"},
                                      {"url": "img_1", "image_id": "img_1"}])
def test_source_transform_fails_closed_on_invalid_or_ambiguous_target(argument):
    with pytest.raises(ValueError):
        canonicalize_source_assistant_text(_block("image_search", argument))
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_source_assistant_text(
            '<tool_call>{"name":"image_search",'
            '"arguments":{"url":"img_1","url":"img_2"}}</tool_call>')


def test_effective_record_preserves_raw_provenance_and_audits_v3_calls():
    raw = {"tools": "[]", "system": "Historical source",
           "images": ["image.png"],
           "conversations": [{"from": "human", "value": "<image>Question?"},
                             {"from": "gpt", "value": _block("image_search", {"url": "img_1"})}]}
    effective = canonicalize_sft_training_record(raw)
    assert raw["conversations"][1]["value"] == _block("image_search", {"url": "img_1"})
    assert effective["_source_tools"] == raw["tools"]
    assert effective["_runtime_tool_protocol_version"] == RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
    counts = effective_image_search_call_counts(effective)
    assert counts["image_search_image_id"] == 1
    assert all(counts[key] == 0 for key in counts if key != "image_search_image_id")
    assert canonicalize_sft_training_record(effective)["conversations"] == effective["conversations"]


@pytest.mark.parametrize("arguments,expected_error", [
    ({"image_id": "img_1"}, None),
    ({"url": "img_1"}, "ValueError"),
    ({"image_id": "https://example.com/a.jpg"}, "unknown_image_id"),
    ({"image_id": "img_999"}, "unknown_image_id"),
])
def test_runtime_rejects_legacy_http_and_unknown_before_backend(arguments, expected_error):
    calls = []

    def backend(actual, context):
        calls.append(actual)
        return ToolResult(status="success", observation="<observation>match</observation>")

    registry = ToolRegistry()
    registry.register(RegisteredTool(TOOL_DECLARATIONS_BY_NAME["image_search"], backend))
    model = ScriptedAgentModel([_block("image_search", arguments), "Done."])
    trajectory = AgentRuntime(model=model, tool_registry=registry, max_agent_turns=2).run(
        question="Question?", images=[Image.new("RGB", (4, 4))])
    turn = trajectory.turns[0]
    assert turn.error == expected_error
    assert calls == ([arguments] if expected_error is None else [])
    if expected_error == "unknown_image_id":
        assert turn.metadata["provider_called"] is False


def test_old_manifests_and_checkpoint_identity_fail_closed(tmp_path):
    pool = tmp_path / "manifest.json"
    pool.write_text(json.dumps({"version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_sft_manifest(pool)
    with pytest.raises(FileExistsError, match="historical SFT pool"):
        prepare_sft_pool(tmp_path / "raw", tmp_path)
    with pytest.raises(ValueError, match="pinned provenance"):
        validate_dev_manifest([], {"version": 1})
    adapter = tmp_path / "checkpoint" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "model"}), encoding="utf-8")
    (adapter.parent / "metadata.json").write_text(json.dumps({
        "checkpoint_complete": True, "model": "model", "model_revision": "revision",
        "sft_input_message_version": "runtime-image-id-grounding-v2",
    }), encoding="utf-8")
    assert SFT_INPUT_MESSAGE_VERSION == RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
    with pytest.raises(ValueError, match="base model/revision"):
        adapter_identity(adapter, base_model="model", base_revision="revision")


def test_v3_config_changes_only_artifact_paths():
    import yaml

    root = Path(__file__).resolve().parents[1] / "configs"
    old = yaml.safe_load((root / "sft_main.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((root / "sft_main_imageid_v3.yaml").read_text(encoding="utf-8"))
    assert {key: value for key, value in new.items() if key not in {"project", "data"}} == {
        key: value for key, value in old.items() if key not in {"project", "data"}}
    assert new["project"]["seed"] == old["project"]["seed"]
    assert {key: value for key, value in new["data"].items() if key != "pool_dir"} == {
        key: value for key, value in old["data"].items() if key != "pool_dir"}
    assert new["project"]["output_dir"] != old["project"]["output_dir"]
    assert new["project"]["report_dir"] != old["project"]["report_dir"]


def test_legacy_targeted_repair_does_not_mix_with_v3_runtime(tmp_path):
    with pytest.raises(RuntimeError, match="legacy v2 targeted repair"):
        run_repair(tmp_path / "unused.yaml")


def test_v3_dev50_requires_exact_historical_ordered_ids():
    ids = [f"fvqa:{index}" for index in range(50)]
    require_unchanged_dev_ids(ids, list(ids))
    with pytest.raises(ValueError, match="ordered IDs"):
        require_unchanged_dev_ids(ids, list(reversed(ids)))


def test_sft_only_audit_does_not_claim_full_dev50_pass(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/audit_image_search_v3.py"
    spec = importlib.util.spec_from_file_location("audit_image_search_v3", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = canonicalize_sft_training_record({
        "tools": "[]", "images": ["image.png"],
        "conversations": [{"from": "human", "value": "<image>Question?"},
                          {"from": "gpt", "value": 'image_search({"url":"img_1"})'}]})
    row["_sample_id"] = "fvqa:0"
    monkeypatch.setattr(module, "SHARD_SIZES", {"tiny": 1})
    monkeypatch.setattr(module, "SOURCE_COUNTS", {"fvqa": 1})
    monkeypatch.setattr(module, "load_sft_manifest", lambda path: {
        "shards": {"tiny": {"path": "tiny.json", "count": 1}},
        "pool_source_counts": {"fvqa": 1}, "membership": [{"sample_id": "fvqa:0"}]})
    monkeypatch.setattr(module, "require_data_quality_exclusions", lambda manifest: None)
    monkeypatch.setattr(module, "iter_json_array", lambda path: [row])
    result = module.audit(tmp_path / "manifest.json")
    assert result["sft_passed"] is True
    assert result["passed"] is False and result["dev50_checked"] is False
    assert result["effective_image_search_counts"]["image_search_image_id"] == 1
