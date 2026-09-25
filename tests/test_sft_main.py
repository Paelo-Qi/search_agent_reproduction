from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from opensearch_vl_repro.data import RoleTokenSpan

from opensearch_vl_repro.agent.tool_contracts import TOOL_DECLARATIONS_BY_NAME
from opensearch_vl_repro.evaluation.run_manifest import create_run_manifest, manifest_mismatches
from opensearch_vl_repro.inference.adapter import adapter_identity
from opensearch_vl_repro.inference.config import load_inference_config
from opensearch_vl_repro.inference.model_loader import load_inference_bundle
from opensearch_vl_repro.sft_long_training import check_checkpoint, checkpoint_metadata
from opensearch_vl_repro.sft_main_data import (
    SHARD_SIZES, SOURCE_COUNTS, canonicalize_tool_declarations, load_sft_manifest, prepare_sft_pool,
    proportional_quotas, runtime_tools, selection_plan, stable_sample_id,
)
from opensearch_vl_repro.sft_preflight import (
    formal_preflight_checks, leakage_audit, sequence_summary, token_audit, tool_contract_audit,
)
from opensearch_vl_repro.sft_tool_audit import sha256_file
from opensearch_vl_repro.sft_train_plan import (
    cosine_factor, load_main_config, plan_stage, validate_resume_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def test_proportional_8k_allocation_is_exact_and_deterministic():
    assert sum(SOURCE_COUNTS.values()) == 36592
    assert proportional_quotas(8000, SOURCE_COUNTS) == {
        "fvqa": 965, "livevqa": 2913, "palace": 650, "webqa": 832,
        "wiki_art": 1113, "wiki_en": 766, "wiki_zh": 761,
    }
    first = selection_plan()
    assert first == selection_plan()
    assert first != selection_plan(seed=20260507)
    identities = []
    for shard, by_source in first.items():
        assert sum(map(len, by_source.values())) == SHARD_SIZES[shard]
        identities.extend(stable_sample_id(source, index)
                          for source, indices in by_source.items() for index in indices)
    assert len(identities) == len(set(identities)) == 8000
    assert sum(len(first[name][source]) for name in first for source in first[name]) == 8000
    assert sum(SHARD_SIZES[name] for name in ("main_a_1k", "main_b_2k")) == 3000
    assert sum(SHARD_SIZES[name] for name in ("main_a_1k", "main_b_2k", "extra_1k")) == 4000
    assert stable_sample_id("fvqa", 42) == "fvqa:42"


def _record(index: int) -> dict:
    return {"conversations": [
        {"from": "human", "value": f"<image> Question {index}?"},
        {"from": "gpt", "value": f"Answer {index}."},
    ], "images": [f"image_{index}.png"], "tools": "[]"}


def test_small_pool_manifest_is_rebuildable_and_detects_tampering(tmp_path, monkeypatch):
    from opensearch_vl_repro import sft_main_data as module

    monkeypatch.setattr(module, "SOURCE_FILES", {"fvqa": "fvqa/records.json", "webqa": "webqa/records.json"})
    monkeypatch.setattr(module, "SOURCE_COUNTS", {"fvqa": 6, "webqa": 6})
    monkeypatch.setattr(module, "SHARD_SIZES", {"main_a_1k": 1, "main_b_2k": 2,
                                            "extra_1k": 1, "reserve_4k": 4})
    monkeypatch.setattr(module, "POOL_SIZE", 8)
    raw = tmp_path / "raw"
    for source in ("fvqa", "webqa"):
        path = raw / source / "records.json"
        path.parent.mkdir(parents=True)
        values = [_record(index) for index in range(6)]
        values[5]["conversations"] = [{"from": "human", "value": "<image> invalid"}]
        path.write_text(json.dumps(values), encoding="utf-8")
    output = tmp_path / "prepared"
    first = prepare_sft_pool(raw, output)
    assert first["total_selected_count"] == 8 and first["disjointness_verified"]
    assert first["version"] == 2 and first["canonicalization_applied"] is True
    assert first["effective_runtime_tool_contract_fingerprint"]
    assert first["source_tool_declaration_fingerprint"]
    assert first["source_validity"]["fvqa"] == {"eligible": 5, "excluded_invalid": 1}
    assert first["images_ready"] is False
    assert "images_ready" not in load_sft_manifest(output / "manifest.json")
    assert json.loads((output / "media_status.json").read_text(encoding="utf-8")) == {"images_ready": False}
    ids = [item["sample_id"] for item in first["membership"]]
    assert len(set(ids)) == 8
    assert all(item["source_index"] < 5 for item in first["membership"])
    assert load_sft_manifest(output / "manifest.json")["shards"] == first["shards"]
    checksums = {name: first["shards"][name]["sha256"] for name in first["shards"]}
    assert {name: entry["sha256"] for name, entry in prepare_sft_pool(raw, output)["shards"].items()} == checksums
    manifest_sha = sha256_file(output / "manifest.json")
    for shard in first["shards"]:
        for record in json.loads((output / f"{shard}.json").read_text(encoding="utf-8")):
            for relative in record["images"]:
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (2, 2), "navy").save(target)
    assert prepare_sft_pool(raw, output)["images_ready"] is True
    assert sha256_file(output / "manifest.json") == manifest_sha
    (output / "main_a_1k.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="shard identity mismatch"):
        load_sft_manifest(output / "manifest.json")


def test_leakage_audit_detects_question_and_image_content_overlap(tmp_path):
    image = Image.new("RGB", (4, 3), "navy")
    image.save(tmp_path / "same.png")
    record = {"_sample_id": "fvqa:1", "_source": "fvqa",
              "conversations": [{"from": "human", "value": "<image> What is this?"}],
              "images": ["same.png"]}
    eval_sample = SimpleNamespace(sample_id="eval-1", benchmark="simplevqa",
                                  question="What is this?", images=[image])
    report = leakage_audit([record], [eval_sample], tmp_path)
    assert report["complete"] is True
    assert report["question_overlap_count"] == report["image_overlap_count"] == 1
    assert report["image_overlaps"][0]["eval_sample_id"] == "eval-1"
    assert report["image_overlaps"][0]["source"] == "fvqa"


def test_tool_contract_audit_reports_drift_without_remapping():
    declaration = TOOL_DECLARATIONS_BY_NAME["image_search"].as_chat_template_tool()
    record = {"_sample_id": "fvqa:2", "_source": "fvqa",
              "_source_tools": [declaration], "tools": runtime_tools(),
              "conversations": [{"from": "gpt", "value":
                                 '<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}</tool_call>'}]}
    assert tool_contract_audit([record])["passed"] is True
    function_style = {**record, "conversations": [{"from": "gpt", "value":
                                                   'image_search({"url":"img_1"})'}]}
    assert tool_contract_audit([function_style])["tool_call_counts"] == {"image_search": 1}
    bad = {**record, "conversations": [{"from": "gpt", "value":
                                         '<tool_call>{"name":"image_search","arguments":{"image":"img_1"}}</tool_call>'}]}
    report = tool_contract_audit([bad])
    assert report["passed"] is False
    assert report["mismatches"][0]["kind"] == "call_schema_drift"
    assert "url" in report["mismatches"][0]["detail"]
    extra = json.loads(json.dumps(declaration))
    extra["function"]["parameters"]["properties"]["file_path"] = {"type": "string"}
    report = tool_contract_audit([{**record, "_source_tools": [extra]}])
    assert report["passed"] is True
    assert report["raw_declaration_mismatches"][0]["kind"] == "declaration_schema_drift"
    assert report["raw_declaration_mismatches"][0]["extra_dataset_properties"] == ["file_path"]
    assert report["effective_declaration_drift_count"] == report["actual_call_drift_count"] == 0
    safe_sequence = {"full_8k": {"zero_supervised_count": 0,
                                 "partial_assistant_span_cut_count": 0,
                                 "partial_tool_call_cut_count": 0,
                                 "complete_assistant_span_dropped_count": 1}}
    safe_leakage = {"complete": True, "question_overlap_count": 0, "image_overlap_count": 0}
    assert all(formal_preflight_checks(report, safe_leakage, safe_sequence).values())
    assert tool_contract_audit([{**record, "tools": [extra]}])["passed"] is False
    assert not formal_preflight_checks(
        tool_contract_audit([{**record, "tools": [extra]}]), safe_leakage,
        safe_sequence)["effective_tool_contract"]
    assert not formal_preflight_checks(
        tool_contract_audit([bad]), safe_leakage, safe_sequence)["actual_tool_calls"]
    assert not formal_preflight_checks(report,
                                      {**safe_leakage, "image_overlap_count": 1},
                                      safe_sequence)["zero_image_overlap"]


def test_tool_canonicalization_preserves_expert_call_observation_and_final():
    raw = {"tools": '[{"type":"function","function":{"name":"text_search"}}]',
           "conversations": [
               {"from": "human", "value": "<image> Identify this."},
               {"from": "gpt", "value": '<tool_call>{"name":"text_search","arguments":{"q":"art"}}</tool_call>'},
               {"from": "observation", "value": "Original search result."},
               {"from": "gpt", "value": "Original final answer."},
           ], "images": ["original.png"]}
    original = json.loads(json.dumps(raw))
    effective = canonicalize_tool_declarations(raw)
    assert raw == original
    assert effective["_source_tools"] == original["tools"]
    assert effective["tools"] == runtime_tools()
    assert effective["conversations"] == original["conversations"]
    assert effective["images"] == original["images"]
    assert canonicalize_tool_declarations(raw) == effective


def test_sequence_audit_counts_zero_targets_and_cut_assistant_span():
    full = list(map(ord, "abXYZ"))
    tokenizer = SimpleNamespace(decode=lambda values, **kwargs: "".join(map(chr, values)))
    spans = [RoleTokenSpan(0, 2, 5)]
    messages = [{"role": "assistant", "content": "XYZ"}]
    intact = token_audit(full, full, spans, messages, tokenizer)
    cut = token_audit(full, full[:3], spans, messages, tokenizer)
    zero = token_audit(full, full[:2], spans, messages, tokenizer)
    assert intact["supervised_tokens_after_truncation"] == 3
    assert cut["partial_assistant_span_cut"] is True
    assert zero["zero_supervised_tokens"] is True
    summary = sequence_summary([intact, cut, zero], max_length=3)
    assert summary["count_over_max_length"] == 3
    assert summary["zero_supervised_count"] == 1
    assert summary["partial_assistant_span_cut_count"] == 1
    assert summary["complete_assistant_span_dropped_count"] == 1


def test_token_cut_boundaries_and_formal_report_only_drop():
    full = list(map(ord, "aaA!bb<tool_call>X</tool_call>Y!"))
    tokenizer = SimpleNamespace(decode=lambda values, **kwargs: "".join(map(chr, values)))
    spans = [RoleTokenSpan(0, 2, 4), RoleTokenSpan(1, 6, len(full))]
    messages = [{"role": "assistant", "content": "A!"},
                {"role": "assistant", "content": "<tool_call>X</tool_call>Y!"}]
    def audit(cutoff):
        return token_audit(full, full[:cutoff], spans, messages, tokenizer)

    assert audit(0)["zero_supervised_tokens"] is True
    assert audit(2)["zero_supervised_tokens"] is True  # exactly at assistant body start
    assert audit(4)["partial_assistant_span_cut"] is False  # exactly after first end
    assert audit(4)["complete_assistant_span_dropped"] == 1
    assert audit(6)["complete_assistant_span_dropped"] == 1  # second body start
    assert audit(3)["partial_assistant_span_cut"] is True
    assert audit(16)["partial_tool_call_cut"] is True
    assert audit(16)["partial_assistant_span_cut"] is True
    assert audit(len(full) - 1)["partial_tool_call_cut"] is False
    assert audit(len(full))["complete_assistant_span_dropped"] == 0
    summary = sequence_summary([audit(4)], 4)
    checks = formal_preflight_checks(
        {"effective_declaration_drift_count": 0, "actual_call_drift_count": 0},
        {"complete": True, "question_overlap_count": 0, "image_overlap_count": 0},
        {"full_8k": summary})
    assert all(checks.values())
    assert summary["complete_assistant_span_dropped_count"] == 1


def test_deterministic_question_and_image_exclusion_replacement(tmp_path, monkeypatch):
    from opensearch_vl_repro import sft_main_data as module

    monkeypatch.setattr(module, "SOURCE_FILES", {"fvqa": "fvqa/records.json", "webqa": "webqa/records.json"})
    monkeypatch.setattr(module, "SOURCE_COUNTS", {"fvqa": 6, "webqa": 6})
    monkeypatch.setattr(module, "SHARD_SIZES", {"main_a_1k": 1, "main_b_2k": 2,
                                            "extra_1k": 1, "reserve_4k": 4})
    monkeypatch.setattr(module, "POOL_SIZE", 8)
    raw_dir = tmp_path / "raw"
    for source in module.SOURCE_FILES:
        path = raw_dir / module.SOURCE_FILES[source]
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps([_record(i) for i in range(6)]), encoding="utf-8")
    original = prepare_sft_pool(raw_dir, tmp_path / "original")
    original_ids = {row["sample_id"] for row in original["membership"]}
    victim = sorted(item for item in original_ids if item.startswith("fvqa:"))[0]
    victim_index = int(victim.split(":")[1])
    monkeypatch.setattr(module, "eval_question_set",
                        lambda _: ({f"question {victim_index}?"}, "frozen-eval-sha"))
    first = prepare_sft_pool(raw_dir, tmp_path / "prepared", eval_path="fake-eval")
    assert victim not in {row["sample_id"] for row in first["membership"]}
    assert first["replacements"][0]["excluded_sample_id"] == victim
    assert first["replacements"][0]["replacement_source"] == "fvqa"
    assert first["replacements"][0]["replacement_sample_id"] not in original_ids
    assert first["pool_source_counts"] == original["pool_source_counts"]
    assert {key: value["source_counts"] for key, value in first["shards"].items()} == {
        key: value["source_counts"] for key, value in original["shards"].items()}
    assert {key: value["count"] for key, value in first["shards"].items()} == module.SHARD_SIZES
    assert len({row["sample_id"] for row in first["membership"]}) == 8
    manifest_sha = sha256_file(tmp_path / "prepared/manifest.json")
    assert sha256_file(tmp_path / "prepared/manifest.json") == manifest_sha
    assert prepare_sft_pool(raw_dir, tmp_path / "prepared", eval_path="fake-eval") == first
    assert sha256_file(tmp_path / "prepared/manifest.json") == manifest_sha
    for shard in module.SHARD_SIZES:
        for row in json.loads((tmp_path / "prepared" / f"{shard}.json").read_text(encoding="utf-8")):
            assert row["tools"] == runtime_tools()
            assert row["_source_tools"] == "[]"
            assert row["conversations"] == _record(row["_source_index"])["conversations"]
    assert load_sft_manifest(tmp_path / "prepared/manifest.json")["version"] == 2
    image_victim = sorted(item for item in original_ids if item.startswith("webqa:")
                          and int(item.split(":")[1]) != victim_index)[0]
    second = prepare_sft_pool(raw_dir, tmp_path / "prepared", eval_path="fake-eval",
                              exclusions={image_victim: "eval300_image_overlap"})
    assert image_victim not in {row["sample_id"] for row in second["membership"]}
    assert any(row["reason"] == "eval300_image_overlap" for row in second["replacements"])
    assert second["pool_source_counts"] == first["pool_source_counts"]
    assert sha256_file(tmp_path / "prepared/manifest.json") != manifest_sha
    assert len({row["sample_id"] for row in second["membership"]}) == 8
    assert prepare_sft_pool(raw_dir, tmp_path / "prepared", eval_path="fake-eval",
                            exclusions={image_victim: "eval300_image_overlap"}) == second
    with pytest.raises(ValueError, match="fingerprint/transform"):
        changed = json.loads((tmp_path / "prepared/manifest.json").read_text(encoding="utf-8"))
        changed["effective_runtime_tool_contract_fingerprint"] = "changed"
        (tmp_path / "prepared/manifest.json").write_text(json.dumps(changed), encoding="utf-8")
        load_sft_manifest(tmp_path / "prepared/manifest.json")


def _config():
    return load_main_config(ROOT / "configs/sft_main.yaml",
                            base_eval_config=ROOT / "configs/eval_base_300.yaml")


def test_independent_4b_smoke_config_keeps_phase0_artifacts_untouched():
    smoke = load_main_config(ROOT / "configs/sft_4b_smoke.yaml",
                             base_eval_config=ROOT / "configs/eval_base_300.yaml")
    assert smoke["data"]["path"] == "data/sft_4b_smoke_100.json"
    assert smoke["data"]["expected_samples"] == 100
    assert smoke["data"]["max_length"] == 32000
    assert smoke["training"]["world_size"] == 2
    assert smoke["training"]["max_steps"] == 20
    assert smoke["project"]["output_dir"] != "outputs/phase0/qwen3_vl_2b_lora"
    assert plan_stage(smoke, "smoke", micro_batch=2,
                      gradient_accumulation=2).effective_global_batch == 8


def _state(global_step: int, phase_step: int, current_lr: float = 0.0001):
    return {"global_step": global_step, "phase_step": phase_step, "epoch": 2,
            "microbatch_offset": 0, "cumulative_samples_seen": global_step * 8,
            "stage_samples_seen": 2000, "current_lr": current_lr}


def _metadata(plan, global_step, phase_step):
    return checkpoint_metadata(plan, _state(global_step, phase_step), config=_config(),
                               pool_sha256="pool-sha", shard_sha256="shard-sha",
                               resumed_from=None, complete_stage=True)


def test_phase_1_steps_and_cross_shard_resume_do_not_reset_scheduler():
    config = _config()
    first = plan_stage(config, "main_a_1k")
    second = plan_stage(config, "main_b_2k")
    extra = plan_stage(config, "extra_1k", micro_batch=2, gradient_accumulation=2)
    assert (first.stage_steps, second.stage_steps, extra.stage_steps) == (250, 500, 250)
    assert (first.global_target_step, second.global_target_step, extra.global_target_step) == (250, 750, 1000)
    assert {first.phase_total_steps, second.phase_total_steps, extra.phase_total_steps} == {1000}
    assert {first.warmup_steps, second.warmup_steps, extra.warmup_steps} == {100}
    assert first.effective_global_batch == extra.effective_global_batch == 8
    assert cosine_factor(750, 100, 1000) > 0
    assert cosine_factor(1000, 100, 1000) == 0
    assert validate_resume_metadata(second, _metadata(first, 250, 250),
                                    model_name=MODEL, revision=REVISION,
                                    pool_sha256="pool-sha") == "next_stage"
    mid_state = _state(125, 125)
    mid_metadata = checkpoint_metadata(first, mid_state, config=config,
                                       pool_sha256="pool-sha", shard_sha256="shard-sha",
                                       resumed_from=None, complete_stage=False)
    assert validate_resume_metadata(first, mid_metadata, model_name=MODEL,
                                    revision=REVISION, pool_sha256="pool-sha") == "same_stage"
    mid_metadata["phase_step"] = 0
    with pytest.raises(ValueError, match="scheduler/global step"):
        validate_resume_metadata(first, mid_metadata, model_name=MODEL,
                                 revision=REVISION, pool_sha256="pool-sha")
    assert validate_resume_metadata(plan_stage(config, "extra_1k"),
                                    _metadata(second, 750, 750), model_name=MODEL,
                                    revision=REVISION, pool_sha256="pool-sha") == "next_stage"
    with pytest.raises(ValueError, match="batch/DDP"):
        validate_resume_metadata(extra, _metadata(second, 750, 750),
                                 model_name=MODEL, revision=REVISION, pool_sha256="pool-sha")


def test_phase_2_requires_completed_4k_and_explicit_peak_lr():
    config = _config()
    with pytest.raises(ValueError, match="explicitly chosen"):
        plan_stage(config, "reserve_4k")
    phase2 = plan_stage(config, "reserve_4k", phase_2_peak_lr=0.0001)
    assert phase2.scheduler_phase == "phase_2" and phase2.phase_start_step == 0
    assert phase2.global_start_step == 1000 and phase2.global_target_step == 2000
    assert phase2.phase_total_steps == 1000 and phase2.warmup_steps == 100
    fourth = plan_stage(config, "extra_1k")
    assert validate_resume_metadata(phase2, _metadata(fourth, 1000, 1000),
                                    model_name=MODEL, revision=REVISION,
                                    pool_sha256="pool-sha") == "next_stage"
    with pytest.raises(ValueError, match="exact preceding"):
        validate_resume_metadata(phase2, _metadata(plan_stage(config, "main_b_2k"), 750, 750),
                                 model_name=MODEL, revision=REVISION, pool_sha256="pool-sha")


def _fake_checkpoint(path: Path, *, weights: bytes = b"weights") -> Path:
    adapter = path / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": MODEL}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(weights)
    for name in ("optimizer.pt", "scheduler.pt", "rng.pt"):
        (path / name).write_bytes(b"state")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 250}), encoding="utf-8")
    checksums = {item.relative_to(path).as_posix(): sha256_file(item)
                 for item in path.rglob("*") if item.is_file()}
    (path / "metadata.json").write_text(json.dumps({
        "checkpoint_complete": True, "model": MODEL, "model_revision": REVISION,
        "stage": "main_a_1k", "lineage": ["main_a_1k"], "global_step": 250,
        "file_sha256": checksums,
    }), encoding="utf-8")
    return adapter


def test_checkpoint_and_adapter_identity_validate_files_revision_and_fingerprint(tmp_path):
    adapter = _fake_checkpoint(tmp_path / "first")
    assert check_checkpoint(adapter.parent)["global_step"] == 250
    identity = adapter_identity(adapter, base_model=MODEL, base_revision=REVISION)
    assert identity["training_cumulative_stage"] == "main_a_1k"
    assert identity["source_checkpoint_lineage"] == ["main_a_1k"]
    assert identity["adapter_config_fingerprint"] == sha256_file(adapter / "adapter_config.json")
    second = _fake_checkpoint(tmp_path / "second", weights=b"changed")
    assert adapter_identity(second, base_model=MODEL, base_revision=REVISION)[
        "adapter_fingerprint"] != identity["adapter_fingerprint"]
    with pytest.raises(ValueError, match="revision"):
        adapter_identity(adapter, base_model=MODEL, base_revision="wrong")
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        check_checkpoint(adapter.parent)


def test_base_and_adapter_eval_manifests_cannot_resume_each_other(tmp_path):
    common = dict(run_id="same-id", model_name_or_path=MODEL, model_revision=REVISION,
                  inference_config_fingerprint="formal-config", dataset_path=tmp_path / "eval.parquet",
                  dataset_identity={"sha256": "fixed"}, start=0, limit=1,
                  max_agent_turns=16, search_config_fingerprint="search",
                  layout_config_fingerprint="layout", created_at="2026-01-01T00:00:00+00:00")
    base = create_run_manifest(**common)
    adapter = create_run_manifest(**common, adapter={"adapter_fingerprint": "new"})
    assert "adapter_identity" not in base
    assert "adapter_identity" in manifest_mismatches(base, adapter)
    assert base["run_config_fingerprint"] != adapter["run_config_fingerprint"]


class _FakeModel:
    def __init__(self):
        self.training = True
        self.grad_enabled = True

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, enabled):
        self.grad_enabled = enabled
        return self


class _FakeModelClass:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return _FakeModel()


class _FakeProcessorClass:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return object()


class _FakePeft:
    adapter_path = None

    @classmethod
    def from_pretrained(cls, model, adapter_path, **kwargs):
        cls.adapter_path = adapter_path
        assert kwargs == {"is_trainable": False}
        return model


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch:
    __version__ = "fake"
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"
    cuda = _FakeCuda()
    version = SimpleNamespace(cuda=None)


def test_adapter_loader_is_optional_and_disables_gradients(tmp_path):
    config = replace(load_inference_config(ROOT / "configs/eval_base_300.yaml"), device="cpu")
    common = dict(model_class=_FakeModelClass, processor_class=_FakeProcessorClass,
                  torch_module=_FakeTorch, transformers_module=SimpleNamespace(__version__="fake"),
                  peft_model_class=_FakePeft)
    base = load_inference_bundle(config, **common)
    assert "adapter_identity" not in base.environment
    adapter = _fake_checkpoint(tmp_path / "checkpoint")
    adapted = load_inference_bundle(replace(config, adapter_path=adapter), **common)
    assert adapted.model.training is False and adapted.model.grad_enabled is False
    assert _FakePeft.adapter_path == adapter
    assert adapted.environment["adapter_identity"]["training_cumulative_stage"] == "main_a_1k"


def test_inference_config_optional_adapter_section_is_resolved(tmp_path):
    config_path = tmp_path / "configs" / "eval.yaml"
    config_path.parent.mkdir()
    original = (ROOT / "configs/eval_base_300.yaml").read_text(encoding="utf-8")
    config_path.write_text(original + "\nadapter:\n  path: outputs/checkpoint-3k/adapter\n",
                           encoding="utf-8")
    config = load_inference_config(config_path)
    assert config.adapter_path == (tmp_path / "outputs/checkpoint-3k/adapter").resolve()
