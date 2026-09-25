from __future__ import annotations

import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_sft_vram", ROOT / "scripts/diagnose_sft_vram.py")
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_smoke_default_and_explicit_sample_index_do_not_mutate_records(tmp_path):
    records = [{"tools": "[]", "conversations": [{"from": "gpt", "value": str(index)}]}
               for index in range(100)]
    original = copy.deepcopy(records)
    config = {"data": {"path": "data/sft_4b_smoke_100.json", "expected_samples": 100}}
    path, count = diagnostic.validate_dataset(
        config, "data/sft_4b_smoke_100.json", project_root=tmp_path)
    assert path == (tmp_path / "data/sft_4b_smoke_100.json").resolve() and count == 100
    index, raw, length = diagnostic.select_sample(
        records, sample_index=None, select_longest=False, token_length=lambda _: 17)
    assert (index, length) == (85, 17)
    assert raw["conversations"][0]["value"] == "85"
    index, raw, _ = diagnostic.select_sample(
        records, sample_index=7, select_longest=False, token_length=lambda _: 17)
    assert index == 7 and raw["conversations"][0]["value"] == "7"
    selected = diagnostic.training_record(raw)
    assert selected["_source_tools"] == "[]" and len(selected["tools"]) > 0
    assert records == original
    with pytest.raises(ValueError, match="exactly 100"):
        diagnostic.validate_dataset(
            {"data": {"path": "data/sft_4b_smoke_100.json", "expected_samples": 99}},
            "data/sft_4b_smoke_100.json", project_root=tmp_path)
    with pytest.raises(ValueError, match="official 4B smoke file"):
        diagnostic.validate_dataset(
            {"data": {"path": "data/other.json", "expected_samples": 100}},
            "data/sft_4b_smoke_100.json", project_root=tmp_path)


def test_formal_shard_path_and_longest_processor_selection(tmp_path):
    config = {"data": {"pool_dir": "data/sft_main"}}
    path, count = diagnostic.validate_dataset(
        config, "data/sft_main/main_a_1k.json", project_root=tmp_path)
    assert path == (tmp_path / "data/sft_main/main_a_1k.json").resolve()
    assert count == 1000
    with pytest.raises(ValueError, match="official smoke file or a formal SFT shard"):
        diagnostic.validate_dataset(config, "data/other.json", project_root=tmp_path)
    records = [{"id": index, "tools": "[]"} for index in range(4)]
    original = copy.deepcopy(records)
    lengths = [6, 30977, 8, 30977]
    index, raw, actual = diagnostic.select_sample(
        records, sample_index=None, select_longest=True,
        token_length=lambda record: lengths[record["id"]])
    assert (index, raw["id"], actual) == (1, 1, 30977)  # Stable first tie.
    assert records == original
    with pytest.raises(ValueError, match="mutually exclusive"):
        diagnostic.select_sample(records, sample_index=1, select_longest=True,
                                 token_length=lambda _: 1)


def test_processor_length_uses_untruncated_multimodal_template(monkeypatch):
    monkeypatch.setattr(diagnostic, "build_messages",
                        lambda record, path: ([{"role": "user"}], ["real image"], [{"name": "tool"}]))
    monkeypatch.setattr(diagnostic, "render_prompt",
                        lambda processor, messages, tools: "rendered official prompt")
    def processor(**kwargs):
        assert kwargs == {"text": ["rendered official prompt"], "images": [["real image"]],
                          "padding": False, "truncation": False, "return_tensors": "pt"}
        return {"input_ids": [[1] * 31]}
    assert diagnostic.processor_token_length(processor, {}, Path("data/shard.json")) == 31


def test_formal_record_canonicalization_preserves_source_metadata():
    raw = {"tools": "[]", "_source_tools": "original declarations", "_sample_id": "fvqa:1"}
    original = copy.deepcopy(raw)
    prepared = diagnostic.training_record(raw)
    assert raw == original
    assert prepared is not raw
    assert prepared["_source_tools"] == "original declarations"
    assert len(prepared["tools"]) > 0


def test_backward_switch_and_pinned_attention_configuration():
    assert diagnostic.should_run_backward(False, None) is False
    assert diagnostic.should_run_backward(True, 1.25) is True
    with pytest.raises(ValueError, match="finite forward loss"):
        diagnostic.should_run_backward(True, float("nan"))
    formal = yaml.safe_load((ROOT / "configs/sft_main.yaml").read_text(encoding="utf-8"))
    evaluation = yaml.safe_load((ROOT / "configs/eval_base_300.yaml").read_text(encoding="utf-8"))
    assert formal["model"]["attn_implementation"] == "flash_attention_2"
    assert formal["training"]["per_device_train_batch_size"] == 1
    assert formal["training"]["gradient_accumulation_steps"] == 4
    assert evaluation["model"]["attn_implementation"] == "sdpa"


def test_find_exact_language_decoder_stack():
    stack = SimpleNamespace(layers=[object() for _ in range(36)])
    model = SimpleNamespace(named_modules=lambda: iter([
        ("", SimpleNamespace()), ("base_model.model.model.language_model", stack)]))
    path, found, layers = diagnostic.find_language_decoder_layers(model)
    assert path.endswith("language_model") and found is stack and len(layers) == 36
    with pytest.raises(RuntimeError, match="36-layer"):
        diagnostic.find_language_decoder_layers(
            SimpleNamespace(named_modules=lambda: iter([("language_model", SimpleNamespace(layers=[]))])))


def test_parameter_and_attention_reports_do_not_mutate_model():
    class Parameter:
        def __init__(self, size, dtype, trainable):
            self.size, self.dtype, self.requires_grad = size, dtype, trainable

        def numel(self):
            return self.size

    params = [("base.weight", Parameter(100, "bf16", False)),
              ("lora_A.weight", Parameter(8, "fp32", True))]
    text = SimpleNamespace(_attn_implementation="sdpa", use_cache=False,
                           output_hidden_states=False, output_attentions=False)
    config = SimpleNamespace(_attn_implementation="sdpa", text_config=text,
                             use_cache=False, output_hidden_states=False,
                             output_attentions=False)
    language = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flash_attention_2"),
                               gradient_checkpointing=True, training=False)
    model = SimpleNamespace(named_parameters=lambda: iter(params), config=config,
                            is_gradient_checkpointing=True, training=False)
    summary = diagnostic.parameter_summary(model)
    assert summary == {"dtype_parameter_counts": {"bf16": 100, "fp32": 8},
                       "total_parameters": 108, "trainable_parameters": 8,
                       "lora_trainable_parameters": 8}
    flags = diagnostic.resolved_model_flags(model, language, "flash_attention_2")
    assert flags["effective_attention_implementation"] == "flash_attention_2"
    assert flags["attention_matches_request"] is True
    assert flags["language_model_gradient_checkpointing"] is True
    assert flags["language_model_training"] is False
    assert diagnostic.resolved_model_flags(model, language, "sdpa")["attention_matches_request"] is False


def test_layer_hooks_track_enter_exit_and_remove_cleanly(capsys):
    class Handle:
        removed = False

        def remove(self):
            self.removed = True

    class Layer:
        def register_forward_pre_hook(self, callback):
            self.before = callback
            return Handle()

        def register_forward_hook(self, callback):
            self.after = callback
            return Handle()

    cuda = SimpleNamespace(memory_allocated=lambda device: 2 * diagnostic.GIB,
                           memory_reserved=lambda device: 3 * diagnostic.GIB,
                           max_memory_allocated=lambda device: 4 * diagnostic.GIB)
    torch = SimpleNamespace(cuda=cuda)
    layers = [Layer() for _ in range(36)]
    handles, state = diagnostic.register_layer_hooks(layers, torch, "cuda:0")
    hidden = SimpleNamespace(shape=(1, 16723, 2560))
    layers[4].before(layers[4], (hidden,))
    assert state["last_entered"]["layer"] == 4
    assert state["last_entered"]["hidden_states_shape"] == [1, 16723, 2560]
    assert state["last_exited"] is None  # An OOM here has no successful exit.
    layers[4].after(layers[4], (hidden,), hidden)
    assert state["last_exited"]["layer"] == 4
    assert state["last_exited"]["max_allocated_gib"] == 4.0
    assert "[LAYER] enter" in capsys.readouterr().out
    for handle in handles:
        handle.remove()
    assert len(handles) == 72 and all(handle.removed for handle in handles)
