from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_sft_vram", ROOT / "scripts/diagnose_sft_vram.py")
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_exact_smoke_index_and_in_memory_canonicalization(tmp_path):
    records = [{"tools": "[]", "conversations": [{"from": "gpt", "value": str(index)}]}
               for index in range(100)]
    config = {"data": {"path": "data/sft_4b_smoke_100.json", "expected_samples": 100}}
    selected = diagnostic.select_official_sample(config, records, project_root=tmp_path)
    assert selected["conversations"][0]["value"] == "85"
    assert selected["_source_tools"] == "[]"
    assert len(selected["tools"]) > 0
    assert records[85]["tools"] == "[]"  # No source-data mutation.
    with pytest.raises(ValueError, match="exactly 100"):
        diagnostic.select_official_sample(config, records[:-1], project_root=tmp_path)
    with pytest.raises(ValueError, match="official 4B smoke file"):
        diagnostic.select_official_sample(
            {"data": {"path": "data/other.json", "expected_samples": 100}},
            records, project_root=tmp_path)


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
