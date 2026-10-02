"""CPU-only provenance/static handoff tests; not a real GPU Gate PASS."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.data import SFT_INPUT_MESSAGE_VERSION
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.inference.adapter import adapter_file_identity
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, REQUIRED_CHECKS, gate_identity, load_gate_config
from opensearch_vl_repro.rl.checkpoint import build_rl_lineage
from opensearch_vl_repro.rl.rollout_sync import (
    check_disk_space, merge_actor_adapter, merge_identity, require_plain_merged_model,
    prepare_qwen_vl_processor_inputs, required_merge_space, validate_actor_adapter, validate_merged_files,
)
from opensearch_vl_repro.sft_tool_audit import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def actor_artifacts(tmp_path):
    sft = yaml.safe_load((ROOT / "configs/sft_main_imageid_v3.yaml").read_text(encoding="utf-8"))
    rl = yaml.safe_load((ROOT / "configs/rl_main.yaml").read_text(encoding="utf-8"))
    source = tmp_path / "checkpoint-3k/adapter"
    source.mkdir(parents=True)
    lora = sft["lora"]
    ac = {"base_model_name_or_path": BASE_MODEL, "peft_type": "LORA", "r": lora["rank"],
          "lora_alpha": lora["alpha"], "lora_dropout": lora["dropout"], "target_modules": lora["target_modules"]}
    (source / "adapter_config.json").write_text(json.dumps(ac), encoding="utf-8")
    (source / "adapter_model.safetensors").write_bytes(b"CPU fixture source weights, not GPU evidence")
    metadata = {"checkpoint_complete": True, "stage_complete": True, "model": BASE_MODEL,
                "model_revision": BASE_REVISION, "stage": "main_b_2k", "lineage": ["main_a_1k", "main_b_2k"],
                "sft_input_message_version": SFT_INPUT_MESSAGE_VERSION,
                "runtime_tool_protocol_version": RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
                "file_sha256": {f"adapter/{p.name}": sha256_file(p) for p in source.iterdir()}}
    (source.parent / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    lineage = build_rl_lineage(config=rl, sft_config=sft, adapter_path=source, run_id="fixture-a22")
    actor = tmp_path / "a22/adapter"
    actor.mkdir(parents=True)
    (actor / "adapter_config.json").write_text(json.dumps(ac), encoding="utf-8")
    (actor / "adapter_model.safetensors").write_bytes(b"CPU fixture updated actor weights, not GPU evidence")
    artifact = adapter_file_identity(actor)
    identity = gate_identity(gate=load_gate_config(ROOT / "configs/rl_gate_a22.yaml"), sft=sft,
                             lineage=lineage.to_dict(), data_manifest={"manifest_sha256": "d" * 64},
                             sample_ids=["rl_000123", "rl_000456"], world_size=2, versions={"verl": "0.6.1"}, seed=20260506)
    hashes = {"adapter/" + k: v for k, v in artifact["file_sha256"].items()}
    manifest = {"passed": True, "fsdp_mode": "fsdp2", "world_size": 2, "model": BASE_MODEL,
                "revision": BASE_REVISION, "identity": identity,
                "input_adapter_fingerprint": lineage.sft_adapter_fingerprint,
                "output_adapter_fingerprint": artifact["adapter_fingerprint"], "checkpoint_file_sha256": hashes,
                "checkpoint_fingerprint": canonical_json_sha256(hashes),
                "checks": {**dict.fromkeys(REQUIRED_CHECKS, True), "all_ranks_report_success": True}}
    path = actor.parent / "gate_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"adapter": actor, "gate_manifest": path, "rl_config": rl, "sft_config": sft, "source_sft_adapter": source}


def test_a22_adapter_and_explicit_sft_fallback_validate_without_mutation(actor_artifacts):
    files = {p: p.read_bytes() for p in actor_artifacts["adapter"].parent.rglob("*") if p.is_file()}
    result = validate_actor_adapter(**actor_artifacts)
    assert result["actor_source_kind"] == "a22_temporary_updated_actor"
    assert result["formal_rl_initialization_allowed"] is False
    assert result["actor_adapter_fingerprint"] != result["source_sft_adapter_fingerprint"]
    fallback = {**actor_artifacts, "gate_manifest": None, "adapter": actor_artifacts["source_sft_adapter"]}
    assert validate_actor_adapter(**fallback)["actor_source_kind"] == "formal_sft_checkpoint_fallback"
    assert {p: p.read_bytes() for p in files} == files
    # An A2.2 adapter cannot masquerade as formal checkpoint-3k simply by omitting its manifest.
    with pytest.raises(ValueError):
        validate_actor_adapter(**{**actor_artifacts, "gate_manifest": None})


@pytest.mark.parametrize("field,value", [("passed", False), ("passed", 1), ("fsdp_mode", "fsdp1"),
                                         ("model", "wrong"), ("revision", "wrong"),
                                         ("input_adapter_fingerprint", "0" * 64),
                                         ("output_adapter_fingerprint", "0" * 64), ("world_size", 1)])
def test_actor_manifest_fail_closed(actor_artifacts, field, value):
    path = actor_artifacts["gate_manifest"]
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_actor_adapter(**actor_artifacts)


def test_actor_corrupted_weights_rejected(actor_artifacts):
    (actor_artifacts["adapter"] / "adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        validate_actor_adapter(**actor_artifacts)


def test_merge_identity_binds_weights_lineage_versions_but_not_machine_paths(actor_artifacts):
    actor = validate_actor_adapter(**actor_artifacts)
    kwargs = {"actor": actor, "versions": {"torch": "2.8.0", "vllm": "0.11.0"},
              "file_hashes": {"config.json": "c" * 64, "model.safetensors": "w" * 64}}
    identity = merge_identity(**kwargs)
    relocated = {**actor, "runtime_locators": {"adapter": "Z:/different"},
                 "actor_gate_manifest_sha256": "0" * 64, "created_at_unix": 123}
    assert merge_identity(**{**kwargs, "actor": relocated}) == identity
    for name, value in (("actor_adapter_fingerprint", "a" * 64), ("source_sft_adapter_fingerprint", "b" * 64)):
        assert merge_identity(**{**kwargs, "actor": {**actor, name: value}}) != identity
    assert merge_identity(**{**kwargs, "versions": {"torch": "2.8.1"}}) != identity
    assert merge_identity(**{**kwargs, "file_hashes": {"config.json": "d" * 64}}) != identity
    assert "Z:/" not in json.dumps(identity) and identity["formal_rl_initialization_allowed"] is False


def test_static_checkpoint_files_and_plain_model_constraints(tmp_path):
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / name).write_bytes(b"fixture")
    assert len(validate_merged_files(tmp_path)) == 4
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="PEFT"):
        validate_merged_files(tmp_path)
    (tmp_path / "adapter_config.json").unlink()
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="unsafe"):
        validate_merged_files(tmp_path)
    class Peft:
        pass
    require_plain_merged_model(SimpleNamespace(named_parameters=lambda: [("weight", None)]), Peft)
    for model in (Peft(), SimpleNamespace(_hf_peft_config_loaded=True),
                  SimpleNamespace(named_parameters=lambda: [("x.lora_A", None)])):
        with pytest.raises(RuntimeError, match="active"):
            require_plain_merged_model(model, Peft)


def test_disk_space_estimate_uses_actual_base_size(tmp_path, monkeypatch):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir(); adapter.mkdir()
    (base / "model.safetensors").write_bytes(b"x" * 100)
    (adapter / "adapter_model.safetensors").write_bytes(b"x" * 10)
    assert required_merge_space(base, adapter) == 210
    monkeypatch.setattr("opensearch_vl_repro.rl.rollout_sync.shutil.disk_usage", lambda _: SimpleNamespace(free=209))
    with pytest.raises(OSError, match="insufficient"):
        check_disk_space(base, adapter, tmp_path)


def test_merge_refuses_existing_output_and_source_overlap_before_heavy_imports(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir(); adapter.mkdir()
    kwargs = dict(base_snapshot=base, adapter=adapter, actor={}, sft_config={}, versions={},
                  validation_messages=[], tools=[])
    with pytest.raises(FileExistsError):
        merge_actor_adapter(**kwargs, output=base)
    for output in (base / "new", adapter / "new"):
        with pytest.raises(ValueError, match="overlap"):
            merge_actor_adapter(**kwargs, output=output)


def test_static_handoff_source_enforces_fresh_reload_and_hf_destruction():
    import inspect
    source = inspect.getsource(merge_actor_adapter)
    assert "merge_and_unload(safe_merge=True)" in source
    assert source.index("if ref() is not None") < source.index("fresh = load_base_model")
    assert source.index("if fresh_ref() is not None") < source.index("staging.rename(output)")
    assert "local_files_only=True" in source and "torch.inference_mode()" in source
    assert "LoRARequest" not in source and "enable_lora" not in source


@pytest.mark.parametrize("image_count", [1, 2])
def test_two_stage_processor_regression_preserves_system_string_and_pil_order(image_count):
    images = [Image.new("RGB", (8 + i, 6), color) for i, color in enumerate(["blue", "red"][:image_count])]
    messages = [{"role": "system", "content": "system string"},
                {"role": "user", "content": [*({"type": "image", "image": image} for image in images),
                                              {"type": "text", "text": "question"}]},
                {"role": "tool", "content": "unchanged tool string"}]
    before = [{**message, "content": [dict(part) for part in message["content"]]
               if isinstance(message["content"], list) else message["content"]} for message in messages]
    tools, calls = [{"type": "function", "function": {"name": "fixture"}}], []
    batch = {"input_ids": SimpleNamespace(shape=(1, 8))}
    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            if kwargs.get("tokenize") is True:
                # Exact offending traversal from Transformers 4.57.1.
                for message in conversation:
                    [content for content in message["content"] if content["type"] in ["image", "video"]]
            assert conversation is messages
            assert kwargs == {"tools": tools, "tokenize": False, "add_generation_prompt": True}
            assert kwargs["tools"] is tools and conversation[0]["content"] == "system string"
            calls.append("render")
            return "CPU rendered prompt"
        def __call__(self, **kwargs):
            assert set(kwargs) == {"text", "images", "return_tensors", "truncation"}
            assert kwargs["text"] == ["CPU rendered prompt"]
            assert len(kwargs["images"]) == 1
            assert all(a is b for a, b in zip(kwargs["images"][0], images, strict=True))
            assert kwargs["return_tensors"] == "pt" and kwargs["truncation"] is False
            calls.append("encode")
            return batch
    processor = Processor()
    with pytest.raises(TypeError, match="string indices"):
        processor.apply_chat_template(messages, tokenize=True)
    prompt, actual, inputs = prepare_qwen_vl_processor_inputs(processor, messages, tools)
    assert calls == ["render", "encode"] and prompt == "CPU rendered prompt" and inputs is batch
    assert all(a is b for a, b in zip(actual, images, strict=True))
    assert messages == before and isinstance(messages[0]["content"], str)
    assert messages[-1]["content"] == "unchanged tool string"


@pytest.mark.parametrize("visual", [None, "image.png", "https://example.com/image.png", "data:image/png;base64,AAAA", b"pixels"])
def test_processor_non_pil_images_fail_closed_before_render(visual):
    processor = SimpleNamespace(apply_chat_template=lambda *a, **k: pytest.fail("must reject before render"))
    messages = [{"role": "system", "content": "system string"},
                {"role": "user", "content": [{"type": "image", "image": visual}]}]
    with pytest.raises(RuntimeError, match="actual PIL"):
        prepare_qwen_vl_processor_inputs(processor, messages, [])


def test_processor_zero_images_and_image_limit_fail_closed():
    processor = SimpleNamespace(apply_chat_template=lambda *a, **k: pytest.fail("must reject before render"))
    with pytest.raises(RuntimeError, match="at least one"):
        prepare_qwen_vl_processor_inputs(processor, [{"role": "system", "content": "system string"}], [])
    images = [{"type": "image", "image": Image.new("RGB", (2, 2))} for _ in range(2)]
    with pytest.raises(RuntimeError, match="limit"):
        prepare_qwen_vl_processor_inputs(processor, [{"role": "user", "content": images}], [], max_images=1)


def test_hf_and_vllm_use_the_same_processor_materialization_helper():
    import inspect
    from opensearch_vl_repro.rl import rollout_gate, rollout_sync
    assert rollout_gate.prepare_qwen_vl_processor_inputs is rollout_sync.prepare_qwen_vl_processor_inputs
    for function in (merge_actor_adapter, rollout_gate.VLLMStaticBackend.generate):
        source = inspect.getsource(function)
        assert "prepare_qwen_vl_processor_inputs(" in source
        assert ".apply_chat_template(" not in source  # neither has a divergent rendering path


def test_actual_peft_static_merge_fresh_plain_reload_on_tiny_cpu_model(tmp_path, monkeypatch):
    """Real PEFT/safetensors, tiny CPU-only fixture; no Qwen/GPU PASS claim."""
    import contextlib
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file
    from opensearch_vl_repro import model as project_model
    class Config(dict):
        def to_dict(self):
            return dict(self)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2, bias=False)
            self.config = Config(model_type="tiny_cpu_fixture")
        def forward(self, input_ids=None, **kwargs):
            return SimpleNamespace(logits=self.q_proj(torch.ones(1, 2, dtype=self.q_proj.weight.dtype)))
        def save_pretrained(self, directory, **kwargs):
            directory = Path(directory)
            (directory / "config.json").write_text(json.dumps(dict(self.config)))
            save_file(self.state_dict(), directory / "model.safetensors")
    initial = Image.new("RGB", (8, 6), "blue")
    derived = initial.crop((2, 1, 6, 4))
    messages = [{"role": "system", "content": "system string"},
                {"role": "user", "content": [{"type": "image", "image": initial}, {"type": "text", "text": "question"}]},
                {"role": "tool", "content": [{"type": "image", "image": derived}, {"type": "text", "text": "observation"}]}]
    tools, processor_calls = [{"type": "function", "function": {"name": "crop"}}], []
    class Processor:
        tokenizer = SimpleNamespace(name_or_path="fixture", init_kwargs={})
        def save_pretrained(self, directory):
            for name in ("preprocessor_config.json", "tokenizer_config.json"):
                (Path(directory) / name).write_text("{}")
        def apply_chat_template(self, conversation, **kwargs):
            assert conversation is messages and conversation[0]["content"] == "system string"
            assert kwargs == {"tools": tools, "tokenize": False, "add_generation_prompt": True}
            processor_calls.append("render")
            return "CPU rendered multimodal prompt"
        def __call__(self, **kwargs):
            assert kwargs == {"text": ["CPU rendered multimodal prompt"], "images": [[initial, derived]],
                              "return_tensors": "pt", "truncation": False}
            assert kwargs["images"][0][0] is initial and kwargs["images"][0][1] is derived
            processor_calls.append("encode")
            return {"input_ids": torch.tensor([[1, 2]])}
    base, adapter, output = tmp_path / "base", tmp_path / "adapter", tmp_path / "merged"
    base.mkdir()
    original = Tiny()
    original.save_pretrained(base)
    wrapped = get_peft_model(original, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"]))
    for name, parameter in wrapped.named_parameters():
        if "lora_B" in name:
            parameter.data.fill_(.2)
    wrapped.save_pretrained(adapter)
    del wrapped, original
    protected_before = {p: p.read_bytes() for directory in (base, adapter) for p in directory.iterdir() if p.is_file()}
    loaded = []
    def load(config, **kwargs):
        path = Path(config["model"]["name_or_path"])
        loaded.append(path)
        model = Tiny().to(dtype=torch.bfloat16)
        model.load_state_dict(load_file(path / "model.safetensors"))
        return model
    real_to = torch.nn.Module.to
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self if a == ("cuda:0",) else real_to(self, *a, **k))
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(project_model, "load_base_model", load)
    monkeypatch.setattr(project_model, "load_processor", lambda *a, **k: Processor())
    monkeypatch.setattr(project_model, "move_batch", lambda batch, device: batch)
    actor = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION,
             "actor_adapter_fingerprint": adapter_file_identity(adapter)["adapter_fingerprint"],
             "source_sft_adapter_fingerprint": "s" * 64, "source_sft_lineage": ["main_a_1k", "main_b_2k"],
             "actor_source_kind": "a22_temporary_updated_actor", "actor_gate_identity_sha256": "a" * 64}
    manifest = merge_actor_adapter(base_snapshot=base, adapter=adapter, actor=actor,
        sft_config={"model": {}}, output=output, versions={"test": "tiny_cpu"}, validation_messages=messages, tools=tools)
    assert processor_calls == ["render", "encode"]
    assert all(manifest[k] is True for k in ("merge_complete", "fresh_hf_forward_finite", "no_active_peft", "merge_hf_destroyed", "reload_hf_destroyed"))
    assert len(loaded) == 2 and loaded[0] == base and loaded[1] != base
    assert output.is_dir() and not loaded[1].exists()  # atomic staging publication
    assert not (output / "adapter_config.json").exists()
    assert (output / "merge_manifest.json").is_file()
    assert not any("lora_" in key for key in load_file(output / "model.safetensors"))
    assert {p: p.read_bytes() for p in protected_before} == protected_before
