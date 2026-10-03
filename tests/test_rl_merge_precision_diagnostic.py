"""CPU fixtures: diagnostic precision/provenance, never a real GPU/Gate result."""
import copy
import inspect
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from opensearch_vl_repro.rl import merge_precision_diagnostic as diag
from opensearch_vl_repro.rl import policy_handoff_diagnostic as handoff
from opensearch_vl_repro.rl.actor_gate import atomic_json
from test_rl_policy_handoff_diagnostic import attempt, Tokenizer
from test_rl_rollout_sync import actor_artifacts

ROOT = Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        for suffix in sorted(diag.TARGETS):
            linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            linear.weight.data.fill_(1.)
            setattr(layer, suffix, linear)
        self.layers = torch.nn.ModuleList([layer])
        self.config = SimpleNamespace(_attn_implementation="flash_attention_2",
                                      to_dict=lambda: {"model_type": "tiny"})

    def save_pretrained(self, path, *, safe_serialization, max_shard_size):
        assert safe_serialization and max_shard_size == "2GB"
        path = Path(path)
        save_file(self.state_dict(), str(path / "model.safetensors"))
        atomic_json(path / "config.json", {"model_type": "tiny"})


def tiny_peft():
    from peft import LoraConfig, get_peft_model
    wrapped = get_peft_model(TinyModel(), LoraConfig(r=1, lora_alpha=1,
        target_modules=sorted(diag.TARGETS), bias="none"))
    with torch.no_grad():
        for _, module, _ in diag.lora_roster(wrapped, expected_layers=1):
            module.lora_A["default"].weight.fill_(.001)
            module.lora_B["default"].weight.fill_(1.)
    return wrapped


def test_real_peft_fp32_merge_actual_dtype_and_known_small_delta():
    wrapped = tiny_peft()
    roster = diag.lora_roster(wrapped, expected_layers=1)
    expected = {key: m.get_base_layer().weight.detach().float().clone()
                + m.get_delta_weight("default").detach().float() for _, m, key in roster}
    initial = diag.lora_dtype_audit(wrapped, expected_layers=1)
    assert initial["by_suffix"]["q_proj"]["lora_A_dtype"] == {"torch.float32": 1}
    merged, stats, audit = diag.fp32_merge_lora_model(wrapped, expected_layers=1)
    assert not any("lora_" in name for name, _ in merged.named_parameters())
    assert audit["merge_device"] == "cpu" and audit["actual_fp32_delta_merge_call_count"] == 7
    assert audit["target_dtype_counts"]["computed_delta_dtype"] == {"torch.float32": 7}
    assert audit["target_dtype_counts"]["base_target_weight_dtype_at_merge"] == {"torch.float32": 7}
    assert audit["merged_parameter_dtypes_before_cast"]["tensor_count_by_dtype"] == {"torch.float32": 7}
    for name, parameter in merged.named_parameters():
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter, expected[diag.module_key(name)])
        # No assumption that FP32->BF16 differs from direct BF16 in every case.
        direct = torch.ones_like(parameter, dtype=torch.bfloat16)
        direct.add_(torch.full_like(parameter, .001).to(torch.bfloat16))
        assert torch.equal(parameter.to(torch.bfloat16), direct)
    assert all(r["unchanged_despite_delta_fraction"] == 1. for r in stats["per_module"])
    assert set(stats["by_suffix"]) == diag.TARGETS
    assert not any("get_delta_weight" in m.__dict__ for _, m, _ in roster)


def test_quantization_stats_nonzero_denominator_and_chunked_residual():
    weight = torch.ones(4, dtype=torch.bfloat16)
    delta = torch.tensor([0., .001, .25, -.001], dtype=torch.float32)
    stats = diag.quantization_stats(weight, delta, chunk_size=1)
    target = weight.float() + delta
    error = target.to(torch.bfloat16).float() - target
    assert stats["delta_abs_mean"] == pytest.approx(delta.abs().double().mean().item())
    assert stats["quant_error_abs_mean"] == pytest.approx(error.abs().double().mean().item())
    assert stats["fraction_delta_rounded_to_zero_effectively"] == .75
    assert stats["nonzero_delta_count"] == 3
    assert stats["fraction_merged_weight_unchanged_despite_nonzero_delta"] == pytest.approx(2 / 3)
    zero = diag.quantization_stats(weight, torch.zeros(4))
    assert zero["unchanged_despite_delta_fraction"] is None
    assert zero["quant_error_mean_relative_to_delta"] is None
    assert zero["fraction_delta_rounded_to_zero_effectively"] == 1.
    json.dumps(stats, allow_nan=False)


def test_suffix_summary_is_element_weighted_and_counts_nonzero_deltas():
    records = []
    for layer, delta in enumerate((torch.tensor([.001]), torch.tensor([.25, .25, .25]))):
        records.append({"layer_index": layer, "target_suffix": "q_proj",
            **diag.quantization_stats(torch.ones_like(delta, dtype=torch.bfloat16), delta)})
    stats = diag.suffix_summary(records)["q_proj"]
    assert stats["element_count"] == 4 and stats["module_count"] == 2
    assert stats["delta_mean_abs"] == pytest.approx((float(torch.tensor(.001)) + .75) / 4)
    assert stats["unchanged_fraction"] == .25


@pytest.mark.parametrize("invalid", ["nonfinite", "wrong_dtype", "wrong_shape", "chunk"])
def test_quantization_stats_fail_closed(invalid):
    w, d, chunk = torch.ones(4, dtype=torch.bfloat16), torch.zeros(4), 1
    if invalid == "nonfinite": d[0] = float("nan")
    if invalid == "wrong_dtype": w = w.float()
    if invalid == "wrong_shape": d = torch.zeros(5)
    if invalid == "chunk": chunk = 0
    with pytest.raises(ValueError): diag.quantization_stats(w, d, chunk_size=chunk)


@pytest.mark.parametrize("invalid", ["roster", "base_dtype", "variant", "dora"])
def test_fp32_merge_fail_closed_for_nonoriginal_representation(invalid):
    wrapped = tiny_peft()
    module = diag.lora_roster(wrapped, expected_layers=1)[0][1]
    if invalid == "base_dtype": module.get_base_layer().float()
    if invalid == "variant": module.lora_variant = {"default": object()}
    if invalid == "dora": module.use_dora["default"] = True
    with pytest.raises(ValueError):
        diag.fp32_merge_lora_model(wrapped, expected_layers=2 if invalid == "roster" else 1)


def test_dtype_header_checks_actual_serialization_not_load_override(tmp_path):
    save_file({"w": torch.ones(2, dtype=torch.bfloat16), "counter": torch.ones(1, dtype=torch.int64)},
              str(tmp_path / "model.safetensors"))
    assert diag.saved_dtype_counts(tmp_path)["tensor_count_by_dtype"] == {"I64": 1, "BF16": 1}
    save_file({"w": torch.ones(2, dtype=torch.float32)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not BF16"): diag.saved_dtype_counts(tmp_path)


@pytest.mark.parametrize("failure", [None, "save", "reload", "serialized_dtype"])
def test_staged_real_cpu_peft_save_fresh_reload_and_cleanup(tmp_path, monkeypatch, failure):
    from opensearch_vl_repro import model as formal_model
    base, assets, dest = (tmp_path / name for name in ("base", "old_merge", "diagnostic"))
    for directory in (base, assets, dest): directory.mkdir()
    (base / "model.safetensors").write_bytes(b"tiny size estimate")
    for name in ("preprocessor_config.json", "tokenizer_config.json"): atomic_json(assets / name, {})
    ctx = {"adapter": base, "merged": assets,
           "sft": {"model": {"attn_implementation": "flash_attention_2"}},
           "identity": {"source_sft_actor": {"source_sft_adapter_fingerprint": "a" * 64}}}
    references = []
    def load_cpu(*args):
        wrapped = tiny_peft(); references.append(weakref.ref(wrapped)); return wrapped
    monkeypatch.setattr(diag, "load_cpu_peft", load_cpu)
    real_merge = diag.fp32_merge_lora_model
    monkeypatch.setattr(diag, "fp32_merge_lora_model", lambda m: real_merge(m, expected_layers=1))
    original_save = TinyModel.save_pretrained
    def save(self, path, **kw):
        if failure == "save": raise OSError("partial save fixture")
        original_save(self, path, **kw)
        if failure == "serialized_dtype":
            save_file({k: v.float() for k, v in self.state_dict().items()}, str(Path(path) / "model.safetensors"))
    monkeypatch.setattr(TinyModel, "save_pretrained", save)
    def reload(config, *, for_training):
        assert not for_training and all(r() is None for r in references)
        path = Path(config["model"]["name_or_path"])
        assert path.name.startswith(".fp32-merge-") and not (dest / "tmp_fp32_merge").exists()
        if failure == "reload": raise ValueError("fresh reload fixture")
        result = TinyModel(); result.load_state_dict(load_file(str(path / "model.safetensors")))
        return result
    monkeypatch.setattr(formal_model, "load_base_model", reload)
    if failure:
        with pytest.raises((ValueError, OSError)):
            diag.diagnostic_fp32_merge(ctx, base, dest, torch.device("cpu"))
        assert not (dest / "tmp_fp32_merge").exists()  # never published half-model
        diag.cleanup_models(dest)
        assert not list(dest.iterdir())
    else:
        result, path, stats, audit = diag.diagnostic_fp32_merge(ctx, base, dest, torch.device("cpu"))
        assert audit["fresh_reload_completed"] and audit["cpu_merge_model_destroyed"]
        assert audit["serialized_tensor_dtypes"]["tensor_count_by_dtype"] == {"BF16": 7}
        assert audit["reload_parameter_dtypes"]["tensor_count_by_dtype"] == {"torch.bfloat16": 7}
        assert len(audit["diagnostic_model_fingerprint"]) == 64
        assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in result.parameters())
        metadata = json.loads((path / "diagnostic_metadata.json").read_text())
        for field, value in diag.META.items(): assert metadata[field] == value
        assert len(stats["per_module"]) == 7
        assert diag.cleanup_models(dest, keep=True) == [] and path.exists()
        assert diag.cleanup_models(dest) == ["tmp_fp32_merge"] and not path.exists()
    assert (base / "model.safetensors").read_bytes() == b"tiny size estimate"


def test_insufficient_space_fails_before_load_or_staging(tmp_path, monkeypatch):
    base = tmp_path / "base"; base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    dest = tmp_path / "diagnostic"; dest.mkdir()
    monkeypatch.setattr(diag.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    monkeypatch.setattr(diag, "load_cpu_peft", lambda *a: pytest.fail("must not load"))
    with pytest.raises(OSError, match="before model load/save"):
        diag.diagnostic_fp32_merge({"adapter": base}, base, dest, torch.device("cpu"))
    assert not list(dest.iterdir())


def test_full_weight_compare_streaming_per_layer_suffix_and_roster(tmp_path):
    a, b = tmp_path / "B0", tmp_path / "B1"
    a.mkdir(); b.mkdir()
    wa = {f"model.language_model.layers.0.{suffix}.weight": torch.ones(4, dtype=torch.bfloat16)
          for suffix in diag.TARGETS}
    wb = {key: value.clone() for key, value in wa.items()}
    for value in wb.values(): value[0] = 2.
    save_file(wa, str(a / "model.safetensors")); save_file(wb, str(b / "model.safetensors"))
    stats = diag.weight_compare_stats(a, b, expected_layers=1)
    assert len(stats["per_module"]) == 7
    for row in stats["per_module"]:
        assert row["mean_abs_weight_diff"] == .25
        assert row["max_abs_weight_diff"] == 1. and row["nonzero_diff_fraction"] == .25
    assert stats["by_suffix"]["q_proj"]["element_count"] == 4
    with pytest.raises(ValueError, match="roster"): diag.weight_compare_stats(a, b)
    json.dumps(stats, allow_nan=False)  # tensors are never exported


@pytest.fixture
def forensic(attempt):
    ctx = handoff.validate_attempt(attempt.root, attempt.run_id, attempt.base)
    tokens = handoff.token_diagnostics(ctx["rows"], [[-.1, -.2]] * 2, [[-.3, -.4]] * 2, Tokenizer())
    dest = attempt.reports / "policy_handoff_diagnostic"; dest.mkdir()
    summary = {**handoff.METADATA, "execution_succeeded": True, "source_artifacts_unchanged": True,
        "gate_run_id": attempt.run_id, "token_count": len(tokens),
        "formal_sft_adapter_fingerprint": ctx["group"]["identity"]["pre_update_policy_fingerprint"],
        "merged_checkpoint_fingerprint": ctx["group"]["merged_checkpoint_fingerprint"],
        "collection_attempt": ctx["group"]["identity"]["collection_attempt"],
        "dynamic_peft_vs_merged_hf": handoff.pair_metrics([r["dynamic_peft_logprob"] for r in tokens],
                                                       [r["merged_hf_logprob"] for r in tokens])}
    atomic_json(dest / "summary.json", summary)
    (dest / "token_diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tokens), encoding="utf-8")
    return attempt, ctx, tokens


@pytest.mark.parametrize("tamper", ["token_id", "lineage", "metrics", "mask_count"])
def test_old_forensic_bound_to_original_group_without_rewriting(forensic, tamper):
    attempt, ctx, tokens = forensic
    dest = attempt.reports / "policy_handoff_diagnostic"
    summary = json.loads((dest / "summary.json").read_text())
    if tamper == "token_id": tokens[0]["token_id"] = 999
    if tamper == "lineage": summary["collection_attempt"] = "wrong"
    if tamper == "metrics": summary["dynamic_peft_vs_merged_hf"]["mean_abs_logprob_diff"] = 0.
    if tamper == "mask_count": tokens.pop()
    atomic_json(dest / "summary.json", summary)
    (dest / "token_diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tokens), encoding="utf-8")
    before = handoff.source_checksums([dest], [])
    with pytest.raises(ValueError): diag.load_previous_forensic(ctx)
    handoff.assert_sources_unchanged(before, handoff.source_checksums([dest], []))


def test_pair_improvement_original_outliers_and_unchanged_token_mask(forensic):
    _, ctx, prior = forensic
    before = copy.deepcopy(ctx["rows"])
    records, analysis = diag.analyze_tokens(ctx["rows"], prior,
        [[-.1, -.2]] * 2, [[-.2, -.3]] * 2, [[-.12, -.22]] * 2, top_n=50)
    assert analysis["improvements"]["mean_abs_reduction_fraction"] == pytest.approx(.8)
    assert analysis["improvements"]["max_abs_reduction_fraction"] == pytest.approx(.8)
    assert analysis["improvements"]["clip_fraction_reduction_fraction"] is None
    assert set(analysis["comparisons"]) == set(diag.PAIRS)
    assert [r["token_id"] for r in records] == [r["token_id"] for r in prior]
    assert analysis["original_production_outliers"][0]["original_outlier_abs_diff_reduction_fraction"] == pytest.approx(.9)
    assert ctx["rows"] == before
    assert diag.reduction_fraction(.1, .02) == pytest.approx(.8)
    assert diag.reduction_fraction(0., .1) is None
    assert diag.reduction_fraction(.1, .2) == -1.
    with pytest.raises(ValueError): diag.reduction_fraction(float("nan"), .1)
    with pytest.raises(ValueError): diag.analyze_tokens(ctx["rows"], prior, [[-.1]] * 2, [[-.2]] * 2, [[-.12]] * 2, top_n=50)
    json.dumps(analysis, allow_nan=False)


@pytest.mark.parametrize("mode", ["success", "keep", "source_mutation", "prior_mutation", "merge_failure", "publication_failure"])
def test_serial_orchestration_readonly_no_overlap_cleanup_no_gate_pass(forensic, monkeypatch, mode):
    attempt, ctx, _ = forensic
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50,
                           local_files_only=True, keep_diagnostic_model=mode == "keep")
    for name in ("is_available", "is_bf16_supported"): monkeypatch.setattr(torch.cuda, name, lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for name in ("set_device", "reset_peak_memory_stats", "empty_cache"): monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("max_memory_allocated", "max_memory_reserved"): monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(diag.importlib.metadata, "version", lambda name: "0.6.1" if name == "verl" else "fixture")
    class Model(torch.nn.Module):
        def __init__(self, kind):
            super().__init__(); self.kind = kind
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
    refs, kinds = [], []
    def new(kind):
        assert all(ref() is None for ref in refs)
        model = Model(kind); refs.append(weakref.ref(model)); kinds.append(kind); return model
    def load(kind, context, base, device):
        assert context["merged"] == attempt.merged  # exact attempt, never regenerated
        return new(kind)
    monkeypatch.setattr(handoff, "load_diagnostic_model", load)
    monkeypatch.setattr(diag, "lora_dtype_audit", lambda m: {"CPU fixture": True})
    def forward(model, rows, directory, device):
        assert directory == attempt.output / "group" and rows == ctx["rows"]
        return {"dynamic": [[-.1, -.2]] * 2, "merged": [[-.3, -.4]] * 2,
                "improved": [[-.12, -.22]] * 2}[model.kind]
    monkeypatch.setattr(handoff, "forward_rows", forward)
    def merge(context, base, destination, device):
        path = destination / (".fp32-merge-failure" if mode == "merge_failure" else "tmp_fp32_merge")
        path.mkdir(); (path / "model.safetensors").write_bytes(b"CPU fixture only")
        if mode == "merge_failure": raise ValueError("partial diagnostic merge")
        if mode == "source_mutation": (attempt.output / "bad-mutation").write_bytes(b"changed")
        if mode == "prior_mutation": (attempt.reports / "policy_handoff_diagnostic/token_diagnostics.jsonl").write_bytes(b"changed")
        return new("improved"), path, {"by_suffix": {}}, {"fresh_reload_completed": True}
    monkeypatch.setattr(diag, "diagnostic_fp32_merge", merge)
    monkeypatch.setattr(diag, "weight_compare_stats", lambda a, b: {"by_suffix": {}})
    real_atomic = diag.atomic_json
    if mode == "publication_failure":
        def publish(path, data):
            if path.name == "weight_compare_stats.json": raise OSError("publication fixture")
            real_atomic(path, data)
        monkeypatch.setattr(diag, "atomic_json", publish)
    protected = [attempt.output, attempt.base, attempt.reports / "policy_handoff_diagnostic"]
    files = [attempt.reports / "gate_c_report.json"]
    before = handoff.source_checksums(protected, files)
    destination = attempt.reports / "merge_precision_diagnostic"
    if mode in ("success", "keep"):
        assert diag.run_diagnostic(args, attempt.root) == 0
        summary = json.loads((destination / "summary.json").read_text())
        assert summary["execution_succeeded"] and summary["token_count"] == 4
        assert summary["models_destroyed"] == dict(dynamic=True, merged=True, fp32_merge_bf16=True)
        assert kinds == ["dynamic", "merged", "improved"] and all(ref() is None for ref in refs)
        assert summary["source_artifacts_unchanged"] and not summary["formal_rl_initialization_allowed"]
        assert len((destination / "token_diagnostics.jsonl").read_text().splitlines()) == 4
        assert (destination / "tmp_fp32_merge").exists() is (mode == "keep")
        handoff.assert_sources_unchanged(before, handoff.source_checksums(protected, files))
        with pytest.raises(FileExistsError): diag.run_diagnostic(args, attempt.root)
    else:
        with pytest.raises((ValueError, RuntimeError, OSError)): diag.run_diagnostic(args, attempt.root)
        summary = json.loads((destination / "summary.json").read_text())
        assert not summary["execution_succeeded"] and not summary["formal_rl_initialization_allowed"]
        assert summary["source_artifacts_unchanged"] is (mode not in ("source_mutation", "prior_mutation"))
        assert not (destination / "tmp_fp32_merge").exists() and not list(destination.glob(".fp32-merge-*"))
    assert not list(attempt.root.rglob("gate_manifest.json"))
    assert not list(destination.glob("*weight*")) or mode in ("success", "keep")


def test_no_cuda_fails_before_model_load(forensic, monkeypatch):
    attempt, _, _ = forensic
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50,
                           local_files_only=True, keep_diagnostic_model=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(handoff, "load_diagnostic_model", lambda *a: pytest.fail("no model load"))
    with pytest.raises(RuntimeError, match="one visible BF16 CUDA"):
        diag.run_diagnostic(args, attempt.root)
    summary = json.loads((attempt.reports / "merge_precision_diagnostic/summary.json").read_text())
    assert not summary["execution_succeeded"] and summary["source_artifacts_unchanged"]


def test_missing_old_forensic_is_not_regenerated(attempt, monkeypatch):
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50,
                           local_files_only=True, keep_diagnostic_model=False)
    monkeypatch.setattr(handoff, "load_diagnostic_model", lambda *a: pytest.fail("no model load"))
    with pytest.raises(FileNotFoundError): diag.run_diagnostic(args, attempt.root)
    assert not (attempt.reports / "policy_handoff_diagnostic").exists()
    # Missing protected inputs fail at the checksum precheck, BEFORE any output.
    assert not (attempt.reports / "merge_precision_diagnostic").exists()


def test_software_mismatch_is_rejected_before_model_load(forensic, monkeypatch):
    attempt, _, _ = forensic
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50,
                           local_files_only=True, keep_diagnostic_model=False)
    for name in ("is_available", "is_bf16_supported"): monkeypatch.setattr(torch.cuda, name, lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(diag.importlib.metadata, "version", lambda name: "different")
    monkeypatch.setattr(handoff, "load_diagnostic_model", lambda *a: pytest.fail("no model load"))
    with pytest.raises(ValueError, match="historical forward environment"):
        diag.run_diagnostic(args, attempt.root)


def test_lazy_import_and_forbidden_operations_absent():
    code = "import sys; import opensearch_vl_repro.rl.merge_precision_diagnostic; assert not any(k in sys.modules for k in ['torch','transformers','peft','verl','vllm','rllm'])"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    source = inspect.getsource(diag) + (ROOT / "scripts/diagnose_rl_merge_precision.py").read_text()
    for forbidden in ("merge_actor_adapter", "VLLMStaticBackend", "LLM(", "SamplingParams", "DeepSeekJudge",
        "live_rewards", "create_phase3_tool_registry", "AgentRuntime.run", "update_policy(", "optimizer.step",
        "save_checkpoint", "prepare_context(", "bind_run(", "finalize(", "gate_manifest.json",
        "apply_chat_template", "load_processor", "tokenizer.encode", "tokenizer("):
        assert forbidden not in source
    assert "merge_and_unload(safe_merge=True)" in source and "handoff.forward_rows" in source
    assert "handoff.pair_metrics" in source and "handoff.validate_attempt" in source
    assert '"HF_HUB_OFFLINE"] = "1"' in source and '"TRANSFORMERS_OFFLINE"] = "1"' in source


def test_fp32_serialized_header_and_wrong_dtype_fail_closed(tmp_path):
    save_file({"w": torch.ones(4)}, str(tmp_path / "model.safetensors"))
    assert diag.saved_dtype_counts(tmp_path, expected="F32")["tensor_count_by_dtype"] == {"F32": 1}
    with pytest.raises(ValueError, match="not BF16"): diag.saved_dtype_counts(tmp_path)
    save_file({"w": torch.ones(4, dtype=torch.bfloat16)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not F32"): diag.saved_dtype_counts(tmp_path, expected="F32")


def test_fp32_merge_save_reload_never_bf16_and_repeated_state_identical(tmp_path, monkeypatch):
    from opensearch_vl_repro import model as formal_model
    base, assets, dest = [tmp_path / n for n in ("base", "assets", "diagnostic")]
    for p in (base, assets, dest): p.mkdir()
    (base / "model.safetensors").write_bytes(b"CPU disk fixture")
    for name in ("preprocessor_config.json", "tokenizer_config.json"): atomic_json(assets / name, {})
    ctx = {"adapter": base, "merged": assets, "sft": {"model": {"attn_implementation": "flash_attention_2"}},
           "identity": {"source_sft_actor": {"source_sft_adapter_fingerprint": "a" * 64}}}
    monkeypatch.setattr(diag, "load_cpu_peft", lambda *a: tiny_peft())
    merge = diag.fp32_merge_lora_model
    monkeypatch.setattr(diag, "fp32_merge_lora_model", lambda m: merge(m, expected_layers=1))
    def reload_bf16(cfg, *, for_training):
        result = TinyModel(); result.load_state_dict(load_file(str(Path(cfg["model"]["name_or_path"]) / "model.safetensors")))
        return result
    def reload_fp32(path, ctx, *, device):
        assert not (dest / "tmp_fp32_merge_fp32").exists()
        result = TinyModel().float()
        weights = load_file(str(path / "model.safetensors"))
        assert all(w.dtype == torch.float32 for w in weights.values())
        result.load_state_dict(weights)
        return result.to(device).eval().requires_grad_(False)
    monkeypatch.setattr(formal_model, "load_base_model", reload_bf16)
    monkeypatch.setattr(diag, "load_fp32_merged_model", reload_fp32)
    bf16, bp, stats, ba = diag.diagnostic_fp32_merge(ctx, base, dest, torch.device("cpu"),
        checkpoint_name="tmp_fp32_merge_bf16", record_merge_source=True)
    del bf16
    fp32, fp, _, fa = diag.diagnostic_fp32_merge(ctx, base, dest, torch.device("cpu"),
        serialized_dtype="fp32", checkpoint_name="tmp_fp32_merge_fp32", record_merge_source=True)
    diag.require_fp32_parameters(fp32)
    assert ba["pre_cast_merge_source_fingerprint"] == fa["pre_cast_merge_source_fingerprint"]
    assert fa["serialized_tensor_dtypes"]["tensor_count_by_dtype"] == {"F32": 7}
    assert fa["reload_parameter_dtypes"]["tensor_count_by_dtype"] == {"torch.float32": 7}
    assert fa["forward_autocast_enabled"] is False and fa["forward_autocast_dtype"] is None
    representation = diag.weight_representation_stats(bp, fp, stats, expected_layers=1)
    for r in representation["per_module"]:
        delta = stats["per_module"][0]
        assert r["mean_abs_representation_error"] == pytest.approx(delta["quant_error_abs_mean"])
        assert r["mean_relative_to_lora_delta"] == pytest.approx(delta["quant_error_mean_relative_to_delta"])
    assert diag.cleanup_models(dest, keep=True) == []
    assert set(diag.cleanup_models(dest)) == {"tmp_fp32_merge_bf16", "tmp_fp32_merge_fp32"}
    assert not list(dest.iterdir())


def test_explicit_fp32_loader_preserves_pins_offline_backend_dtype(tmp_path, monkeypatch):
    import transformers
    captured = {}
    class Loader:
        @staticmethod
        def from_pretrained(path, **kw):
            captured.update(path=path, **kw); return TinyModel().float()
    monkeypatch.setattr(transformers, "Qwen3VLForConditionalGeneration", Loader)
    ctx = {"sft": {"model": {"revision": "pinned", "attn_implementation": "flash_attention_2", "dtype": "bf16"}}}
    model = diag.load_fp32_merged_model(tmp_path, ctx, device=torch.device("cpu"))
    diag.require_fp32_parameters(model)
    assert captured["dtype"] == torch.float32 and captured["revision"] == "pinned"
    assert captured["attn_implementation"] == "flash_attention_2" and captured["local_files_only"]
    assert captured["trust_remote_code"] is False and captured["low_cpu_mem_usage"]


def test_dynamic_fp32_base_and_lora_not_merged(monkeypatch):
    import peft
    from peft import LoraConfig, get_peft_model
    calls = []
    def load(*args, **kw): calls.append(kw); return TinyModel().float()
    monkeypatch.setattr(diag, "load_fp32_merged_model", load)
    def adapter(base, path, **kw):
        assert kw == {"is_trainable": False, "local_files_only": True}
        result = get_peft_model(base, LoraConfig(r=1, target_modules=sorted(diag.TARGETS)))
        for _, m, _ in roster(result, expected_layers=1): m.lora_A["default"].bfloat16()
        return result
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", adapter)
    roster = diag.lora_roster
    monkeypatch.setattr(diag, "lora_roster", lambda m: roster(m, expected_layers=1))
    model = diag.load_dynamic_fp32({"adapter": Path("fixture")}, Path("base"), device=torch.device("cpu"))
    diag.require_fp32_parameters(model)
    assert any("lora_" in name for name, _ in model.named_parameters())
    assert all(not m.merged for _, m, _ in roster(model, expected_layers=1))
    assert calls[0]["attention_backend"] is None


def test_true_fp32_forward_original_inputs_mrope_previous_slice_and_no_autocast(tmp_path):
    from test_rl_policy_handoff_diagnostic import save_tensor_inputs
    from test_rl_group import fixture_group
    rows, _ = handoff.diagnostic_rows_from_group(fixture_group())
    save_tensor_inputs(tmp_path, rows[0])
    # Explicitly exercise promotion of historical BF16 multimodal inputs.
    saved = torch.load(tmp_path / rows[0]["multimodal_file"], weights_only=True)
    saved["pixel_values"] = saved["pixel_values"].bfloat16()
    torch.save(saved, tmp_path / rows[0]["multimodal_file"])
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(_attn_implementation="cpu_fixture")
            self.model = SimpleNamespace(get_rope_index=self.rope)
        def rope(self, **kw):
            assert kw["input_ids"].tolist() == [[1, 2, 3, 4]]
            assert kw["image_grid_thw"].tolist() == [[1, 2, 2]]
            return torch.ones(3, 1, 4, dtype=torch.int64), None
        def forward(self, **kw):
            assert not torch.is_autocast_enabled("cpu") and not torch.is_grad_enabled()
            assert kw["pixel_values"].dtype == torch.float32 and kw["input_ids"].dtype == torch.int64
            assert kw["position_ids"].shape == (3, 1, 4) and kw["use_cache"] is False
            assert not self.training and not self.weight.requires_grad
            logits = torch.zeros(1, 4, 5); logits[0, 1, 3] = .7; logits[0, 2, 4] = 1.4
            logits[0, 3, 4] = 100.
            return SimpleNamespace(logits=logits)
    def logprob(logits, labels, inplace_backward):
        assert logits.dtype == torch.float32 and inplace_backward is False
        assert labels.tolist() == [[3, 4]] and logits[0, 0, 3] == 1. and logits[0, 1, 4] == 2.
        return torch.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    audit = {}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = diag.forward_rows_fp32(Model(), rows, tmp_path, device=torch.device("cpu"), audit=audit, logprob_function=logprob)
    assert len(result) == 2 and len(result[0]) == 2
    assert audit["rows"][0]["saved_vision_dtypes"]["pixel_values"] == "torch.bfloat16"
    assert audit["rows"][0]["model_input_dtypes"]["pixel_values"] == "torch.float32"
    assert audit["forward_autocast_enabled"] is False
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == flags


def test_fa2_kernel_implicit_half_cast_guard_and_restoration(monkeypatch):
    import transformers
    called = []
    def kernel(q, k, v): called.append(q.dtype); return q
    flash = SimpleNamespace(_flash_fn=kernel, _flash_varlen_fn=kernel, lazy_import_flash_attention=lambda backend: None)
    monkeypatch.setattr(transformers, "modeling_flash_attention_utils", flash)
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    for dtype in (torch.float32, torch.bfloat16):
        def invoke():
            with diag.fp32_execution_guard(TinyModel().float(), device=torch.device("cuda"), backend="flash_attention_2", audit={}):
                q = torch.ones(2, dtype=dtype); flash._flash_fn(q, q, q)
        if dtype == torch.bfloat16:
            with pytest.raises(diag.FP32ForwardUnsupported, match="FA2 kernel QKV"): invoke()
        else: invoke()
        assert flash._flash_fn is kernel and flash._flash_varlen_fn is kernel
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == flags
    assert called == [torch.float32]


@pytest.mark.parametrize("error,supported", [
    (RuntimeError("FlashAttention only support fp16 and bf16 data type"), False),
    (diag.FP32ForwardUnsupported("implicit downcast"), False),
    (RuntimeError("unrelated failure"), None), (ValueError("nonfinite HF logits"), None),
    (torch.OutOfMemoryError("flash only supports fp16: OOM"), None),
])
def test_probe_reports_only_fp32_unsupported_not_oom_or_corruption(monkeypatch, error, supported):
    def forward(*a, **kw): raise error
    monkeypatch.setattr(diag, "forward_rows_fp32", forward)
    if supported is None:
        with pytest.raises(type(error)): diag.probe_fp32_attention_support(None, [1], None, device=torch.device("cpu"))
    else:
        status, result = diag.probe_fp32_attention_support(None, [1], None, device=torch.device("cpu"))
        assert status["supported"] is False and result is None and status["error_type"] == type(error).__name__


@pytest.fixture
def precision_history(forensic):
    attempt, ctx, prior = forensic
    tokens, analysis = diag.analyze_tokens(ctx["rows"], prior, [[-.1, -.2]] * 2,
        [[-.3, -.4]] * 2, [[-.12, -.22]] * 2, top_n=50)
    dest = attempt.reports / "merge_precision_diagnostic"; dest.mkdir()
    summary = {**diag.META, **analysis, "execution_succeeded": True, "source_artifacts_unchanged": True,
        "gate_run_id": attempt.run_id, "token_count": 4,
        "base_model": diag.BASE_MODEL, "base_revision": diag.BASE_REVISION, "temperature": .7,
        "collection_attempt": ctx["group"]["identity"]["collection_attempt"],
        "formal_sft_adapter_fingerprint": ctx["group"]["identity"]["pre_update_policy_fingerprint"],
        "production_merged_checkpoint_fingerprint": ctx["group"]["merged_checkpoint_fingerprint"],
        "software_versions": ctx["identity"]["software_versions"]}
    atomic_json(dest / "summary.json", summary)
    (dest / "token_diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tokens), encoding="utf-8")
    return attempt, ctx, prior, tokens


@pytest.mark.parametrize("available", [True, False, "asymmetric"])
def test_fp32_comparisons_null_support_repeat_and_original_outliers(precision_history, available):
    _, ctx, prior, old = precision_history
    a2 = [[-.11, -.21]] * 2 if available else None
    b2 = [[-.111, -.211]] * 2 if available is True else None
    tokens, analysis = diag.analyze_fp32_tokens(ctx["rows"], prior, old, [[-.1, -.2]] * 2,
        [[-.3, -.4]] * 2, [[-.12, -.22]] * 2, a2, b2, top_n=50)
    assert [r["token_id"] for r in tokens] == [r["token_id"] for r in old]
    assert analysis["repeat_delta_vs_v453"]["dynamic_peft_vs_fp32_merge_bf16"]["mean_abs_logprob_diff"] == 0
    for alias, old_key in diag.BF16_PAIR_ALIASES.items():
        assert analysis["comparisons"][alias] == analysis["comparisons"][old_key]
    if available is True:
        assert analysis["comparisons"]["dynamic_fp32_vs_merged_fp32"]["mean_abs_logprob_diff"] == pytest.approx(.001)
        assert analysis["comparisons"]["dynamic_bf16_vs_dynamic_fp32"]["mean_abs_logprob_diff"] == pytest.approx(.01)
        assert analysis["comparisons"]["fp32_merge_bf16_vs_fp32_merge_fp32"]["mean_abs_logprob_diff"] == pytest.approx(.009)
    else:
        assert analysis["comparisons"]["dynamic_fp32_vs_merged_fp32"] is None
        assert analysis["original_production_outliers"][0]["dynamic_fp32_vs_merged_fp32_diff"] is None
    json.dumps(analysis, allow_nan=False)


@pytest.mark.parametrize("tamper", ["lineage", "tokens", "metrics", "count", "software"])
def test_v453_history_fail_closed(precision_history, tamper):
    attempt, ctx, prior, tokens = precision_history
    directory = attempt.reports / "merge_precision_diagnostic"
    path = directory / "summary.json"; summary = json.loads(path.read_text())
    if tamper == "lineage": summary["collection_attempt"] = "wrong"
    if tamper == "tokens": tokens[0]["token_id"] = 999
    if tamper == "metrics": summary["comparisons"]["dynamic_peft_vs_fp32_merge_bf16"]["mean_abs_logprob_diff"] = 999.
    if tamper == "count": summary["token_count"] = 1
    if tamper == "software": summary["software_versions"] = {}
    atomic_json(path, summary)
    (directory / "token_diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tokens), encoding="utf-8")
    before = handoff.source_checksums([directory], [])
    with pytest.raises(ValueError): diag.load_previous_precision(ctx, prior)
    handoff.assert_sources_unchanged(before, handoff.source_checksums([directory], []))


@pytest.mark.parametrize("mode", ["supported", "unsupported", "asymmetric", "keep", "mutation", "merge_drift", "gross_b1", "secondary"])
def test_v454_serial_orchestration_support_readonly_cleanup_no_gate(precision_history, monkeypatch, mode):
    attempt, ctx, _, _ = precision_history
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50,
        local_files_only=True, keep_diagnostic_model=mode == "keep", allow_sdpa_fp32_secondary=mode == "secondary")
    for name in ("is_available", "is_bf16_supported"): monkeypatch.setattr(torch.cuda, name, lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for name in ("set_device", "reset_peak_memory_stats", "empty_cache"): monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("max_memory_allocated", "max_memory_reserved"): monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    monkeypatch.setattr(diag.importlib.metadata, "version", lambda name: "0.6.1" if name == "verl" else "fixture")
    monkeypatch.setenv("WORLD_SIZE", "1")
    ctx["sft"]["model"]["attn_implementation"] = "flash_attention_2"
    real_validate = handoff.validate_attempt
    def validate(*a):
        current = real_validate(*a); current["sft"]["model"]["attn_implementation"] = "flash_attention_2"; return current
    monkeypatch.setattr(handoff, "validate_attempt", validate)
    refs, loaded = [], []
    class Model(torch.nn.Module):
        def __init__(self, kind):
            super().__init__(); self.kind = kind; self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
    def new(kind):
        assert all(ref() is None for ref in refs)
        model = Model(kind); refs.append(weakref.ref(model)); loaded.append(kind); return model
    monkeypatch.setattr(handoff, "load_diagnostic_model", lambda kind, *a: new("A0" if kind == "dynamic" else "B0"))
    monkeypatch.setattr(diag, "load_dynamic_fp32", lambda *a, **kw: new("A2_secondary" if kw.get("attention_backend") else "A2"))
    monkeypatch.setattr(diag, "load_fp32_merged_model", lambda *a, **kw: new("B2_secondary"))
    monkeypatch.setattr(diag, "lora_dtype_audit", lambda *a: {})
    values = {"A0": [[-.1, -.2]] * 2, "B0": [[-.3, -.4]] * 2, "B1": [[-.12, -.22]] * 2,
              "A2": [[-.11, -.21]] * 2, "B2": [[-.111, -.211]] * 2,
              "A2_secondary": [[-.11, -.21]] * 2, "B2_secondary": [[-.111, -.211]] * 2}
    if mode == "gross_b1": values["B1"] = [[-2., -2.]] * 2
    def bf16(model, rows, directory, **kw):
        assert rows == ctx["rows"] and directory == attempt.output / "group"; return values[model.kind]
    monkeypatch.setattr(diag, "bf16_forward_audit", bf16)
    def merge(context, base, destination, device, **kw):
        kind = "B2" if kw.get("serialized_dtype") == "fp32" else "B1"
        path = destination / kw["checkpoint_name"]; path.mkdir()
        (path / "model.safetensors").write_bytes(b"CPU only fixture")
        if mode == "mutation" and kind == "B2": (attempt.reports / "merge_precision_diagnostic/mutation").write_bytes(b"changed")
        return new(kind), path, {"by_suffix": {}}, {"pre_cast_merge_source_fingerprint": "different" if mode == "merge_drift" and kind == "B2" else "same"}
    monkeypatch.setattr(diag, "diagnostic_fp32_merge", merge)
    def probe(model, rows, directory, **kw):
        if mode in ("unsupported", "secondary") or mode == "asymmetric" and model.kind == "B2":
            return diag.unsupported_status(diag.FP32ForwardUnsupported("FA2 FP32 dtype unsupported"), stage="first_row_forward"), None
        return {"supported": True}, values[model.kind][:1]
    monkeypatch.setattr(diag, "probe_fp32_attention_support", probe)
    monkeypatch.setattr(diag, "forward_rows_fp32", lambda model, rows, *a, **kw: values[model.kind][:len(rows)])
    monkeypatch.setattr(diag, "weight_compare_stats", lambda *a, **kw: {})
    monkeypatch.setattr(diag, "weight_representation_stats", lambda *a: {"by_suffix": {}})
    before = handoff.source_checksums([attempt.output, attempt.reports / "merge_precision_diagnostic"], [])
    dest = attempt.reports / "fp32_merged_forward_diagnostic"
    if mode in ("mutation", "merge_drift", "gross_b1"):
        with pytest.raises((ValueError, RuntimeError)): diag.run_fp32_forward_diagnostic(args, attempt.root)
        summary = json.loads((dest / "summary.json").read_text())
        assert not summary["execution_succeeded"]
    else:
        assert diag.run_fp32_forward_diagnostic(args, attempt.root) == 0
        summary = json.loads((dest / "summary.json").read_text())
        assert summary["execution_succeeded"] and summary["token_count"] == 4
        assert summary["fp32_same_backend_supported"] is (mode in ("supported", "keep"))
        assert summary["fp32_support_asymmetric"] is (mode == "asymmetric")
        assert summary["merge_repeat_count"] == 2 and summary["merge_determinism"]["equal"]
        assert summary["models_destroyed"] == {k: True for k in ("A0", "B0", "B1", "A2", "B2")}
        assert (dest / "tmp_fp32_merge_fp32").exists() is (mode == "keep")
        assert not summary["formal_rl_initialization_allowed"]
        if mode in ("unsupported", "asymmetric", "secondary"):
            assert summary["comparisons"]["dynamic_fp32_vs_merged_fp32"] is None
        if mode == "secondary":
            assert summary["secondary"]["secondary_backend_changed"] and summary["secondary"]["excluded_from_primary_metrics"]
            assert summary["secondary"]["comparisons"]["dynamic_fp32_vs_merged_fp32_sdpa"]["mean_abs_logprob_diff"] == pytest.approx(.001)
        else: assert summary["secondary"] is None
        handoff.assert_sources_unchanged(before, handoff.source_checksums([attempt.output, attempt.reports / "merge_precision_diagnostic"], []))
        with pytest.raises(FileExistsError): diag.run_fp32_forward_diagnostic(args, attempt.root)
    assert not list(attempt.root.rglob("gate_manifest.json"))
    assert loaded[:5] == ["A0", "B0", "B1", "A2", "B2"]


def test_fp32_guard_rejects_half_parameters_buffers_and_module_outputs():
    with pytest.raises(ValueError, match="floating parameters"):
        diag.require_fp32_parameters(TinyModel())
    model = TinyModel().float()
    model.register_buffer("bad", torch.ones(1, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="buffers"): diag.require_fp32_parameters(model)
    class HalfOutput(torch.nn.Module):
        def forward(self, value): return value.bfloat16()
    model = HalfOutput()
    with pytest.raises(diag.FP32ForwardUnsupported, match="output"):
        with diag.fp32_execution_guard(model, device=torch.device("cpu"), backend="fixture", audit={}):
            model(torch.ones(2))
    assert not model._forward_hooks and not model._forward_pre_hooks


def test_attention_backend_cannot_silently_change_inside_model():
    class FixtureAttention(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config = SimpleNamespace(_attn_implementation="sdpa")
    model = TinyModel().float(); model.attn = FixtureAttention()
    with pytest.raises(ValueError, match="nested attention"):
        diag.attention_backend_audit(model)


def test_bf16_input_audit_observes_without_changing_original_forward(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        def forward(self, **kw): return kw["input_ids"]
    model = Model()
    ids = torch.ones(1, 4, dtype=torch.int64); pixels = torch.ones(2, 3)
    def original(m, rows, directory, *, device):
        assert m is model and rows == ["original"]
        assert torch.equal(m(input_ids=ids, pixel_values=pixels, position_ids=ids, attention_mask=ids), ids)
        return [[-.1, -.2]]
    monkeypatch.setattr(handoff, "forward_rows", original)
    audit = {}
    assert diag.bf16_forward_audit(model, ["original"], None, device=torch.device("cuda"), audit=audit) == [[-.1, -.2]]
    assert audit["rows"][0]["pixel_values"] == "torch.float32"
    assert audit["rows"][0]["position_ids"] == "torch.int64"
    assert audit["forward_autocast_enabled"] is True and audit["forward_autocast_dtype"] == "torch.bfloat16"
    assert not model._forward_pre_hooks


def test_fp32_entry_reuses_parser_secondary_is_explicit_and_offline():
    result = subprocess.run([sys.executable, "scripts/diagnose_rl_fp32_merged_forward.py", "--help"], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0 and "--allow-sdpa-fp32-secondary" in result.stdout
    source = (ROOT / "scripts/diagnose_rl_fp32_merged_forward.py").read_text()
    for forbidden in ("merge_actor_adapter", "update_policy", "optimizer", "gate_manifest", "vllm"):
        assert forbidden not in source
    assert 'action="store_true"' in source and "run_fp32_forward_diagnostic" in source
