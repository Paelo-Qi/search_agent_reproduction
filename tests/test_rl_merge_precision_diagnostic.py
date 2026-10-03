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
