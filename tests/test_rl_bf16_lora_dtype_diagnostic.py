"""CPU-only actual tiny PEFT and historical fixtures, never a GPU/Gate result."""
import ast
import copy
import inspect
import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opensearch_vl_repro.rl import bf16_lora_dtype_diagnostic as diag
from opensearch_vl_repro.rl import bf16_sdpa_diagnostic as backend
from opensearch_vl_repro.rl import merge_precision_diagnostic as precision
from opensearch_vl_repro.rl import policy_handoff_diagnostic as handoff
from opensearch_vl_repro.rl.actor_gate import atomic_json
from test_rl_bf16_sdpa_diagnostic import history as backend_history
from test_rl_merge_precision_diagnostic import forensic, precision_history, tiny_peft, TinyModel
from test_rl_policy_handoff_diagnostic import attempt
from test_rl_rollout_sync import actor_artifacts

ROOT = Path(__file__).resolve().parents[1]


def dynamic():
    model = tiny_peft()
    model.config._attn_implementation = "sdpa"
    return model.eval().requires_grad_(False)


def values():
    return dict(A3=[[-.1, -.8], [-.4, -.2]], A4=[[-.45, -.5], [-.4, -.4]], B3=[[-.5, -.4], [-.3, -.4]])


@pytest.fixture
def v455_history(backend_history):
    attempt, ctx, prior = backend_history
    data = values()
    tokens, analysis = backend.analyze_tokens(ctx["rows"], prior, dict(A0=[[-.1, -.2]]*2, B0=[[-.3, -.4]]*2,
                                                                    A3=data["A3"], B3=data["B3"]), top_n=50)
    old = json.loads((attempt.reports / "merge_precision_diagnostic/summary.json").read_text())
    model = dynamic()
    dtype_audits = {"A3": {"base_parameter_dtypes": precision.dtype_counts(model, lora=False),
                          "lora_parameter_dtypes": precision.dtype_counts(model, lora=True),
                          "lora_targets": precision.lora_dtype_audit(model, expected_layers=1)},
                   "B3": {"base_parameter_dtypes": precision.dtype_counts(TinyModel())}}
    summary = {**old, **backend.META, **analysis, "variants": backend.VARIANTS,
        "model_dtypes": dtype_audits, "attention_backend_audits": {
            k: {"model": "sdpa", "attention_modules": []} for k in ("A3", "B3")}}
    dest = attempt.reports / "bf16_sdpa_backend_diagnostic"
    dest.mkdir()
    atomic_json(dest / "summary.json", summary)
    (dest / "token_diagnostics.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tokens), encoding="utf-8")
    return attempt, ctx, tokens


def test_actual_peft_cast_only_ab_exact_values_and_no_base_changes():
    from peft import PeftModel
    model = dynamic()
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    state = diag.lora_state(model, expected_layers=1)
    assert state["dtype_counts"] == {"torch.float32": 14}
    assert state["module_count"] == 7
    audit = diag.cast_dynamic_lora_to_bf16(model, expected_layers=1)
    assert audit["source_dtype_counts"] == {"torch.float32": 14}
    assert audit["target_dtype_counts"] == {"torch.bfloat16": 14}
    assert audit["base_unchanged"] and audit["non_lora_before"] == audit["non_lora_after"]
    assert audit["before"]["per_module"] == audit["after"]["per_module"]
    assert isinstance(model, PeftModel)
    for name, parameter in model.named_parameters():
        expected = initial[name].to(torch.bfloat16) if "lora_" in name else initial[name]
        assert parameter.dtype == torch.bfloat16 and torch.equal(parameter, expected)
        assert not parameter.requires_grad
    assert all(r["actual_cast_value_sha256"] == r["expected_cast_value_sha256"] for r in audit["per_tensor"])
    assert not any(m.merged for _, m, _ in precision.lora_roster(model, expected_layers=1))
    assert diag.verify_after_forward(model, audit["after"], audit["non_lora_after"], require_bf16=True, expected_layers=1) == audit["after"]
    json.dumps(audit, allow_nan=False)


def test_full_36_times_7_target_roster_and_504_tensors():
    from peft import LoraConfig, get_peft_model
    base = TinyModel()
    base.layers = torch.nn.ModuleList([copy.deepcopy(base.layers[0]) for _ in range(36)])
    base.config._attn_implementation = "sdpa"
    model = get_peft_model(base, LoraConfig(r=1, lora_alpha=2, lora_dropout=.05,
        target_modules=sorted(precision.TARGETS), bias="none")).eval().requires_grad_(False)
    audit = diag.cast_dynamic_lora_to_bf16(model)
    assert audit["module_count"] == 252 and audit["tensor_count"] == 504
    assert len({r["name"] for r in audit["per_tensor"]}) == 504
    assert {(r["layer_index"], r["target_suffix"]) for r in audit["before"]["per_module"]} == {
        (layer, suffix) for layer in range(36) for suffix in precision.TARGETS}
    assert all(r["rank"] == 1 and r["alpha"] == 2 and r["scaling"] == 2 and r["dropout_p"] == .05
               and not r["dropout_training"] for r in audit["after"]["per_module"])


def test_mixed_source_dtype_inspected_and_already_bf16_is_not_informative():
    model = dynamic()
    first = precision.lora_roster(model, expected_layers=1)[0][1]
    first.lora_A["default"].weight.data = first.lora_A["default"].weight.data.to(torch.bfloat16)
    state = diag.lora_state(model, expected_layers=1)
    assert state["dtype_counts"] == {"torch.bfloat16": 1, "torch.float32": 13}
    diag.cast_dynamic_lora_to_bf16(model, expected_layers=1)
    with pytest.raises(diag.ExperimentNotInformative, match="not_informative"):
        diag.cast_dynamic_lora_to_bf16(model, expected_layers=1)


@pytest.mark.parametrize("bad", ["missing", "extra", "alias", "base_fp32", "backend", "nested", "disabled", "training", "merged", "nonfinite", "plain"])
def test_cast_safety_rejects_invalid_representation_before_modification(bad):
    model = dynamic()
    module = precision.lora_roster(model, expected_layers=1)[0][1]
    if bad == "missing": del model.base_model.model.layers[0].q_proj
    if bad == "extra": model.register_parameter("lora_extra", torch.nn.Parameter(torch.ones(1), requires_grad=False))
    if bad == "alias": module.lora_B["default"].weight = module.lora_A["default"].weight
    if bad == "base_fp32": module.get_base_layer().float()
    if bad == "backend": model.config._attn_implementation = "flash_attention_2"
    if bad == "nested":
        class FixtureAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        model.base_model.model.bad_attention = FixtureAttention()
    if bad == "disabled": module.enable_adapters(False)
    if bad == "training": model.train()
    if bad == "merged": module.merged_adapters.append("default")
    if bad == "nonfinite": module.lora_A["default"].weight.data.fill_(float("nan"))
    if bad == "plain": model = TinyModel()
    with pytest.raises(ValueError): diag.cast_dynamic_lora_to_bf16(model, expected_layers=1)


@pytest.mark.parametrize("bad", ["promotion", "base", "lora_value", "scaling", "dropout"])
def test_after_forward_requires_unchanged_values_semantics_and_bf16(bad):
    model = dynamic()
    audit = diag.cast_dynamic_lora_to_bf16(model, expected_layers=1)
    module = precision.lora_roster(model, expected_layers=1)[0][1]
    if bad == "promotion": module.lora_A["default"].float()
    if bad == "base": module.get_base_layer().weight.data.add_(1.)
    if bad == "lora_value": module.lora_A["default"].weight.data.add_(1.)
    if bad == "scaling": module.scaling["default"] *= 2
    if bad == "dropout": module.lora_dropout["default"].train()
    with pytest.raises(ValueError):
        diag.verify_after_forward(model, audit["after"], audit["non_lora_after"], require_bf16=True, expected_layers=1)


def test_metrics_improvement_direction_clips_and_original_outliers(v455_history):
    _, ctx, prior = v455_history
    data = values()
    tokens, analysis = diag.analyze_tokens(ctx["rows"], prior, data, top_n=50)
    assert len(tokens) == 4
    for name, (a, b) in diag.PAIRS.items():
        assert analysis["comparisons"][name] == handoff.pair_metrics(sum(data[a], []), sum(data[b], []))
    b, m = analysis["comparisons"]["A3_vs_B3"], analysis["comparisons"]["A4_vs_B3"]
    for field, metric in (("mean_abs_reduction_fraction", "mean_abs_logprob_diff"),
                          ("max_abs_reduction_fraction", "max_abs_logprob_diff"),
                          ("clip_fraction_reduction_fraction", "clip_fraction_0p8_1p28")):
        assert analysis["lora_dtype_alignment_improvement"][field] == pytest.approx((b[metric]-m[metric])/b[metric])
    assert analysis["toward_B3_fraction"] == .75 and analysis["away_from_B3_fraction"] == 0 and analysis["equal_fraction"] == .25
    clips = analysis["clip_sets"]
    assert clips["A3_vs_B3"] == [0, 1] and clips["A4_vs_B3"] == []
    assert clips["A3_vs_B3_only"] == [0, 1] and clips["A4_vs_B3_only"] == []
    for names, count in analysis["clip_set_intersections"].items():
        assert count == len(set.intersection(*(set(clips[n]) for n in names.split(" & "))))
    top = analysis["original_A3_vs_B3_outliers"]
    assert [r["global_trainable_index"] for r in top] == sorted(range(4), key=lambda i: (
        -abs(prior[i]["A3_logprob"]-prior[i]["B3_logprob"]), i))
    assert sum(t["original_clipped_to_nonclipped"] for t in tokens) == 2
    assert all(t["original_outlier_abs_reduction"] == pytest.approx(abs(t["prior_v455_A3_vs_B3_diff"])-abs(t["A4_vs_B3_diff"])) for t in top)
    assert [t["decoded_token"] for t in tokens] == [t["decoded_token"] for t in prior]
    json.dumps(analysis, allow_nan=False)


def test_zero_baseline_negative_reduction_correlation_and_away_equal(v455_history):
    _, ctx, prior = v455_history
    _, analysis = diag.analyze_tokens(ctx["rows"], prior, {k: [[0., 0.]]*2 for k in diag.VARIANTS}, top_n=50)
    assert set(analysis["lora_dtype_alignment_improvement"].values()) == {None}
    assert analysis["lora_dtype_shift_correlation_with_A3B3_gap"] is None and analysis["equal_fraction"] == 1.
    assert precision.reduction_fraction(1., 2.) == -1.
    result = diag.direction_analysis([1., -1., 0., 0.], [1., -1., 0., 1.], [0., -2., 0., 1.])
    assert result["toward_B3_fraction"] == .25 and result["away_from_B3_fraction"] == .25 and result["equal_fraction"] == .5
    assert result["direction_denominator"] == 4
    assert diag.direction_analysis([1., 2.], [1., 2.], [0., 0.])["lora_dtype_shift_correlation_with_A3B3_gap"] == pytest.approx(1.)
    with pytest.raises(ValueError): diag.direction_analysis([0.], [0.], [float("nan")])


def test_original_mask_and_invalid_token_inputs(v455_history):
    _, ctx, prior = v455_history
    rows = copy.deepcopy(ctx["rows"])
    rows[0]["response_mask"] = [1, 0]
    selected = [prior[0], {**prior[2], "global_trainable_index": 1}, {**prior[3], "global_trainable_index": 2}]
    tokens, _ = diag.analyze_tokens(rows, selected, values(), top_n=50)
    assert [(t["rollout_index"], t["response_token_position"]) for t in tokens] == [(0, 0), (1, 0), (1, 1)]
    for bad in ("missing", "count", "nonfinite", "order", "top_n"):
        data, old = values(), copy.deepcopy(prior)
        if bad == "missing": data.pop("A4")
        if bad == "count": data["A4"] = [[0.]]*2
        if bad == "nonfinite": data["B3"][0][0] = float("nan")
        if bad == "order": old[0]["token_id"] = 99
        with pytest.raises(ValueError): diag.analyze_tokens(ctx["rows"], old, data, top_n=0 if bad == "top_n" else 50)


@pytest.mark.parametrize("bad", ["run", "fingerprint", "software", "count", "temperature", "order", "nonfinite", "metrics", "backend", "base", "lora", "missing"])
def test_v455_full_token_history_binding_fail_closed(v455_history, bad):
    attempt, ctx, _ = v455_history
    dest = attempt.reports / "bf16_sdpa_backend_diagnostic"
    summary = json.loads((dest / "summary.json").read_text())
    tokens = [json.loads(line) for line in (dest / "token_diagnostics.jsonl").read_text().splitlines()]
    if bad == "run": summary["gate_run_id"] = "wrong"
    if bad == "fingerprint": summary["production_merged_checkpoint_fingerprint"] = "wrong"
    if bad == "software": summary["software_versions"] = {}
    if bad == "count": summary["token_count"] = 99
    if bad == "temperature": summary["temperature"] = 1.
    if bad == "order": tokens[0]["token_id"] = 99
    if bad == "nonfinite": tokens[0]["A3_logprob"] = float("nan")
    if bad == "metrics": summary["comparisons"]["A3_vs_B3"]["mean_abs_logprob_diff"] = 99.
    if bad == "backend": summary["attention_backend_audits"]["A3"]["model"] = "flash_attention_2"
    if bad == "base": summary["model_dtypes"]["A3"]["base_parameter_dtypes"] = {}
    if bad == "lora": summary["model_dtypes"]["A3"].pop("lora_targets")
    atomic_json(dest / "summary.json", summary)
    path = dest / "token_diagnostics.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in tokens), encoding="utf-8")
    if bad == "missing": path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)): diag.load_history(ctx)


def test_v455_repeat_guard_checks_individual_branches_not_just_gap(v455_history):
    _, ctx, prior = v455_history
    history = diag.load_history(ctx)
    tokens, _ = diag.analyze_tokens(ctx["rows"], prior, values(), top_n=50)
    assert diag.repeat_checks(tokens, history)["interpretation_allowed"]
    for token in tokens:
        token["A3_logprob"] += 1.
        token["B3_logprob"] += 1.
    repeat = diag.repeat_checks(tokens, history)
    assert not repeat["interpretation_allowed"]
    assert repeat["pair_metric_deltas"]["mean_abs_logprob_diff"] == pytest.approx(0.)
    assert repeat["per_variant_repeat_deltas"]["A3"]["mean_abs_logprob_diff"] == pytest.approx(1.)


@pytest.mark.parametrize("mode", ["success", "mutation", "repeat", "already_bf16", "different_base", "different_lora", "promotion", "forward", "retain", "publication", "no_cuda", "software"])
def test_serial_inference_cast_lifecycle_readonly_failure_and_no_overwrite(v455_history, monkeypatch, mode):
    attempt, ctx, _ = v455_history
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50, local_files_only=True)
    monkeypatch.setenv("WORLD_SIZE", "1")
    for name in ("is_available", "is_bf16_supported"):
        monkeypatch.setattr(torch.cuda, name, lambda: mode != "no_cuda")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for name in ("set_device", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 123)
    monkeypatch.setattr(diag.importlib.metadata, "version", lambda n: "0.6.1" if n == "verl" else "wrong" if mode == "software" else "fixture")
    state, cast, verify, dtype = diag.lora_state, diag.cast_dynamic_lora_to_bf16, diag.verify_after_forward, precision.lora_dtype_audit
    monkeypatch.setattr(diag, "lora_state", lambda m, **kw: state(m, expected_layers=1))
    monkeypatch.setattr(diag, "cast_dynamic_lora_to_bf16", lambda m: cast(m, expected_layers=1))
    monkeypatch.setattr(diag, "verify_after_forward", lambda m, s, b, **kw: verify(m, s, b, expected_layers=1, **kw))
    monkeypatch.setattr(precision, "lora_dtype_audit", lambda m: dtype(m, expected_layers=1))
    refs, loaded, loader_kinds, retained = [], [], [], []
    def load(kind, context, base, device):
        assert all(ref() is None for ref in refs)
        assert context["merged"] == ctx["merged"] and base == attempt.base and device.type == "cuda"
        name = "A4" if kind == "A3" and loaded else kind
        model = dynamic() if kind == "A3" else TinyModel().eval().requires_grad_(False)
        model.config._attn_implementation = "sdpa"
        model.diagnostic_kind = name
        if mode == "already_bf16" and name == "A3":
            for n, p in model.named_parameters():
                if "lora_" in n: p.data = p.data.to(torch.bfloat16)
        if mode in ("different_base", "different_lora") and name == "A4":
            for n, p in model.named_parameters():
                if ("lora_" in n) == (mode == "different_lora"): p.data.add_(1.); break
        if mode == "retain": retained.append(model)
        loaded.append(name)
        loader_kinds.append(kind)
        refs.append(weakref.ref(model))
        return model
    monkeypatch.setattr(backend, "load_bf16_sdpa", load)
    data = values()
    if mode == "repeat": data["A3"] = [[-2., -2.]]*2
    def forward(model, rows, directory, *, device, audit):
        assert rows == ctx["rows"] and directory == attempt.output / "group"
        audit["resolved_attention"] = precision.attention_backend_audit(model)
        if mode == "forward": raise RuntimeError("fixture forward failure")
        if mode == "mutation": (attempt.output / "mutation").write_bytes(b"fixture")
        if mode == "promotion" and model.diagnostic_kind == "A4":
            precision.lora_roster(model, expected_layers=1)[0][1].lora_A["default"].float()
        return data[model.diagnostic_kind]
    monkeypatch.setattr(backend, "forward_with_audit", forward)
    atomic = diag.atomic_json
    def publish(path, value):
        if mode == "publication" and path.name == "summary.json" and value["execution_succeeded"]:
            raise OSError("fixture publication failure")
        return atomic(path, value)
    monkeypatch.setattr(diag, "atomic_json", publish)
    protected = [attempt.output, attempt.base, attempt.root / "configs", *(attempt.reports / d for d in diag.HISTORY_DIRS)]
    before = handoff.source_checksums(protected, [attempt.reports / "gate_c_report.json"])
    if mode == "success": assert diag.run_diagnostic(args, attempt.root) == 0
    else:
        with pytest.raises((ValueError, RuntimeError, OSError)): diag.run_diagnostic(args, attempt.root)
    destination = attempt.reports / "bf16_lora_dtype_diagnostic"
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["execution_succeeded"] is (mode == "success")
    assert summary["source_artifacts_unchanged"] is (mode != "mutation")
    assert summary["formal_rl_initialization_allowed"] is False
    if mode == "already_bf16":
        assert summary["experiment_informative"] is False and summary["experiment_not_informative"] is True
        assert loaded == ["A3"]
    if mode == "success":
        assert loaded == ["A3", "A4", "B3"] and loader_kinds == ["A3", "A3", "B3"]
        assert all(summary["models_destroyed"].values())
        assert summary["experiment_informative"] and summary["lora_cast_audit"]["base_unchanged"]
        assert summary["lora_dtype_audit"]["A4"]["post_forward_dtype_counts"] == {"torch.bfloat16": 14}
        assert summary["model_sources"]["B3"] == str(ctx["merged"])
        assert summary["token_count"] == 4 and set(summary["comparisons"]) == set(diag.PAIRS)
        assert len((destination / "token_diagnostics.jsonl").read_text().splitlines()) == 4
    if mode != "mutation": handoff.assert_sources_unchanged(before, handoff.source_checksums(protected, [attempt.reports / "gate_c_report.json"]))
    if mode in ("no_cuda", "software"): assert not loaded
    with pytest.raises(FileExistsError): diag.run_diagnostic(args, attempt.root)
    assert not list(attempt.root.rglob("gate_manifest.json"))
    assert not list(destination.rglob("*.safetensors")) and not list(destination.rglob("*.pt"))


def test_static_reuse_no_live_operations_merges_model_cast_or_saved_checkpoints():
    source = inspect.getsource(diag)
    forbidden = {"VLLMStaticBackend", "SamplingParams", "DeepSeekJudge", "live_rewards", "step", "update_policy",
                 "merge_actor_adapter", "diagnostic_fp32_merge", "merge_and_unload", "create_phase3_tool_registry",
                 "AgentRuntime", "save_pretrained", "bfloat16", "float", "half", "save"}
    called = {n.func.id if isinstance(n.func, ast.Name) else n.func.attr for n in ast.walk(ast.parse(source))
              if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert not called & forbidden and "gate_manifest.json" not in source
    assert "backend.load_bf16_sdpa" in source and "backend.forward_with_audit" in source and "backend.load_history" in source
    assert "parameter.data = cast" in source and "set_float32_matmul_precision" not in source and "sdpa_kernel(" not in source
    helper = inspect.getsource(handoff.forward_rows) + inspect.getsource(handoff.sampled_response_logprobs)
    assert "get_rope_index" in helper and "torch.autocast" in helper and "temperature=.7" in helper
    assert "verl.utils.torch_functional" in helper and "-length - 1 : -1" in helper
    assert 'atomic_json(destination / "summary.json", summary)' in source


def test_cli_explicit_offline_no_model_save_flags():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_rl_bf16_lora_dtype.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--local-files-only" in result.stdout
    assert "--top-n" in result.stdout and "--keep" not in result.stdout and "--secondary" not in result.stdout
