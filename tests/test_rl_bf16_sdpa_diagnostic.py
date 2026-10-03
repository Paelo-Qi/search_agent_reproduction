"""CPU-only forensic fixtures; no test claims real CUDA/Gate success."""
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

from opensearch_vl_repro.rl import bf16_sdpa_diagnostic as diag
from opensearch_vl_repro.rl import merge_precision_diagnostic as precision
from opensearch_vl_repro.rl import policy_handoff_diagnostic as handoff
from opensearch_vl_repro.rl.actor_gate import atomic_json
from test_rl_merge_precision_diagnostic import forensic, precision_history, TinyModel
from test_rl_policy_handoff_diagnostic import attempt, save_tensor_inputs
from test_rl_rollout_sync import actor_artifacts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def history(precision_history):
    attempt, ctx, prior, previous = precision_history
    path = attempt.reports / "policy_handoff_diagnostic/summary.json"
    summary = json.loads(path.read_text())
    summary.update(base_model=diag.BASE_MODEL, base_revision=diag.BASE_REVISION,
                   temperature=.7, software_versions=ctx["identity"]["software_versions"])
    atomic_json(path, summary)
    tokens, analysis = precision.analyze_fp32_tokens(ctx["rows"], prior, previous,
        [[-.1, -.2]]*2, [[-.3, -.4]]*2, [[-.12, -.22]]*2, None, None, top_n=50)
    for name in diag.HISTORY_DIRS[2:]:
        dest = attempt.reports / name
        dest.mkdir()
        summary = {**json.loads((attempt.reports / "merge_precision_diagnostic/summary.json").read_text()),
                   **precision.FP32_META, **analysis, "secondary": None}
        if name == diag.HISTORY_DIRS[3]:
            summary["secondary"] = dict(secondary_backend_changed=True, exploratory_only=True,
                excluded_from_primary_metrics=True, attention_backend="sdpa",
                variants={k: {"supported": True} for k in ("A2", "B2")},
                comparisons={"dynamic_fp32_vs_merged_fp32_sdpa": handoff.pair_metrics([-.1]*4, [-.10001]*4)})
        atomic_json(dest / "summary.json", summary)
        (dest / "token_diagnostics.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tokens), encoding="utf-8")
    return attempt, ctx, prior


def values():
    return dict(A0=[[-.1, -.2]]*2, B0=[[-.3, -.4]]*2,
                A3=[[-.11, -.21]]*2, B3=[[-.11001, -.21001]]*2)


def test_real_cpu_peft_loader_sdpa_bf16_local_and_exact_b0(history, monkeypatch):
    from transformers import Qwen3VLForConditionalGeneration
    from peft import PeftModel, LoraConfig, get_peft_model
    attempt, ctx, _ = history
    calls, adapters = [], []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        model = TinyModel()
        model.config._attn_implementation = kwargs["attn_implementation"]
        return model
    def attach(model, path, **kwargs):
        adapters.append((path, kwargs))
        return get_peft_model(model, LoraConfig(r=1, lora_alpha=1,
            target_modules=sorted(precision.TARGETS), bias="none"))
    monkeypatch.setattr(Qwen3VLForConditionalGeneration, "from_pretrained", load)
    monkeypatch.setattr(PeftModel, "from_pretrained", attach)
    a3 = diag.load_bf16_sdpa("A3", ctx, attempt.base, torch.device("cpu"))
    b3 = diag.load_bf16_sdpa("B3", ctx, attempt.base, torch.device("cpu"))
    assert isinstance(a3, PeftModel) and not isinstance(b3, PeftModel)
    assert [p for p, _ in calls] == [str(attempt.base), str(ctx["merged"])]
    for _, kw in calls:
        assert kw == dict(revision=diag.BASE_REVISION, dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=False, low_cpu_mem_usage=True, local_files_only=True)
    assert adapters == [(str(ctx["adapter"]), dict(is_trainable=False, local_files_only=True))]
    assert precision.dtype_counts(a3, lora=False)["tensor_count_by_dtype"] == {"torch.bfloat16": 7}
    # Actual PEFT default inference promotion is preserved, not manually cast.
    assert precision.dtype_counts(a3, lora=True)["tensor_count_by_dtype"] == {"torch.float32": 14}
    assert precision.dtype_counts(b3)["tensor_count_by_dtype"] == {"torch.bfloat16": 7}
    assert not a3.training and not b3.training
    assert not any(p.requires_grad for m in (a3, b3) for p in m.parameters())
    assert all(diag.validate_model(m, backend="sdpa", dynamic=m is a3)["model"] == "sdpa" for m in (a3, b3))


@pytest.mark.parametrize("bad", ["dtype", "backend", "nested_backend", "dynamic", "kind"])
def test_loader_and_model_contract_fail_closed(history, bad):
    _, ctx, _ = history
    model = TinyModel()
    model.config._attn_implementation = "sdpa"
    if bad == "dtype": model.float()
    if bad == "backend": model.config._attn_implementation = "flash_attention_2"
    if bad == "nested_backend":
        class FixtureAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        model.attention = FixtureAttention()
    if bad == "kind":
        with pytest.raises(ValueError): diag.load_bf16_sdpa("B2", ctx, Path("unused"), torch.device("cpu"))
    else:
        with pytest.raises(ValueError): diag.validate_model(model, backend="sdpa", dynamic=bad == "dynamic")


def test_forward_delegates_original_ids_vision_mrope_slice_temperature_and_mask(history, monkeypatch):
    attempt, ctx, prior = history
    rows = copy.deepcopy(ctx["rows"])
    rows[0]["response_mask"] = [1, 0]
    for row in rows: save_tensor_inputs(attempt.output / "group", row)
    seen = []
    class ForwardModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.visual = torch.nn.Identity()
            self.config = SimpleNamespace(_attn_implementation="sdpa")
            self.model = SimpleNamespace(get_rope_index=self.rope)
        def rope(self, **kw):
            seen.append(("rope", kw["input_ids"].tolist(), kw["image_grid_thw"].tolist()))
            return torch.ones(3, 1, kw["input_ids"].shape[1], dtype=torch.int64), None
        def forward(self, **kw):
            assert kw["use_cache"] is False and kw["position_ids"].shape == (3, 1, 4)
            self.visual(kw["pixel_values"])
            seen.append(("forward", kw["input_ids"].tolist()))
            return SimpleNamespace(logits=torch.arange(24, dtype=torch.bfloat16).reshape(1, 4, 6))
    def logprobs(logits, responses, **kw):
        assert kw == {"inplace_backward": False}
        assert torch.equal(logits, torch.arange(24, dtype=torch.bfloat16).reshape(1, 4, 6).div_(.7)[:, -3:-1])
        assert responses.tolist() == [[3, 4]]
        return torch.tensor([[-.1, -.2]])
    original = handoff.forward_rows
    monkeypatch.setattr(handoff, "forward_rows", lambda *a, **kw: original(*a, **kw, logprob_function=logprobs))
    model, audit = ForwardModel(), {}
    result = diag.forward_with_audit(model, rows, attempt.output / "group", device=torch.device("cpu"), audit=audit)
    assert len(result) == 2 and len(seen) == 4
    assert all(item[1] == [[1, 2, 3, 4]] for item in seen)
    assert audit["saved_vision_dtypes"] == [dict(pixel_values="torch.float32", image_grid_thw="torch.int64")]*2
    assert audit["rows"] == [dict(input_ids="torch.int64", attention_mask="torch.int64", position_ids="torch.int64",
                                 pixel_values="torch.float32", image_grid_thw="torch.int64")]*2
    assert audit["vision_tower_input_dtypes"] == [["torch.float32"]]*2
    assert all(s["logits_dtype"] == "torch.bfloat16" and not s["autocast_enabled"] for s in audit["actual_forward_states"])
    assert not model._forward_hooks and not model._forward_pre_hooks and not model.visual._forward_pre_hooks
    selected = [prior[0], {**prior[2], "global_trainable_index": 1}, {**prior[3], "global_trainable_index": 2}]
    tokens, _ = diag.analyze_tokens(rows, selected, {v: result for v in diag.VARIANTS}, top_n=50)
    assert [r["response_token_position"] for r in tokens] == [0, 0, 1]


@pytest.mark.parametrize("bad", [None, "autocast_off", "autocast_fp32", "logits_fp32"])
def test_actual_cuda_autocast_audit_guard_with_cpu_stub(history, monkeypatch, bad):
    attempt, ctx, _ = history
    class Output(torch.nn.Module):
        def forward(self):
            return SimpleNamespace(logits=torch.ones(1, dtype=torch.float32 if bad == "logits_fp32" else torch.bfloat16))
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda d: bad != "autocast_off")
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda d: torch.float32 if bad == "autocast_fp32" else torch.bfloat16)
    monkeypatch.setattr(precision, "bf16_forward_audit", lambda model, *a, **kw: model())
    monkeypatch.setattr(handoff, "load_row_inputs", lambda *a, **kw: (None, None, {"pixel_values": torch.ones(1)}))
    model, audit = Output(), {}
    if bad:
        with pytest.raises(ValueError, match="autocast contract"):
            diag.forward_with_audit(model, ctx["rows"], attempt.output / "group", device=torch.device("cuda"), audit=audit)
    else:
        diag.forward_with_audit(model, ctx["rows"], attempt.output / "group", device=torch.device("cuda"), audit=audit)
        assert audit["actual_forward_states"] == [dict(logits_dtype="torch.bfloat16", autocast_enabled=True, autocast_dtype="torch.bfloat16")]
    assert not model._forward_hooks


def test_metrics_clips_reductions_top_original_mapping_and_json(history):
    _, ctx, prior = history
    data = values()
    data["A0"] = [[-.1, -.8], [-.4, -.2]]
    data["B0"] = [[-.5, -.4], [-.3, -.4]]
    tokens, analysis = diag.analyze_tokens(ctx["rows"], prior, data, top_n=50)
    assert len(tokens) == 4
    for name, (a, b) in diag.PAIRS.items():
        first = [v for row in data[a] for v in row]
        second = [v for row in data[b] for v in row]
        assert analysis["comparisons"][name] == handoff.pair_metrics(first, second)
    clips = analysis["clip_sets"]
    assert clips["A0_vs_B0"] == [0, 1]
    assert len(analysis["clip_set_intersections"]) == 11
    for names, count in analysis["clip_set_intersections"].items():
        assert count == len(set.intersection(*(set(clips[k]) for k in names.split(" & "))))
    assert analysis["backend_alignment_improvement"]["mean_abs_reduction_fraction"] > .99
    originals = analysis["original_A0_vs_B0_outliers"]
    assert [t["global_trainable_index"] for t in originals] == sorted(range(4), key=lambda i: (
        -abs(prior[i]["dynamic_peft_logprob"]-prior[i]["merged_hf_logprob"]), i))
    assert all(t["A3_vs_B3_diff"] == pytest.approx(.00001) for t in originals)
    assert all(t["original_outlier_abs_diff_reduction_fraction"] > .99 for t in originals)
    assert all(set(k+"_logprob" for k in diag.VARIANTS) <= t.keys() for t in tokens)
    json.dumps(analysis, allow_nan=False)


def test_zero_baseline_negative_improvement_and_invalid_inputs(history):
    _, ctx, prior = history
    data = {v: [[0., 0.]]*2 for v in diag.VARIANTS}
    _, analysis = diag.analyze_tokens(ctx["rows"], prior, data, top_n=50)
    assert set(analysis["backend_alignment_improvement"].values()) == {None}
    assert analysis["backend_shift_correlation"] is None
    assert precision.reduction_fraction(1., 2.) == -1.
    for bad in ("missing", "length", "nonfinite", "order", "top_n"):
        altered, old = copy.deepcopy(data), copy.deepcopy(prior)
        if bad == "missing": altered.pop("B3")
        if bad == "length": altered["A3"] = [[0.]]*2
        if bad == "nonfinite": altered["A3"][0][0] = float("nan")
        if bad == "order": old[0]["token_id"] = 99
        with pytest.raises(ValueError): diag.analyze_tokens(ctx["rows"], old, altered, top_n=0 if bad == "top_n" else 50)


def test_signed_direction_all_tokens_denominator_zero_variance_and_nonfinite():
    metrics = diag.direction_metrics([1., -1., 0., 2.], [-1., 1., 0., -2.])
    assert metrics["backend_shift_correlation"] == pytest.approx(-1.)
    assert metrics["backend_shift_opposite_sign_fraction"] == .75
    assert metrics["backend_shift_denominator"] == 4
    assert metrics["backend_shift_zero_pair_count"] == 1
    assert diag.direction_metrics([1., 1.], [1., 2.])["backend_shift_correlation"] is None
    with pytest.raises(ValueError): diag.direction_metrics([float("nan")], [0.])


@pytest.mark.parametrize("directory", range(4))
@pytest.mark.parametrize("tamper", ["identity", "count", "order", "metrics", "temperature", "software"])
def test_all_historical_archives_fail_closed(history, directory, tamper):
    attempt, ctx, _ = history
    dest = attempt.reports / diag.HISTORY_DIRS[directory]
    sp, tp = dest / "summary.json", dest / "token_diagnostics.jsonl"
    summary, tokens = json.loads(sp.read_text()), [json.loads(line) for line in tp.read_text().splitlines()]
    if tamper == "identity": summary["collection_attempt"] = "wrong"
    if tamper == "count": summary["token_count"] = 99
    if tamper == "temperature": summary["temperature"] = 1.
    if tamper == "software": summary["software_versions"] = {}
    if tamper == "order": tokens[0]["token_id"] = 99
    if tamper == "metrics":
        if directory == 0: summary["dynamic_peft_vs_merged_hf"]["mean_abs_logprob_diff"] = 99.
        else: summary["comparisons"]["dynamic_peft_vs_production_merge"]["mean_abs_logprob_diff"] = 99.
    atomic_json(sp, summary)
    tp.write_text("".join(json.dumps(t) + "\n" for t in tokens), encoding="utf-8")
    with pytest.raises(ValueError): diag.load_history(ctx)


@pytest.mark.parametrize("tamper", ["primary_secondary", "backend", "incomplete", "aggregate_count", "aggregate_nan", "alias"])
def test_secondary_and_alias_evidence_not_silently_trusted(history, tamper):
    attempt, ctx, _ = history
    path = attempt.reports / diag.HISTORY_DIRS[2 if tamper == "primary_secondary" else 3] / "summary.json"
    summary = json.loads(path.read_text())
    if tamper == "primary_secondary": summary["secondary"] = {}
    elif tamper == "alias": summary["comparisons"]["dynamic_vs_production_merge"] = {}
    elif tamper == "backend": summary["secondary"]["attention_backend"] = "other"
    elif tamper == "incomplete": summary["secondary"]["variants"]["A2"]["supported"] = False
    else:
        metric = summary["secondary"]["comparisons"]["dynamic_fp32_vs_merged_fp32_sdpa"]
        metric["token_count" if tamper == "aggregate_count" else "mean_abs_logprob_diff"] = 99 if tamper == "aggregate_count" else float("nan")
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError): diag.load_history(ctx)


def test_repeat_history_limits_and_large_repeat_refuses_interpretation(history):
    _, ctx, prior = history
    evidence = diag.load_history(ctx)
    tokens, _ = diag.analyze_tokens(ctx["rows"], prior, values(), top_n=50)
    repeats = diag.repeat_checks(tokens, evidence)
    assert len(repeats) == 4 and all(r["interpretation_allowed"] for r in repeats.values())
    for token in tokens: token["A0_logprob"] += 1.
    assert not any(r["interpretation_allowed"] for r in diag.repeat_checks(tokens, evidence).values())


@pytest.mark.parametrize("mode", ["success", "mutation", "repeat", "lora_dtype", "forward", "retain", "publication", "no_cuda", "software", "flags"])
def test_serial_readonly_orchestration_failure_summary_and_no_overwrite(history, monkeypatch, mode):
    attempt, ctx, _ = history
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50, local_files_only=True)
    monkeypatch.setenv("WORLD_SIZE", "1")
    for name in ("is_available", "is_bf16_supported"):
        monkeypatch.setattr(torch.cuda, name, lambda: mode != "no_cuda")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for name in ("set_device", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 123)
    monkeypatch.setattr(diag.importlib.metadata, "version", lambda name: "0.6.1" if name == "verl" else "wrong" if mode == "software" else "fixture")
    real_validate = handoff.validate_attempt
    def validate(*a):
        current = real_validate(*a)
        current["sft"]["model"]["attn_implementation"] = "flash_attention_2"
        return current
    monkeypatch.setattr(handoff, "validate_attempt", validate)
    monkeypatch.setattr(precision, "lora_dtype_audit", lambda m: {"actual_lora_dtype": str(m.lora_fixture.dtype)})
    refs, loaded, retained = [], [], []
    class Model(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            if kind in ("A0", "A3"):
                self.lora_fixture = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16 if mode == "lora_dtype" and kind == "A3" else torch.float32))
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2" if kind in ("A0", "B0") else "sdpa")
    def load(kind):
        assert all(ref() is None for ref in refs)
        model = Model(kind)
        refs.append(weakref.ref(model))
        loaded.append(kind)
        if mode == "retain": retained.append(model)
        return model
    monkeypatch.setattr(handoff, "load_diagnostic_model", lambda kind, *a: load("A0" if kind == "dynamic" else "B0"))
    monkeypatch.setattr(diag, "load_bf16_sdpa", lambda kind, *a: load(kind))
    data = values()
    if mode == "repeat": data["A0"] = [[-2., -2.]]*2
    def forward(model, rows, directory, *, device, audit):
        assert rows == ctx["rows"] and directory == attempt.output / "group" and device.type == "cuda"
        audit["resolved_attention"] = precision.attention_backend_audit(model)
        if mode == "forward": raise RuntimeError("fixture forward failure")
        if mode == "mutation": (attempt.output / "source-mutated").write_bytes(b"mutation fixture")
        return data[model.kind]
    monkeypatch.setattr(diag, "forward_with_audit", forward)
    if mode == "flags":
        calls = []
        def flags(*a): calls.append(1); return {"observed": len(calls)}
        monkeypatch.setattr(diag, "numeric_environment", flags)
    real_atomic = diag.atomic_json
    def publish(path, value):
        if mode == "publication" and path.name == "summary.json" and value["execution_succeeded"]:
            raise OSError("fixture publication failure")
        return real_atomic(path, value)
    monkeypatch.setattr(diag, "atomic_json", publish)
    protected = [attempt.output, attempt.base, attempt.root / "configs", *(attempt.reports / d for d in diag.HISTORY_DIRS)]
    before = handoff.source_checksums(protected, [attempt.reports / "gate_c_report.json"])
    if mode != "success":
        with pytest.raises((ValueError, RuntimeError, OSError)):
            diag.run_diagnostic(args, attempt.root)
    else:
        assert diag.run_diagnostic(args, attempt.root) == 0
    dest = attempt.reports / "bf16_sdpa_backend_diagnostic"
    summary = json.loads((dest / "summary.json").read_text())
    assert summary["execution_succeeded"] is (mode == "success")
    assert summary["source_artifacts_unchanged"] is (mode != "mutation")
    assert summary["formal_rl_initialization_allowed"] is False
    if mode == "success":
        assert loaded == ["A0", "B0", "A3", "B3"]
        assert all(summary["models_destroyed"].values())
        assert summary["model_sources"]["B0"] == summary["model_sources"]["B3"] == str(ctx["merged"])
        assert set(summary["comparisons"]) == set(diag.PAIRS)
        assert len((dest / "token_diagnostics.jsonl").read_text().splitlines()) == 4
        assert summary["peak_gpu_memory"]["A3"]["peak_allocated_bytes"] == 123
    if mode in ("no_cuda", "software"): assert not loaded
    if mode != "mutation":
        handoff.assert_sources_unchanged(before, handoff.source_checksums(protected, [attempt.reports / "gate_c_report.json"]))
    with pytest.raises(FileExistsError): diag.run_diagnostic(args, attempt.root)
    assert not list(attempt.root.rglob("gate_manifest.json"))
    assert not list(dest.rglob("*.safetensors")) and not list(dest.rglob("*.pt"))


def test_missing_archive_refused_and_checksum_detects_added_deleted_changed(history):
    attempt, ctx, _ = history
    path = attempt.reports / diag.HISTORY_DIRS[-1] / "token_diagnostics.jsonl"
    before = handoff.source_checksums([path.parent], [])
    path.unlink()
    with pytest.raises(FileNotFoundError): diag.load_history(ctx)
    with pytest.raises(RuntimeError): handoff.assert_sources_unchanged(before, handoff.source_checksums([path.parent], []))
    for after in ({**before, "added": "hash"}, {k: "changed" for k in before}):
        with pytest.raises(RuntimeError): handoff.assert_sources_unchanged(before, after)


def test_static_no_live_operations_no_merges_no_weights_or_checkpoints():
    source = inspect.getsource(diag)
    forbidden = {"VLLMStaticBackend", "SamplingParams", "DeepSeekJudge", "live_rewards", "step", "update_policy",
                 "merge_actor_adapter", "diagnostic_fp32_merge", "create_phase3_tool_registry", "AgentRuntime",
                 "merge_and_unload", "save_pretrained", "weight_compare_stats", "forward_rows_fp32", "float", "half"}
    called = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not called & forbidden
    assert "gate_manifest.json" not in source and "torch.float32" not in source
    assert "handoff.load_diagnostic_model" in source and "precision.bf16_forward_audit" in source
    assert "allow_tf32 =" not in source and "sdpa_kernel(" not in source
    helper = inspect.getsource(handoff.forward_rows) + inspect.getsource(handoff.sampled_response_logprobs)
    assert "get_rope_index" in helper and "torch.autocast" in helper and "temperature=.7" in helper
    assert "verl.utils.torch_functional" in helper and "-length - 1 : -1" in helper
    assert 'atomic_json(destination / "summary.json", summary)' in source


def test_cli_is_offline_explicit_no_save_or_backend_secondary_flags():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_rl_bf16_sdpa_backend.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--local-files-only" in result.stdout and "--top-n" in result.stdout
    assert "--keep" not in result.stdout and "--allow" not in result.stdout
