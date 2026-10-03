"""CPU forensic fixtures only; no CUDA, real model, API or Gate PASS."""
import copy
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl import policy_handoff_diagnostic as diag
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json
from opensearch_vl_repro.rl.group import group_identity, publish_group
from opensearch_vl_repro.rl.policy_alignment import alignment_artifact, alignment_checks, compare_policy_logprobs
from opensearch_vl_repro.rl.rollout_sync import merge_identity, validate_actor_adapter
from opensearch_vl_repro.rl.training_batch import training_rows
from opensearch_vl_repro.sft_tool_audit import sha256_file
from test_rl_group import fixture_group
from test_rl_rollout_sync import actor_artifacts

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    all_special_ids = [4]
    def __call__(self, *a, **k): pytest.fail("response/prompt re-tokenization forbidden")
    def encode(self, *a, **k): pytest.fail("response re-encoding forbidden")
    def convert_ids_to_tokens(self, token): return f"token_{token}"
    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is False
        return "" if tokens == [4] else "\n\x00"


@pytest.mark.parametrize("a,b,c,expected", [
    ([-1., -2.], [-1., -2.], [-1., -2.], (0., 0., 0.)),
    ([-.5, -1.5], [-1., -2.], [-1., -2.], (.5, .5, 0.)),
    ([-1., -2.], [-1., -2.], [-1.5, -2.5], (.5, 0., .5)),
    ([-.5, -1.5], [-1., -2.], [-1.5, -2.5], (1., .5, .5)),
])
def test_exact_merge_backend_and_mixed_drift_metrics(a, b, c, expected):
    rows, _ = diag.diagnostic_rows_from_group(fixture_group())
    tokens = diag.token_diagnostics(rows, [a, a], [b, b], Tokenizer())
    # Override the CPU fixture's rollout values for this controlled comparison.
    for i, record in enumerate(tokens): record["vllm_old_logprob"] = c[i % 2]
    original = dict(mean_abs_logprob_diff=.01, max_abs_logprob_diff=.02,
                    mean_signed_logprob_diff=.03, initial_clip_fraction=.04)
    report = diag.summarize_tokens(tokens, top_n=50, original_alignment=original)
    for name, value in zip(diag.PAIRS, expected):
        assert report[name]["mean_abs_logprob_diff"] == pytest.approx(value)
        if value == 0:
            assert report[name]["mean_importance_ratio"] == 1 and report[name]["clip_fraction_0p8_1p28"] == 0
    assert report["fsdp_proxy_metric_deltas"]["delta_mean_abs"] == pytest.approx(expected[0] - .01)


def test_quantiles_threshold_counts_outlier_order_and_empty_decoded_tokens():
    m = diag.pair_metrics([0., .02, .08, .15, .3], [0.] * 5)
    assert m["p50_abs"] == .08 and m["p90_abs"] == pytest.approx(.24)
    assert m["p99_5_abs"] == pytest.approx(.297)
    assert m["p100_abs"] == .3 and m["count_abs_diff_gt_0p10"] == 2
    rows, _ = diag.diagnostic_rows_from_group(fixture_group())
    tokens = diag.token_diagnostics(rows, [[-.1, -.7], [-.3, -.2]], [[-.1, -.2]] * 2, Tokenizer())
    report = diag.summarize_tokens(tokens, top_n=2, original_alignment=dict(
        mean_abs_logprob_diff=0, max_abs_logprob_diff=0, mean_signed_logprob_diff=0, initial_clip_fraction=0))
    outliers = report["top_outliers"]["dynamic_peft_vs_rollout"]
    assert [r["global_trainable_index"] for r in outliers] == [1, 2]
    assert outliers[0]["decoded_token"] == "" and outliers[0]["is_special_token"]
    assert "\\u0000" in json.dumps(tokens, ensure_ascii=True, allow_nan=False)
    assert len(report["per_row"]) == 2


def test_diagnostic_rows_exactly_reuse_training_masks_including_fatal_turn():
    group = fixture_group()
    group["members"][0]["fatal"] = {"fatal": True, "fatal_step": 0}
    group["members"][0]["steps"].append(copy.deepcopy(group["members"][0]["steps"][0]))
    before = copy.deepcopy(group)
    rows, counts = diag.diagnostic_rows_from_group(group)
    formal, formal_counts = training_rows(group, [1., -1.])
    for got, expected in zip(rows, formal, strict=True):
        for key in ("rollout_index", "step_index", "responses", "response_mask", "prompt_ids", "multimodal_file"):
            assert got[key] == expected[key]
    assert counts == formal_counts and counts["supervised_response_tokens"] == 4
    assert all(r["step_index"] == 0 for r in rows) and counts["masked_post_fatal_tokens"] == 2
    assert group == before  # expert/source conversations untouched


def save_tensor_inputs(directory, row, *, wrong_ids=False):
    torch.save({"input_ids": torch.tensor([[9, 9]] if wrong_ids else [row["prompt_ids"]]),
        "pixel_values": torch.ones(2, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}, directory / row["multimodal_file"])


def test_saved_multimodal_prompt_and_response_lengths_fail_closed(tmp_path):
    row = diag.diagnostic_rows_from_group(fixture_group())[0][0]
    save_tensor_inputs(tmp_path, row, wrong_ids=True)
    with pytest.raises(ValueError, match="prompt IDs mismatch"):
        diag.load_row_inputs(tmp_path, row, device=torch.device("cpu"))
    save_tensor_inputs(tmp_path, row)
    with pytest.raises(ValueError, match="length mismatch"):
        diag.load_row_inputs(tmp_path, {**row, "old_log_probs": [-.1]}, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="unsafe"):
        diag.load_row_inputs(tmp_path, {**row, "multimodal_file": "../outside.pt"}, device=torch.device("cpu"))


def test_tiny_forward_mrope_and_previous_position_slice(tmp_path):
    rows, _ = diag.diagnostic_rows_from_group(fixture_group())
    save_tensor_inputs(tmp_path, rows[0])
    calls = []
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.model = SimpleNamespace(get_rope_index=self.rope)
        def rope(self, **kw):
            assert kw["input_ids"].tolist() == [[1, 2, 3, 4]]
            assert kw["attention_mask"].tolist() == [[1, 1, 1, 1]]
            return torch.full((3, 1, 4), 42), None
        def forward(self, **kw):
            assert not self.training and not self.weight.requires_grad and not torch.is_grad_enabled()
            assert kw["use_cache"] is False and kw["position_ids"].shape == (3, 1, 4)
            assert (kw["position_ids"] == 42).all() and kw["pixel_values"].shape == (2, 3)
            logits = torch.zeros(1, 4, 5)
            logits[0, 1, 3] = .7  # previous position for response token 3
            logits[0, 2, 4] = 1.4  # previous position for response token 4
            logits[0, 3, 4] = 100.  # must NOT be used
            return SimpleNamespace(logits=logits)
    def fixture_logprob(logits, labels, inplace_backward):
        assert inplace_backward is False and labels.tolist() == [[3, 4]]
        assert logits[0, 0, 3].item() == pytest.approx(1.)
        assert logits[0, 1, 4].item() == pytest.approx(2.)
        calls.append(logits.clone())
        return torch.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    result = diag.forward_rows(Model(), rows, tmp_path, device=torch.device("cpu"), logprob_function=fixture_logprob)
    assert len(result) == len(rows) and len(calls) == 2 and len(result[0]) == 2


@pytest.mark.parametrize("kind", ["logits", "sampled", "shape"])
def test_nonfinite_or_misaligned_hf_forward_fails(kind):
    logits = torch.zeros(1, 3, 6)
    if kind == "logits": logits[0, 0, 0] = float("inf")
    callback = lambda *a, **k: torch.tensor([[float("nan")]]) if kind == "sampled" else torch.zeros(1, 2)
    with pytest.raises(ValueError): diag.sampled_response_logprobs(logits, torch.tensor([[4]]), logprob_function=callback)


@pytest.fixture
def attempt(tmp_path, actor_artifacts):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    for name in ("rl_main.yaml", "sft_main_imageid_v3.yaml", "eval_base_300.yaml"):
        shutil.copyfile(ROOT / "configs" / name, root / "configs" / name)
    adapter = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    shutil.copytree(actor_artifacts["source_sft_adapter"].parent, adapter.parent)
    actor = validate_actor_adapter(adapter=adapter, gate_manifest=None, rl_config=actor_artifacts["rl_config"],
        sft_config=actor_artifacts["sft_config"], source_sft_adapter=adapter)
    base = tmp_path / "base"; base.mkdir()
    atomic_json(base / "config.json", {"model_type": "qwen3_vl", "text_config": {"num_hidden_layers": 36}})
    (base / "model.safetensors").write_bytes(b"CPU fixture, not real weights")
    run_id = "gate-c-v451-attempt1"
    output, reports = diag.diagnostic_paths(root, run_id)
    output.mkdir(parents=True); reports.mkdir(parents=True)
    from opensearch_vl_repro.rl.gate_c import load_gate_c_config
    identity = dict(run_id=run_id, gate_version="minimum-rl-integration-c-v1", base_model=BASE_MODEL,
        base_revision=BASE_REVISION, formal_rl_initialization_allowed=False,
        source_sft_actor=actor, logprobs_mode="processed_logprobs",
        gate_config={**load_gate_c_config(ROOT / "configs/rl_gate_c.yaml"), "gate_version": "minimum-rl-integration-c-v1"},
        formal_config_sha256=sha256_file(root / "configs/rl_main.yaml"),
        sft_config_sha256=sha256_file(root / "configs/sft_main_imageid_v3.yaml"),
        offline_base_file_sha256={p.name: sha256_file(p) for p in base.iterdir()},
        integration_source_sha256={"gate_c.py": "HISTORICAL-NOT-CURRENT"},
        software_versions={"torch": "fixture", "transformers": "fixture", "peft": "fixture", "verl": "0.6.1"})
    identity["identity_sha256"] = canonical_json_sha256(identity)
    atomic_json(output / "run_manifest.json", identity)
    gi = group_identity(prompt_id="rl_000001", policy_fingerprint=actor["source_sft_adapter_fingerprint"],
        rollout_fingerprint="b" * 64, attempt="exact-old-attempt", context=identity["identity_sha256"])
    group = fixture_group(); group["identity"] = gi
    for member in group["members"]: member["identity"] = gi
    merged = output / "merged-exact-old-attempt"; merged.mkdir()
    atomic_json(merged / "config.json", {"model_type": "qwen3_vl"})
    for name in ("preprocessor_config.json", "tokenizer_config.json"): atomic_json(merged / name, {})
    (merged / "model.safetensors").write_bytes(b"CPU fixture static weights")
    mi = merge_identity(actor=actor, versions={}, file_hashes={p.name: sha256_file(p) for p in merged.iterdir()})
    atomic_json(merged / "merge_manifest.json", {"identity": mi, "actor_provenance": actor,
        "merge_complete": True, "no_active_peft": True})
    group["merged_checkpoint_fingerprint"] = mi["merged_checkpoint_fingerprint"]
    staging = tmp_path / "group-staging"; staging.mkdir()
    save_tensor_inputs(staging, diag.diagnostic_rows_from_group(group)[0][0])
    group = publish_group(staging, output / "group", group)
    local = compare_policy_logprobs(torch.tensor([[-.4, -.5], [-.4, -.5]]),
        torch.tensor([[-.1, -.2], [-.1, -.2]]), torch.ones(2, 2),
        clip_ratio_low=.2, clip_ratio_high=.28, expected_masked_token_count=4)
    local.update(logprobs_computed=True, temperature=.7, rollout_temperature=.7)
    local["checks"] = alignment_checks(local); local["passed"] = False
    alignment = alignment_artifact([{**local, "rank": 0}, {**local, "rank": 1}],
        gate_version="minimum-rl-integration-c-v1", identity=identity,
        trajectory_group_id=gi["trajectory_group_id"], policy_fingerprint=actor["source_sft_adapter_fingerprint"])
    atomic_json(output / "pre_update_policy_alignment.json", alignment)
    atomic_json(output / "update_started.json", {"identity": identity})
    atomic_json(reports / "gate_c_report.json", dict(stage="pre_update_policy_alignment", passed=False,
        optimizer_step_count=0, identity=identity, pre_update_policy_alignment=alignment))
    return SimpleNamespace(root=root, base=base, output=output, reports=reports, run_id=run_id, merged=merged)


def test_historical_attempt_validation_and_exact_path_without_current_hash_binding(attempt):
    before = diag.source_checksums([attempt.output], [attempt.reports / "gate_c_report.json"])
    ctx = diag.validate_attempt(attempt.root, attempt.run_id, attempt.base)
    assert ctx["merged"] == attempt.merged and ctx["alignment"]["masked_token_count"] == 4
    assert ctx["identity"]["integration_source_sha256"]["gate_c.py"] == "HISTORICAL-NOT-CURRENT"
    diag.assert_sources_unchanged(before, diag.source_checksums([attempt.output], [attempt.reports / "gate_c_report.json"]))


@pytest.mark.parametrize("bad", ["missing_merged", "updated", "step", "passed", "merge_hash", "adapter", "base", "alignment_stats"])
def test_artifact_validation_fail_closed_without_repairing_sources(attempt, bad):
    if bad == "missing_merged": attempt.merged.rename(attempt.output / "merged-wrong-attempt")
    elif bad == "updated": atomic_json(attempt.output / "update_verified.json", {})
    elif bad in ("step", "passed"):
        path = attempt.reports / "gate_c_report.json"; report = json.loads(path.read_text())
        report["optimizer_step_count" if bad == "step" else "passed"] = 1 if bad == "step" else True
        atomic_json(path, report)
    elif bad == "merge_hash": (attempt.merged / "model.safetensors").write_bytes(b"changed")
    elif bad == "adapter":
        (attempt.root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter/adapter_model.safetensors").write_bytes(b"changed")
    elif bad == "base": (attempt.base / "model.safetensors").write_bytes(b"changed")
    elif bad == "alignment_stats":
        path = attempt.output / "pre_update_policy_alignment.json"
        alignment = json.loads(path.read_text())
        alignment["max_abs_logprob_diff"] = .001  # contradicts recorded rank statistics
        atomic_json(path, alignment)
        report_path = attempt.reports / "gate_c_report.json"
        report = json.loads(report_path.read_text()); report["pre_update_policy_alignment"] = alignment
        atomic_json(report_path, report)
    before = diag.source_checksums([attempt.output], [attempt.reports / "gate_c_report.json"])
    with pytest.raises((ValueError, FileNotFoundError)):
        diag.validate_attempt(attempt.root, attempt.run_id, attempt.base)
    diag.assert_sources_unchanged(before, diag.source_checksums([attempt.output], [attempt.reports / "gate_c_report.json"]))
    assert not (attempt.output / "gate_manifest.json").exists()


def test_source_checksum_mutation_addition_and_deletion_detection(tmp_path):
    atomic_json(tmp_path / "old.json", {"a": 1})
    before = diag.source_checksums([tmp_path], [])
    atomic_json(tmp_path / "old.json", {"a": 2})
    with pytest.raises(RuntimeError): diag.assert_sources_unchanged(before, diag.source_checksums([tmp_path], []))
    with pytest.raises(RuntimeError): diag.assert_sources_unchanged(before, {})
    with pytest.raises(RuntimeError): diag.assert_sources_unchanged(before, {**before, "new": "hash"})


def test_diagnostic_publication_all_tokens_json_safe_never_gate_manifest_or_old_report(attempt):
    ctx = diag.validate_attempt(attempt.root, attempt.run_id, attempt.base)
    tokens = diag.token_diagnostics(ctx["rows"], [[-.1, -.2]] * 2, [[-.1, -.2]] * 2, Tokenizer())
    measured = diag.summarize_tokens(tokens, top_n=50, original_alignment=ctx["alignment"])
    dest = attempt.reports / "policy_handoff_diagnostic"; dest.mkdir()
    old = (attempt.reports / "gate_c_report.json").read_bytes()
    before = diag.source_checksums([attempt.output], [])
    diag.publish_diagnostic(dest, measured, tokens, measured["per_row"], {"before": before, "after": before})
    assert len((dest / "token_diagnostics.jsonl").read_text().splitlines()) == 4
    for path in dest.glob("*.json"):
        assert json.loads(path.read_text())["formal_rl_initialization_allowed"] is False
    assert not list(attempt.root.rglob("gate_manifest.json"))
    assert (attempt.reports / "gate_c_report.json").read_bytes() == old
    diag.assert_sources_unchanged(before, diag.source_checksums([attempt.output], []))
    with pytest.raises(FileExistsError): diag.publish_diagnostic(dest, measured, tokens, [], {})


@pytest.mark.parametrize("mode", ["success_with_drift", "mutation", "forward_failure"])
def test_serial_orchestration_cleanup_readonly_reports_and_drift_is_not_gate_failure(attempt, monkeypatch, mode):
    import weakref
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50, local_files_only=True)
    for name in ("is_available", "is_bf16_supported"):
        monkeypatch.setattr(torch.cuda, name, lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for name in ("set_device", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(diag.importlib.metadata, "version", lambda name: "0.6.1" if name == "verl" else "fixture")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **k: Tokenizer())))
    references, loaded = [], []
    def load(kind, *a):
        assert not references or references[-1]() is None  # no simultaneous residency
        model = SimpleModel(kind)
        references.append(weakref.ref(model)); loaded.append(kind)
        return model
    class SimpleModel:
        def __init__(self, kind): self.kind = kind
    monkeypatch.setattr(diag, "load_diagnostic_model", load)
    def forward(model, rows, *a, **k):
        if mode == "forward_failure": raise ValueError("nonfinite HF sampled-token logprobs")
        if mode == "mutation" and model.kind == "dynamic":
            atomic_json(attempt.output / "run_manifest.json", {"deliberate CPU fixture mutation": True})
        return [[v - (.3 if model.kind == "dynamic" else 0.) for v in r["old_log_probs"]] for r in rows]
    monkeypatch.setattr(diag, "forward_rows", forward)
    report_before = (attempt.reports / "gate_c_report.json").read_bytes()
    source_before = diag.source_checksums([attempt.output], [])
    dest = attempt.reports / "policy_handoff_diagnostic"
    if mode == "success_with_drift":
        assert diag.run_diagnostic(args, attempt.root) == 0
        summary = json.loads((dest / "summary.json").read_text())
        assert summary["execution_succeeded"] and summary["token_count"] == 4
        assert summary["dynamic_peft_vs_rollout"]["max_abs_logprob_diff"] > .1  # drift is a measurement
        assert summary["dynamic_model_destroyed"] and summary["merged_model_destroyed"]
        assert loaded == ["dynamic", "merged"]
        diag.assert_sources_unchanged(source_before, diag.source_checksums([attempt.output], []))
        with pytest.raises(FileExistsError): diag.run_diagnostic(args, attempt.root)
    else:
        with pytest.raises((RuntimeError, ValueError)): diag.run_diagnostic(args, attempt.root)
        summary = json.loads((dest / "summary.json").read_text())
        assert summary["execution_succeeded"] is False
        assert summary["source_artifacts_unchanged"] is (mode != "mutation")
        assert summary["stage"] == ("dynamic_hf_forward" if mode == "forward_failure" else "token_analysis")
    assert all(ref() is None for ref in references)
    assert (attempt.reports / "gate_c_report.json").read_bytes() == report_before
    assert not list(attempt.root.rglob("gate_manifest.json"))


def test_imports_are_lazy_and_forbidden_operations_absent():
    code = "import sys; import opensearch_vl_repro.rl.policy_handoff_diagnostic; assert not any(k in sys.modules for k in ['torch','transformers','peft','verl','vllm','rllm'])"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    source = inspect.getsource(diag) + (ROOT / "scripts/diagnose_rl_policy_handoff.py").read_text()
    for forbidden in ("merge_actor_adapter", "VLLMStaticBackend", "LLM(", "SamplingParams", "DeepSeekJudge",
                      "live_rewards", "create_phase3_tool_registry", "AgentRuntime.run", "update_policy(",
                      "optimizer.step", "save_checkpoint", "prepare_context(", "bind_run(", "finalize(",
                      "apply_chat_template", "load_processor", "tokenizer.encode", "tokenizer("):
        assert forbidden not in source
    assert 'weights_only=True' in source and 'from verl.utils.torch_functional import logprobs_from_logits' in source
