"""CPU evidence only. No mock result is a real GPU feasibility conclusion."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opensearch_vl_repro.rl import actor_old_logprob_diagnostic as diag
from opensearch_vl_repro.rl import bf16_lora_dtype_diagnostic as lora
from opensearch_vl_repro.rl import bf16_sdpa_diagnostic as backend
from opensearch_vl_repro.rl import verl_old_logprob_semantics as semantics
from opensearch_vl_repro.rl.policy_alignment import compare_policy_logprobs
from opensearch_vl_repro.rl.actor_gate import atomic_json
from test_rl_bf16_lora_dtype_diagnostic import v455_history, values, dynamic
from test_rl_bf16_sdpa_diagnostic import history as backend_history
from test_rl_merge_precision_diagnostic import forensic, precision_history, tiny_peft, TinyModel
from test_rl_policy_handoff_diagnostic import attempt
from test_rl_rollout_sync import actor_artifacts

ROOT = Path(__file__).resolve().parents[1]


class Data:
    def __init__(self):
        self.batch = dict(input_ids=torch.tensor([[1, 2, 3, 4]]), responses=torch.tensor([[3, 4]]),
            response_mask=torch.tensor([[1, 0]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
            position_ids=torch.tensor([[[0, 1, 2, 3]]]*3).transpose(0, 1),
            old_log_probs=torch.tensor([[-.1, -.2]]), advantages=torch.zeros(1, 2))
        modal = np.empty(1, dtype=object)
        modal[0] = dict(pixel_values=torch.tensor([[.25, .5]]), image_grid_thw=torch.tensor([[1, 1, 1]]))
        self.non_tensor_batch = {"multi_modal_inputs": modal}
        self.meta_info = dict(temperature=.7, micro_batch_size=1, use_dynamic_bsz=False)


class Actor:
    def __init__(self, mutation=None):
        self.actor_module = torch.nn.Linear(2, 2)
        self.actor_module.weight.data.fill_(.25)
        self.actor_module.bias.data.zero_()
        self.mutation, self.calls, self.batches = mutation, 0, []
        self.actor_optimizer = torch.optim.AdamW(self.actor_module.parameters())
    def update_policy(self, data):
        raise AssertionError("never execute update_policy")
    def compute_log_prob(self, data, calculate_entropy=False):
        self.calls += 1
        self.batches.append(data)
        self.actor_module.eval()
        result = -self.actor_module(torch.ones(1, 2))
        if self.mutation == "data":
            data.batch["input_ids"][0, 0] += 1
        elif self.mutation == "vision":
            data.non_tensor_batch["multi_modal_inputs"][0]["pixel_values"][0, 0] += 1
        elif self.mutation == "metadata":
            data.meta_info["temperature"] = .8
        elif self.mutation == "parameter":
            self.actor_module.weight.add_(.1)
        elif self.mutation == "rng":
            torch.rand(1)
        elif self.mutation == "training":
            self.actor_module.train()
            result = self.actor_module(torch.ones(1, 2))
        return result, None


def compute(actor, data, label="O"):
    return diag.independent_compute(actor, data, label, expected_input=diag.input_fingerprint(data),
                                    expected_parameters=diag.actor_fingerprint(actor.actor_module))


def test_two_actual_independent_forwards_and_clones_without_parameter_input_or_rng_change():
    actor, data = Actor(), Data()
    params, inputs = diag.actor_fingerprint(actor.actor_module), diag.input_fingerprint(data)
    O, oa = compute(actor, data)
    C, ca = compute(actor, data, "C")
    assert actor.calls == 2 and len(actor.batches) == 2
    assert actor.batches[0] is not actor.batches[1] and all(b is not data for b in actor.batches)
    assert actor.batches[0].batch["input_ids"].data_ptr() != actor.batches[1].batch["input_ids"].data_ptr()
    assert O.data_ptr() != C.data_ptr() and torch.equal(O, C)
    assert oa["forward_count"] == ca["forward_count"] == 1
    assert oa["parameters_before"] == ca["parameters_after"] == params
    assert oa["input_before"] == ca["input_after"] == inputs
    assert oa["rng_before"] == oa["rng_after"] == ca["rng_before"] == ca["rng_after"]
    assert diag.input_fingerprint(data) == inputs
    assert not actor.actor_optimizer.state
    assert all(p.grad is None for p in actor.actor_module.parameters())


@pytest.mark.parametrize("mutation,match", [("data", "mutated DataProto"), ("vision", "mutated DataProto"),
    ("metadata", "mutated DataProto"), ("parameter", "parameter mutation"), ("rng", "RNG changed"),
    ("training", "deterministic eval")])
def test_fail_closed_mutations(mutation, match):
    with pytest.raises(ValueError, match=match):
        compute(Actor(mutation), Data())


def test_no_forward_or_gradients_or_unexpected_inputs_rejected():
    actor, data = Actor(), Data()
    actor.compute_log_prob = lambda data, **kwargs: (torch.ones(1, 2), None)
    with pytest.raises(ValueError, match="actual actor module forward"):
        compute(actor, data)
    actor = Actor()
    actor.actor_module.weight.grad = torch.ones_like(actor.actor_module.weight)
    with pytest.raises(ValueError, match="gradients"):
        compute(actor, data)
    with pytest.raises(ValueError, match="input fingerprint"):
        diag.independent_compute(Actor(), data, "C", expected_input={}, expected_parameters={})


def test_gradient_storage_dtype_and_tensor_fingerprint_are_not_changed():
    actor = Actor()
    actor.actor_module.to(torch.bfloat16)
    pre = diag.actor_fingerprint(actor.actor_module)
    assert all(p["dtype"] == "torch.bfloat16" for p in pre["local_shards"])
    with torch.no_grad():
        actor.actor_module.bias.add_(torch.tensor(.125, dtype=torch.bfloat16))
    assert pre != diag.actor_fingerprint(actor.actor_module)


def test_stable_recursive_nested_multimodal_hash_and_mutation_sensitivity():
    data = Data()
    pre = diag.input_fingerprint(data)
    assert pre == diag.input_fingerprint(copy.deepcopy(data))
    for field in ("input_ids", "responses", "response_mask", "attention_mask", "position_ids", "old_log_probs"):
        other = copy.deepcopy(data)
        other.batch[field].reshape(-1)[0] += 1
        assert pre != diag.input_fingerprint(other)
    with pytest.raises(TypeError, match="unsupported"):
        diag.recursive_fingerprint(object())


def row():
    return dict(prompt_ids=[1, 2], responses=[3, 4], response_mask=[1, 0], old_log_probs=[-.1, -.2], rollout_index=0, step_index=0)


def token_prior(rows):
    result = []
    for r in rows:
        for pos, mask in enumerate(r["response_mask"]):
            if mask:
                result.append(dict(global_trainable_index=len(result), rollout_index=r["rollout_index"], step_index=r["step_index"],
                    response_token_position=pos, token_id=r["responses"][pos], token_repr="x", decoded_token="x", is_special_token=False))
    return result


def test_exact_saved_response_mask_mrope_multimodal_and_rollout_carrier_reused():
    data = Data()
    assert diag.audit_batch(data, [row()], 1)["mrope_shape"] == [1, 3, 4]
    for key in ("responses", "response_mask", "old_log_probs", "position_ids"):
        other = copy.deepcopy(data)
        other.batch[key] = other.batch[key].reshape(1, -1) if key == "position_ids" else other.batch[key] + 1
        with pytest.raises(ValueError):
            diag.audit_batch(other, [row()], 1)
    other = copy.deepcopy(data)
    other.non_tensor_batch.clear()
    with pytest.raises(ValueError, match="multimodal"):
        diag.audit_batch(other, [row()], 1)
    other = copy.deepcopy(data)
    other.meta_info["temperature"] = .8
    with pytest.raises(ValueError, match="temperature"):
        diag.audit_batch(other, [row()], 1)


def test_R_O_C_metrics_ratio_direction_mask_top50_and_no_fabricated_historical_tokens():
    rows = [row()]
    R, O, C = torch.tensor([[-.8, float("nan")]]), torch.tensor([[-.5, float("inf")]]), torch.tensor([[-.5, float("nan")]])
    tokens, summary = diag.analyze_tokens(rows, token_prior(rows), dict(R=R, O=O, C=C), torch.tensor([[1, 0]]), top_n=50)
    assert len(tokens) == 1
    assert tokens[0]["R_minus_O_diff"] == pytest.approx(-.3)
    assert tokens[0]["R_to_O_ratio"] == pytest.approx(math_exp(.3))
    assert summary["actor_side_initial_ratio"] == dict(formula="exp(C - O)", mean=1., min=1., max=1., clip_fraction=0.)
    for name, (current, old) in diag.PAIRS.items():
        assert summary["comparisons"][name] == diag.handoff.pair_metrics(
            [dict(R=R, O=O, C=C)[current][0, 0].item()], [dict(R=R, O=O, C=C)[old][0, 0].item()])
    assert summary["comparisons"]["rollout_vs_actor_current"]["clip_fraction_0p8_1p28"] == 1
    assert summary["historical_R_C_top50"]["available"] is False
    assert tokens[0]["historical_R_minus_C_diff"] is None
    assert set(summary["top_differences"]) == {"R_vs_C", "R_vs_O", "O_vs_C"}
    assert all(len(values) == 1 for values in summary["top_differences"].values())
    json.dumps(summary, allow_nan=False)


def math_exp(x):
    import math
    return math.exp(x)


@pytest.mark.parametrize("kind", ["order", "count", "nonfinite", "shape"])
def test_token_alignment_failures(kind):
    rows, prior = [row()], token_prior([row()])
    values = {k: torch.zeros(1, 2) for k in diag.VARIANTS}
    if kind == "order":
        prior[0]["token_id"] += 1
    elif kind == "count":
        prior = []
    elif kind == "nonfinite":
        values["C"][0, 0] = float("nan")
    else:
        values["C"] = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        diag.analyze_tokens(rows, prior, values, torch.tensor([[1, 0]]), top_n=50)


def test_top50_stable_sort_and_complete_metric_fields():
    rows = [dict(row(), responses=list(range(60)), response_mask=[1]*60)]
    R = torch.zeros(1, 60)
    C = torch.arange(60).reshape(1, 60).float() * -.01
    tokens, summary = diag.analyze_tokens(rows, token_prior(rows), dict(R=R, O=C, C=C), torch.ones(1, 60), top_n=50)
    assert len(tokens) == 60 and len(summary["top_differences"]["R_vs_C"]) == 50
    assert summary["top_differences"]["R_vs_C"][0]["global_trainable_index"] == 59
    assert summary["top_differences"]["O_vs_C"][0]["global_trainable_index"] == 0
    assert {"p50_abs", "p99_5_abs", "count_abs_diff_gt_0p20", "mean_signed_logprob_diff"} <= summary["comparisons"]["rollout_vs_actor_current"].keys()


def test_numeric_feasibility_is_diagnostic_not_formal_threshold_or_exact_equality():
    historical = dict(masked_token_count=1, max_abs_logprob_diff=.524)
    metric = diag.handoff.pair_metrics([1e-5], [0.])
    assert diag.feasibility(metric, historical)["actor_recompute_numerically_feasible"]
    assert diag.feasibility(metric, historical)["diagnostic_max_abs_limit"] == .001
    for delta in (.005, .01, .09):
        assert not diag.feasibility(diag.handoff.pair_metrics([delta], [0.]), historical)["actor_recompute_numerically_feasible"]
    assert not diag.feasibility(metric, dict(historical, masked_token_count=2))["actor_recompute_numerically_feasible"]


def test_historical_projection_repeat_guard_without_modifying_gate_threshold():
    R, C, mask = torch.tensor([[-.5, -.8]]), torch.tensor([[-.4, -.4]]), torch.ones(1, 2)
    history = compare_policy_logprobs(C, R, mask, clip_ratio_low=.2, clip_ratio_high=.28)
    assert not history["passed"]
    assert diag.repeat_guard(history, history)["passed"]
    other = compare_policy_logprobs(R, R, mask, clip_ratio_low=.2, clip_ratio_high=.28)
    assert not diag.repeat_guard(other, history)["passed"]


@pytest.fixture
def v456_history(v455_history):
    attempt, ctx, prior = v455_history
    tokens, analysis = lora.analyze_tokens(ctx["rows"], prior, values(), top_n=50)
    old = json.loads((attempt.reports / "bf16_sdpa_backend_diagnostic/summary.json").read_text())
    dest = attempt.reports / "bf16_lora_dtype_diagnostic"
    dest.mkdir()
    summary = {**old, **lora.META, **analysis, "variants": lora.VARIANTS, "experiment_informative": True}
    atomic_json(dest / "summary.json", summary)
    (dest / "token_diagnostics.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tokens))
    return attempt, ctx, tokens


def test_all_six_history_validators_and_readonly_source_protection(v456_history):
    attempt, ctx, tokens = v456_history
    assert diag.load_history(ctx) == tokens
    trees, files = diag.protected_sources(attempt.root, ctx["identity"]["run_id"], attempt.base)
    assert all(str(attempt.reports / d) in {str(t) for t in trees} for d in diag.HISTORY_DIRS)
    before = diag.handoff.source_checksums(trees, files)
    dest, _, _, protected = diag.reserve_reports(attempt.root, ctx["identity"]["run_id"], attempt.base)
    assert protected == before
    assert json.loads((dest / "summary.json").read_text())["execution_succeeded"] is False
    assert all(not Path(p).is_relative_to(dest) for p in before)
    assert diag.handoff.source_checksums(trees, files) == before
    with pytest.raises(FileExistsError, match="overwrite forbidden"):
        diag.reserve_reports(attempt.root, ctx["identity"]["run_id"], attempt.base)
    (attempt.reports / diag.HISTORY_DIRS[-1] / "extra.json").write_text("{}")
    with pytest.raises(RuntimeError, match="changed"):
        diag.handoff.assert_sources_unchanged(before, diag.handoff.source_checksums(trees, files))


@pytest.mark.parametrize("field,value", [("token_count", 1294), ("formal_sft_adapter_fingerprint", "other"),
    ("temperature", 1), ("execution_succeeded", False)])
def test_historical_lineage_fails_closed(v456_history, field, value):
    attempt, ctx, _ = v456_history
    path = attempt.reports / "bf16_lora_dtype_diagnostic/summary.json"
    summary = json.loads(path.read_text())
    summary[field] = value
    atomic_json(path, summary)
    with pytest.raises(ValueError, match="mismatch"):
        diag.load_history(ctx)


def test_software_matches_history_not_mock_claimed_gpu_result(monkeypatch):
    version = {k: "test" for k in backend.PACKAGES}
    version["verl"] = "0.6.1"
    monkeypatch.setattr(diag.importlib.metadata, "version", lambda k: version[k])
    assert diag.software_versions(dict(software_versions=version)) == version
    with pytest.raises(ValueError, match="software differs"):
        diag.software_versions(dict(software_versions=dict(version, torch="other")))


ACTOR_SOURCE = '''
class DataParallelPPOActor:
    def _forward_micro_batch(self, data, temperature):
        with torch.autocast(device_type="cuda", dtype=self.param_dtype):
            logits.div_(temperature)
    def compute_log_prob(self, data, calculate_entropy=False):
        self.actor_module.eval()
        with torch.no_grad():
            entropy, log_probs = self._forward_micro_batch(data, temperature=.7)
        entropys = None
        return log_probs, entropys
    def update_policy(self, data):
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        entropy, log_prob = self._forward_micro_batch(model_inputs, temperature)
        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
            old_log_prob = model_inputs["old_log_probs"]
        else:
            if on_policy:
                old_log_prob = log_prob.detach()
            else:
                old_log_prob = model_inputs["old_log_probs"]
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss = policy_loss_fn(old_log_prob=old_log_prob, log_prob=log_prob)
'''
LOSS_SOURCE = '''
def vanilla(old_log_prob, log_prob):
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
'''
CONFIG_SOURCE = '''
class FSDPActorConfig:
    use_rollout_log_probs: bool = False
'''
WORKER_SOURCE = '''
class ActorRolloutRefWorker:
    def compute_log_prob(self, batch):
        log_probs, entropys = self.actor.compute_log_prob(batch)
        return DataProto.from_dict(tensors={"old_log_probs": log_probs})
'''
TRAINER_SOURCE = '''
class RayPPOTrainer:
    def fit(self):
        if rollout_correction_config is not None and rollout_correction_config.get("bypass_mode", False):
            batch = apply_rollout_correction(batch)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)
'''


def probe(actor_source=ACTOR_SOURCE, loss_source=LOSS_SOURCE):
    return semantics.analyze_sources(actor_source=actor_source, actor_class="DataParallelPPOActor",
        loss_source=loss_source, loss_name="vanilla", config_source=CONFIG_SOURCE, config_class="FSDPActorConfig",
        worker_source=WORKER_SOURCE, trainer_source=TRAINER_SOURCE)


def test_semantic_probe_traces_conditional_false_denominator_and_separate_trainer_carrier():
    result = probe()
    assert result["use_rollout_log_probs_symbol_found"]
    assert result["ppo_semantic_acceptability"] == "not_supported_by_verl_implementation"
    assert "detach" in result["when_false"] and "model_inputs" in result["when_true"]
    assert result["actor_recompute_path_exists"]
    assert result["facts"]["temperature_division_count"] == 1
    assert result["facts"]["config_default"] == "False"
    assert result["evidence_nodes"]


@pytest.mark.parametrize("source", ["# use_rollout_log_probs=False means recompute\nclass Other: pass",
    ACTOR_SOURCE.replace('model_inputs["old_log_probs"]', 'model_inputs["rollout_log_probs"]'),
    ACTOR_SOURCE.replace("log_prob.detach()", "unknown_recompute()"),
    ACTOR_SOURCE.replace("old_log_prob=old_log_prob", "old_log_prob=other"),
    ACTOR_SOURCE.replace("self.config.ppo_epochs == 1", "self.config.ppo_epochs == 2"),
    ACTOR_SOURCE.replace('get_policy_loss_fn(loss_mode)', 'unrelated_loss_factory(loss_mode)')])
def test_semantic_source_drift_or_names_alone_are_undetermined(source):
    assert probe(source)["ppo_semantic_acceptability"] == "undetermined"


def test_ratio_source_drift_is_undetermined_and_source_evidence_hashes():
    assert probe(loss_source=LOSS_SOURCE.replace("log_prob - old_log_prob", "old_log_prob - log_prob"))["ppo_semantic_acceptability"] == "undetermined"
    node = semantics.find_node(ACTOR_SOURCE, "DataParallelPPOActor.compute_log_prob")
    record = semantics.evidence(Path("installed.py"), ACTOR_SOURCE, node, "compute_log_prob")
    assert record["line_end"] > record["line_start"]
    assert len(record["source_sha256"]) == len(record["snippet_sha256"]) == 64
    assert "torch.no_grad" in record["snippet"]


def test_official_extra_initial_assignment_and_runtime_selected_temperature_branch():
    source = ACTOR_SOURCE.replace("entropy, log_prob = self._forward_micro_batch(model_inputs, temperature)",
        'old_log_prob = model_inputs["old_log_probs"]\n        entropy, log_prob = self._forward_micro_batch(model_inputs, temperature)')
    source = source.replace('logits.div_(temperature)', '''if self.use_remove_padding:
                logits_rmpad.div_(temperature)
            else:
                if self.use_fused_kernels:
                    logits = fused(temperature)
                else:
                    logits.div_(temperature)''')
    trainer = TRAINER_SOURCE.replace('if rollout_correction_config is not None and rollout_correction_config.get("bypass_mode", False):',
        'bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)\n        if bypass_recomputing_logprobs:')
    result = semantics.analyze_sources(actor_source=source, actor_class="DataParallelPPOActor", loss_source=LOSS_SOURCE,
        loss_name="vanilla", config_source=CONFIG_SOURCE, config_class="FSDPActorConfig", worker_source=WORKER_SOURCE,
        trainer_source=trainer, correction_source='''
def apply_rollout_correction(batch):
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
''', execution_flags={"self.use_remove_padding": False, "self.use_fused_kernels": False})
    assert result["ppo_semantic_acceptability"] == "not_supported_by_verl_implementation"
    assert result["facts"]["temperature_division_count"] == 1
    assert result["actor_recompute_path_exists"] is True
    assert "rollout_log_probs" in result["facts"]["bypass_rollout_carrier"]


@pytest.mark.parametrize("fail_stage", [None, "independent_O_compute", "independent_C_compute", "report_publication"])
def test_cpu_distributed_orchestration_summary_and_source_integrity(v456_history, monkeypatch, fail_stage):
    attempt, ctx, _ = v456_history
    # Simulated distributed/CUDA plumbing, NOT a real FSDP/feasibility result.
    ctx = copy.deepcopy(ctx)
    ctx["identity"]["seed"] = 20260506
    monkeypatch.setattr(diag.handoff, "validate_attempt", lambda *a: ctx)
    monkeypatch.setattr(diag, "software_versions", lambda identity: identity["software_versions"])
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    import torch.distributed as dist
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "broadcast_object_list", lambda *a, **k: None)
    monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
    def gather(rows, local):
        rows[:] = [local, dict(copy.deepcopy(local), rank=1)]
    monkeypatch.setattr(dist, "all_gather_object", gather)
    import torch.distributed.device_mesh as mesh
    monkeypatch.setattr(mesh, "init_device_mesh", lambda *a, **k: "CPU fixture")
    from opensearch_vl_repro import model
    monkeypatch.setattr(model, "load_processor", lambda *a, **k: SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)))
    monkeypatch.setattr(diag, "load_gate_config", lambda *a: {})
    actor = Actor()
    actor.actor_module.weight.data.copy_(torch.diag(torch.tensor([.4, .5])))
    monkeypatch.setattr(diag, "construct_actor", lambda **k: (actor, dict(model_training=True, language_model_training=True,
        decoder_layers_training=36, decoder_layers_gradient_checkpointing=36, effective_attention_implementation="flash_attention_2")))
    monkeypatch.setattr(diag, "configure_one_update", lambda actor, *a: setattr(actor, "config", SimpleNamespace(strategy="fsdp2", use_rollout_log_probs=True)))
    def batch(**kwargs):
        result = Data()
        result.batch = {k: v.repeat(2, *([1]*(v.ndim-1))) for k, v in result.batch.items()}
        result.batch["response_mask"] = torch.ones(2, 2, dtype=torch.long)
        result.non_tensor_batch["multi_modal_inputs"] = np.concatenate([result.non_tensor_batch["multi_modal_inputs"]]*2)
        return result
    monkeypatch.setattr(diag, "build_dataproto", lambda rows, **k: batch(**k))
    # Two independent actual tiny model computations, same formal tuple order.
    def compute(data, calculate_entropy):
        actor.calls += 1
        actor.actor_module.eval()
        return -actor.actor_module(torch.ones(2, 2)), None
    actor.compute_log_prob = compute
    monkeypatch.setattr(diag, "inspect_verl_old_logprob_semantics", lambda *a, **k: dict(
        verl_version="0.6.1", source_hashes={}, use_rollout_log_probs_symbol_found=True,
        runtime_gate_config=dict(use_rollout_log_probs=True), ppo_semantic_acceptability="undetermined"))
    monkeypatch.setattr(diag.backend, "numeric_environment", lambda *a: {})
    class Stages:
        def __init__(self, *a, **k):
            pass
        def run(self, stage, operation):
            if stage == fail_stage and stage != "report_publication":
                raise ValueError("simulated peer failure")
            return operation()
    monkeypatch.setattr(diag, "CollectiveStages", Stages)
    original_json = diag.atomic_json
    def write(path, value):
        if fail_stage == "report_publication" and path.name == "summary.json" and value.get("execution_succeeded"):
            raise OSError("simulated final report failure")
        return original_json(path, value)
    monkeypatch.setattr(diag, "atomic_json", write)
    protected = diag.protected_sources(attempt.root, attempt.run_id, attempt.base)
    before = diag.handoff.source_checksums(*protected)
    args = SimpleNamespace(run_id=attempt.run_id, base_model_path=attempt.base, top_n=50, local_files_only=True)
    if fail_stage:
        with pytest.raises((OSError, ValueError), match="simulated"):
            diag.run_diagnostic(args, attempt.root)
    else:
        assert diag.run_diagnostic(args, attempt.root) == 0
        assert actor.calls == 2
    dest = attempt.reports / "actor_old_logprob_recompute_diagnostic"
    summary = json.loads((dest / "summary.json").read_text())
    assert summary["execution_succeeded"] is (fail_stage is None)
    assert summary["formal_rl_initialization_allowed"] is False and summary["optimizer_step_count"] == 0
    assert not (dest / "gate_manifest.json").exists()
    assert before == diag.handoff.source_checksums(*protected)
    if not fail_stage:
        assert summary["token_count"] == 4 and len(summary["per_rank"]) == 2
        assert summary["historical_alignment_repeat"]["passed"]
        assert summary["ppo_semantic_acceptability"] == "undetermined"
        assert (dest / "verl_source_audit.json").exists() and (dest / "token_diagnostics.jsonl").exists()


def test_runtime_guard_forbids_any_accidental_optimizer_or_update_and_is_restorable():
    actor = Actor()
    originals = diag.forbid_updates(actor)
    for owner, name, _ in originals:
        with pytest.raises(RuntimeError, match="forbidden"):
            getattr(owner, name)(None)
    for owner, name, method in originals:
        setattr(owner, name, method)
    assert actor.calls == 0 and not actor.actor_optimizer.state


def test_runtime_inspector_actual_files_class_factory_worker_discovery_and_sha(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    package = tmp_path / "verl"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("# installed-package CPU fixture\n")
    fake_verl = ModuleType("verl")
    fake_verl.__file__ = str(init)
    monkeypatch.setitem(sys.modules, "verl", fake_verl)
    def module(label, source):
        path = package / (label + ".py")
        path.write_text(source)
        value = ModuleType("test_installed_" + label)
        value.__file__ = str(path)
        monkeypatch.setitem(sys.modules, value.__name__, value)
        exec(compile(source, str(path), "exec"), value.__dict__)
        return value
    loss = module("loss", LOSS_SOURCE)
    actor_mod = module("actor", ACTOR_SOURCE + '\ndef get_policy_loss_fn(name):\n    return vanilla\n')
    actor_mod.vanilla = loss.vanilla
    config_mod = module("config", CONFIG_SOURCE)
    actor = actor_mod.DataParallelPPOActor()
    actor.config = config_mod.FSDPActorConfig()
    actor.config.policy_loss = {"loss_mode": "vanilla"}
    actor.config.use_rollout_log_probs = True
    actor.config.ppo_epochs = 1
    actor.config.ppo_mini_batch_size = 2
    actor.use_remove_padding = actor.use_fused_kernels = False
    (package / "worker.py").write_text('from verl.utils.fsdp_utils import fsdp_version\n' + WORKER_SOURCE)
    # Same class name in an unrelated worker must not defeat FSDP discovery.
    (package / "meg_peer.py").write_text(WORKER_SOURCE)
    (package / "trainer.py").write_text(TRAINER_SOURCE)
    result = semantics.inspect_verl_old_logprob_semantics(actor, version="0.6.1")
    assert result["ppo_semantic_acceptability"] == "not_supported_by_verl_implementation"
    assert result["actor_recompute_path_exists"] is True
    assert result["runtime_gate_config"]["use_rollout_log_probs"] is True
    assert set(result["source_files"]) == {"actor", "config", "loss", "worker", "trainer"}
    assert result["definition_locations"] and result["evidence"]
    for path, digest in result["source_hashes"].items():
        assert digest == diag.sha256_file(Path(path))
    assert semantics.inspect_verl_old_logprob_semantics(actor, version="other")["ppo_semantic_acceptability"] == "undetermined"


def test_real_gpu_path_static_reuse_and_forbidden_actual_calls():
    for module in (diag, semantics):
        source = inspect.getsource(module)
        tree = ast.parse(source)
        forbidden = {"VLLMStaticBackend", "SamplingParams", "DeepSeekJudge", "live_rewards", "step", "update_policy",
            "merge_actor_adapter", "merge_and_unload", "create_phase3_tool_registry", "save_pretrained", "backward", "save_checkpoint"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                assert name not in forbidden
        assert "gate_manifest.json" not in source and "update_started" not in source
    source = inspect.getsource(diag.run_diagnostic)
    for helper in ("construct_actor", "configure_one_update", "build_dataproto", "compare_policy_logprobs", "load_processor"):
        assert helper + "(" in source
    assert 'temperature=.7' in source and "init_device_mesh" in source and '"nccl"' in source
    assert diag.META["formal_rl_initialization_allowed"] is False and diag.META["optimizer_step_count"] == 0


def test_cli_defaults_and_requires_explicit_local_offline_inputs():
    import runpy
    script = runpy.run_path(str(ROOT / "scripts/diagnose_rl_actor_old_logprob_recompute.py"))
    parser = script["build_parser"]()
    args = parser.parse_args(["--run-id", "gate-c-v451-attempt1", "--base-model-path", "base", "--local-files-only"])
    assert args.top_n == 50
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-id", "x", "--base-model-path", "base"])
