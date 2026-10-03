"""CPU-only audit/ordering fixtures, never evidence of real vLLM/FSDP PASS."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opensearch_vl_repro.rl import gate_c, verl_policy_update
from opensearch_vl_repro.rl.actor_gate import atomic_json
from opensearch_vl_repro.rl.policy_alignment import (
    ALIGNMENT_CHECKS, alignment_artifact, alignment_checks, compare_policy_logprobs,
    require_policy_alignment, temperature_matches,
)
from opensearch_vl_repro.sft_tool_audit import sha256_file
from test_rl_gate_c import toy_actor
from test_rl_group import fixture_group
from opensearch_vl_repro.rl.old_logprob import ALIGNMENT_META, actor_alignment_artifact

ROOT = Path(__file__).resolve().parents[1]


def compare(current, old=(-1., -2.), mask=(1, 1), **kwargs):
    return compare_policy_logprobs(torch.tensor(current, dtype=torch.float64),
        torch.tensor(old, dtype=torch.float64), torch.tensor(mask),
        clip_ratio_low=.2, clip_ratio_high=.28, **kwargs)


def test_exact_and_small_difference_pass_without_bitwise_equality():
    audit = compare((-1., -2.))
    assert audit["passed"] and audit["mean_importance_ratio"] == 1
    assert audit["initial_clip_fraction"] == 0 and audit["masked_token_count"] == 2
    assert compare((-.99, -2.))["passed"]


@pytest.mark.parametrize("current,old,mask", [
    ((-.7,), (-1.,), (1,)),  # ratio > 1.28
    ((-1.3,), (-1.,), (1,)),  # ratio < .8
    ((-.85,), (-1.,), (1,)),  # inside clipping bounds but exceeds sanity .1
    ((.1,), (0.,), (1,)),  # STRICT < .1
    ((float("nan"),), (-1.,), (1,)),
    ((-1.,), (float("nan"),), (1,)),
    ((float("inf"),), (-1.,), (1,)),
    ((-1.,), (float("-inf"),), (1,)),
    ((1000.,), (-1.,), (1,)),  # finite logs, overflowing exp
    ((-1.,), (-1.,), (0,)),
    ((-1., -2.), (-1.,), (1,)),
    ((-1.,), (-1.,), (1, 1)),
    ((-1.,), (-1.,), (2,)),
])
def test_invalid_alignment_fails_closed_and_json_safe(current, old, mask):
    value = compare(current, old, mask)
    assert not value["passed"]
    json.dumps(value, allow_nan=False)


@pytest.mark.parametrize("padding", [1e10, float("nan"), float("inf")])
def test_padding_and_post_fatal_mask_zero_are_excluded(padding):
    audit = compare((-1., padding, padding), (-1., -padding, -padding), (1, 0, 0))
    assert audit["passed"] and audit["masked_token_count"] == 1 and audit["max_abs_logprob_diff"] == 0


def test_expected_token_count_mismatch_fails():
    audit = compare((-1., -2.), expected_masked_token_count=3)
    assert not audit["passed"] and not audit["token_count_match"]


def cpu_data(temperature=.7):
    return SimpleNamespace(batch={"old_log_probs": torch.tensor([[-1., -2.]]),
        "response_mask": torch.tensor([[1, 1]]), "responses": torch.tensor([[10, 11]])},
        meta_info={"temperature": temperature})


def cpu_audit(rank=0, delta=0., temperature=.7):
    data = cpu_data(temperature)
    actor = SimpleNamespace(compute_log_prob=lambda d, **kw: (d.batch["old_log_probs"] + delta, None))
    value = verl_policy_update.audit_pre_update_policy(actor, data,
        gate_c.load_gate_c_config(ROOT / "configs/rl_gate_c.yaml"), expected_masked_token_count=2)
    return {**value, "rank": rank}


def artifact(rows=None):
    return alignment_artifact(rows or [cpu_audit(0), cpu_audit(1)], gate_version=gate_c.GATE_C_VERSION,
        identity={"identity_sha256": "context"}, trajectory_group_id="group", policy_fingerprint="policy")


def test_temperature_must_equal_frozen_rollout_point_seven():
    assert temperature_matches(.7, .7) and cpu_audit()["passed"]
    assert not temperature_matches(1., .7)
    bad = cpu_audit(temperature=1.)
    assert not bad["checks"]["pre_update_policy_temperature_match"]
    with pytest.raises(RuntimeError): require_policy_alignment(artifact([bad, cpu_audit(1)]))


def test_audit_is_no_grad_read_only_and_success_update_uses_identical_data():
    actor = toy_actor()
    data = cpu_data()
    original_old = data.batch["old_log_probs"].clone()
    original_ids = data.batch["responses"].clone()
    before = actor.actor_module.q_proj.lora_B.detach().clone()
    before_state = copy.deepcopy(actor.actor_optimizer.state_dict())
    calls = []
    def compute(received, calculate_entropy):
        assert received is data and calculate_entropy is False and not torch.is_grad_enabled()
        actor.actor_module.eval()  # fixture reproduces pinned verl compute's mode change
        calls.append("compute")
        return received.batch["old_log_probs"] + .01, None
    actor.compute_log_prob = compute
    local = verl_policy_update.audit_pre_update_policy(actor, data,
        gate_c.load_gate_c_config(ROOT / "configs/rl_gate_c.yaml"), expected_masked_token_count=2)
    assert torch.equal(before, actor.actor_module.q_proj.lora_B)
    assert actor.actor_optimizer.state_dict() == before_state
    assert all(p.grad is None for p in actor.actor_module.parameters())
    def update(received):
        assert received is data
        assert torch.equal(received.batch["old_log_probs"], original_old)
        assert torch.equal(received.batch["responses"], original_ids)
        calls.append("update")
        actor.actor_module.train()  # pinned update_policy restores train
        actor.actor_module.q_proj.lora_B.sum().backward()
        actor.actor_optimizer.step(); actor.actor_optimizer.zero_grad(set_to_none=True)
        return {"actor/pg_loss": [.1], "actor/pg_clipfrac": [0.], "actor/ppo_kl": [.01],
                "actor/pg_clipfrac_lower": [0.], "future/actual_metric": [2.]}
    actor.update_policy = update
    alignment = artifact([{**local, "rank": 0}, {**local, "rank": 1}])
    # This legacy read-only audit does NOT authorize production updates anymore.
    with pytest.raises(ValueError, match="receipt"):
        verl_policy_update.policy_update_after_alignment(actor, data, alignment)
    metrics, grad = verl_policy_update.audited_policy_update(actor, data)
    assert calls == ["compute", "update"] and grad["optimizer_step_count"] == 1
    assert metrics["future/actual_metric"] == [2.]  # no metrics discarded/invented
    assert torch.equal(original_old, data.batch["old_log_probs"])


def test_failure_never_enters_audited_update_or_optimizer(monkeypatch):
    actor = toy_actor()
    calls = []
    actor.update_policy = lambda data: calls.append("actor.update_policy")
    monkeypatch.setattr(verl_policy_update, "audited_policy_update", lambda *a: calls.append("audited update"))
    actor.actor_optimizer.step = lambda: calls.append("optimizer.step")
    before = actor.actor_module.q_proj.lora_B.detach().clone()
    with pytest.raises(RuntimeError, match="0 steps"):
        verl_policy_update.policy_update_after_alignment(actor, cpu_data(), artifact([cpu_audit(0), cpu_audit(1, .3)]))
    assert calls == [] and not actor.actor_optimizer.state
    assert torch.equal(before, actor.actor_module.q_proj.lora_B)


def test_real_update_orchestration_peer_failure_persists_zero_step_report(tmp_path, monkeypatch):
    """Exercise the update entry point with CPU-injected model/collectives only."""
    import sys
    import torch.distributed as dist
    from opensearch_vl_repro import model
    from opensearch_vl_repro.rl import verl_actor_gate
    from opensearch_vl_repro.rl.group import publish_group

    args = SimpleNamespace(run_id="cpu-peer-failure", seed=1)
    output, reports = gate_c.paths_for(tmp_path, args.run_id)
    group = fixture_group()
    identity = {"identity_sha256": group["identity"]["context"], "gate_version": gate_c.GATE_C_VERSION,
                "effective_pre_update_policy_fingerprint": group["identity"]["pre_update_policy_fingerprint"],
                "rl_policy_execution_contract": {"CPU collective isolation": True}}
    gate_c.bind_run(output, reports, identity)
    staging = tmp_path / "group-staging"; staging.mkdir()
    torch.save({"fixture": torch.ones(1)}, staging / "mm.pt")
    publish_group(staging, output / "group", group)
    ctx = dict(identity=identity, gate=gate_c.load_gate_c_config(ROOT / "configs/rl_gate_c.yaml"),
        runtime_sft={}, sft={}, a22={}, adapter=tmp_path / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter",
        actor={"source_sft_adapter_fingerprint": group["identity"]["pre_update_policy_fingerprint"]})
    monkeypatch.setattr(verl_policy_update, "prepare_context", lambda *a: ctx)
    monkeypatch.setattr(verl_policy_update, "verify_same_policy_lineage", lambda *a: {"same_policy_lineage_verified": True})
    for name in ("is_available", "is_bf16_supported"):
        monkeypatch.setattr(torch.cuda, name, lambda: True)
    for name in ("set_device", "manual_seed_all", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed.device_mesh, "init_device_mesh", lambda *a, **k: "CPU mesh fixture")
    class Stages:
        def __init__(self, *a, **k): self.log = []
        def run(self, stage, operation):
            self.log.append(stage)
            return operation()
    monkeypatch.setattr(verl_actor_gate, "CollectiveStages", Stages)
    monkeypatch.setattr(model, "load_processor", lambda *a, **k: SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)))
    actor = toy_actor(); actor.config = SimpleNamespace(strategy="fsdp2")
    actor_audit = dict(model_training=True, language_model_training=True, decoder_layers_training=36,
        decoder_layers_gradient_checkpointing=36, effective_attention_implementation="flash_attention_2")
    monkeypatch.setattr(verl_actor_gate, "construct_rl_actor", lambda **k: (actor, actor_audit))
    monkeypatch.setattr(verl_policy_update, "require_rl_lora_dropout_runtime", lambda *a:
                        dict(nonzero_dropout_count=0, source_adapter_lora_dropout=.05,
                             rl_policy_execution_contract=identity["rl_policy_execution_contract"]))
    monkeypatch.setattr(verl_policy_update, "official_rloo", lambda *a, **k: ([1., -1.], [1., -1.]))
    monkeypatch.setattr(verl_policy_update, "configure_one_update", lambda *a: None)
    class CPUProto:
        def __init__(self):
            self.batch = {"old_log_probs": torch.tensor([[-.1, -.2], [-.1, -.2]]),
                          "response_mask": torch.ones(2, 2, dtype=torch.long)}
            self.meta_info = {"temperature": .7}
    data = CPUProto()
    monkeypatch.setitem(sys.modules, "verl", SimpleNamespace(DataProto=CPUProto))
    monkeypatch.setattr(verl_policy_update, "build_dataproto", lambda *a, **k: data)
    calls = []
    def compute(received, **kw):
        assert received is data and not torch.is_grad_enabled()
        calls.append("compute")
        return data.batch["old_log_probs"] + .01, None
    actor.compute_log_prob = compute
    actor.update_policy = lambda *a: calls.append("update")
    monkeypatch.setattr(verl_policy_update, "audited_policy_update", lambda *a: calls.append("audited_update"))
    actor.actor_optimizer.step = lambda: calls.append("step")
    def prepare(*a, **kw):
        # This orchestration test isolates the peer collective. Real O/C carrier
        # guards/forward observations have separate CPU production-helper tests.
        local = cpu_audit(0)
        local.update(ALIGNMENT_META)
        calls.extend(["compute O", "compute C"])
        proof = dict(rank=0, independent_actor_compute_count=2, rollout_log_probs_sha256="r",
            rl_policy_execution_contract=identity["rl_policy_execution_contract"],
            rl_dropout_boundary_audits={"old_before": {"nonzero_dropout_count": 0}})
        return dict(alignment=local, receipt=SimpleNamespace(artifact=proof, old_tensor=data.batch["old_log_probs"]),
                    handoff=dict(rank=0, rollout_log_probs_sha256="r", metrics=dict(all_finite=True, token_count_match=True)))
    monkeypatch.setattr(verl_policy_update, "prepare_actor_old_log_probs", prepare)
    def gather(rows, local):
        peer = copy.deepcopy(local)
        peer.update(rank=1)
        if local.get("comparison") == ALIGNMENT_META["comparison"]:
            peer.update(passed=False, max_abs_logprob_diff=.3, max_importance_ratio=1.35, initial_clip_fraction=.5)
            peer["checks"] = alignment_checks(peer)
        rows[:] = [local, peer]
    monkeypatch.setattr(dist, "all_gather_object", gather)
    before = actor.actor_module.q_proj.lora_B.detach().clone()
    with pytest.raises(RuntimeError, match="0 steps"):
        verl_policy_update.update(args, tmp_path)
    assert calls == ["compute O", "compute C"] and not actor.actor_optimizer.state
    assert torch.equal(before, actor.actor_module.q_proj.lora_B)
    failure = json.loads((reports / "gate_c_report.json").read_text())
    evidence = json.loads((output / "pre_update_policy_alignment.json").read_text())
    assert failure["stage"] == "pre_update_policy_alignment" and failure["optimizer_step_count"] == 0
    assert failure["passed"] is False and not failure["checks"]["pre_update_policy_no_initial_clipping"]
    assert evidence["per_rank"][0]["passed"] and not evidence["per_rank"][1]["passed"]
    assert (output / "update_started.json").exists()
    assert not (output / "gate_manifest.json").exists() and not (output / "update_verified.json").exists()
    assert not (output / "updated_actor").exists()


def receipt(output, rows=None):
    rows = [{**r, **ALIGNMENT_META} for r in (rows or [cpu_audit(0), cpu_audit(1)])]
    value = actor_alignment_artifact(rows, gate_version=gate_c.GATE_C_VERSION,
        identity={"identity_sha256": "context"}, trajectory_group_id="group", policy_fingerprint="policy")
    atomic_json(output / "pre_update_policy_alignment.json", value)
    return {"pre_update_policy_alignment_sha256": sha256_file(output / "pre_update_policy_alignment.json"),
        "pre_update_policy_alignment": value, "pre_update_policy_fingerprint": "policy",
        "training_token_counts": {"supervised_response_tokens": 2},
        "per_rank": [{"rank": r["rank"], "checks": r["checks"], "pre_update_policy_alignment": r} for r in value["per_rank"]]}


def verify(output, update):
    return gate_c.verify_policy_alignment_artifact(output, update,
        {"identity": {"trajectory_group_id": "group", "pre_update_policy_fingerprint": "policy"}},
        {"identity_sha256": "context"}, "policy")


def test_two_rank_artifact_binding_and_numeric_checks(tmp_path):
    update = receipt(tmp_path)
    assert verify(tmp_path, update)["passed"]
    # Even truthful re-hashing cannot turn rank1 failure into a valid audit.
    update = receipt(tmp_path, [cpu_audit(0), cpu_audit(1, .3)])
    with pytest.raises(RuntimeError): verify(tmp_path, update)
    update = receipt(tmp_path)
    update["per_rank"][1]["checks"]["pre_update_logprobs_finite"] = False
    with pytest.raises(ValueError): verify(tmp_path, update)


@pytest.mark.parametrize("failure", ["tamper", "rank1", "count", "policy"])
def test_finalize_rejects_alignment_failure_without_pass_manifest(tmp_path, monkeypatch, failure):
    args = SimpleNamespace(run_id="cpu-alignment-finalize")
    output, reports = gate_c.paths_for(tmp_path, args.run_id)
    identity = {"identity_sha256": "context", "effective_pre_update_policy_fingerprint": "policy"}
    gate_c.bind_run(output, reports, identity)
    group = {"identity": {"context": "context", "trajectory_group_id": "group",
                          "pre_update_policy_fingerprint": "policy"}}
    atomic_json(output / "group/group.json", group)
    atomic_json(output / "training_masks.json", {})
    update = receipt(output, [cpu_audit(0), cpu_audit(1, .3)] if failure == "rank1" else None)
    update.update(identity=identity, group_sha256=sha256_file(output / "group/group.json"),
                  training_masks_sha256=sha256_file(output / "training_masks.json"))
    if failure == "count": update["training_token_counts"]["supervised_response_tokens"] = 99
    if failure == "policy": update["pre_update_policy_fingerprint"] = "other"
    atomic_json(output / "update_verified.json", update)
    if failure == "tamper":
        atomic_json(output / "pre_update_policy_alignment.json", {"passed": True, "tampered": True})
    monkeypatch.setattr(gate_c, "prepare_context", lambda *a: {"identity": identity,
        "actor": {"source_sft_adapter_fingerprint": "policy"}})
    monkeypatch.setattr(gate_c, "read_group", lambda *a: group)
    with pytest.raises((ValueError, RuntimeError)): gate_c.finalize(args, tmp_path)
    assert not (output / "gate_manifest.json").exists()
    assert json.loads((reports / "gate_c_report.json").read_text())["passed"] is False


def test_checks_never_bulk_assigned_true_and_alignment_precedes_real_update():
    source = inspect.getsource(verl_policy_update)
    assert "dict.fromkeys(UPDATE_CHECKS, True)" not in source
    assert "dict.fromkeys(UPDATE_CHECKS, False)" in source
    # Catch the equivalent comprehension/loop forms, not just one spelling.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "fromkeys":
            if node.args and ast.unparse(node.args[0]) == "UPDATE_CHECKS":
                assert len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value is False
        if isinstance(node, (ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.For)):
            generators = node.generators if hasattr(node, "generators") else [node]
            if any(ast.unparse(g.iter) == "UPDATE_CHECKS" for g in generators):
                if isinstance(node, ast.For):
                    values = [child.value for child in ast.walk(node) if isinstance(child, (ast.Assign, ast.AnnAssign))]
                else:
                    values = [node.value if isinstance(node, ast.DictComp) else node.elt]
                assert not any(isinstance(value, ast.Constant) and value.value is True for value in values)
    assert source.index('before = run("pre_update_lora_snapshot"') < source.index('local_alignment = run(')
    assert source.index('run("pre_update_policy_alignment", lambda: require_policy_alignment') < source.index('metrics, grad_audit = run(')
    assert source.count("build_dataproto(rows,") == 1
    assert "current_log_probs, _ = actor.compute_log_prob(data, calculate_entropy=False)" in source
    assert "use_rollout_log_probs=True" in source
    assert all(name in gate_c.UPDATE_CHECKS for name in ALIGNMENT_CHECKS)
    assert '"policy_alignment.py"' in inspect.getsource(gate_c.prepare_context)
