"""CPU/mock execution-contract tests, not real verl/GPU validation."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from _rl_actor_fixture import actor, model, target, first_target, first_weight, sft_config
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl import rl_actor_semantics as semantics, verl_actor_gate, verl_policy_update, gate_c


def configured():
    value = model()
    return value, semantics.configure_rl_lora_dropout_runtime(value, sft_config())


def test_runtime_changes_only_p_preserves_source_weights_freezing_modes_checkpointing():
    value = model()
    weights = {n: p.detach().clone() for n, p in value.named_parameters()}
    frozen = {n: p.requires_grad for n, p in value.named_parameters()}
    modes = {n: m.training for n, m in value.named_modules()}
    dropout_ids = [id(m) for m in value.modules() if isinstance(m, torch.nn.Dropout)]
    config = copy.deepcopy(value.peft_config)
    audit = semantics.configure_rl_lora_dropout_runtime(value, sft_config())
    assert audit["lora_dropout_target_count"] == 252 and audit["nonzero_dropout_count"] == 0
    assert value.peft_config == config and value.peft_config["default"].lora_dropout == .05
    assert dropout_ids == [id(m) for m in value.modules() if isinstance(m, torch.nn.Dropout)]
    assert modes == {n: m.training for n, m in value.named_modules()}
    assert frozen == {n: p.requires_grad for n, p in value.named_parameters()}
    assert all(torch.equal(p, weights[n]) for n, p in value.named_parameters())
    assert all(m.training and m.gradient_checkpointing for m in value.language_model.layers)
    assert value.training and value.language_model.training
    assert semantics.configure_rl_lora_dropout_runtime(value, sft_config()) == audit


@pytest.mark.parametrize("bad", ["missing_target", "extra_layer", "duplicate_layer", "vision", "projector",
    "missing_A", "missing_B", "missing_dropout", "extra_A", "extra_B", "extra_dropout", "wrong_adapter",
    "disabled", "merged", "frozen_A", "frozen_B", "identity_dropout", "wrong_p", "wrong_cfg_p",
    "wrong_cfg_r", "wrong_cfg_alpha", "wrong_cfg_targets", "extra_cfg", "model_adapter", "aliased_dropout"])
def test_source_roster_fail_closed_without_zeroing_unknown_modules(bad):
    value = model()
    t = first_target(value)
    if bad == "missing_target": del value.language_model.layers[0].self_attn.q_proj
    elif bad == "extra_layer": value.language_model.layers.append(copy.deepcopy(value.language_model.layers[0]))
    elif bad == "duplicate_layer":
        value.other = torch.nn.Module(); value.other.language_model = copy.deepcopy(value.language_model)
    elif bad == "vision": value.visual.q_proj = target()
    elif bad == "projector": value.visual.merger.q_proj = target()
    elif bad.startswith("missing_"): delattr(t, {"missing_A": "lora_A", "missing_B": "lora_B", "missing_dropout": "lora_dropout"}[bad])
    elif bad in {"extra_A", "extra_B", "extra_dropout"}:
        getattr(t, {"extra_A": "lora_A", "extra_B": "lora_B", "extra_dropout": "lora_dropout"}[bad])["other"] = torch.nn.Identity()
    elif bad == "wrong_adapter": t.active_adapters = ["other"]
    elif bad == "disabled": t._disable_adapters = True
    elif bad == "merged": t.merged_adapters = ["default"]
    elif bad == "frozen_A": t.lora_A["default"].weight.requires_grad_(False)
    elif bad == "frozen_B": t.lora_B["default"].weight.requires_grad_(False)
    elif bad == "identity_dropout": t.lora_dropout["default"] = torch.nn.Identity()
    elif bad == "wrong_p": t.lora_dropout["default"].p = .1
    elif bad == "wrong_cfg_p": value.peft_config["default"].lora_dropout = 0.
    elif bad == "wrong_cfg_r": value.peft_config["default"].r = 8
    elif bad == "wrong_cfg_alpha": value.peft_config["default"].lora_alpha = 16
    elif bad == "wrong_cfg_targets": value.peft_config["default"].target_modules = ["q_proj"]
    elif bad == "extra_cfg": value.peft_config["other"] = copy.deepcopy(value.peft_config["default"])
    elif bad == "model_adapter": value.active_adapters = ["other"]
    else: value.language_model.layers[1].self_attn.q_proj.lora_dropout["default"] = t.lora_dropout["default"]
    with pytest.raises(ValueError): semantics.configure_rl_lora_dropout_runtime(value, sft_config())
    assert value.peft_config["default"].lora_dropout == (0. if bad == "wrong_cfg_p" else .05)


@pytest.mark.parametrize("where", ["base", "visual", "projector", "attention_module", "attention_config"])
def test_nonlora_policy_dropout_rejected_not_auto_zeroed(where):
    value = model()
    if where in {"base", "visual", "projector"}:
        parent = value.base if where == "base" else value.visual if where == "visual" else value.visual.merger
        parent.dropout = torch.nn.Dropout(.1)
    elif where == "attention_module": value.language_model.layers[0].self_attn.attention_dropout = .1
    else: value.config = SimpleNamespace(text_config=SimpleNamespace(attention_dropout=.1))
    with pytest.raises(ValueError, match="dropout"): semantics.configure_rl_lora_dropout_runtime(value, sft_config())
    assert first_target(value).lora_dropout["default"].p == .05


@pytest.mark.parametrize("phase", ["initial", "fresh", "native"])
def test_explicit_rl_constructor_leaves_raw_constructor_semantics_unchanged(monkeypatch, phase):
    raw = model()
    calls = []
    def construct(**kwargs):
        calls.append(kwargs)
        assert first_target(raw).lora_dropout["default"].p == .05
        return SimpleNamespace(actor_module=raw), {"raw_gate_a": True}
    monkeypatch.setattr(verl_actor_gate, "construct_actor", construct)
    value, audit = verl_actor_gate.construct_rl_actor(config=sft_config(), gate={}, adapter=Path(phase), mesh=None)
    assert len(calls) == 1 and audit["raw_gate_a"]
    assert value.actor_module is raw and audit["rl_lora_dropout_runtime"]["nonzero_dropout_count"] == 0
    assert raw.peft_config["default"].lora_dropout == .05


def test_runtime_semantic_fingerprint_stable_in_train_eval_and_rejects_reset():
    value, audit = configured()
    value.eval()
    other = semantics.require_rl_lora_dropout_runtime(value, sft_config())
    assert not other["model_training"] and other["lora_dropout_runtime_sha256"] == audit["lora_dropout_runtime_sha256"]
    value.train()
    assert semantics.require_rl_lora_dropout_runtime(value, sft_config())["lora_dropout_runtime_sha256"] == other["lora_dropout_runtime_sha256"]
    first_target(value).lora_dropout["default"].p = .05
    with pytest.raises(ValueError): semantics.require_rl_lora_dropout_runtime(value, sft_config())


@pytest.mark.parametrize("mutation", ["p_reset_before", "p_reset_forward", "p_reset_after_forward", "eval_forward",
    "no_grad_forward", "no_forward", "frozen_lora", "wrong_adapter", "base_dropout", "caught_bad_forward"])
def test_actual_update_forward_or_step_fail_closed_with_zero_completed_steps(mutation):
    value = actor()
    audit = semantics.configure_rl_lora_dropout_runtime(value.actor_module, sft_config())
    receipt = dict(rl_policy_execution_contract=audit["rl_policy_execution_contract"],
                   lora_dropout_runtime_sha256=audit["lora_dropout_runtime_sha256"])
    p = first_weight(value.actor_module)
    before = p.detach().clone()
    t = first_target(value.actor_module)
    if mutation == "p_reset_before": t.lora_dropout["default"].p = .05
    def update(data):
        value.actor_module.train()
        if mutation == "p_reset_forward": t.lora_dropout["default"].p = .05
        elif mutation == "eval_forward": value.actor_module.eval()
        elif mutation == "frozen_lora": t.lora_A["default"].weight.requires_grad_(False)
        elif mutation == "wrong_adapter": t.active_adapters = ["other"]
        elif mutation == "base_dropout": value.actor_module.base.dropout = torch.nn.Dropout(.1)
        elif mutation == "caught_bad_forward":
            value.actor_module.eval()
            with pytest.raises(ValueError): value.actor_module(torch.ones(1))
            value.actor_module.train()  # a later valid forward must not erase the rejection
        if mutation == "no_grad_forward":
            with torch.no_grad(): value.actor_module(torch.ones(1))
        elif mutation != "no_forward": value.actor_module(torch.ones(1)).sum().backward()
        else: p.sum().backward()
        if mutation == "p_reset_after_forward": t.lora_dropout["default"].p = .05
        value.actor_optimizer.step()
        return {"actor/pg_loss": [.1]}
    value.update_policy = update
    with pytest.raises((ValueError, RuntimeError)) as failure:
        verl_policy_update.audited_policy_update(value, None, runtime_receipt=receipt)
    assert getattr(failure.value, "optimizer_step_count", 0) == 0
    assert not value.actor_optimizer.state and torch.equal(p, before)
    assert not value.actor_module._forward_pre_hooks


def test_actual_train_forward_p0_audited_and_official_api_step_once():
    value = actor()
    audit = semantics.configure_rl_lora_dropout_runtime(value.actor_module, sft_config())
    receipt = dict(rl_policy_execution_contract=audit["rl_policy_execution_contract"],
                   lora_dropout_runtime_sha256=audit["lora_dropout_runtime_sha256"])
    def update(data):
        value.actor_module.train()
        for _ in range(2): value.actor_module(torch.ones(1)).sum().backward()
        value.actor_optimizer.step()
        value.actor_optimizer.zero_grad(set_to_none=True)
        return {"actor/pg_loss": [.1]}
    value.update_policy = update
    _, result = verl_policy_update.audited_policy_update(value, None, runtime_receipt=receipt)
    assert result["optimizer_step_count"] == 1
    evidence = result["rl_dropout_execution_audit"]
    assert evidence["forward_count"] == 2 and evidence["train_mode_forward_seen"]
    assert all(row["model_training"] and row["grad_enabled"] for row in evidence["forwards"])
    assert all(all(d["training"] and d["p"] == 0 for d in row["dropout_modules"]) for row in evidence["forwards"])
    semantics.require_update_forward_audit(evidence, receipt["rl_policy_execution_contract"], receipt["lora_dropout_runtime_sha256"])


@pytest.mark.parametrize("bad", ["contract", "group_fp", "identity_fp", "context", "group_policy"])
def test_same_weights_different_runtime_semantics_rejected_before_actor_old(tmp_path, bad):
    from test_rl_verl_policy_update import lineage_fixture
    from opensearch_vl_repro.rl.old_logprob import verify_same_policy_lineage
    ctx, group, output = lineage_fixture(tmp_path)
    if bad == "contract": group["rl_policy_execution_contract"]["runtime_effective_lora_dropout"] = .05
    elif bad == "group_fp": group["effective_pre_update_policy_fingerprint"] = "f" * 64
    elif bad == "identity_fp": ctx["identity"]["effective_pre_update_policy_fingerprint"] = "f" * 64
    elif bad == "context": group["identity"]["context"] = "f" * 64
    else: group["identity"]["pre_update_policy_fingerprint"] = ctx["actor"]["source_sft_adapter_fingerprint"]
    with pytest.raises(ValueError): verify_same_policy_lineage(ctx, group, output)


@pytest.mark.parametrize("boundary", ["O_before", "O_after", "C_before", "C_after", "before_update"])
def test_old_current_and_live_receipt_reject_dropout_reset(tmp_path, boundary):
    from test_rl_verl_policy_update import lineage_fixture, carrier_fixture, prepared_fixture
    from opensearch_vl_repro.rl import old_logprob
    if boundary == "before_update":
        *_, value, data, calls, counts, proof = prepared_fixture(tmp_path)
        first_target(value.actor_module).lora_dropout["default"].p = .05
        with pytest.raises(ValueError): old_logprob.verify_update_receipt(value, data, proof["alignment"], proof["receipt"])
    else:
        ctx, group, output = lineage_fixture(tmp_path)
        lineage = old_logprob.verify_same_policy_lineage(ctx, group, output)
        value, data, calls, counts = carrier_fixture(group)
        if boundary == "O_before": first_target(value.actor_module).lora_dropout["default"].p = .05
        else:
            compute = value.compute_log_prob
            def reset(*args, **kwargs):
                if boundary == "C_before" and len(calls) == 1:
                    first_target(value.actor_module).lora_dropout["default"].p = .05
                result = compute(*args, **kwargs)
                if boundary == "O_after" and len(calls) == 1 or boundary == "C_after" and len(calls) == 2:
                    first_target(value.actor_module).lora_dropout["default"].p = .05
                return result
            value.compute_log_prob = reset
        with pytest.raises(ValueError): old_logprob.prepare_actor_old_log_probs(value, data, ctx["gate"], lineage=lineage,
            expected_masked_token_count=counts["supervised_response_tokens"], rank=0)
    assert not value.actor_optimizer.state


@pytest.mark.parametrize("bad", ["signature", "roster", "source", "p", "nonzero", "contract"])
def test_rehashed_forged_audit_not_accepted_as_finalizer_evidence(bad):
    _, audit = configured()
    contract = copy.deepcopy(audit["rl_policy_execution_contract"])
    if bad == "signature": audit["lora_dropout_runtime_sha256"] = "x"
    elif bad == "roster": audit["lora_target_roster"].pop()
    elif bad == "source": audit["source_adapter_lora_dropout"] = 0.
    elif bad == "p": audit["dropout_roster"][0]["p"] = .05
    elif bad == "nonzero": audit["nonzero_dropout_count"] = 1
    else: audit["rl_policy_execution_contract"]["runtime_effective_lora_dropout"] = .05
    with pytest.raises(ValueError): semantics.require_runtime_audit(audit, contract)


def test_production_constructor_and_finalizer_boundaries_are_explicit():
    raw = inspect.getsource(verl_actor_gate.construct_actor)
    assert "configure_rl_lora_dropout_runtime" not in raw
    source = inspect.getsource(verl_policy_update.update)
    assert "construct_actor(" not in source and source.count("construct_rl_actor(") == 2
    assert "require_rl_lora_dropout_runtime(fresh.actor_module" in source
    assert source.index('run("rl_dropout_before_update"') < source.index("update_entered = True")
    assert "verify_dropout_artifacts" in inspect.getsource(gate_c.finalize)
    for name in ("rl_actor_semantics.py", "verl_actor_gate.py"):
        assert name in inspect.getsource(gate_c.prepare_context)
    assert gate_c.GATE_C_VERSION == "minimum-rl-integration-c-v3-actor-old-zero-lora-dropout"


@pytest.mark.parametrize("bad", ["missing_contract", "wrong_contract", "missing_initial", "missing_fresh", "missing_native",
    "missing_forward", "zero_forward_count", "eval_forward", "nonzero_forward", "source_config", "saved_config",
    "inline_mismatch", "aggregate", "false_check"])
def test_finalizer_requires_live_dropout_evidence_not_boolean_flags(tmp_path, bad):
    from test_rl_verl_policy_update import artifact_fixture
    ctx, group, output, _, update = artifact_fixture(tmp_path)
    row = update["per_rank"][0]
    if bad == "missing_contract": del update["rl_policy_execution_contract"]
    elif bad == "wrong_contract": update["rl_policy_execution_contract"] = {"runtime_effective_lora_dropout": .05}
    elif bad in {"missing_initial", "missing_fresh", "missing_native"}:
        del row[{"missing_initial": "initial_runtime_audit", "missing_fresh": "fresh_runtime_audit", "missing_native": "native_runtime_audit"}[bad]]
    elif bad == "missing_forward": row["rl_dropout_execution_audit"]["forwards"] = []
    elif bad == "zero_forward_count": row["rl_dropout_execution_audit"]["forward_count"] = 0
    elif bad == "eval_forward": row["rl_dropout_execution_audit"]["forwards"][0]["model_training"] = False
    elif bad == "nonzero_forward": row["rl_dropout_execution_audit"]["forwards"][0]["dropout_modules"][0]["p"] = .05
    elif bad in {"source_config", "saved_config"}:
        path = ctx["adapter"] if bad == "source_config" else output / "updated_actor/adapter"
        (path / "adapter_config.json").write_text(json.dumps({"lora_dropout": 0.}), encoding="utf-8")
    elif bad == "inline_mismatch": row["gradient_audit"] = {"rl_dropout_execution_audit": {}}
    elif bad == "aggregate": update["rl_update_train_mode_forward_seen"] = False
    else: row["checks"]["rl_lora_dropout_zero_during_update"] = False
    with pytest.raises(ValueError): gate_c.verify_dropout_artifacts(ctx, group, update, output)
    assert not (output / "gate_manifest.json").exists()


def test_missing_runtime_setup_rejected_before_old_and_optimizer(tmp_path):
    from test_rl_verl_policy_update import lineage_fixture, carrier_fixture
    from opensearch_vl_repro.rl import old_logprob
    ctx, group, output = lineage_fixture(tmp_path)
    lineage = old_logprob.verify_same_policy_lineage(ctx, group, output)
    value, data, calls, counts = carrier_fixture(group)
    value.actor_module = model()  # actual source .05, no configure call
    with pytest.raises(ValueError): old_logprob.prepare_actor_old_log_probs(value, data, ctx["gate"], lineage=lineage,
        expected_masked_token_count=counts["supervised_response_tokens"], rank=0)
    assert not calls and not value.actor_optimizer.state


def test_failure_after_real_step_does_not_falsely_report_zero_steps():
    value = actor()
    audit = semantics.configure_rl_lora_dropout_runtime(value.actor_module, sft_config())
    def update(data):
        value.actor_module.train()
        value.actor_module(torch.ones(1)).sum().backward()
        value.actor_optimizer.step()
        raise RuntimeError("post-step failure")
    value.update_policy = update
    with pytest.raises(RuntimeError) as exc:
        verl_policy_update.audited_policy_update(value, None, runtime_receipt=audit)
    assert exc.value.optimizer_step_count == 1 and exc.value.optimizer_step_started


def test_real_installed_peft_save_fresh_load_preserves_source_metadata(tmp_path):
    """Actual installed PEFT ModuleDict/save path, tiny CPU model, no HF downloads."""
    peft = pytest.importorskip("peft")
    def base():
        value = torch.nn.Module()
        value.config = {"model_type": "cpu_test"}
        value.language_model = torch.nn.Module()
        value.language_model.layers = torch.nn.ModuleList()
        for _ in range(36):
            layer = torch.nn.Module(); layer.self_attn = torch.nn.Module(); layer.mlp = torch.nn.Module()
            for name in semantics.LORA_TARGETS:
                setattr(layer.self_attn if name in semantics.LORA_TARGETS[:4] else layer.mlp, name, torch.nn.Linear(1, 1, bias=False))
            value.language_model.layers.append(layer)
        return value
    value = peft.get_peft_model(base(), peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=list(semantics.LORA_TARGETS)))
    semantics.configure_rl_lora_dropout_runtime(value, sft_config())
    value.save_pretrained(tmp_path / "adapter", safe_serialization=True)
    assert json.loads((tmp_path / "adapter/adapter_config.json").read_text())["lora_dropout"] == .05
    source_bytes = {p.name: p.read_bytes() for p in (tmp_path / "adapter").iterdir() if p.is_file()}
    fresh = peft.PeftModel.from_pretrained(base(), tmp_path / "adapter", is_trainable=True, local_files_only=True)
    raw_dropouts = [m for m in fresh.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(raw_dropouts) == 252 and all(m.p == .05 for m in raw_dropouts)
    audit = semantics.configure_rl_lora_dropout_runtime(fresh, sft_config())
    fresh.train()
    assert audit["nonzero_dropout_count"] == 0 and fresh.peft_config["default"].lora_dropout == .05
    assert source_bytes == {p.name: p.read_bytes() for p in (tmp_path / "adapter").iterdir() if p.is_file()}
