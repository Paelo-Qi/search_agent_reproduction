"""CPU/stub production carrier regression; never real GPU/verl Gate evidence."""
import copy
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opensearch_vl_repro.rl.verl_policy_update import configure_one_update, audit_pre_update_policy
from opensearch_vl_repro.rl import gate_c, old_logprob as production, verl_policy_update
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, atomic_json
from opensearch_vl_repro.rl.group import group_identity
from opensearch_vl_repro.rl.rollout_sync import merge_identity, validate_merged_files
from opensearch_vl_repro.rl.training_batch import mask_artifact, training_rows
from opensearch_vl_repro.inference.adapter import adapter_file_identity
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.sft_tool_audit import sha256_file
from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from _rl_actor_fixture import actor as toy_actor, first_weight, sft_config
from opensearch_vl_repro.rl.rl_actor_semantics import execution_contract, effective_policy_fingerprint, configure_rl_lora_dropout_runtime
from test_rl_group import fixture_group

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    ppo_mini_batch_size: int = 0
    ppo_micro_batch_size_per_gpu: int = 0
    ppo_epochs: int = 0
    shuffle: bool = True
    clip_ratio_low: float = 0.
    clip_ratio_high: float = 0.
    entropy_coeff: float = 1.
    use_kl_loss: bool = True
    use_rollout_log_probs: bool = False
    loss_agg_mode: str = ""


def test_formal_runtime_config_still_true_one_epoch_no_new_diagnostic_config():
    actor = SimpleNamespace(config=Config())
    configure_one_update(actor, 3, dict(clip_ratio_low=.2, clip_ratio_high=.28))
    assert actor.config.use_rollout_log_probs is True
    assert actor.config.ppo_mini_batch_size == 3 and actor.config.ppo_epochs == 1
    assert actor.config.ppo_micro_batch_size_per_gpu == 1
    assert (actor.config.clip_ratio_low, actor.config.clip_ratio_high) == (.2, .28)
    with pytest.raises(ValueError):
        configure_one_update(actor, 0, {})


def test_logprobs_first_entropy_second_preserves_formal_alignment_return_contract():
    calls = []
    old = torch.tensor([[-.1, -.2]])
    def compute(data, calculate_entropy):
        assert not torch.is_grad_enabled() and calculate_entropy is False
        calls.append(data)
        return old.clone(), None  # pinned verl returns (log_probs, entropy)
    actor = SimpleNamespace(compute_log_prob=compute)
    data = SimpleNamespace(batch=dict(old_log_probs=old, response_mask=torch.ones_like(old)), meta_info=dict(temperature=.7))
    result = audit_pre_update_policy(actor, data, dict(clip_ratio_low=.2, clip_ratio_high=.28, vllm=dict(temperature=.7)), expected_masked_token_count=2)
    assert len(calls) == 1 and result["passed"] and result["mean_importance_ratio"] == 1


def lineage_fixture(root):
    """Real file hashes/merge identity, deliberately fake weights and NO model load."""
    adapter = root / "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
    atomic_json(adapter / "adapter_config.json", {"lora_dropout": .05})
    (adapter / "adapter_model.safetensors").write_bytes(b"CPU source fixture")
    fp = adapter_file_identity(adapter)["adapter_fingerprint"]
    actor = dict(actor_adapter_fingerprint=fp, source_sft_adapter_fingerprint=fp,
        actor_source_kind="formal_sft_checkpoint_fallback", source_sft_lineage=["main_a_1k", "main_b_2k"],
        actor_gate_identity_sha256=None, base_model=BASE_MODEL, base_revision=BASE_REVISION,
        formal_rl_initialization_allowed=False)
    gate = gate_c.load_gate_c_config(ROOT / "configs/rl_gate_c.yaml")
    contract = execution_contract(sft_config())
    effective_fp = effective_policy_fingerprint(fp, contract)
    identity = dict(run_id="cpu-v461", gate_version=gate_c.GATE_C_VERSION, source_sft_actor=actor,
        rl_policy_execution_contract=contract, effective_pre_update_policy_fingerprint=effective_fp,
        base_model=BASE_MODEL, base_revision=BASE_REVISION, logprobs_mode="processed_logprobs",
        prompt_id="rl_000001", runtime_protocol=RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION)
    identity["identity_sha256"] = canonical_json_sha256(identity)
    output = root / "outputs/rl_gate_c/cpu-v461"
    gid = group_identity(prompt_id=identity["prompt_id"], policy_fingerprint=effective_fp,
        rollout_fingerprint=canonical_json_sha256(gate), attempt="new-collection", context=identity["identity_sha256"])
    group = fixture_group(); group["identity"] = gid
    group.update(rl_policy_execution_contract=contract, effective_pre_update_policy_fingerprint=effective_fp)
    for member in group["members"]:
        member["identity"] = copy.deepcopy(gid)
    merged = output / "merged-new-collection"
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        atomic_json(merged / name, {})
    (merged / "model.safetensors").write_bytes(b"CPU static fixture")
    merge = merge_identity(actor=actor, versions={}, file_hashes=validate_merged_files(merged))
    atomic_json(merged / "merge_manifest.json", dict(identity=merge, actor_provenance=actor,
        merge_complete=True, no_active_peft=True, fresh_hf_forward_finite=True, merge_hf_destroyed=True, reload_hf_destroyed=True))
    group["merged_checkpoint_fingerprint"] = merge["merged_checkpoint_fingerprint"]
    atomic_json(output / "group/group.json", group)
    ctx = dict(actor=actor, identity=identity, gate=gate, adapter=adapter, versions={}, sft=sft_config())
    return ctx, group, output


def carrier_fixture(group, *, delta=.5, current_delta=0., mutation=None):
    """Actual CPU torch forwards, official actor API mocked, no external package."""
    rows, counts = training_rows(group, [1., -1.], rollout_schema=True)
    data = SimpleNamespace(batch=dict(input_ids=torch.tensor([[1, 2, 3, 4]] * 2),
        attention_mask=torch.ones(2, 4, dtype=torch.long), position_ids=torch.arange(4).repeat(2, 3, 1),
        responses=torch.tensor([r["responses"] for r in rows]),
        response_mask=torch.tensor([r["response_mask"] for r in rows]),
        rollout_log_probs=torch.tensor([r["rollout_log_probs"] for r in rows], dtype=torch.float32),
        advantages=torch.tensor([[r["advantage"]] * 2 for r in rows])),
        non_tensor_batch=dict(multi_modal_inputs=[dict(pixel_values=torch.ones(2, 8), image_grid_thw=torch.tensor([[1, 2, 2]]))] * 2),
        meta_info=dict(temperature=.7, micro_batch_size=1, use_dynamic_bsz=False))
    actor = toy_actor(); actor.config = Config()
    configure_rl_lora_dropout_runtime(actor.actor_module, sft_config())
    configure_one_update(actor, len(rows), dict(clip_ratio_low=.2, clip_ratio_high=.28))
    calls = []
    def compute(received, calculate_entropy):
        assert received is not data and not torch.is_grad_enabled() and calculate_entropy is False
        calls.append(received)
        actor.actor_module.eval()
        result = actor.actor_module(received.batch["rollout_log_probs"] - delta)
        if mutation is not None:
            mutation(actor, received)
        return result + (current_delta if len(calls) == 2 else 0.), None
    actor.compute_log_prob = compute
    return actor, data, calls, counts


def prepared_fixture(root, **kwargs):
    ctx, group, output = lineage_fixture(root)
    lineage = production.verify_same_policy_lineage(ctx, group, output)
    actor, data, calls, counts = carrier_fixture(group, **kwargs)
    proof = production.prepare_actor_old_log_probs(actor, data, ctx["gate"], lineage=lineage,
        expected_masked_token_count=counts["supervised_response_tokens"], rank=0)
    return ctx, group, output, lineage, actor, data, calls, counts, proof


def test_large_legal_handoff_two_independent_forwards_strict_actor_alignment_and_one_update(tmp_path):
    ctx, group, output, lineage, actor, data, calls, counts, proof = prepared_fixture(tmp_path)
    assert len(calls) == 2 and calls[0] is not calls[1]
    assert "old_log_probs" not in calls[0].batch and "old_log_probs" in calls[1].batch
    rollout = torch.tensor([[-.1, -.2]] * 2)
    assert torch.equal(data.batch["rollout_log_probs"], rollout)
    assert torch.equal(data.batch["old_log_probs"], rollout - .5)
    assert data.batch["old_log_probs"].untyped_storage().data_ptr() != data.batch["rollout_log_probs"].untyped_storage().data_ptr()
    assert proof["alignment"]["passed"] and proof["alignment"]["max_abs_logprob_diff"] == 0
    assert proof["alignment"]["mean_importance_ratio"] == 1 and proof["alignment"]["initial_clip_fraction"] == 0
    assert proof["alignment"]["max_abs_logprob_diff_bound"] == .1
    assert (proof["alignment"]["ratio_lower_bound"], proof["alignment"]["ratio_upper_bound"]) == (.8, 1.28)
    assert proof["handoff"]["metrics"]["max_abs_logprob_diff"] == pytest.approx(.5)
    assert proof["handoff"]["metrics"]["initial_clip_fraction"] == 1
    assert "passed" not in proof["handoff"] and "passed" not in proof["handoff"]["metrics"]
    assert proof["handoff"]["informational_only"] and not proof["handoff"]["gate_blocking"]
    assert proof["receipt"].artifact["formal_rl_initialization_allowed"] is False
    assert proof["receipt"].artifact["parameters_unchanged"] and proof["receipt"].artifact["rng_unchanged"]
    before = production.fingerprint(data.batch["advantages"])
    updates = []
    def update(received):
        assert received is data and actor.config.use_rollout_log_probs is True
        assert torch.equal(received.batch["old_log_probs"], rollout - .5)
        updates.append("official API stub")
        actor.actor_module.train()
        actor.actor_module(torch.ones(1)).sum().backward()
        actor.actor_optimizer.step(); actor.actor_optimizer.zero_grad(set_to_none=True)
        return {"actor/pg_loss": [.1], "actor/pg_clipfrac": [0.]}
    actor.update_policy = update
    metrics, grad = verl_policy_update.policy_update_after_alignment(actor, data, proof["alignment"], proof["receipt"])
    assert updates == ["official API stub"] and grad["optimizer_step_count"] == 1
    assert metrics["actor/pg_loss"] == [.1] and production.fingerprint(data.batch["advantages"]) == before
    assert torch.equal(data.batch["rollout_log_probs"], rollout)


@pytest.mark.parametrize("bad", ["group_policy", "actor_fp", "actor_kind", "adapter_path", "adapter_file", "context", "rollout_config", "base", "revision", "merge_source", "merge_files", "merge_fp"])
def test_stale_policy_and_wrong_source_fail_before_any_actor_compute(tmp_path, bad):
    ctx, group, output = lineage_fixture(tmp_path)
    if bad in {"group_policy", "context", "rollout_config"}:
        old = group["identity"]
        gid = group_identity(prompt_id=old["prompt_id"],
            policy_fingerprint="0" * 64 if bad == "group_policy" else old["pre_update_policy_fingerprint"],
            rollout_fingerprint="1" * 64 if bad == "rollout_config" else old["rollout_config_fingerprint"],
            attempt=old["collection_attempt"], context="wrong" if bad == "context" else old["context"])
        group["identity"] = gid
        for member in group["members"]: member["identity"] = copy.deepcopy(gid)
    elif bad == "actor_fp": ctx["actor"]["actor_adapter_fingerprint"] = "2" * 64
    elif bad == "actor_kind": ctx["actor"]["actor_source_kind"] = "gate_a_actor"
    elif bad == "adapter_path": ctx["adapter"] = output / "updated_actor/adapter"
    elif bad == "adapter_file": (ctx["adapter"] / "adapter_model.safetensors").write_bytes(b"different policy")
    elif bad == "base": ctx["identity"]["base_model"] = "wrong"
    elif bad == "revision": ctx["actor"]["base_revision"] = "wrong"
    elif bad == "merge_files": (output / "merged-new-collection/model.safetensors").write_bytes(b"different merge")
    elif bad == "merge_fp": group["merged_checkpoint_fingerprint"] = "3" * 64
    else:
        path = output / "merged-new-collection/merge_manifest.json"
        value = json.loads(path.read_text()); value["identity"]["actor_adapter_fingerprint"] = "4" * 64
        # Re-hashing a wrong source cannot make it same-policy.
        value["identity"]["merged_checkpoint_fingerprint"] = canonical_json_sha256({k: v for k, v in value["identity"].items() if k != "merged_checkpoint_fingerprint"})
        group["merged_checkpoint_fingerprint"] = value["identity"]["merged_checkpoint_fingerprint"]
        atomic_json(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        production.verify_same_policy_lineage(ctx, group, output)


@pytest.mark.parametrize("bad", ["missing_R", "already_old", "shape", "count", "mask", "nan", "inf", "temperature", "dtype", "fake_lineage"])
def test_invalid_initial_carriers_fail_before_forward(tmp_path, bad):
    ctx, group, output = lineage_fixture(tmp_path)
    lineage = production.verify_same_policy_lineage(ctx, group, output)
    actor, data, calls, counts = carrier_fixture(group)
    n = counts["supervised_response_tokens"]
    if bad == "missing_R": del data.batch["rollout_log_probs"]
    elif bad == "already_old": data.batch["old_log_probs"] = data.batch["rollout_log_probs"]
    elif bad == "shape": data.batch["rollout_log_probs"] = data.batch["rollout_log_probs"][:, :1]
    elif bad == "count": n += 1
    elif bad == "mask": data.batch["response_mask"][0, 0] = 2
    elif bad == "nan": data.batch["rollout_log_probs"][0, 0] = float("nan")
    elif bad == "inf": data.batch["rollout_log_probs"][0, 0] = float("inf")
    elif bad == "temperature": data.meta_info["temperature"] = 1.
    elif bad == "dtype": data.batch["rollout_log_probs"] = data.batch["rollout_log_probs"].to(torch.bfloat16)
    else: lineage = dict(lineage)
    with pytest.raises(ValueError):
        production.prepare_actor_old_log_probs(actor, data, ctx["gate"], lineage=lineage, expected_masked_token_count=n, rank=0)
    assert calls == [] and not actor.actor_optimizer.state


@pytest.mark.parametrize("bad", ["R_alias", "R_clone", "old_mutation", "R_mutation", "mask", "tokens", "positions", "vision", "advantage", "temperature", "param", "receipt", "missing_receipt", "old_fp", "alignment_fp", "false_flag", "different_data", "group_file", "actor_file"])
def test_update_boundary_rejects_substitution_stale_or_missing_receipt(tmp_path, monkeypatch, bad):
    *_, actor, data, calls, counts, proof = prepared_fixture(tmp_path)
    receipt, alignment = proof["receipt"], proof["alignment"]
    if bad == "R_alias": data.batch["old_log_probs"] = data.batch["rollout_log_probs"]
    elif bad == "R_clone": data.batch["old_log_probs"] = data.batch["rollout_log_probs"].clone()
    elif bad == "old_mutation": data.batch["old_log_probs"][0, 0] += 1
    elif bad == "R_mutation": data.batch["rollout_log_probs"][0, 0] += 1
    elif bad == "mask": data.batch["response_mask"][0, 0] = 0
    elif bad == "tokens": data.batch["responses"][0, 0] += 1
    elif bad == "positions": data.batch["position_ids"][0, 0, 0] += 1
    elif bad == "vision": data.non_tensor_batch["multi_modal_inputs"][0]["pixel_values"][0, 0] += 1
    elif bad == "advantage": data.batch["advantages"][0, 0] += 1
    elif bad == "temperature": data.meta_info["temperature"] = 1.
    elif bad == "param":
        with torch.no_grad(): first_weight(actor.actor_module).add_(1)
    elif bad == "receipt": receipt = receipt.artifact  # serialized receipt is not an in-process capability
    elif bad == "missing_receipt": receipt = None
    elif bad == "old_fp": receipt.artifact["old_log_probs_sha256"] = "a" * 64
    elif bad == "alignment_fp": alignment["old_log_probs_sha256"] = "a" * 64
    elif bad == "false_flag": actor.config.use_rollout_log_probs = False
    elif bad in {"group_file", "actor_file"}:
        ending = "group.json" if bad == "group_file" else "adapter_model.safetensors"
        path = next(Path(p) for p in receipt.artifact["source_file_sha256"] if p.endswith(ending))
        path.write_bytes(b"stale source at update boundary")
    else: data = copy.deepcopy(data)
    invoked = []
    monkeypatch.setattr(verl_policy_update, "audited_policy_update", lambda *a: invoked.append("update"))
    with pytest.raises(ValueError):
        verl_policy_update.policy_update_after_alignment(actor, data, alignment, receipt)
    assert invoked == [] and not actor.actor_optimizer.state


def test_actor_alignment_point_eleven_still_blocks_optimizer(tmp_path, monkeypatch):
    *_, actor, data, calls, counts, proof = prepared_fixture(tmp_path, current_delta=.11)
    assert len(calls) == 2 and not proof["alignment"]["passed"]
    invoked = []
    monkeypatch.setattr(verl_policy_update, "audited_policy_update", lambda *a: invoked.append("update"))
    with pytest.raises(RuntimeError, match="0 steps"):
        verl_policy_update.policy_update_after_alignment(actor, data, proof["alignment"], proof["receipt"])
    assert not invoked and not actor.actor_optimizer.state


def test_equal_numeric_r_and_o_still_cannot_alias_or_replace_installed_tensor(tmp_path):
    *_, actor, data, calls, counts, proof = prepared_fixture(tmp_path, delta=0.)
    assert torch.equal(data.batch["old_log_probs"], data.batch["rollout_log_probs"])
    data.batch["old_log_probs"] = data.batch["rollout_log_probs"]
    with pytest.raises(ValueError, match="denominator"):
        production.verify_update_receipt(actor, data, proof["alignment"], proof["receipt"])


@pytest.mark.parametrize("kwargs", [dict(delta=float("nan")), dict(current_delta=float("inf"))])
def test_nonfinite_actor_o_or_c_is_not_a_numerical_handoff_exception(tmp_path, kwargs):
    with pytest.raises(ValueError, match="nonfinite"):
        prepared_fixture(tmp_path, **kwargs)


def test_stale_group_cannot_be_hidden_by_deterministic_zero_actor_difference(tmp_path):
    ctx, group, output = lineage_fixture(tmp_path)
    actor, data, calls, counts = carrier_fixture(group, current_delta=0.)
    old = group["identity"]
    wrong = group_identity(prompt_id=old["prompt_id"], policy_fingerprint="f" * 64,
        rollout_fingerprint=old["rollout_config_fingerprint"], attempt=old["collection_attempt"], context=old["context"])
    group["identity"] = wrong
    for member in group["members"]: member["identity"] = copy.deepcopy(wrong)
    with pytest.raises(ValueError, match="stale|contract"):
        lineage = production.verify_same_policy_lineage(ctx, group, output)
        production.prepare_actor_old_log_probs(actor, data, ctx["gate"], lineage=lineage,
            expected_masked_token_count=counts["supervised_response_tokens"], rank=0)
    assert calls == [] and not actor.actor_optimizer.state


def test_source_changed_after_lineage_verification_refuses_recompute(tmp_path):
    ctx, group, output = lineage_fixture(tmp_path)
    lineage = production.verify_same_policy_lineage(ctx, group, output)
    actor, data, calls, counts = carrier_fixture(group)
    atomic_json(output / "group/group.json", {"stale": True})
    with pytest.raises(ValueError, match="evidence changed"):
        production.prepare_actor_old_log_probs(actor, data, ctx["gate"], lineage=lineage,
            expected_masked_token_count=counts["supervised_response_tokens"], rank=0)
    assert calls == []


@pytest.mark.parametrize("bad", ["input", "R", "old", "parameter", "RNG", "nonfinite", "backward", "optimizer"])
def test_independent_computations_reject_mutation_and_nonfinite(tmp_path, bad):
    def mutate(actor, data):
        if bad == "input": data.batch["input_ids"][0, 0] += 1
        elif bad == "R": data.batch["rollout_log_probs"][0, 0] += 1
        elif bad == "old":
            if "old_log_probs" in data.batch: data.batch["old_log_probs"][0, 0] += 1
        elif bad == "parameter": first_weight(actor.actor_module).add_(1)
        elif bad == "RNG": torch.rand(1)
        elif bad == "nonfinite": first_weight(actor.actor_module).fill_(float("nan"))
        elif bad == "backward": first_weight(actor.actor_module).grad = torch.ones(1, 1)
        else: actor.actor_optimizer.state[first_weight(actor.actor_module)] = {"step": 1}
    with pytest.raises(ValueError): prepared_fixture(tmp_path, mutation=mutate)


def artifact_fixture(root):
    ctx, group, output, lineage, actor, data, calls, counts, proof = prepared_fixture(root)
    actor2, data2, _, _ = carrier_fixture(group)
    proof2 = production.prepare_actor_old_log_probs(actor2, data2, ctx["gate"], lineage=lineage,
        expected_masked_token_count=counts["supervised_response_tokens"], rank=1)
    proofs = [proof, proof2]
    alignment = production.actor_alignment_artifact([p["alignment"] for p in proofs],
        gate_version=gate_c.GATE_C_VERSION, identity=ctx["identity"], trajectory_group_id=group["identity"]["trajectory_group_id"],
        policy_fingerprint=ctx["identity"]["effective_pre_update_policy_fingerprint"])
    atomic_json(output / "pre_update_policy_alignment.json", alignment)
    for kind, field in (("actor_old_logprob_receipt", "receipt"), ("rollout_actor_handoff", "handoff")):
        values = [p[field].artifact if field == "receipt" else p[field] for p in proofs]
        atomic_json(output / (kind + ".json"), production.paired_artifact(values, identity=ctx["identity"], kind=kind))
    atomic_json(output / "training_masks.json", mask_artifact(group, [1., -1.], rollout_schema=True))
    update = dict(identity=ctx["identity"], final_advantages=[1., -1.], training_token_counts=counts,
        group_sha256=sha256_file(output / "group/group.json"), training_masks_sha256=sha256_file(output / "training_masks.json"),
        pre_update_policy_fingerprint=ctx["identity"]["effective_pre_update_policy_fingerprint"],
        pre_update_policy_alignment=alignment, pre_update_policy_alignment_sha256=sha256_file(output / "pre_update_policy_alignment.json"),
        pre_update_actor_alignment_sha256=sha256_file(output / "pre_update_policy_alignment.json"),
        old_logprob_source=production.OLD_SOURCE, rollout_logprob_source=production.UPDATE_ROLLOUT_SOURCE,
        rollout_log_probs_preserved=True, ppo_denominator_source='data.batch["old_log_probs"]',
        same_policy_lineage_verified=True, optimizer_step_count=1, formal_rl_initialization_allowed=False,
        checks=dict.fromkeys(gate_c.UPDATE_CHECKS, True),
        per_rank=[dict(rank=i, checks=dict.fromkeys(gate_c.UPDATE_CHECKS, True), optimizer_step_count=1,
            pre_update_policy_alignment=p["alignment"], actor_old_logprob_receipt=p["receipt"].artifact,
            rollout_actor_handoff=p["handoff"]) for i, p in enumerate(proofs)])
    for kind in ("actor_old_logprob_receipt", "rollout_actor_handoff"):
        update[kind + "_sha256"] = sha256_file(output / (kind + ".json"))
    update.update(rl_policy_execution_contract=ctx["identity"]["rl_policy_execution_contract"],
        effective_pre_update_policy_fingerprint=ctx["identity"]["effective_pre_update_policy_fingerprint"],
        source_adapter_lora_dropout=.05, runtime_effective_lora_dropout=0., lora_dropout_target_count=252)
    atomic_json(output / "updated_actor/adapter/adapter_config.json", {"lora_dropout": .05})
    for value, batch, proof, report in zip((actor, actor2), (data, data2), proofs, update["per_rank"], strict=True):
        def cpu_update(unused):
            value.actor_module.train()
            value.actor_module(torch.ones(1)).sum().backward()
            value.actor_optimizer.step()
            value.actor_optimizer.zero_grad(set_to_none=True)
            return {"actor/pg_loss": [.1]}
        value.update_policy = cpu_update
        _, gradient = verl_policy_update.audited_policy_update(value, batch, runtime_receipt=proof["receipt"].artifact)
        evidence = gradient["rl_dropout_execution_audit"]
        runtime = proof["receipt"].artifact["rl_dropout_boundary_audits"]["old_before"]
        report.update(rl_lora_dropout_runtime_verified=True, rl_update_train_mode_forward_seen=True,
            rl_update_forward_count=evidence["forward_count"], rl_update_nonzero_dropout_count=0,
            rl_dropout_execution_audit=evidence, rl_dropout_execution_audit_sha256=evidence["audit_sha256"],
            gradient_audit=gradient, **{k: update[k] for k in ("rl_policy_execution_contract",
                "effective_pre_update_policy_fingerprint", "source_adapter_lora_dropout", "runtime_effective_lora_dropout", "lora_dropout_target_count")})
        for key in ("initial_runtime_audit", "before_update_runtime_audit", "fresh_runtime_audit", "native_runtime_audit", "after_reload_runtime_audit"):
            report[key] = copy.deepcopy(runtime)
    update.update(rl_lora_dropout_runtime_verified=True, rl_update_train_mode_forward_seen=True,
        rl_update_forward_count=sum(r["rl_update_forward_count"] for r in update["per_rank"]),
        rl_update_nonzero_dropout_count=0,
        rl_dropout_execution_audit=[r["rl_dropout_execution_audit"] for r in update["per_rank"]])
    return ctx, group, output, lineage, update


def test_finalizer_accepts_new_receipts_and_large_finite_handoff(tmp_path):
    ctx, group, output, lineage, update = artifact_fixture(tmp_path)
    assert gate_c.verify_policy_alignment_artifact(output, update, group, ctx["identity"], ctx["identity"]["effective_pre_update_policy_fingerprint"])["passed"]
    artifacts = production.verify_old_logprob_artifacts(output, update, group, ctx["identity"], lineage)
    gate_c.verify_dropout_artifacts(ctx, group, update, output)
    assert artifacts["rollout_actor_handoff"]["per_rank"][0]["metrics"]["max_abs_logprob_diff"] > .1


@pytest.mark.parametrize("bad", ["missing_handoff", "missing_receipt", "source", "alias_checksum", "old_fp", "group", "count", "mask", "step", "nonfinite_handoff", "handoff_pass_flag", "missing_metric"])
def test_finalizer_requires_bound_new_artifacts_fail_closed(tmp_path, bad):
    ctx, group, output, lineage, update = artifact_fixture(tmp_path)
    if bad == "missing_handoff": (output / "rollout_actor_handoff.json").unlink()
    elif bad == "missing_receipt": (output / "actor_old_logprob_receipt.json").unlink()
    elif bad == "source": update["old_logprob_source"] = "rollout"
    elif bad == "alias_checksum": update["pre_update_actor_alignment_sha256"] = "a" * 64
    elif bad == "mask":
        value = json.loads((output / "training_masks.json").read_text()); value["rows"][0]["response_mask"][0] = 0
        atomic_json(output / "training_masks.json", value)
    elif bad == "step": update["per_rank"][1]["optimizer_step_count"] = 0
    else:
        kind = "rollout_actor_handoff" if bad in {"nonfinite_handoff", "handoff_pass_flag", "missing_metric"} else "actor_old_logprob_receipt"
        path = output / (kind + ".json"); value = json.loads(path.read_text())
        evidence = value["per_rank"][0]
        if bad == "old_fp": evidence["old_log_probs_sha256"] = "0" * 64
        elif bad == "group": evidence["trajectory_group_id"] = "different"
        elif bad == "count": evidence["token_count"] += 1
        elif bad == "nonfinite_handoff": evidence["metrics"]["max_abs_logprob_diff"] = None
        elif bad == "handoff_pass_flag": evidence["passed"] = True
        else: del evidence["metrics"]["outlier_counts"]
        if kind == "actor_old_logprob_receipt": evidence["receipt_sha256"] = canonical_json_sha256({k: v for k, v in evidence.items() if k != "receipt_sha256"})
        atomic_json(path, value)
        # Truthful file rehash and matching inline row must not waive contracts.
        update[kind + "_sha256"] = sha256_file(path)
        update["per_rank"][0][kind] = evidence
    with pytest.raises(ValueError): production.verify_old_logprob_artifacts(output, update, group, ctx["identity"], lineage)
    assert not (output / "gate_manifest.json").exists()


def test_old_attempt_read_only_and_new_version_identity_not_reused(tmp_path):
    output = tmp_path / "outputs/rl_gate_c/gate-c-v451-attempt1"
    reports = tmp_path / "reports/rl_gate_c/gate-c-v451-attempt1"
    atomic_json(output / "run_manifest.json", dict(gate_version="minimum-rl-integration-c-v1"))
    atomic_json(output / "group/group.json", {"historical": True})
    atomic_json(reports / "gate_c_report.json", {"passed": False})
    before = {str(p): sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="read-only"): gate_c.paths_for(tmp_path, "gate-c-v451-attempt1")
    assert before == {str(p): sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert gate_c.GATE_C_VERSION == "minimum-rl-integration-c-v3-actor-old-zero-lora-dropout"
    assert gate_c.load_gate_c_config(ROOT / "configs/rl_gate_c.yaml")["gate_version"] == gate_c.GATE_C_VERSION
    with pytest.raises(ValueError, match="read-only"):
        gate_c.paths_for(tmp_path, "gate-c-v460-attempt1")
    assert gate_c.paths_for(tmp_path, "gate-c-v461-attempt1")[0] != output


def test_production_has_no_forensic_dependency_no_custom_loss_and_new_artifacts_required():
    source = inspect.getsource(production)
    assert "actor_old_logprob_diagnostic" not in source and "verl_old_logprob_semantics" not in source
    assert ".backward(" not in source and ".step(" not in source
    assert 'data.batch["old_log_probs"] = old.detach().clone()' in source
    assert "use_rollout_log_probs=True" in inspect.getsource(verl_policy_update.configure_one_update)
    assert "verify_update_receipt" in inspect.getsource(verl_policy_update.policy_update_after_alignment)
    assert "verify_old_logprob_artifacts" in inspect.getsource(gate_c.finalize)
    assert all(k in gate_c.UPDATE_CHECKS for k in production.OLD_LOGPROB_CHECKS)
