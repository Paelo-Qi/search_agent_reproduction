"""S2 CPU-only production-helper tests. No Qwen, GPU, rollout or provider calls."""
import copy
import inspect
import json
import random
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from _rl_actor_fixture import model, sft_config, first_weight
from test_rl_formal_contracts import fixture_run, draft_group, cpu_estimator
from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl import formal_policy_update as formal
from opensearch_vl_repro.rl import training_batch as batches
from opensearch_vl_repro.rl import old_logprob as old
from opensearch_vl_repro.rl import policy_alignment as alignment
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION, trainable_policy
from opensearch_vl_repro.rl.offline_snapshot import offline_snapshot_files
from opensearch_vl_repro.rl.group import publish_formal_group
from opensearch_vl_repro.rl.rloo import assemble_window_rloo
from opensearch_vl_repro.rl.rl_actor_semantics import configure_rl_lora_dropout_runtime, execution_contract
from opensearch_vl_repro.rl.run_state import (
    new_update_attempt, read_update_attempt, new_trainer_state, transition_trainer_state,
    checkpoint_policy,
)
from opensearch_vl_repro.rl.training_window import build_training_window, deterministic_rank_plan
from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.data import SFT_INPUT_MESSAGE_VERSION
from opensearch_vl_repro.inference.adapter import adapter_file_identity
from opensearch_vl_repro.sft_tool_audit import sha256_file


class CPUProto:
    def __init__(self, tensors, non_tensors, meta_info):
        self.batch, self.non_tensor_batch, self.meta_info = tensors, non_tensors, meta_info

    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**kwargs)

    def split(self, size):
        count = self.batch["responses"].shape[0]
        return [CPUProto({k: v[i:i+size] for k, v in self.batch.items()},
            {k: v[i:i+size] for k, v in self.non_tensor_batch.items()}, copy.deepcopy(self.meta_info))
            for i in range(0, count, size)]


@dataclass
class ActorConfig:
    ppo_mini_batch_size: int = 1
    ppo_micro_batch_size_per_gpu: int = 1
    ppo_epochs: int = 1
    shuffle: bool = False
    clip_ratio_low: float = .2
    clip_ratio_high: float = .28
    entropy_coeff: float = 0.
    use_kl_loss: bool = False
    use_rollout_log_probs: bool = True
    loss_agg_mode: str = "seq-mean-token-mean"
    use_dynamic_bsz: bool = False
    policy_loss: dict = field(default_factory=lambda: {"loss_mode": "vanilla"})


def export_adapter(actor, directory):
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(json.dumps({"lora_dropout": .05,
        "base_model_name_or_path": BASE_MODEL, "peft_type": "LORA", "r": 16,
        "lora_alpha": 32, "target_modules": sft_config()["lora"]["target_modules"]}), encoding="utf-8")
    save_file({n: p.detach().clone() for n, p in actor.actor_module.named_parameters() if p.requires_grad},
              str(directory / "adapter_model.safetensors"))


class CPUActor:
    """PEFT-shaped 252-target torch model; fake actor never substitutes a production loss."""
    def __init__(self, *, config, gate, adapter, mesh):
        self.runtime_locator = config["model"]["name_or_path"]
        torch.manual_seed(111)
        self.actor_module = model()
        params = dict(self.actor_module.named_parameters())
        with torch.no_grad():
            for n, value in load_file(str(adapter / "adapter_model.safetensors")).items():
                params[n].copy_(value)
        configure_rl_lora_dropout_runtime(self.actor_module, config)
        self.actor_optimizer = torch.optim.AdamW([p for p in self.actor_module.parameters() if p.requires_grad],
            lr=gate["optimizer"]["learning_rate"], weight_decay=gate["optimizer"]["weight_decay"])
        self.config, self.calls, self.update_calls = ActorConfig(), 0, 0
        self.offset = 0.
        self.before_optimizer = None
        self.steps_requested = 1

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
        assert micro_batch["responses"].shape[0] == 1
        assert all(t.device.type == "cpu" for v in micro_batch["multi_modal_inputs"] for t in v.values())
        x = micro_batch["responses"].float()
        logp = self.actor_module(-x * .1 / temperature) + self.offset
        return None, logp

    def compute_log_prob(self, data, calculate_entropy=False):
        self.calls += 1
        self.actor_module.eval()
        with torch.no_grad():
            values = [self._forward_micro_batch({**m.batch, **m.non_tensor_batch}, data.meta_info["temperature"])[1]
                      for m in data.split(data.meta_info["micro_batch_size"])]
        return torch.cat(values), None

    def update_policy(self, data):
        self.update_calls += 1
        if self.before_optimizer:
            self.before_optimizer()
        self.actor_module.train()
        self.actor_optimizer.zero_grad()
        losses = []
        for micro in data.split(1):
            _, logp = self._forward_micro_batch({**micro.batch, **micro.non_tensor_batch}, data.meta_info["temperature"])
            ratio = (logp - micro.batch["old_log_probs"]).exp()
            mask = micro.batch["response_mask"]
            # Test-only objective mirroring pinned per-row normalization.
            loss = (-ratio * micro.batch["advantages"] * mask).sum() / (mask.sum() + 1e-8)
            (loss / self.config.ppo_mini_batch_size).backward()
            losses.append(loss.detach().item() / self.config.ppo_mini_batch_size)
        for _ in range(self.steps_requested):
            self.actor_optimizer.step()
        return {"actor/pg_loss": losses}


class CPUManager:
    def __init__(self, actor, processor=None):
        self.actor = actor

    @staticmethod
    def get_rng_state():
        return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())

    @staticmethod
    def load_rng_state(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])

    def load_checkpoint(self, path, del_local_after_load=False):
        assert del_local_after_load is False
        path = Path(path)
        self.actor.actor_module.load_state_dict(torch.load(path / "model_world_size_1_rank_0.pt", weights_only=True))
        self.actor.actor_optimizer.load_state_dict(torch.load(path / "optim_world_size_1_rank_0.pt", weights_only=True))
        extra = torch.load(path / "extra_state_world_size_1_rank_0.pt", weights_only=False)
        assert extra["lr_scheduler"] is None
        self.load_rng_state(extra["rng"])


def cpu_construct(**kwargs):
    return CPUActor(**kwargs), {}


def cpu_save(actor, processor, directory, *, global_step):
    native = directory / "distributed"
    native.mkdir()
    torch.save(actor.actor_module.state_dict(), native / "model_world_size_1_rank_0.pt")
    torch.save(actor.actor_optimizer.state_dict(), native / "optim_world_size_1_rank_0.pt")
    torch.save(dict(lr_scheduler=None, rng=CPUManager.get_rng_state()), native / "extra_state_world_size_1_rank_0.pt")
    export_adapter(actor, directory / "adapter")
    return adapter_file_identity(directory / "adapter")


class Rope:
    def get_rope_index(self, *, input_ids, attention_mask, image_grid_thw, mm_token_type_ids):
        assert input_ids.device.type == image_grid_thw.device.type == "cpu"
        assert input_ids.shape == attention_mask.shape == mm_token_type_ids.shape
        # Distinct 3-D positions verify no generic arange replaces the real model callback.
        base = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        return torch.stack((base, base * 2, base * 3)), None


ROPE_MODEL = SimpleNamespace(get_base_model=lambda: SimpleNamespace(model=Rope()))


def publish_groups(root, run, policy, prompts, *, fatal=False, extreme_rollout=False):
    groups, directories = [], {}
    for prompt in prompts:
        draft = draft_group(run, policy, prompt)
        for member in draft["members"]:
            original = member["steps"][0]
            original["response_ids"] = [3 + member["rollout_index"], 4 + member["rollout_index"]]
            original["model_output"]["completion_ids"] = original["response_ids"]
            second = copy.deepcopy(original)
            second["prompt_ids"] = [1, 2, *original["response_ids"], 8, 9]  # prior generation + observation
            second["response_ids"] += [6]
            second["logprobs"] += [-.3]
            second["model_output"] = dict(prompt_ids=second["prompt_ids"], completion_ids=second["response_ids"], logprobs=second["logprobs"])
            second["multimodal_file"] = f"second-{member['rollout_index']}.pt"
            member["steps"].append(second)
            if extreme_rollout:
                for step in member["steps"]:
                    step["logprobs"] = [-1000.] * len(step["response_ids"])
                    step["model_output"]["logprobs"] = step["logprobs"]
            if fatal and member["rollout_index"] == 0:
                member["fatal"], member["fatal_step"] = True, 0
        stage = root / "groups" / f".stage-{uuid.uuid4()}"
        stage.mkdir(parents=True)
        for member in draft["members"]:
            (stage / member["trajectory_file"]).write_text(json.dumps(member), encoding="utf-8")
            for step in member["steps"]:
                ids = torch.tensor([step["prompt_ids"]])
                torch.save(dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                    mm_token_type_ids=torch.tensor([[0, 1] + [0] * (ids.shape[-1] - 2)]),
                    pixel_values=torch.ones(2, 3), image_grid_thw=torch.tensor([[1, 2, 2]]),
                    pixel_values_videos=torch.ones(1, 3)), stage / step["multimodal_file"])
        destination = root / "groups" / draft["identity"]["trajectory_group_id"]
        group = publish_formal_group(stage, destination, draft, cpu_fixture=True)
        groups.append(group)
        directories[group["identity"]["trajectory_group_id"]] = destination
    return groups, directories


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "verl", SimpleNamespace(DataProto=CPUProto))
    config = {**sft_config(), "model": dict(name_or_path=BASE_MODEL, revision=BASE_REVISION,
        freeze_vision_tower=True, freeze_multimodal_projector=True,
        attn_implementation="flash_attention_2", dtype="bfloat16", image_max_pixels=262144)}
    snapshot = tmp_path / "offline-snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps(dict(model_type="qwen3_vl",
        text_config=dict(num_hidden_layers=36), _commit_hash=BASE_REVISION)), encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text('{"cpu_fixture": true}', encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"CPU snapshot identity fixture; not Qwen weights")
    runtime_config = copy.deepcopy(config)
    runtime_config["model"]["name_or_path"] = str(snapshot)
    source = tmp_path / "sft_main_imageid_v3" / "checkpoint-3k" / "adapter"
    torch.manual_seed(111)
    export_adapter(SimpleNamespace(actor_module=model()), source)
    metadata = dict(checkpoint_complete=True, stage_complete=True, model=BASE_MODEL, model_revision=BASE_REVISION,
        sft_input_message_version=SFT_INPUT_MESSAGE_VERSION, runtime_tool_protocol_version=RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
        stage="main_b_2k", lineage=["main_a_1k", "main_b_2k"],
        file_sha256={"adapter/" + k: v for k, v in adapter_file_identity(source)["file_sha256"].items()})
    (source.parent / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    template = fixture_run(world_size=1, groups_per_window=2, prompt_count=4)
    semantics = copy.deepcopy(template["semantics"])
    semantics["base_model"] = dict(name=BASE_MODEL, revision=BASE_REVISION,
        offline_snapshot_sha256=cp.canonical_json_sha256(offline_snapshot_files(snapshot, revision=BASE_REVISION)))
    semantics["execution_contract"] = execution_contract(config)
    semantics["source_sft"].update(adapter_sha256=adapter_file_identity(source)["adapter_fingerprint"],
        metadata_sha256=sha256_file(source.parent / "metadata.json"), lineage=metadata["lineage"], stage=metadata["stage"])
    run = cp.build_training_run_identity("s2-cpu-fixture", semantics=semantics, prompt_ids=template["prompt_ids"],
                                         prompt_sources=template["prompt_sources"])
    loaded = formal.load_formal_actor(run, canonical_config=config, runtime_config=runtime_config,
        source_adapter=source, processor=None, mesh=None,
        initial_seed=7, construct=cpu_construct, manager_factory=CPUManager, cpu_fixture=True)
    root = tmp_path / "formal-fixture"
    cp.initialize_formal_run(root, run, loaded.policy, cpu_fixture=True)
    groups, directories = publish_groups(root, run, loaded.policy, ["p0", "p1"])
    window = build_training_window(run, loaded.policy, groups, window_id="w0")
    reward = assemble_window_rloo(window, run, loaded.policy, groups, estimator=cpu_estimator)
    data, batch_receipt = batches.build_rank_local_dataproto(window, run, loaded.policy, groups, reward,
        group_directories=directories, rank=0, model=ROPE_MODEL, pad_id=0, temperature=.7)
    return SimpleNamespace(root=root, run=run, config=config, runtime_config=runtime_config, snapshot=snapshot,
                           source=source, loaded=loaded, groups=groups,
                           directories=directories, window=window, reward=reward, data=data, batch=batch_receipt)


def prepare(ctx):
    return old.prepare_formal_old_log_probs(ctx.loaded.actor, ctx.data, window=ctx.window, policy=ctx.loaded.policy,
                                           reload_receipt=ctx.loaded.reload_receipt, batch_receipt=ctx.batch)


def update(ctx, **kwargs):
    return formal.update_formal_window(ctx.loaded, ctx.data, ctx.batch, root=ctx.root, run=ctx.run,
        groups=ctx.groups, window=ctx.window, reward_window=ctx.reward, attempt=new_update_attempt(ctx.window), **kwargs)


def staged(ctx):
    evidence = update(ctx)
    directory = ctx.root / "checkpoints" / f".staging-{uuid.uuid4()}"
    description = formal.save_formal_staging(ctx.loaded, None, directory, window=ctx.window, update_evidence=evidence,
                                            manager_factory=CPUManager, save=cpu_save)
    return directory, description, evidence


def test_multi_group_rows_unique_and_advantage_broadcast(ctx):
    rows, audit = batches.formal_training_rows(ctx.window, ctx.run, ctx.loaded.policy, ctx.groups, ctx.reward)
    assert len(rows) == len(set(r["logical_row_id"] for r in rows)) == 8
    for group in ctx.groups:
        for member in group["members"]:
            selected = [r for r in rows if r["member_id"] == member["member_id"]]
            assert len(selected) == 2 and len({r["advantage"] for r in selected}) == 1
    assert audit["weighting"] == "per_generation_row_mean_v1"
    assert all("old_log_probs" not in r for r in rows)


def test_context_observation_padding_masks_and_saved_mrope(ctx):
    data = ctx.data
    assert data.batch["input_ids"].shape == data.batch["attention_mask"].shape
    assert data.batch["position_ids"].shape[1] == 3
    assert (data.batch["response_mask"].sum(1) > 0).all()
    assert (data.batch["advantages"][data.batch["response_mask"] == 0] == 0).all()
    for row in ctx.batch.artifact["mask_audit"]["per_step_masks"]:
        assert row["loss_mask"][:len(row["loss_mask"]) - (2 if row["step_index"] == 0 else 3)] == [0] * (2 if row["step_index"] == 0 else 6)
    assert data.batch["position_ids"][1, 2, -1] == 24
    assert data.non_tensor_batch["multi_modal_inputs"][0]["mm_token_type_ids"].shape[-1] == data.batch["input_ids"].shape[-1]


def test_fatal_prefix_retained_postfatal_zero_not_dummy(ctx):
    groups, dirs = publish_groups(ctx.root, ctx.run, ctx.loaded.policy, ["p0", "p1"], fatal=True)
    window = build_training_window(ctx.run, ctx.loaded.policy, groups, window_id="fatal")
    reward = assemble_window_rloo(window, ctx.run, ctx.loaded.policy, groups, estimator=cpu_estimator)
    rows, audit = batches.formal_training_rows(window, ctx.run, ctx.loaded.policy, groups, reward)
    assert len(rows) == 6
    post = [r for r in audit["per_step_masks"] if r["fatal_step"] == 0 and r["step_index"] == 1]
    assert len(post) == 2 and all(not any(r["loss_mask"]) for r in post)
    assert all(any(r["response_mask"]) for r in rows)
    assert len([r for r in rows if r["fatal_step"] == 0]) == 2


@pytest.mark.parametrize("world,count", [(1, 5), (2, 5), (3, 5), (7, 2), (4, 8)])
def test_rank_plan_objective_and_gradient_equivalence(world, count):
    rows = [{"logical_row_id": str(i)} for i in range(count)]
    plan = deterministic_rank_plan([r["logical_row_id"] for r in rows], world)
    x = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    logical_losses = [(x * (i + 1) - i / 7).square() for i in range(count)]
    reference = torch.stack(logical_losses).mean()
    local_means = []
    for rank in range(world):
        local, evidence = batches.rank_local_rows(rows, plan, rank)
        # Official actor micro=1 scaling: 1 / local rows, FSDP W-way gradient mean.
        local_means.append(sum(logical_losses[int(r["logical_row_id"])] for r in local) / len(local))
        assert len(local) == plan["local_row_count"] and len(evidence["logical_to_physical"]) == plan["physical_count"]
    distributed = sum(local_means) / world
    assert torch.allclose(reference, distributed, atol=1e-12, rtol=0)
    assert torch.allclose(torch.autograd.grad(reference, x, retain_graph=True)[0], torch.autograd.grad(distributed, x)[0], atol=1e-12, rtol=0)


def test_only_active_microbatch_materialized_and_original_cpu_unchanged(ctx):
    actor = ctx.loaded.actor
    before = old.input_fingerprint(ctx.data)
    seen = []
    original = actor._forward_micro_batch
    def forward(micro, *args, **kwargs):
        assert len(micro["multi_modal_inputs"]) == 1
        seen.append(micro["multi_modal_inputs"][0])
        return original(micro, *args, **kwargs)
    actor._forward_micro_batch = forward
    with batches.materialize_actor_microbatches(actor):
        actor.compute_log_prob(ctx.data)
    assert actor._forward_micro_batch is forward
    assert len(seen) == 8 and all(s is not original for s, original in zip(seen, ctx.data.non_tensor_batch["multi_modal_inputs"]))
    assert old.input_fingerprint(ctx.data) == before
    assert all(v.device.type == "cpu" for mm in ctx.data.non_tensor_batch["multi_modal_inputs"] for v in mm.values())
    source = inspect.getsource(batches.build_rank_local_dataproto)
    assert ".to(device)" not in source and "processor(" not in source


def test_microbatch_guard_restores_on_failure(ctx):
    original = ctx.loaded.actor._forward_micro_batch
    with pytest.raises(ValueError, match="microbatch=1"):
        with batches.materialize_actor_microbatches(ctx.loaded.actor):
            ctx.loaded.actor._forward_micro_batch({**ctx.data.batch, **ctx.data.non_tensor_batch}, .7)
    assert ctx.loaded.actor._forward_micro_batch == original


def test_batched_formal_O_and_exact_C_receipt(ctx):
    result = prepare(ctx)
    assert ctx.loaded.actor.calls == 2
    assert result["alignment"]["passed"] and result["alignment"]["max_abs_logprob_diff"] == 0
    assert result["receipt"].artifact["row_ids"] == ctx.data.meta_info["row_ids"]
    assert result["receipt"].artifact["token_count"] == int(ctx.data.batch["response_mask"].sum())
    assert not ctx.data.batch["old_log_probs"].requires_grad
    assert ctx.data.batch["old_log_probs"].data_ptr() != ctx.data.batch["rollout_log_probs"].data_ptr()
    old.verify_formal_old_receipt(ctx.loaded.actor, ctx.data, result["receipt"], result["alignment"], window=ctx.window,
        policy=ctx.loaded.policy, reload_receipt=ctx.loaded.reload_receipt, batch_receipt=ctx.batch)


def test_independent_C_mismatch_blocks_before_optimizer(ctx):
    actor = ctx.loaded.actor
    original = actor.compute_log_prob
    def compute(*a, **k):
        output, entropy = original(*a, **k)
        return output + (.2 if actor.calls == 2 else 0.), entropy
    actor.compute_log_prob = compute
    with pytest.raises(RuntimeError, match="0 steps"):
        update(ctx)
    assert actor.update_calls == 0 and not actor.actor_optimizer.state
    assert not list((ctx.root / "attempts").glob("*/*.json"))


def test_R_O_magnitude_is_informational_only(ctx):
    ctx.groups, ctx.directories = publish_groups(ctx.root, ctx.run, ctx.loaded.policy, ["p0", "p1"], extreme_rollout=True)
    ctx.window = build_training_window(ctx.run, ctx.loaded.policy, ctx.groups, window_id="extreme-R")
    ctx.reward = assemble_window_rloo(ctx.window, ctx.run, ctx.loaded.policy, ctx.groups, estimator=cpu_estimator)
    ctx.data, ctx.batch = batches.build_rank_local_dataproto(ctx.window, ctx.run, ctx.loaded.policy, ctx.groups, ctx.reward,
        group_directories=ctx.directories, rank=0, model=ROPE_MODEL, pad_id=0, temperature=.7)
    result = prepare(ctx)
    assert result["alignment"]["passed"]
    handoff = result["handoff"]
    assert handoff["informational_only"] and not handoff["gate_blocking"]
    assert "passed" not in handoff and "absolute_diff_percentiles" not in handoff
    assert handoff["max_abs_logprob_diff"] > 900 and handoff["ratio_finite"] is False
    json.dumps(handoff, allow_nan=False)


@pytest.mark.parametrize("field", ["parent_checkpoint_identity", "parent_policy_fingerprint", "policy_iteration"])
def test_loaded_window_lineage_mismatch(ctx, field):
    bad = dict(ctx.window)
    bad[field] = "f" * 64 if field != "policy_iteration" else 99
    bad = cp.seal({k: v for k, v in bad.items() if k != "window_sha256"}, "window_sha256")
    with pytest.raises(ValueError, match="lineage"):
        formal.require_formal_reload(ctx.loaded.actor, ctx.loaded.policy, ctx.loaded.reload_receipt, window=bad)


@pytest.mark.parametrize("change", ["optimizer", "rng", "model", "dropout", "frozen", "global_step"])
def test_live_reload_identity_changes_fail_closed(ctx, change):
    actor = ctx.loaded.actor
    if change == "optimizer": actor.actor_optimizer.param_groups[0]["lr"] *= 2
    elif change == "rng": random.random()
    elif change == "model":
        with torch.no_grad(): first_weight(actor.actor_module).add_(1)
    elif change == "dropout": actor.actor_module.language_model.layers[0].self_attn.q_proj.lora_dropout["default"].p = .05
    elif change == "frozen": actor.actor_module.visual.merger.weight.requires_grad_(True)
    else: actor._formal_global_optimizer_step = 9
    with pytest.raises((ValueError, RuntimeError)):
        formal.require_formal_reload(actor, ctx.loaded.policy, ctx.loaded.reload_receipt, window=ctx.window)


def test_recovery_flag_requires_actual_live_receipt_not_json(ctx):
    state = new_trainer_state(ctx.run, ctx.loaded.policy)
    state = transition_trainer_state(transition_trainer_state(state, "collecting"), "ready_to_update")
    state = cp.seal({**{k: v for k, v in state.items() if k != "state_sha256"}, "recovery_reload_required": True}, "state_sha256")
    with pytest.raises(ValueError, match="reload verification"):
        transition_trainer_state(state, "updating")
    with pytest.raises(ValueError, match="receipt"):
        transition_trainer_state(state, "updating", reload_receipt={"model_loaded": True}, actor=ctx.loaded.actor)
    verified = transition_trainer_state(state, "updating", reload_receipt=ctx.loaded.reload_receipt, actor=ctx.loaded.actor)
    assert verified["recovery_reload_required"] is True  # audit history retained
    result = update(ctx, trainer_state=state)
    assert result["after_step"] == 1


def test_uncertainty_marker_precedes_any_optimizer_operation_and_one_step(ctx):
    actor = ctx.loaded.actor
    seen = []
    def before():
        events = list((ctx.root / "attempts").glob("*/*step_may_have_run.json"))
        assert len(events) == 1
        assert json.loads(events[0].read_text())["phase"] == "step_may_have_run"
        seen.append("durable before update/zero_grad")
    actor.before_optimizer = before
    original = actor.actor_optimizer.step
    def step(*a, **k):
        before()
        return original(*a, **k)
    actor.actor_optimizer.step = step
    result = update(ctx)
    assert len(seen) == 2 and result["before_step"] == 0 and result["after_step"] == 1
    assert result["update_audit"]["optimizer_step_count"] == 1
    assert read_update_attempt(ctx.root, result["attempt"]["attempt_id"])["phase"] == "checkpoint_staging"
    audit = result["update_audit"]["rl_dropout_execution_audit"]
    assert audit["forward_count"] == 8 and audit["train_mode_forward_seen"]
    assert audit["dropout_training_true_count_per_forward"] == [252] * 8
    assert all(p.grad is None for p in actor.actor_module.parameters())


@pytest.mark.parametrize("count", [0, 2])
def test_zero_or_two_optimizer_steps_rejected(ctx, count):
    ctx.loaded.actor.steps_requested = count
    with pytest.raises(RuntimeError, match="optimizer step|second"):
        update(ctx)
    event = next((ctx.root / "attempts").glob("*/*failed.json"))
    assert json.loads(event.read_text())["phase"] == "failed"
    assert not list((ctx.root / "checkpoints").glob("policy-*"))


def test_gate_still_rejects_nonempty_optimizer(ctx):
    actor = ctx.loaded.actor
    first_weight(actor.actor_module).sum().backward()
    actor.actor_optimizer.step()
    actor.actor_optimizer.zero_grad(set_to_none=True)
    with pytest.raises(ValueError, match="optimizer state"):
        old.independent_actor_compute(actor, ctx.data, parameters=old.parameter_fingerprint(actor),
            inputs=old.input_fingerprint(ctx.data), rollout=old.fingerprint(ctx.data.batch["rollout_log_probs"]),
            rng=old.rng_fingerprint(), sft_config=ctx.config)


def test_generic_alignment_allows_unequal_tokens_and_requires_all_ranks(ctx):
    result = prepare(ctx)["alignment"]
    rows = [dict(result, rank=i, masked_token_count=i+1, expected_masked_token_count=i+1) for i in range(3)]
    for i, row in enumerate(rows): row["mean_abs_logprob_diff"] = i * .01
    audit = alignment.formal_alignment_artifact(rows, world_size=3, window_sha256=ctx.window["window_sha256"])
    assert audit["passed"] and audit["masked_token_count"] == 6
    assert audit["mean_abs_logprob_diff"] == pytest.approx(.08 / 6)
    with pytest.raises(ValueError): alignment.formal_alignment_artifact(rows[:2], world_size=3, window_sha256=ctx.window["window_sha256"])


def test_saved_staging_roles_complete_and_fresh_native_reload_evidence(ctx):
    directory, description, result = staged(ctx)
    fresh, evidence = formal.fresh_reload_staging(ctx.loaded, ctx.data, directory, description,
        canonical_config=ctx.config, runtime_config=ctx.runtime_config,
        processor=None, mesh=None, construct=cpu_construct, manager_factory=CPUManager)
    assert ctx.loaded.actor is None and bool(fresh.actor_optimizer.state)
    assert evidence["original_actor_destroyed"] and evidence["fresh_multimodal_forward_finite"]
    assert evidence["scope"] == "cpu_fixture"
    assert description["artifact_roles"] == cp.artifact_role_identities(description["artifact_role_files"], description["file_sha256"])
    manifest = cp.build_checkpoint_manifest(ctx.run, ctx.loaded.policy, ctx.groups, ctx.window, result["attempt"], ctx.reward,
        artifact_role_files=description["artifact_role_files"], file_sha256=description["file_sha256"],
        kind="smoke_continuation", reload_evidence=evidence, cpu_fixture=True)
    cp.commit_verified_checkpoint(ctx.root, directory, manifest, cpu_fixture=True)
    successor = checkpoint_policy(manifest)
    del fresh
    loaded = formal.load_formal_actor(ctx.run, canonical_config=ctx.config, runtime_config=ctx.runtime_config,
        source_adapter=ctx.source, processor=None, mesh=None,
        policy=successor, checkpoint_directory=ctx.root / "checkpoints/policy-000001", cpu_fixture=True,
        construct=cpu_construct, manager_factory=CPUManager)
    assert loaded.policy["global_optimizer_step"] == 1 and loaded.actor.actor_optimizer.state
    formal.require_formal_reload(loaded.actor, successor, loaded.reload_receipt)
    groups, dirs = publish_groups(ctx.root, ctx.run, successor, ["p2", "p3"])
    window = build_training_window(ctx.run, successor, groups, window_id="w1")
    reward = assemble_window_rloo(window, ctx.run, successor, groups, estimator=cpu_estimator)
    data, receipt = batches.build_rank_local_dataproto(window, ctx.run, successor, groups, reward,
        group_directories=dirs, rank=0, model=ROPE_MODEL, pad_id=0, temperature=.7)
    result = formal.update_formal_window(loaded, data, receipt, root=ctx.root, run=ctx.run, groups=groups,
        window=window, reward_window=reward, attempt=new_update_attempt(window))
    assert result["before_step"] == 1 and result["after_step"] == 2
    assert loaded.actor.calls == 2  # nonempty optimizer supported for O/C


@pytest.mark.parametrize("field", ["adapter_reloaded", "native_reloaded", "optimizer_reloaded", "rng_reloaded", "execution_contract_verified"])
def test_missing_fresh_reload_evidence_cannot_commit(ctx, field):
    directory, description, result = staged(ctx)
    fresh, evidence = formal.fresh_reload_staging(ctx.loaded, ctx.data, directory, description,
        canonical_config=ctx.config, runtime_config=ctx.runtime_config,
        processor=None, mesh=None, construct=cpu_construct, manager_factory=CPUManager)
    del evidence[field]
    with pytest.raises(ValueError, match="reload verification"):
        cp.build_checkpoint_manifest(ctx.run, ctx.loaded.policy, ctx.groups, ctx.window, result["attempt"], ctx.reward,
            artifact_role_files=description["artifact_role_files"], file_sha256=description["file_sha256"],
            kind="smoke_continuation", reload_evidence=evidence, cpu_fixture=True)


def test_fresh_reload_requires_old_actor_destruction(ctx):
    directory, description, _ = staged(ctx)
    retained = ctx.loaded.actor
    with pytest.raises(ValueError, match="still alive"):
        formal.fresh_reload_staging(ctx.loaded, ctx.data, directory, description,
            canonical_config=ctx.config, runtime_config=ctx.runtime_config,
            processor=None, mesh=None, construct=cpu_construct, manager_factory=CPUManager)
    assert retained.actor_optimizer.state


def test_multi_rank_role_maps_hash_all_shards(tmp_path):
    for rank in range(3):
        for prefix in ("model", "optim", "extra_state"):
            file = tmp_path / "distributed" / f"{prefix}_world_size_3_rank_{rank}.pt"
            file.parent.mkdir(exist_ok=True)
            file.write_bytes(f"CPU native {prefix}/{rank}".encode())
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter/adapter_config.json").write_text('{"lora_dropout": .05}')
    (tmp_path / "adapter/adapter_model.safetensors").write_bytes(b"CPU fixture")
    roles, files = formal.checkpoint_staging_roles(tmp_path, 3)
    hashes = cp.artifact_role_identities(roles, files)
    assert len(roles["native"]) == len(roles["optimizer"]) == len(roles["rng"]) == 3
    (tmp_path / "distributed/optim_world_size_3_rank_2.pt").write_bytes(b"changed rank2")
    new_roles, new_files = formal.checkpoint_staging_roles(tmp_path, 3)
    assert cp.artifact_role_identities(new_roles, new_files)["optimizer"] != hashes["optimizer"]


def test_no_coordinator_scheduler_gate_init_or_pass_publication():
    source = inspect.getsource(formal)
    assert "run_rl_training" not in source and "scheduler.step" not in source
    assert "gate_manifest.json" not in source and "passed=True" not in source
    assert "audited_policy_update(actor, data" in source
    assert "configure_rl_lora_dropout_runtime" in source
    assert "construct_rl_actor" in source and "read_verified_checkpoint" in source


def test_updated_policy_cannot_load_adapter_without_native_checkpoint(ctx):
    bad = cp.seal({**{k: v for k, v in ctx.loaded.policy.items() if k != "effective_policy_fingerprint"},
        "policy_iteration": 1, "global_optimizer_step": 1, "parent_checkpoint_identity": "f" * 64,
        "cumulative_consumed_group_ids": ["g"]}, "effective_policy_fingerprint")
    with pytest.raises(ValueError, match="native checkpoint"):
        formal.load_formal_actor(ctx.run, canonical_config=ctx.config, runtime_config=ctx.runtime_config,
            source_adapter=ctx.source, processor=None, mesh=None,
            policy=bad, cpu_fixture=True, initial_seed=7, construct=cpu_construct, manager_factory=CPUManager)


@pytest.mark.parametrize("identity", ["optimizer_identity", "rng_identity"])
def test_policy_optimizer_and_rng_identity_mismatch(ctx, identity):
    bad = cp.seal({**{k: v for k, v in ctx.loaded.policy.items() if k != "effective_policy_fingerprint"},
                   identity: "f" * 64}, "effective_policy_fingerprint")
    with pytest.raises(ValueError, match="policy identity"):
        formal.require_formal_reload(ctx.loaded.actor, bad, ctx.loaded.reload_receipt)
    with pytest.raises(ValueError, match="optimizer/RNG PolicyIdentity"):
        formal.load_formal_actor(ctx.run, canonical_config=ctx.config, runtime_config=ctx.runtime_config,
            source_adapter=ctx.source, processor=None, mesh=None,
            policy=bad, initial_seed=7, cpu_fixture=True, construct=cpu_construct, manager_factory=CPUManager)


def test_hand_cleared_recovery_flag_does_not_authorize_update(ctx):
    state = new_trainer_state(ctx.run, ctx.loaded.policy)
    state = transition_trainer_state(transition_trainer_state(state, "collecting"), "ready_to_update")
    state = cp.seal({**{k: v for k, v in state.items() if k != "state_sha256"}, "recovery_reload_required": False}, "state_sha256")
    ctx.loaded.reload_receipt = {"model_loaded": True, "ready_to_update": True}
    with pytest.raises(ValueError, match="reload receipt"):
        update(ctx, trainer_state=state)
    assert ctx.loaded.actor.update_calls == 0


@pytest.mark.parametrize("field", ["rollout_log_probs", "response_mask", "advantages", "input_ids"])
def test_batch_mutation_cannot_enter_formal_O(ctx, field):
    ctx.data.batch[field][0, 0] += 1
    with pytest.raises(ValueError, match="batch receipt"):
        prepare(ctx)
    assert ctx.loaded.actor.calls == 0


def test_sealed_O_mutation_cannot_enter_optimizer(ctx):
    prepared = prepare(ctx)
    ctx.data.batch["old_log_probs"][0, 0] += .01
    with pytest.raises(ValueError, match="carrier changed"):
        old.verify_formal_old_receipt(ctx.loaded.actor, ctx.data, prepared["receipt"], prepared["alignment"],
            window=ctx.window, policy=ctx.loaded.policy, reload_receipt=ctx.loaded.reload_receipt, batch_receipt=ctx.batch)


def test_saved_attention_preserved_without_retokenization(ctx):
    # Write a new actual immutable group fixture with a masked context token.
    groups, directories = publish_groups(ctx.root, ctx.run, ctx.loaded.policy, ["p0", "p1"])
    # A production mutation would fail hashes; only fixture staging is republished here.
    draft = draft_group(ctx.run, ctx.loaded.policy, "p0")
    stage = ctx.root / "groups" / f".stage-{uuid.uuid4()}"
    stage.mkdir()
    for m in draft["members"]:
        (stage / m["trajectory_file"]).write_text(json.dumps(m))
        torch.save(dict(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[0, 1]]),
            mm_token_type_ids=torch.tensor([[0, 1]]), pixel_values=torch.ones(2, 3),
            image_grid_thw=torch.tensor([[1, 2, 2]])), stage / m["steps"][0]["multimodal_file"])
    dest = ctx.root / "groups" / draft["identity"]["trajectory_group_id"]
    groups[0] = publish_formal_group(stage, dest, draft, cpu_fixture=True)
    directories[groups[0]["identity"]["trajectory_group_id"]] = dest
    window = build_training_window(ctx.run, ctx.loaded.policy, groups, window_id="masked-context")
    reward = assemble_window_rloo(window, ctx.run, ctx.loaded.policy, groups, estimator=cpu_estimator)
    data, _ = batches.build_rank_local_dataproto(window, ctx.run, ctx.loaded.policy, groups, reward,
        group_directories=directories, rank=0, model=ROPE_MODEL, pad_id=0, temperature=.7)
    assert data.batch["attention_mask"][0, :6].tolist() == [0, 0, 0, 0, 0, 1]


def test_real_transformers_qwen_mrope_method_without_loading_model(monkeypatch):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    assert hasattr(Qwen3VLModel, "get_rope_index")
    def forbid_weights(*args, **kwargs):
        pytest.fail("M-RoPE regression must not initialize or load Qwen weights")
    monkeypatch.setattr(Qwen3VLModel, "__init__", forbid_weights)
    monkeypatch.setattr(Qwen3VLModel, "from_pretrained", forbid_weights)
    # Metadata-only self: bypass Qwen __init__, retaining installed class methods.
    # This needs no version-specific helper binding (4.57.1 computes M-RoPE inline).
    rope = Qwen3VLModel.__new__(Qwen3VLModel)
    torch.nn.Module.__init__(rope)
    rope.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),
        image_token_id=103, video_token_id=104, vision_start_token_id=101)
    assert not list(rope.parameters()) and not list(rope.children())
    assert rope.get_rope_index.__func__ is Qwen3VLModel.get_rope_index

    # One image: 4x4 patches / merge_size=2 -> four image placeholders (2x2).
    # Prefix and suffix include ordinary text and vision boundary tokens.
    ids = torch.tensor([[11, 101, 103, 103, 103, 103, 102, 12]], dtype=torch.long)
    kwargs = dict(input_ids=ids, image_grid_thw=torch.tensor([[1, 4, 4]], dtype=torch.long),
                  attention_mask=torch.ones_like(ids))
    # Same signature-aware modality handling as formal training_batch.py.
    # Pinned 4.57.1 uses the special IDs above; newer installed APIs also need types.
    if "mm_token_type_ids" in inspect.signature(rope.get_rope_index).parameters:
        kwargs["mm_token_type_ids"] = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.long)
    positions, deltas = rope.get_rope_index(**kwargs)
    assert isinstance(positions, torch.Tensor) and positions.shape == (3, 1, ids.shape[-1])
    assert isinstance(deltas, torch.Tensor) and deltas.shape == (1, 1)
    for tensor in (positions, deltas):
        assert tensor.device.type == "cpu" and tensor.dtype == torch.long
        assert bool(torch.isfinite(tensor).all())
    assert bool((positions >= 0).all())
    # Distinct height/width axes prove the real visual branch ran, not text arange.
    assert positions[:, 0, 2:6].tolist() == [[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]]
    assert positions[:, 0, [0, 1, 6, 7]].tolist() == [[0, 1, 4, 5]] * 3
    assert deltas.tolist() == [[-2]]


@pytest.mark.parametrize("field", ["optimizer_state_sha256", "native_rng_sha256", "parameter_sha256"])
def test_fresh_reload_actual_state_mismatch_rejected(ctx, field):
    directory, description, _ = staged(ctx)
    path = directory / "runtime_state_rank_0.json"
    state = json.loads(path.read_text())
    state = cp.seal({**{k: v for k, v in state.items() if k != "runtime_state_sha256"}, field: "f" * 64}, "runtime_state_sha256")
    path.write_text(json.dumps(state))
    roles, files = formal.checkpoint_staging_roles(directory, 1)
    description = cp.seal({**{k: v for k, v in description.items() if k != "staging_sha256"},
        "artifact_role_files": roles, "file_sha256": files, "artifact_roles": cp.artifact_role_identities(roles, files)}, "staging_sha256")
    with pytest.raises(ValueError, match="state mismatch"):
        formal.fresh_reload_staging(ctx.loaded, ctx.data, directory, description,
            canonical_config=ctx.config, runtime_config=ctx.runtime_config,
            processor=None, mesh=None, construct=cpu_construct, manager_factory=CPUManager)


def test_staging_missing_native_rank_fail_closed(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter/adapter_config.json").write_text('{}')
    with pytest.raises(ValueError, match="missing.*shard"):
        formal.checkpoint_staging_roles(tmp_path, 2)


def test_generic_rank_rejects_nonuniform_replication():
    rows = [{"logical_row_id": str(i)} for i in range(5)]
    plan = deterministic_rank_plan([r["logical_row_id"] for r in rows], 3)
    plan["rank_assignments"][0][0] = "1"
    with pytest.raises(ValueError, match="rank plan"):
        batches.rank_local_rows(rows, plan, 0)


def test_save_default_keeps_gate_global_step_one():
    from opensearch_vl_repro.rl.verl_actor_gate import save_checkpoint
    assert inspect.signature(save_checkpoint).parameters["global_step"].default == 1
    assert "global_step=global_step" in inspect.getsource(save_checkpoint)


def initial_reload(ctx, **kwargs):
    args = dict(canonical_config=ctx.config, runtime_config=ctx.runtime_config,
        source_adapter=ctx.source, processor=None, mesh=None, initial_seed=7, cpu_fixture=True,
        construct=cpu_construct, manager_factory=CPUManager)
    args.update(kwargs)
    return formal.load_formal_actor(ctx.run, **args)


def relocated_runtime(ctx):
    destination = ctx.snapshot.parent / "another-machine-snapshot"
    shutil.copytree(ctx.snapshot, destination)
    config = copy.deepcopy(ctx.runtime_config)
    config["model"]["name_or_path"] = str(destination)
    return config


def test_canonical_identity_and_actual_runtime_locator_are_separate(ctx):
    assert ctx.config["model"]["name_or_path"] == BASE_MODEL
    assert ctx.loaded.actor.runtime_locator == str(ctx.snapshot)
    assert ctx.loaded.reload_receipt.artifact["base_model"]["name"] == BASE_MODEL
    assert ctx.loaded.reload_receipt.artifact["base_model"]["revision"] == BASE_REVISION
    formal.require_formal_reload(ctx.loaded.actor, ctx.loaded.policy, ctx.loaded.reload_receipt)
    assert str(ctx.snapshot) not in json.dumps(ctx.run["semantics"])
    assert str(ctx.snapshot) not in json.dumps(ctx.loaded.policy)
    # Runtime config is never an acceptable canonical identity, even when run is unchanged.
    with pytest.raises(ValueError, match="canonical"):
        initial_reload(ctx, canonical_config=ctx.runtime_config)


def test_relocated_identical_snapshot_preserves_run_behavior_and_policy_identities(ctx):
    runtime_config = relocated_runtime(ctx)
    run = cp.build_training_run_identity(ctx.run["run_id"], semantics=ctx.run["semantics"],
        prompt_ids=ctx.run["prompt_ids"], prompt_sources=ctx.run["prompt_sources"],
        locators={"base_snapshot": runtime_config["model"]["name_or_path"]})
    cp.require_same_training_run(ctx.run, run)
    assert run["run_identity_sha256"] == ctx.run["run_identity_sha256"]
    assert run["training_behavior_fingerprint"] == ctx.run["training_behavior_fingerprint"]
    loaded = formal.load_formal_actor(run, canonical_config=ctx.config, runtime_config=runtime_config,
        source_adapter=ctx.source, processor=None, mesh=None, initial_seed=7, cpu_fixture=True,
        construct=cpu_construct, manager_factory=CPUManager)
    assert loaded.policy == ctx.loaded.policy
    assert loaded.actor.runtime_locator == runtime_config["model"]["name_or_path"]
    formal.require_formal_reload(loaded.actor, loaded.policy, loaded.reload_receipt)


@pytest.mark.parametrize("file", ["model.safetensors", "tokenizer_config.json", "new_config.json"])
def test_snapshot_content_mutation_fails_before_construction(ctx, file):
    (ctx.snapshot / file).write_bytes(b'{}')
    constructed = []
    with pytest.raises(ValueError, match="snapshot content identity"):
        initial_reload(ctx, construct=lambda **kw: constructed.append(kw))
    assert not constructed


def test_snapshot_mutation_after_load_revokes_live_update_capability(ctx):
    (ctx.snapshot / "model.safetensors").write_bytes(b"changed snapshot")
    with pytest.raises(ValueError, match="files/execution identity changed"):
        update(ctx)
    assert ctx.loaded.actor.update_calls == 0 and not ctx.loaded.actor.actor_optimizer.state


@pytest.mark.parametrize("case", ["model_type", "layers", "commit", "null_commit", "weights", "empty_weights",
                                  "missing_config", "missing_directory"])
def test_invalid_offline_snapshot_fails_closed(ctx, case):
    config_path = ctx.snapshot / "config.json"
    value = json.loads(config_path.read_text())
    if case == "model_type":
        value["model_type"] = "qwen3"
    elif case == "layers":
        value["text_config"]["num_hidden_layers"] = 32
    elif case == "commit":
        value["_commit_hash"] = "different-revision"
    elif case == "null_commit":
        value["_commit_hash"] = None
    if case in {"model_type", "layers", "commit", "null_commit"}:
        config_path.write_text(json.dumps(value))
    elif case == "weights":
        (ctx.snapshot / "model.safetensors").unlink()
    elif case == "empty_weights":
        (ctx.snapshot / "model.safetensors").write_bytes(b"")
    elif case == "missing_config":
        config_path.unlink()
    else:
        ctx.runtime_config["model"]["name_or_path"] = str(ctx.snapshot / "does-not-exist")
    with pytest.raises(ValueError, match="offline|snapshot|revision"):
        initial_reload(ctx)


def test_snapshot_without_declared_commit_is_content_bound_not_directory_name(ctx):
    path = ctx.snapshot / "config.json"
    value = json.loads(path.read_text())
    del value["_commit_hash"]
    path.write_text(json.dumps(value))
    semantics = copy.deepcopy(ctx.run["semantics"])
    semantics["base_model"]["offline_snapshot_sha256"] = cp.canonical_json_sha256(
        offline_snapshot_files(ctx.snapshot, revision=BASE_REVISION, strict=True))
    run = cp.build_training_run_identity(ctx.run["run_id"], semantics=semantics,
        prompt_ids=ctx.run["prompt_ids"], prompt_sources=ctx.run["prompt_sources"])
    loaded = formal.load_formal_actor(run, canonical_config=ctx.config, runtime_config=ctx.runtime_config,
        source_adapter=ctx.source, processor=None, mesh=None, initial_seed=7, cpu_fixture=True,
        construct=cpu_construct, manager_factory=CPUManager)
    assert loaded.policy["effective_policy_fingerprint"] != ctx.loaded.policy["effective_policy_fingerprint"]
    assert run["run_identity_sha256"] != ctx.run["run_identity_sha256"]
    assert run["training_behavior_fingerprint"] != ctx.run["training_behavior_fingerprint"]


@pytest.mark.parametrize("field,value", [("revision", "wrong"), ("attn_implementation", "sdpa"),
    ("image_max_pixels", 524288), ("dtype", "float32"), ("freeze_vision_tower", False),
    ("freeze_multimodal_projector", False), ("lora.rank", 8), ("lora.dropout", .0),
    ("other_training_semantic", "changed")])
def test_runtime_semantics_cannot_drift_except_locator(ctx, field, value):
    runtime = copy.deepcopy(ctx.runtime_config)
    if field.startswith("lora."):
        runtime["lora"][field.split(".")[1]] = value
    else:
        runtime["model"][field] = value
    with pytest.raises(ValueError, match="beyond snapshot locator"):
        initial_reload(ctx, runtime_config=runtime)


def test_runtime_semantic_comparison_is_order_independent_but_typed(ctx):
    runtime = dict(reversed(list(ctx.runtime_config.items())))
    runtime["model"] = dict(reversed(list(runtime["model"].items())))
    loaded = initial_reload(ctx, runtime_config=runtime)
    assert loaded.policy == ctx.loaded.policy
    runtime["model"]["freeze_vision_tower"] = 1  # int is not the frozen boolean.
    with pytest.raises(ValueError, match="beyond snapshot locator"):
        initial_reload(ctx, runtime_config=runtime)


def test_iteration_zero_lineage_uses_canonical_config_and_rejects_gate_source(ctx, monkeypatch):
    observed = []
    original = cp.build_rl_lineage
    def lineage(**kwargs):
        observed.append(copy.deepcopy(kwargs["sft_config"]))
        return original(**kwargs)
    monkeypatch.setattr(cp, "build_rl_lineage", lineage)
    initial_reload(ctx)
    assert observed == [ctx.config]
    gate_source = ctx.snapshot.parent / "gate-adapter-copy" / "adapter"
    shutil.copytree(ctx.source.parent, gate_source.parent)
    with pytest.raises(ValueError, match="Gate artifacts forbidden"):
        initial_reload(ctx, source_adapter=gate_source)


def test_unbound_snapshot_run_cannot_be_silently_upgraded(ctx):
    semantics = copy.deepcopy(ctx.run["semantics"])
    del semantics["base_model"]["offline_snapshot_sha256"]
    run = cp.build_training_run_identity(ctx.run["run_id"], semantics=semantics,
        prompt_ids=ctx.run["prompt_ids"], prompt_sources=ctx.run["prompt_sources"])
    cp.validate_training_run_identity(run)  # S1 backwards compatibility unchanged.
    with pytest.raises(ValueError, match="SHA256 identity required"):
        formal.load_formal_actor(run, canonical_config=ctx.config, runtime_config=ctx.runtime_config,
            source_adapter=ctx.source, processor=None, mesh=None, initial_seed=7, cpu_fixture=True,
            construct=cpu_construct, manager_factory=CPUManager)


def test_fresh_reload_and_native_continuation_use_validated_relocated_snapshot(ctx):
    directory, description, result = staged(ctx)
    runtime = relocated_runtime(ctx)
    canonical = copy.deepcopy(ctx.config)
    canonical["model"]["revision"] = "wrong"
    with pytest.raises(ValueError, match="canonical"):
        formal.fresh_reload_staging(ctx.loaded, ctx.data, directory, description,
            canonical_config=canonical, runtime_config=runtime, processor=None, mesh=None,
            construct=cpu_construct, manager_factory=CPUManager)
    assert ctx.loaded.actor is not None  # fail BEFORE destruction/construction.
    fresh, evidence = formal.fresh_reload_staging(ctx.loaded, ctx.data, directory, description,
        canonical_config=ctx.config, runtime_config=runtime, processor=None, mesh=None,
        construct=cpu_construct, manager_factory=CPUManager)
    assert fresh.runtime_locator == runtime["model"]["name_or_path"]
    assert evidence["offline_snapshot_sha256"] == ctx.run["semantics"]["base_model"]["offline_snapshot_sha256"]
    assert "another-machine-snapshot" not in json.dumps(evidence)
    del fresh
    manifest = cp.build_checkpoint_manifest(ctx.run, ctx.loaded.policy, ctx.groups, ctx.window,
        result["attempt"], ctx.reward, artifact_role_files=description["artifact_role_files"],
        file_sha256=description["file_sha256"], kind="smoke_continuation", reload_evidence=evidence, cpu_fixture=True)
    cp.commit_verified_checkpoint(ctx.root, directory, manifest, cpu_fixture=True)
    successor = checkpoint_policy(manifest)
    options = dict(source_adapter=ctx.source, processor=None, mesh=None, policy=successor,
        checkpoint_directory=ctx.root / "checkpoints/policy-000001", cpu_fixture=True,
        construct=cpu_construct, manager_factory=CPUManager)
    with pytest.raises(ValueError, match="canonical"):
        formal.load_formal_actor(ctx.run, canonical_config=canonical, runtime_config=runtime, **options)
    run = cp.build_training_run_identity(ctx.run["run_id"], semantics=ctx.run["semantics"],
        prompt_ids=ctx.run["prompt_ids"], prompt_sources=ctx.run["prompt_sources"],
        locators={"base_snapshot": runtime["model"]["name_or_path"]})
    loaded = formal.load_formal_actor(run, canonical_config=ctx.config, runtime_config=runtime, **options)
    assert loaded.policy == successor and loaded.actor.runtime_locator == runtime["model"]["name_or_path"]
    formal.require_formal_reload(loaded.actor, successor, loaded.reload_receipt)
    (Path(runtime["model"]["name_or_path"]) / "model.safetensors").write_bytes(b"changed continuation base")
    with pytest.raises(ValueError, match="snapshot content identity"):
        formal.load_formal_actor(run, canonical_config=ctx.config, runtime_config=runtime, **options)


def test_fresh_reload_rejects_changed_snapshot_before_destroying_old_actor(ctx):
    directory, description, _ = staged(ctx)
    (ctx.snapshot / "model.safetensors").write_bytes(b"changed snapshot")
    with pytest.raises(ValueError, match="snapshot content identity"):
        formal.fresh_reload_staging(ctx.loaded, ctx.data, directory, description,
            canonical_config=ctx.config, runtime_config=ctx.runtime_config, processor=None, mesh=None,
            construct=cpu_construct, manager_factory=CPUManager)
    assert ctx.loaded.actor is not None


def test_vanilla_loss_explicitly_allowed(ctx):
    formal._require_update_config(SimpleNamespace(config=ActorConfig()), 1)
    result = update(ctx)
    assert result["after_step"] == 1 and ctx.loaded.actor.update_calls == 1


@pytest.mark.parametrize("mode", ["gpg", "rollout_correction", "clip_cov", "other", None])
def test_nonvanilla_or_implicit_loss_fails_before_optimizer(ctx, mode):
    actor = ctx.loaded.actor
    actor.config.policy_loss = {} if mode is None else {"loss_mode": mode}
    with pytest.raises(ValueError, match="explicitly be vanilla"):
        update(ctx)
    assert actor.update_calls == actor.calls == 0
    assert not actor.actor_optimizer.state and not list((ctx.root / "attempts").glob("*/*started.json"))


@pytest.mark.parametrize("policy_loss", [None, "vanilla", SimpleNamespace(loss_mode="vanilla")])
def test_missing_or_wrong_type_loss_config_fails_closed(policy_loss):
    config = ActorConfig(policy_loss=policy_loss)
    with pytest.raises(ValueError, match="explicitly be vanilla"):
        formal._require_update_config(SimpleNamespace(config=config), 1)


def test_shared_snapshot_validator_preserves_gate_file_identity(ctx):
    from opensearch_vl_repro.rl import gate_c
    assert gate_c.offline_snapshot_files is offline_snapshot_files
    assert offline_snapshot_files(ctx.snapshot, revision=BASE_REVISION) == {
        path.name: sha256_file(path) for path in sorted(ctx.snapshot.iterdir())
        if path.is_file() and path.suffix in {".json", ".safetensors"}}
