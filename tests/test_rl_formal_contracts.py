"""S1 fixtures are CPU metadata only, NEVER evidence of an actual RL/GPU update."""
import copy
import inspect
import json
import uuid
from collections import Counter

import pytest

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl.group import (
    formal_group_identity, publish_formal_group, read_formal_group, validate_formal_group,
)
from opensearch_vl_repro.rl.rloo import assemble_window_rloo
from opensearch_vl_repro.rl.run_state import (
    advance_update_attempt, checkpoint_policy, new_update_attempt, persist_update_attempt,
    reconstruct_consumed_ledger, retry_update_plan,
    read_update_attempt,
)
from opensearch_vl_repro.rl.training_window import (
    build_training_window, deterministic_rank_plan, expected_window_prompts, validate_training_window,
)


def digest(value):
    return canonical_json_sha256(value)


def fixture_run(n=2, world_size=2, *, groups_per_window=4, prompt_count=8):
    semantics = {
        "dataset": {"sha256": digest("dataset"), "split": "smoke"},
        "base_model": {"name": "pinned-base", "revision": "pinned-revision"},
        "source_sft": {"adapter_sha256": digest("sft"), "metadata_sha256": digest("metadata"),
                       "stage": "checkpoint-3k", "lineage": ["main_a_1k", "main_b_2k"]},
        "execution_contract": {"dropout": .05, "lora_rank": 16},
        "rollout": {"behavior_version": 1, "config": {"temperature": .7, "seed": 7}},
        "rollout_n": n, "weighting": cp.FORMAL_WEIGHTING,
        "groups_per_window": groups_per_window, "require_complete_windows": True,
        "optimizer": {"name": "AdamW", "lr": 1e-6, "weight_decay": 0},
        "ppo": {"epochs": 1, "clip_ratio_low": .2, "clip_ratio_high": .28},
        "world_size": world_size, "reward": {"version": 1, "semantics": "format*(.8*accuracy+.2*query)"},
        "tool_protocol_version": "tool-v3", "image_protocol_version": "runtime-image-id-grounding-v3",
        "integration_source_hashes": {"verl": digest("verl-code"), "rllm": digest("rllm-code")},
    }
    prompt_ids = [f"p{i}" for i in range(prompt_count)]
    return cp.build_training_run_identity("cpu-contract", semantics=semantics, prompt_ids=prompt_ids,
                                          prompt_sources=[{"prompt_id": p, "source_identity": digest(p)}
                                                          for p in prompt_ids])


def fixture_policy(run):
    return cp.initial_policy(run, optimizer_identity=digest("empty-optimizer"), rng_identity=digest("initial-rng"))


def draft_group(run, policy, prompt="p0"):
    identity = formal_group_identity(run, policy, prompt_id=prompt, source_identity=digest(prompt),
                                     attempt_id=str(uuid.uuid4()), attempt_index=0)
    members = []
    for index in range(run["semantics"]["rollout_n"]):
        accuracy = index / (run["semantics"]["rollout_n"] - 1)
        step = {"prompt_ids": [1, 2], "response_ids": [3, 4], "logprobs": [-.1, -.2],
                "model_output": {"prompt_ids": [1, 2], "completion_ids": [3, 4], "logprobs": [-.1, -.2]},
                "info": {"token_origin": "vllm.RequestOutput", "logprobs_mode": "processed_logprobs"},
                "multimodal_file": f"mm-{index}.fixture"}
        members.append({"identity": copy.deepcopy(identity), "rollout_index": index,
                        "member_id": f"{identity['trajectory_group_id']}:{index}",
                        "complete": True, "fatal": False, "steps": [step],
                        "trajectory_file": f"trajectory-{index}.json",
                        "reward": {"format": 1., "accuracy": accuracy, "query": .5, "total": .8 * accuracy + .1}})
    return {"identity": identity, "members": members}


def commit_group(root, run, policy, prompt="p0", *, draft=None):
    group = draft or draft_group(run, policy, prompt)
    staging = root / "groups" / f".stage-{uuid.uuid4()}"
    staging.mkdir(parents=True)
    for member in group["members"]:
        (staging / member["trajectory_file"]).write_text(json.dumps(member), encoding="utf-8")
        (staging / member["steps"][0]["multimodal_file"]).write_bytes(b"CPU fixture: NOT a tensor")
    destination = root / "groups" / group["identity"]["trajectory_group_id"]
    value = publish_formal_group(staging, destination, group, cpu_fixture=True)
    assert read_formal_group(destination) == value
    return value


def cpu_estimator(rewards, mask, index):
    """Explicit fake official callback for schemas, not a production fallback."""
    import torch
    result = torch.zeros_like(rewards)
    for gid in set(index.tolist()):
        rows = [i for i, item in enumerate(index) if item == gid]
        for i in rows:
            result[i, 0] = rewards[i, 0] - sum(rewards[j, 0] for j in rows if j != i) / (len(rows) - 1)
    assert mask.shape == rewards.shape
    return result, result.clone()


def staging_attempt(root, window):
    attempt = new_update_attempt(window)
    persist_update_attempt(root, attempt, cpu_fixture=True)
    for phase in ("started", "step_may_have_run", "checkpoint_staging"):
        attempt = advance_update_attempt(attempt, phase)
        persist_update_attempt(root, attempt, cpu_fixture=True)
    return attempt


def prepare_checkpoint(root, run, policy, groups, window, *, attempt=None, kind="smoke_continuation"):
    attempt = attempt or staging_attempt(root, window)
    rewards = assemble_window_rloo(window, run, policy, groups, estimator=cpu_estimator)
    staging = root / "checkpoints" / f".stage-{uuid.uuid4()}"
    staging.mkdir()
    names = ["adapter/adapter_config.json.fixture", "adapter/adapter_model.safetensors.fixture"]
    names += [f"{role}/{stem}_rank_{rank}.fixture"
              for role, stem in (("native", "model"), ("optimizer", "optim"), ("rng", "extra"))
              for rank in range(run["semantics"]["world_size"])]
    for name in names:
        path = staging / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"{name}:{attempt['attempt_id']}:CPU ONLY".encode())
    inventory = cp.artifact_inventory(staging)
    role_files = {role: {name: sha for name, sha in inventory.items() if name.startswith(role + "/")}
                  for role in ("adapter", "native", "optimizer", "rng")}
    roles = cp.artifact_role_identities(role_files, inventory)
    evidence = {"scope": "cpu_fixture", "reloaded_artifact_roles": roles, "adapter_reloaded": True,
                "native_reloaded": True, "optimizer_reloaded": True, "rng_reloaded": True,
                "execution_contract_verified": True}
    manifest = cp.build_checkpoint_manifest(run, policy, groups, window, attempt, rewards,
                                            artifact_role_files=role_files, file_sha256=inventory, kind=kind,
                                            reload_evidence=evidence, cpu_fixture=True)
    return staging, manifest


@pytest.fixture
def context(tmp_path):
    run = fixture_run()
    policy = fixture_policy(run)
    root = tmp_path / "formal-cpu-fixture"
    cp.initialize_formal_run(root, run, policy, cpu_fixture=True)
    groups = [commit_group(root, run, policy, f"p{i}") for i in range(4)]
    window = build_training_window(run, policy, groups, window_id="window-0")
    return root, run, policy, groups, window


@pytest.mark.parametrize("n", [2, 4, 7])
def test_parameterized_group_commits(tmp_path, n):
    run = fixture_run(n)
    policy = fixture_policy(run)
    group = commit_group(tmp_path, run, policy)
    validate_formal_group(group, committed=True)
    assert {m["rollout_index"] for m in group["members"]} == set(range(n))


@pytest.mark.parametrize("error", ["missing", "duplicate_index", "bad_index", "mixed_policy", "mixed_attempt",
                                 "duplicate_member", "incomplete", "nan_reward", "missing_artifact"])
def test_invalid_groups_fail_closed(context, error):
    _, run, policy, groups, _ = context
    value = copy.deepcopy(groups[0])
    if error == "missing": value["members"].pop()
    if error == "duplicate_index": value["members"][1]["rollout_index"] = 0
    if error == "bad_index": value["members"][1]["rollout_index"] = 2
    if error == "mixed_policy": value["members"][1]["identity"]["pre_update_policy_fingerprint"] = digest("wrong")
    if error == "mixed_attempt": value["members"][1]["identity"]["collection_attempt"] = str(uuid.uuid4())
    if error == "duplicate_member": value["members"][1]["member_id"] = value["members"][0]["member_id"]
    if error == "incomplete": value["members"][0]["complete"] = False
    if error == "nan_reward": value["members"][0]["reward"]["total"] = float("nan")
    if error == "missing_artifact": value["members"][0]["trajectory_file"] = "missing.json"
    if error != "nan_reward":
        value = cp.seal({k: v for k, v in value.items() if k != "group_payload_sha256"}, "group_payload_sha256")
    with pytest.raises(ValueError):
        validate_formal_group(value, committed=True)


@pytest.mark.parametrize("error", ["duplicate", "uncommitted", "mixed_iteration", "mixed_policy", "foreign_rollout"])
def test_window_rejects_invalid_groups(context, error):
    _, run, policy, groups, _ = context
    values = copy.deepcopy(groups)
    if error == "duplicate": values[1] = values[0]
    if error == "uncommitted": values[0]["committed"] = False
    if error in {"mixed_iteration", "mixed_policy", "foreign_rollout"}:
        identity = values[1]["identity"]
        field = {"mixed_iteration": "policy_iteration", "mixed_policy": "pre_update_policy_fingerprint",
                 "foreign_rollout": "rollout_config_fingerprint"}[error]
        identity[field] = 1 if error == "mixed_iteration" else digest("foreign")
        identity["trajectory_group_id"] = digest({k: v for k, v in identity.items() if k != "trajectory_group_id"})
        for member in values[1]["members"]: member["identity"] = copy.deepcopy(identity)
    for i, group in enumerate(values):
        values[i] = cp.seal({k: v for k, v in group.items() if k != "group_payload_sha256"}, "group_payload_sha256")
    with pytest.raises(ValueError):
        build_training_window(run, policy, values, window_id="bad")


def test_window_identity_tamper(context):
    _, run, policy, groups, window = context
    assert len(window["ordered_group_ids"]) == 4
    bad = cp.seal({**{k: v for k, v in window.items() if k != "window_sha256"},
                   "expected_optimizer_step": 4}, "window_sha256")
    with pytest.raises(ValueError): validate_training_window(bad, run, policy, groups)


@pytest.mark.parametrize("n", [2, 4])
def test_grouped_rloo_and_fatal_order(tmp_path, n):
    run, groups = fixture_run(n, groups_per_window=2), []
    policy = fixture_policy(run)
    for i in range(2):
        draft = draft_group(run, policy, f"p{i}")
        draft["members"][0]["fatal"] = True
        if i == 1:
            for m in draft["members"]:
                m["reward"] = {"format": 1., "accuracy": .5, "query": .5, "total": .5}
        groups.append(commit_group(tmp_path, run, policy, draft=draft))
    window = build_training_window(run, policy, groups, window_id="rloo")
    result = assemble_window_rloo(window, run, policy, groups, estimator=cpu_estimator)
    assert result["status"] == "signal"
    assert len(result["rows"]) == n * 2
    assert result["rows"][0]["raw_advantage"] < 0
    assert result["rows"][0]["final_advantage"] == 0
    assert all(abs(r["raw_advantage"]) < 1e-12 for r in result["rows"][n:])
    # Fatal remained in the baseline: highest healthy reward retains a positive advantage.
    assert result["rows"][n - 1]["final_advantage"] > 0


def test_zero_signal_and_no_checkpoint(tmp_path):
    run = fixture_run(4, groups_per_window=1)
    policy = fixture_policy(run)
    draft = draft_group(run, policy)
    for member in draft["members"]:
        member["reward"] = {"format": 1., "accuracy": .5, "query": .5, "total": .5}
    group = commit_group(tmp_path, run, policy, draft=draft)
    window = build_training_window(run, policy, [group], window_id="zero")
    rewards = assemble_window_rloo(window, run, policy, [group], estimator=cpu_estimator)
    assert rewards["status"] == "zero_signal"
    assert all(r["final_advantage"] == 0 for r in rewards["rows"])
    attempt = new_update_attempt(window)
    for phase in ("started", "step_may_have_run", "checkpoint_staging"):
        attempt = advance_update_attempt(attempt, phase)
    with pytest.raises(ValueError, match="zero-signal"):
        cp.build_checkpoint_manifest(run, policy, [group], window, attempt, rewards,
                                     artifact_role_files={}, file_sha256={}, kind="smoke_final", reload_evidence={})


def test_no_silent_rloo_fallback(context, monkeypatch):
    import sys
    _, run, policy, groups, window = context
    monkeypatch.setitem(sys.modules, "verl.trainer.ppo.core_algos", None)
    with pytest.raises(ModuleNotFoundError): assemble_window_rloo(window, run, policy, groups)
    def bad(rewards, mask, index): return rewards * 0, rewards * 0
    with pytest.raises(RuntimeError): assemble_window_rloo(window, run, policy, groups, estimator=bad)


def test_cpu_end_to_end_and_index_rebuild(context):
    root, run, policy, groups, window = context
    ledger = reconstruct_consumed_ledger(run, policy, groups, [], window=window)
    assert sum(v["status"] == "assigned_to_window" for v in ledger["prompts"].values()) == 4
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    # A metadata manifest alone cannot authorize consumption.
    with pytest.raises(ValueError, match="publication|published"):
        reconstruct_consumed_ledger(run, policy, groups, [manifest])
    cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    assert manifest["eligibility"]["eligible_for_main_init"] is False
    assert manifest["eligibility"]["same_run_resume"] is True
    destination = root / "checkpoints" / "policy-000001"
    cp.read_verified_checkpoint(destination)
    verified = advance_update_attempt(manifest["update_attempt"], "verified", checkpoint=manifest,
                                     checkpoint_directory=destination)
    persist_update_attempt(root, verified, cpu_fixture=True)
    for path in (root / "latest.json", root / "state.json"):
        path.write_text('{"untrusted": true}', encoding="utf-8")
    recovered = cp.recover_formal_run(root, run, cpu_fixture=True)
    assert recovered["policy"]["policy_iteration"] == 1
    assert len(recovered["ledger"]["consumed_group_ids"]) == 4
    assert len(recovered["policy"]["cumulative_consumed_group_ids"]) == 4
    next_groups = [commit_group(root, run, recovered["policy"], f"p{i}") for i in range(4, 8)]
    next_window = build_training_window(run, recovered["policy"], next_groups, window_id="window-1")
    assert next_window["expected_optimizer_step"] == 2
    with pytest.raises(ValueError): build_training_window(run, policy, next_groups, window_id="old")
    with pytest.raises(ValueError): build_training_window(run, recovered["policy"], groups, window_id="reuse")
    assert json.loads((root / "latest.json").read_text())["policy"] == recovered["policy"]


def test_ambiguous_retry_reuses_groups_without_consumption(context):
    root, run, policy, groups, window = context
    attempt = new_update_attempt(window)
    persist_update_attempt(root, attempt, cpu_fixture=True)
    for phase in ("started", "step_may_have_run"):
        attempt = advance_update_attempt(attempt, phase)
        persist_update_attempt(root, attempt, cpu_fixture=True)
    before = (root / "attempts" / attempt["attempt_id"]).glob("*.json")
    historical = {p.name: p.read_bytes() for p in before}
    recovered = cp.recover_formal_run(root, run, cpu_fixture=True)
    plan = retry_update_plan(attempt, window, recovered["policy"], verified_checkpoints=recovered["checkpoints"])
    assert plan["reload_checkpoint_identity"] == policy["checkpoint_identity"]
    assert plan["reload_optimizer_identity"] == policy["optimizer_identity"]
    assert plan["reload_rng_identity"] == policy["rng_identity"]
    assert plan["reuse_group_ids"] == window["ordered_group_ids"]
    fresh = plan["new_attempt"]
    assert fresh["attempt_id"] != attempt["attempt_id"] and plan["discard_uncertain_memory"]
    failed = advance_update_attempt(attempt, "failed", failure_reason="crash: optimizer outcome unknown")
    persist_update_attempt(root, failed, cpu_fixture=True)
    assert not cp.recover_formal_run(root, run, cpu_fixture=True)["ledger"]["consumed_group_ids"]
    persist_update_attempt(root, fresh, cpu_fixture=True)
    for phase in ("started", "step_may_have_run", "checkpoint_staging"):
        fresh = advance_update_attempt(fresh, phase)
        persist_update_attempt(root, fresh, cpu_fixture=True)
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window, attempt=fresh)
    cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    for name, data in historical.items():
        assert (root / "attempts" / attempt["attempt_id"] / name).read_bytes() == data
    with pytest.raises(ValueError, match="verified successor"):
        retry_update_plan(attempt, window, policy, verified_checkpoints=[manifest])


@pytest.mark.parametrize("failure", ["manifest", "rename", "parent_fsync", "latest", "state"])
def test_checkpoint_atomic_failure_and_recovery(context, monkeypatch, failure):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    real_json, real_rename, real_sync = cp.durable_json, cp.os.rename, cp.fsync_directory
    def writer(path, value, **kwargs):
        if (failure == "manifest" and path.name == "checkpoint.json"
                or failure == "latest" and path.name == "latest.json"
                or failure == "state" and path.name == "state.json"):
            raise OSError("injected publication failure")
        return real_json(path, value, **kwargs)
    def rename(source, destination):
        if failure == "rename": raise OSError("injected rename failure")
        return real_rename(source, destination)
    def sync(path, **kwargs):
        if (failure == "parent_fsync" and path == root / "checkpoints"
                and (root / "checkpoints" / "policy-000001").exists()):
            raise OSError("injected post-rename parent fsync failure")
        return real_sync(path, **kwargs)
    monkeypatch.setattr(cp, "durable_json", writer)
    monkeypatch.setattr(cp.os, "rename", rename)
    monkeypatch.setattr(cp, "fsync_directory", sync)
    with pytest.raises(OSError): cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    monkeypatch.setattr(cp, "durable_json", real_json)
    monkeypatch.setattr(cp.os, "rename", real_rename)
    monkeypatch.setattr(cp, "fsync_directory", real_sync)
    recovered = cp.recover_formal_run(root, run, cpu_fixture=True)
    expected_iteration = 0 if failure in {"manifest", "rename"} else 1
    assert recovered["policy"]["policy_iteration"] == expected_iteration
    assert len(recovered["ledger"]["consumed_group_ids"]) == 4 * expected_iteration
    if expected_iteration == 0:
        with pytest.raises(ValueError, match="staging"): cp.read_verified_checkpoint(staging)
    else:
        with pytest.raises(ValueError, match="verified successor"):
            retry_update_plan(manifest["update_attempt"], window, policy, verified_checkpoints=recovered["checkpoints"])


def test_checkpoint_order_and_no_overwrite(context, monkeypatch):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    events = []
    real_json, real_rename, real_sync = cp.durable_json, cp.os.rename, cp.fsync_directory
    def writer(path, value, **kwargs):
        events.append(("json", path.name)); return real_json(path, value, **kwargs)
    def rename(source, dest):
        events.append(("rename", dest.name)); return real_rename(source, dest)
    def sync(path, **kwargs):
        events.append(("fsync", path.name)); return real_sync(path, **kwargs)
    monkeypatch.setattr(cp, "durable_json", writer)
    monkeypatch.setattr(cp.os, "rename", rename)
    monkeypatch.setattr(cp, "fsync_directory", sync)
    cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    assert events.index(("json", "checkpoint.json")) < events.index(("rename", "policy-000001"))
    assert ("fsync", staging.name) in events[:events.index(("rename", "policy-000001"))]
    assert events.index(("rename", "policy-000001")) < events.index(("json", "latest.json"))
    another, conflict = prepare_checkpoint(root, run, policy, groups, window)
    with pytest.raises(ValueError, match="successor"): cp.commit_verified_checkpoint(root, another, conflict, cpu_fixture=True)
    assert cp.read_verified_checkpoint(root / "checkpoints" / "policy-000001") == manifest


@pytest.mark.parametrize("role", ["adapter", "native", "optimizer", "rng"])
def test_artifact_reload_failure_closed(context, role):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    bad = copy.deepcopy(manifest)
    bad["reload_evidence"][f"{role}_reloaded"] = False
    bad = cp.seal({k: v for k, v in bad.items() if k != "checkpoint_manifest_sha256"}, "checkpoint_manifest_sha256")
    with pytest.raises(ValueError): cp.commit_verified_checkpoint(root, staging, bad, cpu_fixture=True)
    assert not list((root / "checkpoints").glob("policy-*"))


def test_artifact_tamper_and_failed_attempt_cannot_commit(context):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    (staging / "rng" / "extra_rank_1.fixture").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="inventory"): cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    failed = advance_update_attempt(manifest["update_attempt"], "failed", failure_reason="simulated crash")
    persist_update_attempt(root, failed, cpu_fixture=True)
    with pytest.raises(ValueError, match="staging attempt"): cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    assert not cp.recover_formal_run(root, run, cpu_fixture=True)["ledger"]["consumed_group_ids"]


@pytest.mark.parametrize("count,world", [(8, 2), (6, 3), (3, 2), (5, 4), (1, 7)])
def test_rank_plan_equal_multiplicity_no_drop(count, world):
    ids = [f"row-{i}" for i in range(count)]
    plan = deterministic_rank_plan(ids, world)
    assert plan == deterministic_rank_plan(ids, world)
    assert len(plan["rank_assignments"]) == world
    physical = [r for rows in plan["rank_assignments"] for r in rows]
    assert Counter(physical) == plan["multiplicity"]
    assert set(physical) == set(ids)
    assert len(set(len(rows) for rows in plan["rank_assignments"])) == 1
    assert plan["physical_count"] == count * plan["replication_factor"]
    if count % world == 0: assert plan["replication_factor"] == 1
    else: assert len(set(plan["multiplicity"].values())) == 1


@pytest.mark.parametrize("ids,world", [([], 2), (["a", "a"], 2), (["a"], 0), (["a"], True)])
def test_invalid_rank_inputs(ids, world):
    with pytest.raises(ValueError): deterministic_rank_plan(ids, world)


def test_run_identity_deterministic_locators_not_identity():
    run = fixture_run()
    other = cp.build_training_run_identity(run["run_id"], semantics=copy.deepcopy(run["semantics"]),
                                           prompt_ids=run["prompt_ids"], prompt_sources=run["prompt_sources"],
                                           locators={"adapter_path": "different-machine"})
    cp.require_same_training_run(run, other)
    assert run["training_behavior_fingerprint"] == other["training_behavior_fingerprint"]
    assert run["run_identity_sha256"] == other["run_identity_sha256"]
    assert run == fixture_run()


@pytest.mark.parametrize("key", list(fixture_run()["semantics"]))
def test_every_semantic_identity_change_blocks_resume(key):
    run = fixture_run()
    semantics = copy.deepcopy(run["semantics"])
    old = semantics[key]
    if isinstance(old, dict):
        if key == "integration_source_hashes": old["verl"] = digest("changed")
        else: old["changed_semantics"] = True
    elif type(old) is bool: semantics[key] = not old
    elif type(old) is int: semantics[key] += 1
    else: semantics[key] += "-changed"
    try:
        changed = cp.build_training_run_identity(run["run_id"], semantics=semantics, prompt_ids=run["prompt_ids"],
                                                 prompt_sources=run["prompt_sources"])
    except ValueError:
        return  # unsupported semantics are also fail-closed
    with pytest.raises(ValueError, match="resume semantics"): cp.require_same_training_run(run, changed)


def test_order_schema_and_behavior_are_bound():
    run = fixture_run()
    reordered = cp.build_training_run_identity(run["run_id"], semantics=run["semantics"],
                                               prompt_ids=list(reversed(run["prompt_ids"])),
                                               prompt_sources=list(reversed(run["prompt_sources"])))
    with pytest.raises(ValueError): cp.require_same_training_run(run, reordered)
    for key in ("schema_version", "training_behavior_version"):
        bad = copy.deepcopy(run); bad[key] += 1
        with pytest.raises(ValueError): cp.validate_training_run_identity(bad)
    semantics = copy.deepcopy(run["semantics"])
    semantics["source_sft"]["adapter_path"] = "forbidden"
    with pytest.raises(ValueError, match="paths"):
        cp.build_training_run_identity("x", semantics=semantics, prompt_ids=run["prompt_ids"],
                                       prompt_sources=run["prompt_sources"])


def test_gate_output_isolation_and_eligibility(tmp_path):
    run = fixture_run()
    with pytest.raises(ValueError, match="Gate outputs"):
        cp.initialize_formal_run(tmp_path / "outputs" / "rl_gate_c" / "run", run, fixture_policy(run), cpu_fixture=True)
    for kind in ("gate_artifact", "smoke_continuation", "smoke_final", "main_checkpoint"):
        assert cp.checkpoint_eligibility(kind)["eligible_for_main_init"] is False
    assert cp.checkpoint_eligibility("gate_artifact")["same_run_resume"] is False
    for module in (cp, __import__("opensearch_vl_repro.rl.training_window", fromlist=["x"])):
        source = inspect.getsource(module)
        assert "scheduler.step(" not in source and "optimizer.step(" not in source
        assert "from vllm" not in source and "from rllm" not in source


@pytest.mark.parametrize("n,world", [(2, 2), (4, 3)])
def test_two_iterations_consumption_is_cumulative(tmp_path, n, world):
    root, run = tmp_path / "formal", fixture_run(n, world, groups_per_window=2)
    initial = policy = fixture_policy(run)
    cp.initialize_formal_run(root, run, policy, cpu_fixture=True)
    all_groups, checkpoints = [], []
    for iteration in range(2):
        groups = [commit_group(root, run, policy, f"p{iteration * 2 + i}") for i in range(2)]
        all_groups.extend(groups)
        window = build_training_window(run, policy, groups, window_id=f"window-{iteration}")
        staging, checkpoint = prepare_checkpoint(root, run, policy, groups, window)
        cp.commit_verified_checkpoint(root, staging, checkpoint, cpu_fixture=True)
        checkpoints.append(checkpoint)
        policy = cp.recover_formal_run(root, run, cpu_fixture=True)["policy"]
        assert policy["policy_iteration"] == policy["global_optimizer_step"] == iteration + 1
        assert len(policy["cumulative_consumed_group_ids"]) == (iteration + 1) * 2
    with pytest.raises(ValueError, match="chain"):
        reconstruct_consumed_ledger(run, initial, all_groups, [checkpoints[0], checkpoints[0]],
                                     checkpoint_root=root / "checkpoints")
    # A newly collected group for an already consumed prompt must not become another update.
    recollected = [commit_group(root, run, policy, p) for p in ("p0", "p1")]
    with pytest.raises(ValueError, match="ordered prompt slice"):
        build_training_window(run, policy, recollected, window_id="illegal-reconsume")
    # Audit refuses the extra group even BEFORE checkpoint publication.
    with pytest.raises(ValueError, match="consumed prompt"):
        cp.recover_formal_run(root, run, cpu_fixture=True)


def test_committed_artifacts_rechecked_on_recovery(context):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    (root / "checkpoints" / "policy-000001" / "optimizer" / "optim_rank_1.fixture").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="inventory"):
        cp.recover_formal_run(root, run, cpu_fixture=True)


def test_attempt_history_missing_event_refuses_commit(context):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    directory = root / "attempts" / manifest["update_attempt"]["attempt_id"]
    # Simulate a torn historical log without deleting any user artifact.
    (directory / "000001-started.json").rename(directory / "lost-event.fixture")
    with pytest.raises(ValueError, match="history"):
        read_update_attempt(root, manifest["update_attempt"]["attempt_id"])
    with pytest.raises(ValueError, match="history"):
        cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    assert not (root / "checkpoints" / "policy-000001").exists()


def test_windows_nonfixture_transactions_fail_before_manifest(context, monkeypatch):
    from types import SimpleNamespace
    root, run, policy, groups, window = context
    # Exercise the OS branch explicitly; it cannot accidentally publish runtime PASS.
    monkeypatch.setattr(cp, "os", SimpleNamespace(name="nt"))
    with pytest.raises(OSError, match="POSIX"):
        cp.fsync_directory(root)


def test_cpu_evidence_cannot_be_relabeled_runtime(context):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    evidence = copy.deepcopy(manifest["reload_evidence"])
    evidence["scope"] = "runtime"
    with pytest.raises(ValueError, match="CPU fixture"):
        cp.build_checkpoint_manifest(run, policy, groups, window, manifest["update_attempt"], manifest["reward_window"],
                                     artifact_role_files=manifest["artifact_role_files"], file_sha256=manifest["file_sha256"],
                                     kind="smoke_final", reload_evidence=evidence)


def test_groups_per_window_changes_both_fingerprints_and_blocks_resume():
    a = fixture_run(groups_per_window=2)
    b = fixture_run(groups_per_window=4)
    assert a["training_behavior_fingerprint"] != b["training_behavior_fingerprint"]
    assert a["run_identity_sha256"] != b["run_identity_sha256"]
    with pytest.raises(ValueError, match="resume semantics"):
        cp.require_same_training_run(a, b)


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "4"])
def test_k_is_positive_integer(k):
    with pytest.raises(ValueError): fixture_run(groups_per_window=k)


def test_incomplete_windows_rejected_without_dropping_prompts():
    with pytest.raises(ValueError, match="complete K-sized"):
        fixture_run(groups_per_window=4, prompt_count=7)
    run = fixture_run()
    semantics = {**run["semantics"], "require_complete_windows": False}
    with pytest.raises(ValueError, match="must be true"):
        cp.build_training_run_identity(run["run_id"], semantics=semantics,
                                       prompt_ids=run["prompt_ids"], prompt_sources=run["prompt_sources"])


@pytest.mark.parametrize("size", [3, 5])
def test_window_group_count_must_be_exactly_k(context, size):
    root, run, policy, groups, _ = context
    values = groups[:size]
    if size == 5: values = groups + [commit_group(root, run, policy, "p4")]
    with pytest.raises(ValueError, match="group count"):
        build_training_window(run, policy, values, window_id="wrong-count")


@pytest.mark.parametrize("case", ["reverse", "skip", "future"])
def test_window_order_is_enforced(context, case):
    root, run, policy, groups, window = context
    assert window["ordered_prompt_ids"] == ["p0", "p1", "p2", "p3"]
    assert window["groups_per_window"] == run["semantics"]["groups_per_window"] == 4
    assert window["require_complete_windows"] is True
    if case == "reverse": values = list(reversed(groups))
    if case == "skip": values = groups[:3] + [commit_group(root, run, policy, "p4")]
    if case == "future": values = [commit_group(root, run, policy, f"p{i}") for i in range(4, 8)]
    with pytest.raises(ValueError, match="ordered prompt slice"):
        build_training_window(run, policy, values, window_id="bad-order")


@pytest.mark.parametrize("count,k,n", [(20, 4, 2), (400, 4, 4), (12, 3, 2)])
def test_ordered_slice_is_generic_and_exhaustion_fails(count, k, n):
    run = fixture_run(n, groups_per_window=k, prompt_count=count)
    actual = [p for iteration in range(count // k) for p in expected_window_prompts(run, iteration)]
    assert actual == run["prompt_ids"]
    assert expected_window_prompts(run, 1) == run["prompt_ids"][k:2 * k]
    with pytest.raises(ValueError, match="no complete"):
        expected_window_prompts(run, count // k)


def test_authoritative_source_identity_required():
    run = fixture_run()
    policy = fixture_policy(run)
    assert draft_group(run, policy)["identity"]["source_identity"] == run["prompt_sources"][0]["source_identity"]
    with pytest.raises(ValueError, match="authoritative"):
        formal_group_identity(run, policy, prompt_id="p0", source_identity=digest("wrong-but-valid-sha"),
                              attempt_id=str(uuid.uuid4()), attempt_index=0)


@pytest.mark.parametrize("case", ["missing", "duplicate", "reordered", "wrong_prompt", "bad_sha", "extra_field"])
def test_source_membership_cannot_drift(case):
    run = fixture_run()
    sources = copy.deepcopy(run["prompt_sources"])
    if case == "missing": sources.pop()
    if case == "duplicate": sources[1] = sources[0]
    if case == "reordered": sources.reverse()
    if case == "wrong_prompt": sources[0]["prompt_id"] = "other"
    if case == "bad_sha": sources[0]["source_identity"] = "not-sha"
    if case == "extra_field": sources[0]["locator"] = "not-membership"
    with pytest.raises(ValueError):
        cp.build_training_run_identity(run["run_id"], semantics=run["semantics"],
                                       prompt_ids=run["prompt_ids"], prompt_sources=sources)


def test_source_change_is_a_behavior_and_run_identity_change():
    run = fixture_run()
    sources = copy.deepcopy(run["prompt_sources"])
    sources[0]["source_identity"] = digest("different source record")
    changed = cp.build_training_run_identity(run["run_id"], semantics=run["semantics"],
                                            prompt_ids=run["prompt_ids"], prompt_sources=sources)
    assert run["prompt_membership_sha256"] != changed["prompt_membership_sha256"]
    assert run["training_behavior_fingerprint"] != changed["training_behavior_fingerprint"]
    with pytest.raises(ValueError): cp.require_same_training_run(run, changed)


def test_resealed_group_with_wrong_source_rejected_by_window_and_ledger(context):
    _, run, policy, groups, _ = context
    bad = copy.deepcopy(groups[0])
    identity = bad["identity"]
    identity["source_identity"] = digest("wrong source")
    identity["trajectory_group_id"] = digest({k: v for k, v in identity.items() if k != "trajectory_group_id"})
    for member in bad["members"]: member["identity"] = copy.deepcopy(identity)
    bad = cp.seal({k: v for k, v in bad.items() if k != "group_payload_sha256"}, "group_payload_sha256")
    with pytest.raises(ValueError, match="authoritative"):
        build_training_window(run, policy, [bad] + groups[1:], window_id="forged")
    with pytest.raises(ValueError, match="authoritative"):
        reconstruct_consumed_ledger(run, policy, [bad], [])


@pytest.mark.parametrize("role", ["adapter", "native", "optimizer", "rng"])
def test_multifile_role_identity_is_canonical_and_used_by_policy(context, role):
    root, run, policy, groups, window = context
    _, manifest = prepare_checkpoint(root, run, policy, groups, window)
    files = manifest["artifact_role_files"]
    assert len(files[role]) == 2
    reordered = {r: dict(reversed(list(v.items()))) for r, v in reversed(list(files.items()))}
    identities = cp.artifact_role_identities(reordered, dict(reversed(list(manifest["file_sha256"].items()))))
    assert identities == manifest["artifact_roles"]
    assert identities[role] == digest(files[role])
    assert identities[role] not in manifest["file_sha256"].values()
    successor = checkpoint_policy(manifest)
    field = {"adapter": "adapter_fingerprint", "native": "native_identity",
             "optimizer": "optimizer_identity", "rng": "rng_identity"}[role]
    assert successor[field] == identities[role]


@pytest.mark.parametrize("role", ["adapter", "native", "optimizer", "rng"])
def test_changing_any_rank_or_adapter_file_changes_whole_role_identity(context, role):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    name = sorted(manifest["artifact_role_files"][role])[-1]
    (staging / name).write_bytes(b"changed rank artifact")
    with pytest.raises(ValueError, match="inventory"):
        cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    inventory = cp.artifact_inventory(staging)
    files = copy.deepcopy(manifest["artifact_role_files"])
    files[role][name] = inventory[name]
    identities = cp.artifact_role_identities(files, inventory)
    assert identities[role] != manifest["artifact_roles"][role]
    for other in set(identities) - {role}: assert identities[other] == manifest["artifact_roles"][other]


@pytest.mark.parametrize("error", ["missing_rank", "missing_role", "unknown_role", "empty_role", "nonexistent_file",
                                 "conflicting_owner", "hash_mismatch", "global_missing", "unowned_metadata"])
def test_role_partition_fail_closed(context, error):
    root, run, policy, groups, window = context
    _, manifest = prepare_checkpoint(root, run, policy, groups, window)
    files, inventory = copy.deepcopy(manifest["artifact_role_files"]), dict(manifest["file_sha256"])
    name = "native/model_rank_1.fixture"
    if error == "missing_rank": del files["native"][name]
    if error == "missing_role": del files["native"]
    if error == "unknown_role": files["unknown"] = {"other.fixture": digest("other")}
    if error == "empty_role": files["rng"] = {}
    if error == "nonexistent_file": files["native"]["native/missing.fixture"] = digest("missing")
    if error == "conflicting_owner": files["optimizer"][name] = inventory[name]
    if error == "hash_mismatch": files["native"][name] = digest("wrong")
    if error == "global_missing": del inventory[name]
    if error == "unowned_metadata": inventory["metadata.json.fixture"] = digest("metadata")
    with pytest.raises(ValueError): cp.artifact_role_identities(files, inventory)


def test_missing_physical_rank_file_fails_commit(context):
    root, run, policy, groups, window = context
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    (staging / "native/model_rank_1.fixture").rename(root / "missing-rank.fixture")
    with pytest.raises(ValueError, match="inventory"):
        cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    assert not (root / "checkpoints" / "policy-000001").exists()


def test_metadata_role_is_explicit(context):
    root, run, policy, groups, window = context
    _, manifest = prepare_checkpoint(root, run, policy, groups, window)
    files, inventory = copy.deepcopy(manifest["artifact_role_files"]), dict(manifest["file_sha256"])
    inventory["metadata.json.fixture"] = digest("metadata")
    files["metadata"] = {"metadata.json.fixture": inventory["metadata.json.fixture"]}
    assert cp.artifact_role_identities(files, inventory)["metadata"] == digest(files["metadata"])


def test_resealed_checkpoint_cannot_claim_rank0_as_complete_role(context):
    root, run, policy, groups, window = context
    _, manifest = prepare_checkpoint(root, run, policy, groups, window)
    bad = copy.deepcopy(manifest)
    bad["artifact_roles"]["native"] = bad["file_sha256"]["native/model_rank_0.fixture"]
    bad["reload_evidence"]["reloaded_artifact_roles"]["native"] = bad["artifact_roles"]["native"]
    bad = cp.seal({k: v for k, v in bad.items() if k != "checkpoint_manifest_sha256"}, "checkpoint_manifest_sha256")
    with pytest.raises(ValueError): cp.validate_checkpoint_manifest(bad)


def test_formal_versions_changed_only_for_changed_contracts():
    assert cp.RL_TRAINING_BEHAVIOR_VERSION == cp.RL_RUN_SCHEMA_VERSION == 2
    assert cp.RL_CHECKPOINT_SCHEMA_VERSION == cp.RL_WINDOW_SCHEMA_VERSION == 2
    assert cp.RL_GROUP_SCHEMA_VERSION == 1


@pytest.mark.parametrize("name", ["../escape.fixture", "/absolute.fixture", "folder\\file", "C:/file",
                                "./file", ".", "checkpoint.json"])
def test_role_filenames_cannot_escape_or_include_final_receipt(context, name):
    root, run, policy, groups, window = context
    _, manifest = prepare_checkpoint(root, run, policy, groups, window)
    files, inventory = copy.deepcopy(manifest["artifact_role_files"]), dict(manifest["file_sha256"])
    files["metadata"] = {name: digest("file")}
    inventory[name] = digest("file")
    with pytest.raises(ValueError, match="relative artifact"):
        cp.artifact_role_identities(files, inventory)


def test_old_formal_run_and_checkpoint_schema_fail_closed(context):
    root, run, policy, groups, window = context
    old = copy.deepcopy(run)
    old["schema_version"] = 1
    del old["prompt_sources"]
    with pytest.raises(ValueError, match="schema"):
        cp.validate_training_run_identity(old)
    _, manifest = prepare_checkpoint(root, run, policy, groups, window)
    old = copy.deepcopy(manifest)
    old["schema_version"] = 1
    del old["artifact_role_files"]
    with pytest.raises(ValueError, match="schema"):
        cp.validate_checkpoint_manifest(old)
