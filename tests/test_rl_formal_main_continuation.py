"""Continuation contracts; CPU fixtures NEVER authorize real Main/GPU PASS."""
import copy
import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl import formal_main as main
from opensearch_vl_repro.rl import formal_main_continuation as handoff
from opensearch_vl_repro.rl import formal_main_coordinator as coordinator
from opensearch_vl_repro.rl import formal_main_retention as retention
from opensearch_vl_repro.rl import formal_main_update as update
from opensearch_vl_repro.rl import formal_policy_update as dataplane
from opensearch_vl_repro.rl.run_state import checkpoint_policy, reconstruct_consumed_ledger
from test_rl_formal_s4_main import (main_run, chain_fixture, merge_fixture, group_fixture,
                                   window_fixture, update_fixture)
from test_rl_formal_contracts import fixture_policy, digest


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def legacy_context(tmp_path, monkeypatch, count=2, version="formal-s4-main400-v4"):
    original = main_run()
    s = copy.deepcopy(original["semantics"])
    s["coordinator_version"] = s["rollout"]["behavior_version"] = version
    if version.endswith("-v4"):
        del s["provider_reliability"]
        del s["search_behavior_version"]
    run = cp.build_training_run_identity("parent-cpu-main", semantics=s,
        prompt_ids=original["prompt_ids"], prompt_sources=original["prompt_sources"])
    output, reports = main.main_paths(tmp_path, run["run_id"])
    cp.initialize_formal_run(output, run, fixture_policy(run), cpu_fixture=True)
    (output / "merges").mkdir()
    reports.mkdir(parents=True)
    parent = SimpleNamespace(root=tmp_path, output=output, reports=reports, run=run)
    with monkeypatch.context() as m:
        m.setattr(main, "VERSION", version)
        chain_fixture(parent, count)
    return parent


def child_context(parent, plan=None, run_id="child-cpu-main"):
    plan = plan or handoff.resolve_parent(parent.root, parent.run["run_id"], cpu_fixture=True)
    s = copy.deepcopy(main_run()["semantics"])
    s["continuation"] = handoff.binding(plan)
    run = cp.build_training_run_identity(run_id, semantics=s,
        prompt_ids=parent.run["prompt_ids"], prompt_sources=parent.run["prompt_sources"])
    output, reports = main.main_paths(parent.root, run_id)
    output.mkdir(parents=True)
    reports.mkdir(parents=True)
    receipt = handoff.materialize(parent.root, output, run, cpu_fixture=True)
    return SimpleNamespace(root=parent.root, run=run, output=output, reports=reports, receipt=receipt)


@pytest.mark.parametrize("count", [2, 7])
def test_dynamic_full_parent_materialization_and_first_child_update(tmp_path, monkeypatch, count):
    parent = legacy_context(tmp_path, monkeypatch, count)
    before = snapshot(parent.output)
    plan = handoff.resolve_parent(tmp_path, parent.run["run_id"], cpu_fixture=True)
    assert plan["selected"]["policy_iteration"] == count
    child = child_context(parent, plan)
    value = main.recover_main(child.output, child.run, cpu_fixture=True)
    policy = value["policy"]
    assert policy["policy_iteration"] == policy["global_optimizer_step"] == count
    assert value["missing_prompts"] == parent.run["prompt_ids"][4 * count:4 * count + 4]
    assert [p for p, e in value["ledger"]["prompts"].items()
            if e["status"] == "consumed_by_verified_checkpoint"] == parent.run["prompt_ids"][:4 * count]
    assert not value["groups"] and not value["checkpoints"]
    original = plan["proof"]["policy"]
    for key in ("adapter_fingerprint", "native_identity", "optimizer_identity", "rng_identity"):
        assert policy[key] == original[key]
    assert policy["checkpoint_identity"] != original["checkpoint_identity"]
    assert policy["parent_checkpoint_identity"] == original["checkpoint_identity"]
    assert child.receipt["parent_evidence"]["selected"] == plan["selected"]
    for p in value["missing_prompts"]:
        for index in range(4):
            assert main.member_seed(child.run, p, index) == main.member_seed(parent.run, p, index)
    window_fixture(child)
    successor = update_fixture(child)
    assert successor["policy_iteration"] == successor["global_optimizer_step"] == count + 1
    assert (child.output / "checkpoints" / f"policy-{count + 1:06d}").is_dir()
    assert not (child.output / "checkpoints/policy-000001").exists()
    assert successor["parent_policy"] == policy
    assert main.recover_main(child.output, child.run, cpu_fixture=True)["policy"] == checkpoint_policy(successor)
    assert snapshot(parent.output) == before


def test_resume_freezes_parent_and_never_resolves_again(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch)
    child = child_context(parent)
    frozen = child.receipt
    with monkeypatch.context() as m:
        m.setattr(main, "VERSION", "formal-s4-main400-v4")
        plan = handoff.resolve_parent(tmp_path, parent.run["run_id"], cpu_fixture=True)
        policy = plan["proof"]["policy"]
        merge = merge_fixture(parent, policy)
        groups = [group_fixture(parent, policy, p, merge) for p in main.expected_window_prompts(parent.run, 2)]
        from test_rl_formal_contracts import prepare_checkpoint
        from opensearch_vl_repro.rl.training_window import build_training_window
        from opensearch_vl_repro.rl.run_state import advance_update_attempt, persist_update_attempt
        retention.retire_merge(parent.output, parent.run, policy, groups, workers_dead=True, cpu_fixture=True)
        window = build_training_window(parent.run, policy, groups, window_id=main.window_id(2))
        staging, c = prepare_checkpoint(parent.output, parent.run, policy, groups, window, kind="main_checkpoint")
        dest = parent.output / "checkpoints/policy-000003"
        cp.publish_directory(staging, dest, "checkpoint.json", c, cpu_fixture=True)
        persist_update_attempt(parent.output, advance_update_attempt(c["update_attempt"], "verified",
            checkpoint=c, checkpoint_directory=dest), cpu_fixture=True)
        retention.compact_history(parent.output, parent.run, plan["checkpoints"] + [c], cpu_fixture=True)
    monkeypatch.setattr(handoff, "resolve_parent", lambda *a, **k: pytest.fail("parent drift resolver called"))
    s = copy.deepcopy(child.run["semantics"])
    s.pop("continuation")
    handoff.attach_binding(SimpleNamespace(run_id=child.run["run_id"], continue_from_run=parent.run["run_id"]),
                           tmp_path, s, cpu_fixture=True)
    assert s == child.run["semantics"]
    assert handoff.materialize(tmp_path, child.output, child.run, cpu_fixture=True) == frozen
    assert main.recover_main(child.output, child.run, cpu_fixture=True)["policy"]["policy_iteration"] == 2
    assert handoff.final_prefix_audit(tmp_path, frozen, child.run, cpu_fixture=True)["windows"] == 2


@pytest.mark.parametrize("case", ["native_corrupt", "manifest_corrupt", "staging", "ambiguous", "unverified_higher"])
def test_parent_latest_rejects_corruption_and_ambiguity(tmp_path, monkeypatch, case):
    parent = legacy_context(tmp_path, monkeypatch)
    latest = parent.output / "checkpoints/policy-000002"
    if case == "native_corrupt":
        c = cp.read_verified_checkpoint(latest)
        (latest / next(iter(c["artifact_role_files"]["native"]))).write_bytes(b"corrupt")
    elif case == "manifest_corrupt":
        p = latest / "checkpoint.json"
        c = json.loads(p.read_text())
        c["verified"] = False
        p.write_text(json.dumps(c))
    elif case == "staging":
        (parent.output / "checkpoints/.update-unpublished").mkdir()
    elif case == "unverified_higher":
        (parent.output / "checkpoints/policy-000003").mkdir()
    else:
        c = cp.read_verified_checkpoint(latest)
        directory = parent.output / "attempts" / c["update_attempt"]["attempt_id"]
        next(directory.glob("*-verified.json")).unlink()  # CPU temporary fixture ONLY
    with pytest.raises((ValueError, FileNotFoundError)):
        handoff.resolve_parent(tmp_path, parent.run["run_id"], cpu_fixture=True)


def test_parent_partial_current_window_excluded(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch)
    plan = handoff.resolve_parent(tmp_path, parent.run["run_id"], cpu_fixture=True)
    policy = plan["proof"]["policy"]
    with monkeypatch.context() as m:
        m.setattr(main, "VERSION", "formal-s4-main400-v4")
        merge = merge_fixture(parent, policy)
        group_fixture(parent, policy, parent.run["prompt_ids"][8], merge)
    before = snapshot(parent.output)
    child = child_context(parent)
    recovered = main.recover_main(child.output, child.run, cpu_fixture=True)
    assert not recovered["groups"] and recovered["missing_prompts"] == child.run["prompt_ids"][8:12]
    assert snapshot(parent.output) == before


@pytest.mark.parametrize("key", ["dataset", "base_model", "source_sft", "optimizer", "ppo", "reward",
                                 "initial_seed", "software_versions", "execution_contract", "layout_config_sha256"])
def test_training_semantic_delta_rejected(tmp_path, monkeypatch, key):
    parent = legacy_context(tmp_path, monkeypatch)
    s = copy.deepcopy(main_run()["semantics"])
    if key == "initial_seed": s[key] += 1
    elif key in {"software_versions", "layout_config_sha256"}: s[key] = {"changed": True} if key == "software_versions" else digest("changed")
    else: s[key]["changed"] = True
    child = cp.build_training_run_identity("child", semantics=s, prompt_ids=parent.run["prompt_ids"],
                                         prompt_sources=parent.run["prompt_sources"])
    with pytest.raises(ValueError): handoff.semantic_delta(parent.run, child)


def test_only_enumerated_source_changes_allowed(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch)
    s = copy.deepcopy(main_run()["semantics"])
    allowed = "src/opensearch_vl_repro/rl/formal_main_continuation.py"
    s["integration_source_hashes"][allowed] = digest("new control plane")
    child = cp.build_training_run_identity("child", semantics=s, prompt_ids=parent.run["prompt_ids"], prompt_sources=parent.run["prompt_sources"])
    assert handoff.semantic_delta(parent.run, child) == [allowed]
    s["integration_source_hashes"]["src/opensearch_vl_repro/rl/reward.py"] = digest("bad")
    child = cp.build_training_run_identity("child", semantics=s, prompt_ids=parent.run["prompt_ids"], prompt_sources=parent.run["prompt_sources"])
    with pytest.raises(ValueError, match="illegal"): handoff.semantic_delta(parent.run, child)


@pytest.mark.parametrize("case", ["receipt", "bootstrap", "identity", "prefix_order"])
def test_child_authority_tamper_fails(tmp_path, monkeypatch, case):
    parent = legacy_context(tmp_path, monkeypatch)
    child = child_context(parent)
    if case == "bootstrap":
        d = handoff.bootstrap_directory(child.output, child.receipt)
        (d / next(iter(child.receipt["parent_evidence"]["selected"]["artifact_role_files"]["optimizer"]))).write_bytes(b"corrupt")
    elif case == "identity":
        cp.durable_json(child.output / "identity/run.json", {}, cpu_fixture=True)
    else:
        p = child.output / "continuation/receipt.json"
        value = json.loads(p.read_text())
        if case == "receipt": value["child_run_id"] = "other"
        else:
            value["parent_evidence"]["proof"]["rows"].reverse()
            value = cp.seal({k: v for k, v in value.items() if k != "continuation_receipt_sha256"}, "continuation_receipt_sha256")
        cp.durable_json(p, value, cpu_fixture=True)
    with pytest.raises((ValueError, KeyError)):
        main.recover_main(child.output, child.run, cpu_fixture=True)


def test_no_capability_no_nonzero_anchor_and_no_arbitrary_init(tmp_path):
    run = main_run()
    initial = fixture_policy(run)
    altered = cp.seal({**{k: v for k, v in initial.items() if k != "effective_policy_fingerprint"},
        "policy_iteration": 2, "global_optimizer_step": 2, "parent_checkpoint_identity": digest("parent"),
        "cumulative_consumed_group_ids": ["cpu-group"]}, "effective_policy_fingerprint")
    with pytest.raises(ValueError): cp.initialize_formal_run(tmp_path / "bad", run, altered, cpu_fixture=True)
    with pytest.raises(ValueError): reconstruct_consumed_ledger(run, altered, [], [])
    with pytest.raises(ValueError): reconstruct_consumed_ledger(run, altered, [], [], inherited_prefix={})
    assert cp.checkpoint_eligibility("main_checkpoint")["eligible_for_main_init"] is False


def test_worker_continuation_plumbing_and_no_optimizer_reset():
    assert main.VERSION == "formal-s4-main400-v6" and handoff.VERSION == "formal-main-continuation-v1"
    load = inspect.getsource(update.MainUpdateSession.load)
    assert "bootstrap_reload_capability" in load and "continuation_capability=capability" in load
    assert "super().load(policy)" in load
    native = inspect.getsource(dataplane.load_formal_actor)
    assert "require_same_training_run(run, value" in native
    assert "authorize_actor_reload" in native and "manager.load_checkpoint" in native
    assert "_snapshot(actor, manager, rank, policy[\"global_optimizer_step\"])" in native


@pytest.mark.parametrize("phase", ["copy", "receipt", "anchor", "after_anchor"])
def test_bootstrap_crash_never_enters_collection_and_receipt_resume_is_frozen(tmp_path, monkeypatch, phase):
    parent = legacy_context(tmp_path, monkeypatch)
    before = snapshot(parent.output)
    plan = handoff.resolve_parent(tmp_path, parent.run["run_id"], cpu_fixture=True)
    s = copy.deepcopy(main_run()["semantics"])
    s["continuation"] = handoff.binding(plan)
    run = cp.build_training_run_identity("crash-child", semantics=s, prompt_ids=parent.run["prompt_ids"],
                                         prompt_sources=parent.run["prompt_sources"])
    output, _ = main.main_paths(tmp_path, run["run_id"])
    output.mkdir(parents=True)
    original = cp.publish_directory
    with monkeypatch.context() as m:
        if phase == "copy":
            m.setattr(handoff.shutil, "copyfile", lambda *a, **k: (_ for _ in ()).throw(OSError("copy crash")))
        else:
            def publish(staging, dest, *a, **kw):
                if dest.name == ("identity" if phase == "anchor" else "continuation"):
                    if phase != "after_anchor":
                        raise OSError("publication crash")
                result = original(staging, dest, *a, **kw)
                if phase == "after_anchor" and dest.name == "identity":
                    raise OSError("crash after anchor rename")
                return result
            m.setattr(cp, "publish_directory", publish)
        with pytest.raises(OSError): handoff.materialize(tmp_path, output, run, cpu_fixture=True)
    assert (output / "identity").exists() is (phase == "after_anchor")
    assert not (output / "manifest.json").exists()
    with pytest.raises((FileNotFoundError, ValueError)):
        main.recover_main(output, run, cpu_fixture=True)
    if phase in {"anchor", "after_anchor"}:
        monkeypatch.setattr(handoff, "_resolve_locked", lambda *a, **k: pytest.fail("receipt already froze parent"))
    handoff.materialize(tmp_path, output, run, cpu_fixture=True)
    assert main.recover_main(output, run, cpu_fixture=True)["policy"]["policy_iteration"] == 2
    assert snapshot(parent.output) == before


def test_disk_guard_before_any_copy(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch)
    plan = handoff.resolve_parent(tmp_path, parent.run["run_id"], cpu_fixture=True)
    s = copy.deepcopy(main_run()["semantics"])
    s["continuation"] = handoff.binding(plan)
    run = cp.build_training_run_identity("disk-child", semantics=s, prompt_ids=parent.run["prompt_ids"], prompt_sources=parent.run["prompt_sources"])
    output, _ = main.main_paths(tmp_path, run["run_id"])
    output.mkdir(parents=True)
    copied = []
    monkeypatch.setattr(handoff.shutil, "copyfile", lambda *a, **k: copied.append(a))
    needed = []
    def guard(size):
        needed.append(size)
        raise OSError("headroom")
    with pytest.raises(OSError): handoff.materialize(tmp_path, output, run, cpu_fixture=True, guard=guard)
    assert needed[0] > 0 and not copied and not list(output.glob(".continuation-*"))


@pytest.mark.parametrize("mutation", ["duplicate", "skip", "order"])
def test_prefix_proof_rejects_splicing(tmp_path, monkeypatch, mutation):
    parent = legacy_context(tmp_path, monkeypatch)
    plan = handoff.resolve_parent(tmp_path, parent.run["run_id"], cpu_fixture=True)
    checkpoints = plan["checkpoints"]
    values = {"duplicate": checkpoints + checkpoints[-1:], "skip": checkpoints[1:],
              "order": list(reversed(checkpoints))}[mutation]
    with pytest.raises(ValueError): handoff.prefix_proof(parent.run, plan["parent_anchor"], values)


def test_cli_conflicts_and_worker_flag_propagation(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch)
    child = child_context(parent)
    with pytest.raises(ValueError, match="SAME"):
        handoff.attach_binding(SimpleNamespace(run_id=child.run["run_id"], continue_from_run="other"),
                               tmp_path, copy.deepcopy(main_run()["semantics"]), cpu_fixture=True)
    with pytest.raises(ValueError, match="existing/unrelated/self"):
        handoff.attach_binding(SimpleNamespace(run_id=parent.run["run_id"], continue_from_run=parent.run["run_id"]),
                               tmp_path, copy.deepcopy(main_run()["semantics"]), cpu_fixture=True)
    from opensearch_vl_repro.rl.formal_main_cli import build_parser
    args = build_parser(tmp_path).parse_args(["--run-id", "new-child", "--continue-from-run", parent.run["run_id"],
        "--source-root", "/source", "--source-parquet", "/source/rl.parquet", "--base-model-path", "/base",
        "--eval-overlap-manifest", "/eval", "--sft-overlap-manifest", "/sft", "--stop-after-window", "25"])
    assert args.stop_after_window == 25 and args.continue_from_run == parent.run["run_id"]
    command = coordinator.worker_command(args, tmp_path, "update")
    assert command[command.index("--continue-from-run") + 1] == parent.run["run_id"]
    assert "--continue-from-policy" not in command


def test_coordinator_absolute_stop_and_resume(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch, 7)
    child = child_context(parent)
    args = SimpleNamespace(run_id=child.run["run_id"], continue_from_run=parent.run["run_id"],
        config=tmp_path / "configs/rl_main.yaml", data=tmp_path / "data/rl/main400.json", source_root=tmp_path / "source",
        source_parquet=tmp_path / "source/rl.parquet", base_model_path=tmp_path / "base", sft_adapter=tmp_path / main.SFT_ADAPTER,
        eval_overlap_manifest=tmp_path / "eval", sft_overlap_manifest=tmp_path / "sft", judge_config=tmp_path / "judge",
        search_config=tmp_path / "search", layout_config=tmp_path / "layout", tool_cache_dir=None, reward_cache_dir=None,
        collection_gpus="0,1,2,3", collection_parallelism=4, update_gpus="0,1,2,3", stop_after_window=8,
        max_run_gib=250., min_free_gib=0., merge_headroom_gib=0., update_headroom_gib=0.)
    phases = []
    def single(command, **kw):
        value = main.recover_main(child.output, child.run, cpu_fixture=True)
        if "prepare_rl_formal_main_merge.py" in str(command):
            phases.append("merge")
            merge_fixture(child, value["policy"])
        else:
            phases.append("update")
            update_fixture(child)
        return 0
    def parallel(jobs, **kw):
        value = main.recover_main(child.output, child.run, cpu_fixture=True)
        merge = merge_fixture(child, value["policy"])
        for job in jobs:
            command = job["command"]
            group_fixture(child, value["policy"], command[command.index("--prompt-id") + 1], merge)
        return [0] * len(jobs)
    report = coordinator.orchestrate(args, tmp_path, dict(run=child.run), single=single, parallel=parallel, cpu_fixture=True)
    assert phases == ["merge", "update"]
    assert report["progress"]["policy"]["global_optimizer_step"] == 8
    assert report["inherited_windows"] == 7 and report["local_windows"] == 1
    assert report["continuation"] and report["passed"] is False
    phases.clear()
    coordinator.orchestrate(args, tmp_path, dict(run=child.run), single=single, parallel=parallel, cpu_fixture=True)
    assert not phases


def test_activation_required_and_cpu_scope_cannot_authorize_real_reload(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch)
    child = child_context(parent)
    with pytest.raises(ValueError, match="scope"):
        handoff.bootstrap_reload_capability(child.output, child.run, child.receipt["inherited_policy"])
    with pytest.raises(ValueError, match="capability"):
        handoff.authorize_actor_reload({}, child.run, child.receipt["inherited_policy"], "/bad", cpu_fixture=True)
    source = inspect.getsource(update.MainUpdateSession.bootstrap)
    assert "require_reload_capability(self.loaded.reload_receipt" in source
    assert "publish_live_reload" in source
    assert "self.loaded = None" in source


def test_parent_lock_conflict_is_read_only(tmp_path, monkeypatch):
    from opensearch_vl_repro.rl.group import run_lock
    parent = legacy_context(tmp_path, monkeypatch)
    lock = parent.output / ".coordinator.lock"
    with run_lock(lock):  # initialize temporary fixture only, NOT handoff
        pass
    before = snapshot(parent.output)
    with run_lock(lock):
        with pytest.raises(OSError): handoff.resolve_parent(tmp_path, parent.run["run_id"], cpu_fixture=True)
    # Windows byte-range locks forbid reading the locked byte until release.
    assert snapshot(parent.output) == before


def test_v5_parent_is_supported_but_not_gate_smoke_or_cpu_promoted_to_runtime(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch, version="formal-s4-main400-v5")
    child = child_context(parent)
    assert handoff.semantic_delta(parent.run, child.run) == []
    assert child.receipt["parent_evidence"]["parent_run"]["semantics"]["search_behavior_version"] == 3
    with pytest.raises(ValueError, match="CPU fixture anchor"):
        main.recover_main(parent.output, parent.run, _historical=True, reconcile=False, cleanup=False)


def test_routine_child_recovery_never_reads_parent_artifact_bytes(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch)
    child = child_context(parent)
    original = handoff.sha256_file
    def guarded(path):
        assert not Path(path).resolve().is_relative_to(parent.output)
        return original(path)
    monkeypatch.setattr(handoff, "sha256_file", guarded)
    monkeypatch.setattr(handoff, "resolve_parent", lambda *a, **k: pytest.fail("parent reopened"))
    main.recover_main(child.output, child.run, cpu_fixture=True)


def test_final_prefix_rejects_parent_checkpoint_and_group_tamper(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch)
    child = child_context(parent)
    directory = parent.output / "checkpoints/policy-000002"
    c = cp.read_verified_checkpoint(directory)
    p = directory / next(iter(c["artifact_role_files"]["native"]))
    original = p.read_bytes()
    p.write_bytes(b"tampered parent native")
    with pytest.raises(ValueError): handoff.final_prefix_audit(tmp_path, child.receipt, child.run, cpu_fixture=True)
    p.write_bytes(original)
    g = c["groups"][0]
    (parent.output / "groups" / g["identity"]["trajectory_group_id"] / g["members"][0]["steps"][0]["multimodal_file"]).write_bytes(b"tampered")
    with pytest.raises(ValueError): handoff.final_prefix_audit(tmp_path, child.receipt, child.run, cpu_fixture=True)


def test_milestone_crossing_and_suffix_compaction_absolute_steps(tmp_path, monkeypatch):
    parent = legacy_context(tmp_path, monkeypatch, 24)
    before = snapshot(parent.output)
    child = child_context(parent)
    for expected in (25, 26, 27):
        window_fixture(child)
        c = update_fixture(child)
        assert c["policy_iteration"] == expected
        recovered = main.recover_main(child.output, child.run, cpu_fixture=True)
        retention.compact_history(child.output, child.run, recovered["checkpoints"], cpu_fixture=True)
    cp.read_verified_checkpoint(child.output / "checkpoints/policy-000025")
    cp.read_verified_checkpoint(child.output / "checkpoints/policy-000027")
    assert retention.receipt_path(child.output, "compact", 26).exists()
    assert not retention.receipt_path(child.output, "compact", 25).exists()
    assert snapshot(parent.output) == before


def test_combined_final_policy100_prefix_suffix_and_parent_milestones(tmp_path, monkeypatch):
    # A real immutable 99-window fixture prefix, not a fabricated counter/summary.
    parent = legacy_context(tmp_path, monkeypatch, 99)
    child = child_context(parent)
    assert handoff.final_prefix_audit(tmp_path, child.receipt, child.run, cpu_fixture=True)["groups"] == 396
    window_fixture(child)
    c = update_fixture(child)
    assert c["policy_iteration"] == c["global_optimizer_step"] == 100
    final = main.final_reconstruction(child.output, child.run, cpu_fixture=True)
    assert final["passed"] is False and final["continuation"] is True
    assert final["inherited_windows"] == 99 and final["inherited_prompts"] == 396 and final["local_windows"] == 1
    assert final["groups_completed"] == 400 and final["trajectories_completed"] == 1600
    assert final["optimizer_steps"] == list(range(1, 101))
    assert final["final_policy"]["policy_iteration"] == 100
    assert [p.name for p in (child.output / "checkpoints").iterdir()] == ["policy-000100"]
    for step in (25, 50, 75):
        cp.read_verified_checkpoint(parent.output / "checkpoints" / f"policy-{step:06d}")


@pytest.mark.parametrize("reset", [None, "model", "optimizer", "rng"])
def test_actual_loader_cross_run_capability_restores_all_states_not_fresh_init(tmp_path, monkeypatch, reset):
    """Execute real loader/authority guards with inert CPU state/shard mocks, no Qwen."""
    import test_rl_formal_s4_main as fixtures
    from opensearch_vl_repro.rl import actor_gate
    original_prepare = fixtures.prepare_checkpoint
    def native_prepare(*a, **kw):
        staging, c = original_prepare(*a, **kw)
        step = c["global_optimizer_step"]
        for role in ("native", "optimizer", "rng"):
            for name in c["artifact_role_files"][role]: (staging / name).unlink()
        distributed = staging / "distributed"
        distributed.mkdir()
        for rank in range(4):
            values = dict(parameter_sha256=digest(["model", step]), optimizer_state_sha256=digest(["AdamW", step]),
                          native_rng_sha256=digest(["RNG", step]), global_optimizer_step=step)
            for prefix, key in (("model", "parameter_sha256"), ("optim", "optimizer_state_sha256"), ("extra_state", "native_rng_sha256")):
                (distributed / f"{prefix}_world_size_4_rank_{rank}.pt").write_text(json.dumps({key: values[key]}))
            state = cp.seal(dict(rank=rank, execution_contract=c["execution_contract"],
                window_sha256=c["window"]["window_sha256"], **values), "runtime_state_sha256")
            cp.durable_json(staging / f"runtime_state_rank_{rank}.json", state, cpu_fixture=True)
        roles, inventory = dataplane.checkpoint_staging_roles(staging, 4)
        role_ids = cp.artifact_role_identities(roles, inventory)
        evidence = {**c["reload_evidence"], "reloaded_artifact_roles": role_ids}
        c = cp.build_checkpoint_manifest(c["run"], c["parent_policy"], c["groups"], c["window"],
            c["update_attempt"], c["reward_window"], artifact_role_files=roles, file_sha256=inventory,
            kind="main_checkpoint", reload_evidence=evidence, cpu_fixture=True)
        return staging, c
    monkeypatch.setattr(fixtures, "prepare_checkpoint", native_prepare)
    parent = legacy_context(tmp_path, monkeypatch)
    child = child_context(parent)
    policy = child.receipt["inherited_policy"]
    directory = handoff.bootstrap_directory(child.output, child.receipt)
    capability = handoff.bootstrap_reload_capability(child.output, child.run, policy, cpu_fixture=True)
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "fixture.json").write_bytes(b"CPU-only snapshot")
    monkeypatch.setattr(dataplane, "validated_runtime_snapshot", lambda **kw: (snapshot_dir, {"fixture.json": handoff.sha256_file(snapshot_dir / "fixture.json")}))
    monkeypatch.setattr(dataplane, "_rank", lambda *a: 0)
    monkeypatch.setattr(dataplane, "_boundary", lambda op, *a: op())
    monkeypatch.setattr(dataplane, "rng_fingerprint", lambda: digest("CPU RNG fixture"))
    monkeypatch.setattr(dataplane, "require_saved_source_dropout", lambda *a: None)
    monkeypatch.setattr(dataplane, "configure_rl_lora_dropout_runtime", lambda *a: None)
    monkeypatch.setattr(dataplane, "require_rl_lora_dropout_runtime", lambda *a: {"lora_dropout_runtime_sha256": digest("dropout")})
    monkeypatch.setattr(actor_gate, "lora_snapshot", lambda *a: {"exported_lora": 2})
    monkeypatch.setattr(actor_gate, "reload_matches", lambda a, b: a == b)
    calls = []
    def construct(**kw):
        assert kw["adapter"] == directory / "adapter"
        return SimpleNamespace(actor_module=object(), restored={}, actor_optimizer=SimpleNamespace(
            param_groups=[{"lr": 1e-6, "weight_decay": 0.}])), {}
    class Manager:
        def __init__(self, actor, processor): self.actor = actor
        def load_checkpoint(self, path, del_local_after_load=False):
            assert del_local_after_load is False and Path(path) == directory / "distributed"
            calls.append("native/AdamW/RNG load")
            for prefix, kind in (("model", "model"), ("optim", "optimizer"), ("extra_state", "rng")):
                if reset != kind:
                    self.actor.restored.update(json.loads((Path(path) / f"{prefix}_world_size_4_rank_0.pt").read_text()))
    def actual_snapshot(actor, manager, rank, global_step):
        return dict(rank=rank, global_optimizer_step=global_step,
            **{key: actor.restored.get(key, "RESET/EMPTY") for key in (
                "parameter_sha256", "optimizer_state_sha256", "native_rng_sha256")})
    monkeypatch.setattr(dataplane, "_snapshot", actual_snapshot)
    options = dict(canonical_config={}, runtime_config={}, source_adapter=tmp_path / "unused-sft", processor=None,
        mesh=None, policy=policy, checkpoint_directory=directory, cpu_fixture=True, construct=construct, manager_factory=Manager)
    with pytest.raises(ValueError, match="resume semantics differ"):
        dataplane.load_formal_actor(child.run, **options)
    assert not calls
    if reset:
        with pytest.raises(ValueError, match="actual native model/adapter/AdamW/RNG reload mismatch"):
            dataplane.load_formal_actor(child.run, **options, continuation_capability=capability)
    else:
        loaded = dataplane.load_formal_actor(child.run, **options, continuation_capability=capability)
        assert loaded.policy == policy and loaded.actor._formal_global_optimizer_step == 2
        receipt = dataplane.require_reload_capability(loaded.reload_receipt, policy)
        assert receipt["scope"] == "cpu_fixture" and receipt["global_optimizer_step"] == 2
        assert receipt["policy"]["optimizer_identity"] == child.receipt["parent_evidence"]["proof"]["policy"]["optimizer_identity"]
    assert calls == ["native/AdamW/RNG load"]
