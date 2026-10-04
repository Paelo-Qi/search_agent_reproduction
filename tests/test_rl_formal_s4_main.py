"""CPU-only S4 receipts/control-plane fixtures; NEVER Main GPU/API PASS evidence."""
import ast
import copy
import inspect
import io
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl import formal_main as main
from opensearch_vl_repro.rl import formal_main_retention as retention
from opensearch_vl_repro.rl import formal_main_coordinator as coordinator
from opensearch_vl_repro.rl import formal_main_collection as collection
from opensearch_vl_repro.rl import formal_main_update as update
from opensearch_vl_repro.rl import formal_main_process as process
from opensearch_vl_repro.rl import formal_smoke as smoke
from opensearch_vl_repro.rl.formal_main_cli import build_parser
from opensearch_vl_repro.rl.group import read_formal_group, read_formal_group_manifest_only, validate_formal_group
from opensearch_vl_repro.rl.run_state import advance_update_attempt, persist_update_attempt, checkpoint_policy
from opensearch_vl_repro.rl.training_window import expected_window_prompts, build_training_window, deterministic_rank_plan
from opensearch_vl_repro.rl.actor_gate import BASE_MODEL, BASE_REVISION
from opensearch_vl_repro.rl.rl_actor_semantics import execution_contract, LORA_TARGETS
from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from test_rl_formal_contracts import fixture_run, fixture_policy, draft_group, commit_group, prepare_checkpoint, digest


def main_run():
    original = fixture_run(n=4, world_size=4, prompt_count=400)
    s = copy.deepcopy(original["semantics"])
    s.update(coordinator_version=main.VERSION, initial_seed=20260506,
        retention=copy.deepcopy(retention.CONTRACT), rollout_seed_scheme=main.SEED_SCHEME,
        optimizer=dict(name="AdamW", learning_rate=1e-6, weight_decay=0.),
        ppo=dict(epochs=1, microbatch=1, clip_ratio_low=.2, clip_ratio_high=.28, entropy=0., loss_mode="vanilla"),
        tool_protocol_version=RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION,
        execution_contract=execution_contract(dict(lora=dict(rank=16, alpha=32, dropout=.05, target_modules=LORA_TARGETS))))
    s["dataset"]["split"] = "main"
    s["source_sft"]["stage"] = "main_b_2k"
    s["base_model"] = dict(name=BASE_MODEL, revision=BASE_REVISION, offline_snapshot_sha256=digest("snapshot"))
    s["rollout"] = dict(behavior_version=main.VERSION, config=copy.deepcopy(main.ROLLOUT))
    return cp.build_training_run_identity("main-cpu-fixture", semantics=s,
        prompt_ids=original["prompt_ids"], prompt_sources=original["prompt_sources"])


@pytest.fixture
def ctx(tmp_path):
    run = main_run()
    output, reports = main.main_paths(tmp_path, run["run_id"])
    cp.initialize_formal_run(output, run, fixture_policy(run), cpu_fixture=True)
    (output / "merges").mkdir()
    reports.mkdir(parents=True)
    args = SimpleNamespace(run_id=run["run_id"], config=tmp_path / "configs/rl_main.yaml",
        data=tmp_path / "data/rl/main400.json", source_root=tmp_path / "source", source_parquet=tmp_path / "source/rl.parquet",
        base_model_path=tmp_path / "base", sft_adapter=tmp_path / main.SFT_ADAPTER,
        eval_overlap_manifest=tmp_path / "eval.json", sft_overlap_manifest=tmp_path / "sft.json",
        judge_config=tmp_path / "judge.yaml", search_config=tmp_path / "search.yaml", layout_config=tmp_path / "layout.yaml",
        collection_gpus="0,1,2,3", collection_parallelism=4, update_gpus="0,1,2,3",
        tool_cache_dir=None, reward_cache_dir=None, stop_after_window=None,
        max_run_gib=250., min_free_gib=0., merge_headroom_gib=0., update_headroom_gib=0.)
    return SimpleNamespace(root=tmp_path, output=output, reports=reports, run=run, args=args)


def merge_fixture(ctx, policy):
    directory = collection.merged_path(ctx.output, policy)
    if directory.exists():
        return json.loads((directory / "merge_manifest.json").read_text())["identity"]
    directory.mkdir(parents=True)
    (directory / "weights.fixture").write_bytes(b"CPU ONLY, NOT HF OR GPU EVIDENCE")
    binding = dict(version=main.VERSION, run_identity_sha256=ctx.run["run_identity_sha256"],
        policy_iteration=policy["policy_iteration"], parent_checkpoint_identity=policy["checkpoint_identity"],
        effective_policy_fingerprint=policy["effective_policy_fingerprint"],
        base_snapshot_sha256=ctx.run["semantics"]["base_model"]["offline_snapshot_sha256"])
    ident = collection.formal_merge_identity(actor=dict(formal_binding=binding), versions={}, file_hashes=cp.artifact_inventory(directory))
    cp.durable_json(directory / "merge_manifest.json", dict(identity=ident, merge_complete=True,
        fresh_hf_forward_finite=True, no_active_peft=True, merge_hf_destroyed=True, reload_hf_destroyed=True), cpu_fixture=True)
    return ident


def group_fixture(ctx, policy, prompt, merge):
    draft = draft_group(ctx.run, policy, prompt)
    draft["static_merge"] = merge
    for member in draft["members"]:
        member["member_seed"] = main.member_seed(ctx.run, prompt, member["rollout_index"])
    return commit_group(ctx.output, ctx.run, policy, prompt, draft=draft)


def window_fixture(ctx, policy=None):
    recovered = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    policy = policy or recovered["policy"]
    merge = merge_fixture(ctx, policy)
    for prompt in recovered["missing_prompts"]:
        group_fixture(ctx, policy, prompt, merge)
    return main.recover_main(ctx.output, ctx.run, cpu_fixture=True)


def update_fixture(ctx, *, publication_crash=False):
    value = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    retention.retire_merge(ctx.output, ctx.run, value["policy"], value["current_groups"], workers_dead=True, cpu_fixture=True)
    window = build_training_window(ctx.run, value["policy"], value["current_groups"], window_id=main.window_id(value["policy"]["policy_iteration"]))
    staging, manifest = prepare_checkpoint(ctx.output, ctx.run, value["policy"], value["current_groups"], window, kind="main_checkpoint")
    main.commit_main_checkpoint(ctx.output, staging, manifest, cpu_fixture=True)
    if not publication_crash:
        main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    return manifest


def chain_fixture(ctx, count):
    """Build valid immutable CPU chain without quadratic recovery between fixtures."""
    policy, checkpoints = fixture_policy(ctx.run), []
    for iteration in range(count):
        merge = merge_fixture(ctx, policy)
        groups = [group_fixture(ctx, policy, p, merge) for p in expected_window_prompts(ctx.run, iteration)]
        retention.retire_merge(ctx.output, ctx.run, policy, groups, workers_dead=True, cpu_fixture=True)
        window = build_training_window(ctx.run, policy, groups, window_id=main.window_id(iteration))
        staging, manifest = prepare_checkpoint(ctx.output, ctx.run, policy, groups, window, kind="main_checkpoint")
        directory = ctx.output / "checkpoints" / f"policy-{iteration + 1:06d}"
        cp.publish_directory(staging, directory, "checkpoint.json", manifest, cpu_fixture=True)
        event = advance_update_attempt(manifest["update_attempt"], "verified", checkpoint=manifest, checkpoint_directory=directory)
        persist_update_attempt(ctx.output, event, cpu_fixture=True)
        checkpoints.append(manifest)
        policy = checkpoint_policy(manifest)
    retention.compact_history(ctx.output, ctx.run, checkpoints, cpu_fixture=True)
    return checkpoints


def test_exact_main_identity_and_cli():
    run = main_run()
    main.require_main_run(run)
    assert len(run["prompt_ids"]) == 400
    assert [p for i in range(100) for p in expected_window_prompts(run, i)] == run["prompt_ids"]
    p = build_parser(Path("/fixture"))
    args = p.parse_args(["--run-id", "test", "--source-root", "/source", "--source-parquet", "/source/data.parquet",
        "--base-model-path", "/base", "--eval-overlap-manifest", "/eval", "--sft-overlap-manifest", "/sft", "--stop-after-window", "1"])
    assert args.config.name == "rl_main.yaml" and args.data.name == "main400.json"
    assert args.collection_parallelism == 4 and args.stop_after_window == 1
    assert args.max_run_gib == 250 and args.min_free_gib == 30


@pytest.mark.parametrize("field,value", [("rollout_n", 2), ("world_size", 2), ("groups_per_window", 2),
    ("coordinator_version", smoke.VERSION), ("retention", {}), ("rollout_seed_scheme", "worker order"),
    ("initial_seed", 7), ("diagnostic_version", "Gate/S2"), ("optimizer", dict(name="AdamW", learning_rate=2e-6, weight_decay=0.))])
def test_reject_changed_main_semantics(field, value):
    original = main_run()
    run = cp.build_training_run_identity(original["run_id"], semantics={**original["semantics"], field: value},
        prompt_ids=original["prompt_ids"], prompt_sources=original["prompt_sources"])
    with pytest.raises(ValueError): main.require_main_run(run)


def test_reject_smoke_twenty():
    from test_rl_formal_s3_smoke import smoke_run
    with pytest.raises(ValueError): main.require_main_run(smoke_run())


@pytest.mark.parametrize("adapter", ["outputs/rl_formal_smoke/smoke/checkpoints/policy-000005/adapter",
    "outputs/rl_gate_c/adapter", "outputs/rl_formal_s2_validation/run/adapter",
    "outputs/rl_formal_main/run/checkpoints/.stage/adapter", "outputs/rl_formal_main/foreign/checkpoints/policy-000001/adapter"])
def test_foreign_initialization_rejected_before_data_model(ctx, monkeypatch, adapter):
    # Test real context guard without loading source/weights or installed frameworks.
    ctx.args.sft_adapter = ctx.root / adapter
    monkeypatch.setattr(smoke, "validate_cache_locators", lambda *a: None)
    with pytest.raises(ValueError, match="ORIGINAL SFT"):
        main.prepare_context(ctx.args, ctx.root)


def test_seed_independent_completion_and_gpu_order(ctx):
    ids = ctx.run["prompt_ids"][:4]
    expected = {(p, i): main.member_seed(ctx.run, p, i) for p in ids for i in range(4)}
    actual = {(p, i): main.member_seed(ctx.run, p, i) for p in reversed(ids) for i in reversed(range(4))}
    assert actual == expected
    assert list(expected.values()) == list(range(20260506, 20260522))
    assert coordinator.collection_waves(ids, ["3", "2", "1", "0"], 4) == [list(zip(ids, ["3", "2", "1", "0"]))]


@pytest.mark.parametrize("parallelism", [1, 2, 3, 4])
def test_four_prompts_scheduled_once(parallelism):
    waves = coordinator.collection_waves(["p0", "p1", "p2", "p3"], ["0", "1", "2", "3"], parallelism)
    assert [p for wave in waves for p, _ in wave] == ["p0", "p1", "p2", "p3"]
    assert all(len(wave) <= parallelism for wave in waves)


def test_whole_group_no_splicing(ctx):
    state = window_fixture(ctx)
    assert len(state["current_groups"]) == 4
    for g in state["current_groups"]:
        assert len(g["members"]) == 4
        assert {m["rollout_index"] for m in g["members"]} == set(range(4))
        assert all(m["identity"] == g["identity"] for m in g["members"])
    bad = copy.deepcopy(state["current_groups"][0])
    bad["members"][0] = state["current_groups"][1]["members"][0]
    with pytest.raises(ValueError): validate_formal_group(bad)


def test_committed_group_resume_and_partial_fresh_uuid(ctx, monkeypatch):
    policy = fixture_policy(ctx.run)
    merge = merge_fixture(ctx, policy)
    committed = group_fixture(ctx, policy, "p0", merge)
    state = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    assert state["current_groups"] == [committed] and state["missing_prompts"] == ["p1", "p2", "p3"]
    from opensearch_vl_repro.rl import formal_collection as shared
    monkeypatch.setattr(shared, "source_identity_for_row", lambda run, row: digest(row["prompt_id"]))
    original = cp.durable_json
    monkeypatch.setattr(cp, "durable_json", lambda p, v, **kw: original(p, v, cpu_fixture=True))
    a, one = collection.reserve_collection(ctx.output, ctx.run, policy, dict(prompt_id="p1"))
    (a / "partial.fixture").write_bytes(b"forensic ONLY")
    b, two = collection.reserve_collection(ctx.output, ctx.run, policy, dict(prompt_id="p1"))
    assert a != b and one["collection_attempt"] != two["collection_attempt"]
    assert two["collection_attempt_index"] == 1
    assert main.recover_main(ctx.output, ctx.run, cpu_fixture=True)["current_groups"] == [committed]


@pytest.mark.parametrize("dead,size", [(False, 4), (True, 3), (True, 0)])
def test_retirement_forbidden_without_complete_teardown(ctx, dead, size):
    state = window_fixture(ctx)
    with pytest.raises(ValueError): retention.retire_merge(ctx.output, ctx.run, state["policy"],
        state["current_groups"][:size], workers_dead=dead, cpu_fixture=True)
    assert collection.merged_path(ctx.output, state["policy"]).exists()


def test_one_shared_merge_and_authorized_absence(ctx):
    state = window_fixture(ctx)
    assert len(list((ctx.output / "merges").iterdir())) == 1
    assert len({g["static_merge"]["merged_checkpoint_fingerprint"] for g in state["current_groups"]}) == 1
    receipt = retention.retire_merge(ctx.output, ctx.run, state["policy"], state["current_groups"], workers_dead=True, cpu_fixture=True)
    assert len(receipt["group_ids"]) == 4 and receipt["merged_file_sha256"]
    assert not collection.merged_path(ctx.output, state["policy"]).exists()
    assert main.recover_main(ctx.output, ctx.run, cpu_fixture=True)["missing_prompts"] == []


def test_merge_absent_without_authorization_rejected(ctx):
    state = window_fixture(ctx)
    merge = collection.merged_path(ctx.output, state["policy"])
    merge.rename(merge.parent / ".missing-fixture")
    with pytest.raises(ValueError, match="absent without"): main.recover_main(ctx.output, ctx.run, cpu_fixture=True)


def test_merge_retirement_crash_cleanup(ctx, monkeypatch):
    state = window_fixture(ctx)
    real = retention.delete_authorized
    monkeypatch.setattr(retention, "delete_authorized", lambda *a, **k: (_ for _ in ()).throw(OSError("after authorization")))
    with pytest.raises(OSError): retention.retire_merge(ctx.output, ctx.run, state["policy"], state["current_groups"], workers_dead=True, cpu_fixture=True)
    assert retention.receipt_path(ctx.output, "merge", 0).exists()
    monkeypatch.setattr(retention, "delete_authorized", real)
    main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    assert not collection.merged_path(ctx.output, state["policy"]).exists()


def test_publication_crash_rolls_forward(ctx):
    window_fixture(ctx)
    manifest = update_fixture(ctx, publication_crash=True)
    state = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    assert state["policy"]["policy_iteration"] == 1
    assert state["attempts"][0]["phase"] == "verified"
    assert state["policy"]["checkpoint_identity"] == manifest["checkpoint_manifest_sha256"]


def test_retry_requires_live_native_capability(ctx):
    state = window_fixture(ctx)
    window = build_training_window(ctx.run, state["policy"], state["current_groups"], window_id=main.window_id(0))
    from opensearch_vl_repro.rl.run_state import new_update_attempt, transition_trainer_state
    attempt = new_update_attempt(window)
    persist_update_attempt(ctx.output, attempt, cpu_fixture=True)
    for phase in ("started", "step_may_have_run"):
        attempt = advance_update_attempt(attempt, phase)
        persist_update_attempt(ctx.output, attempt, cpu_fixture=True)
    state = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    transaction = update.prepare_update_transaction(ctx.output, ctx.run, state, window, cpu_fixture=True)
    assert transaction["attempt"]["attempt_id"] != attempt["attempt_id"]
    assert transaction["state"]["recovery_reload_required"] is True
    assert transaction["retry_plan"]["reload_optimizer_identity"] == state["policy"]["optimizer_identity"]
    with pytest.raises(ValueError, match="reload verification"):
        transition_trainer_state(transaction["state"], "updating")


@pytest.mark.parametrize("step", [1, 25, 50, 75, 100])
def test_current_and_milestone_cannot_compact(step):
    # Pure policy protection exercised before physical authorization/deletion.
    c = dict(policy_iteration=step)
    with pytest.raises(ValueError, match="protected"):
        retention.compaction_value(main_run(), c, {}, step if step == 1 else 101)


def test_compaction_requires_verified_successor(ctx):
    window_fixture(ctx)
    first = update_fixture(ctx)
    window_fixture(ctx)
    second = update_fixture(ctx, publication_crash=True)
    with pytest.raises(ValueError, match="verified successor attempt"):
        retention.compact_history(ctx.output, ctx.run, [first, second], cpu_fixture=True)
    assert not retention.receipt_path(ctx.output, "compact", 1).exists()


def test_role_driven_compaction_retains_adapter_and_exact_inventory(ctx):
    checkpoints = chain_fixture(ctx, 2)
    receipt = retention.read_receipt(retention.receipt_path(ctx.output, "compact", 1))
    assert set(receipt["removable_role_files"]) == {"native", "optimizer", "rng"}
    assert receipt["original_file_sha256"] == checkpoints[0]["file_sha256"]
    directory = ctx.output / "checkpoints/policy-000001"
    assert retention.physical_files(directory, exclude=("checkpoint.json",)) == set(checkpoints[0]["artifact_role_files"]["adapter"])
    cp.read_verified_checkpoint(ctx.output / "checkpoints/policy-000002")
    with pytest.raises(ValueError): cp.read_verified_checkpoint(directory)  # old FULL API never weakened


def test_partial_compaction_crash_is_recoverable(ctx, monkeypatch):
    window_fixture(ctx)
    first = update_fixture(ctx)
    window_fixture(ctx)
    second = update_fixture(ctx)
    real = retention.delete_authorized
    def partial(directory, files, **kw):
        (directory / next(iter(files))).unlink()
        raise OSError("partial authorized deletion")
    monkeypatch.setattr(retention, "delete_authorized", partial)
    with pytest.raises(OSError): retention.compact_history(ctx.output, ctx.run, [first, second], cpu_fixture=True)
    assert retention.receipt_path(ctx.output, "compact", 1).exists()
    monkeypatch.setattr(retention, "delete_authorized", real)
    state = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    assert state["policy"]["policy_iteration"] == 2
    assert retention.physical_files(ctx.output / "checkpoints/policy-000001", exclude=("checkpoint.json",)) == set(first["artifact_role_files"]["adapter"])


@pytest.mark.parametrize("role", ["adapter", "native"])
def test_unauthorized_missing_history_fails(ctx, role):
    window_fixture(ctx)
    first = update_fixture(ctx)
    window_fixture(ctx)
    update_fixture(ctx)
    filename = next(iter(first["artifact_role_files"][role]))
    (ctx.output / "checkpoints/policy-000001" / filename).rename(ctx.output / "lost.fixture")
    with pytest.raises(ValueError, match="missing historical"): main.recover_main(ctx.output, ctx.run, cpu_fixture=True)


def test_routine_sha_call_boundary_current_only(ctx, monkeypatch):
    chain_fixture(ctx, 3)
    window_fixture(ctx)
    import opensearch_vl_repro.sft_tool_audit as sha
    real, calls = sha.sha256_file, []
    monkeypatch.setattr(sha, "sha256_file", lambda p: calls.append(Path(p)) or real(p))
    result = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    assert result["policy"]["policy_iteration"] == 3
    assert any("policy-000003" in p.parts and p.suffix == ".fixture" for p in calls)
    assert not any(any(part in {"policy-000001", "policy-000002"} for part in p.parts) for p in calls)
    historical = {g["identity"]["trajectory_group_id"] for g in result["groups"] if g["identity"]["policy_iteration"] < 3}
    assert not any(any(gid in p.parts for gid in historical) for p in calls)
    current = {g["identity"]["trajectory_group_id"] for g in result["current_groups"]}
    assert all(any(gid in p.parts and p.name.startswith("mm-") for p in calls) for gid in current)


@pytest.mark.parametrize("kind", ["current_parent", "current_group"])
def test_current_corruption_fails(ctx, kind):
    chain_fixture(ctx, 2)
    state = window_fixture(ctx)
    if kind == "current_parent":
        path = ctx.output / "checkpoints/policy-000002" / next(iter(state["checkpoints"][-1]["artifact_role_files"]["native"]))
    else:
        g = state["current_groups"][0]
        path = ctx.output / "groups" / g["identity"]["trajectory_group_id"] / g["members"][0]["steps"][0]["multimodal_file"]
    path.write_bytes(b"CORRUPTED CPU FIXTURE")
    with pytest.raises(ValueError, match="inventory"): main.recover_main(ctx.output, ctx.run, cpu_fixture=True)


def test_metadata_readers_explicit_and_full_defaults_unchanged(ctx, monkeypatch):
    chain_fixture(ctx, 1)
    directory = ctx.output / "checkpoints/policy-000001"
    c = cp.read_checkpoint_manifest_only(directory)
    gdir = ctx.output / "groups" / c["consumed_group_ids"][0]
    monkeypatch.setattr(cp, "verify_artifacts", lambda *a, **k: pytest.fail("metadata reader must not hash bytes"))
    assert cp.read_checkpoint_manifest_only(directory) == c
    assert read_formal_group_manifest_only(gdir) == c["groups"][0]
    with pytest.raises(pytest.fail.Exception): cp.read_verified_checkpoint(directory)
    with pytest.raises(pytest.fail.Exception): read_formal_group(gdir)


def test_w4_plan_and_update_session_specialization():
    plan = deterministic_rank_plan([f"row-{i}" for i in range(16)], 4)
    assert plan["replication_factor"] == 1 and plan["local_row_count"] == 4
    assert len(plan["rank_assignments"]) == 4
    assert update.MainUpdateSession.update is update.UpdateSession.update
    assert update.MainUpdateSession.world_size == 4 and update.UpdateSession.world_size == 2
    assert update.MainUpdateSession.window_count == 100 and update.UpdateSession.window_count == 5
    assert update.MainUpdateSession.checkpoint_kind(None, 100) == "main_checkpoint"
    assert update.require_launcher(dict(WORLD_SIZE="4", RANK="3", LOCAL_RANK="3")) == (3, 3)


@pytest.mark.parametrize("world", [1, 2, 3, 5])
def test_reject_not_w4_launcher(world):
    with pytest.raises(ValueError): update.require_launcher(dict(WORLD_SIZE=str(world), RANK="0", LOCAL_RANK="0"))


@pytest.mark.parametrize("case", ["missing_rank", "two_steps", "unchanged_lora", "mismatched_lora", "bad_OC", "bad_reload"])
def test_w4_rank_evidence_fail_closed(tmp_path, case):
    from test_rl_formal_s2_validation import oc_evidence
    from opensearch_vl_repro.rl.policy_alignment import formal_alignment_artifact
    rows = oc_evidence(0)["per_rank"]
    rows += [{**rows[0], "rank": i} for i in (2, 3)]
    alignment = formal_alignment_artifact(rows, world_size=4, window_sha256="0" * 64)
    c = dict(global_optimizer_step=1, update_attempt={}, window=dict(window_sha256="0" * 64), parent_policy={})
    plan = deterministic_rank_plan(["a", "b", "c", "d"], 4)
    for rank in range(4):
        saved = dict(rank=rank, parameter_sha256="a" * 64)
        (tmp_path / f"runtime_state_rank_{rank}.json").write_text(json.dumps(saved))
        evidence = dict(scope="runtime", rank=rank, update=dict(before_step=0, after_step=1,
            update_audit=dict(optimizer_step_count=1), attempt={}, window_sha256="0" * 64, alignment=alignment,
            metrics={"actor/pg_loss": [.01]}), loaded_parent=dict(policy={}), actor_contract=dict(passed=True),
            full_lora_sha256="b" * 64, previous_full_lora_sha256="a" * 64, deterministic_plan=plan,
            fresh_reload=dict(original_actor_destroyed=True, fresh_multimodal_forward_finite=True,
                adapter_reloaded=True, native_reloaded=True, optimizer_reloaded=True, rng_reloaded=True,
                execution_contract_verified=True, per_rank=[dict(rank=rank, state=saved)]))
        if rank == 3:
            if case == "missing_rank": continue
            if case == "two_steps": evidence["update"]["update_audit"]["optimizer_step_count"] = 2
            if case == "unchanged_lora": evidence["full_lora_sha256"] = evidence["previous_full_lora_sha256"]
            if case == "mismatched_lora": evidence["full_lora_sha256"] = "c" * 64
            if case == "bad_OC": evidence["update"]["alignment"] = {**alignment, "passed": False}
            if case == "bad_reload": evidence["fresh_reload"]["native_reloaded"] = False
        (tmp_path / f"formal_update_rank_{rank}.json").write_text(json.dumps(evidence))
    with pytest.raises((ValueError, FileNotFoundError)): smoke.require_runtime_update_evidence(tmp_path, c, world_size=4)


@pytest.mark.parametrize("case", ["valid", "missing_metadata", "missing_native_rank", "wrong_parent_world",
                                  "wrong_rank", "bad_decoder", "wrong_dropout", "missing_reload_rank", "changed_metadata"])
def test_main_immutable_four_rank_evidence_binding(tmp_path, monkeypatch, case):
    # The shared S2/S3 semantic checks have their own tests; isolate S4 binding here.
    calls = []
    monkeypatch.setattr(smoke, "require_runtime_update_evidence", lambda *a, **k: calls.append(k["world_size"]))
    roles = {"metadata": {}}
    for prefix, role in (("model", "native"), ("optim", "optimizer"), ("extra_state", "rng")):
        roles[role] = {f"distributed/{prefix}_world_size_4_rank_{r}.pt": "0" * 64 for r in range(4)}
    for rank in range(4):
        actor = dict(fsdp2=True, bf16=True, model_training=True, language_model_training=True,
            decoder_layers=36, decoder_layers_training=36, decoder_layers_gradient_checkpointing=36,
            effective_attention_implementation="flash_attention_2",
            dropout=dict(source_adapter_lora_dropout=.05, runtime_effective_lora_dropout=0.))
        row = dict(actor_contract=actor, loaded_parent=dict(world_size=4, rank=rank),
                   rank_plan=dict(world_size=4, rank=rank, window_sha256="1" * 64),
                   fresh_reload=dict(per_rank=[dict(rank=i) for i in range(4)]))
        if rank == 3:
            if case == "wrong_parent_world": row["loaded_parent"]["world_size"] = 2
            if case == "wrong_rank": row["rank_plan"]["rank"] = 0
            if case == "bad_decoder": actor["decoder_layers_training"] = 35
            if case == "wrong_dropout": actor["dropout"]["runtime_effective_lora_dropout"] = .05
            if case == "missing_reload_rank": row["fresh_reload"]["per_rank"].pop()
        (tmp_path / f"formal_update_rank_{rank}.json").write_text(json.dumps(row))
        (tmp_path / f"runtime_state_rank_{rank}.json").write_text("{}")
    roles["metadata"] = cp.artifact_inventory(tmp_path)
    files = {n: sha for role in roles.values() for n, sha in role.items()}
    checkpoint = dict(artifact_role_files=roles, file_sha256=files, window=dict(window_sha256="1" * 64))
    if case == "missing_metadata": roles["metadata"].pop("formal_update_rank_3.json")
    if case == "missing_native_rank": roles["native"].pop("distributed/model_world_size_4_rank_3.pt")
    if case == "changed_metadata": (tmp_path / "runtime_state_rank_3.json").write_text('{"tamper": true}')
    if case == "valid": main.require_main_update_evidence(tmp_path, checkpoint)
    else:
        with pytest.raises(ValueError): main.require_main_update_evidence(tmp_path, checkpoint)
    assert calls == [4]


def test_disk_accounting_and_limit_floor(ctx):
    chain_fixture(ctx, 2)
    state = window_fixture(ctx)
    disk = retention.disk_accounting(ctx.output, previous_peak=999999)
    assert disk["compacted_checkpoints"] == ["policy-000001"]
    assert disk["retained_full_checkpoints"] == ["policy-000002"]
    assert disk["groups_bytes"] > 0 and disk["active_merge_bytes"] > 0
    assert disk["peak_observed_run_bytes"] >= disk["current_run_bytes"]
    with pytest.raises(OSError): retention.disk_guard(disk, needed_bytes=1, max_run_bytes=disk["current_run_bytes"])
    with pytest.raises(OSError): retention.disk_guard(disk, needed_bytes=1, min_free_bytes=disk["free_filesystem_bytes"])
    retention.disk_guard(disk, needed_bytes=0, min_free_bytes=0)
    (ctx.reports / "worker.log").write_bytes(b"audit log")
    with_reports = retention.disk_accounting(ctx.output, report_root=ctx.reports)
    assert with_reports["current_run_bytes"] == disk["current_run_bytes"] + len(b"audit log")
    assert with_reports["report_bytes"] == len(b"audit log")


def fixture_runners(ctx, *, provider_failure=False):
    calls = []
    def single(command, **kwargs):
        phase = command[command.index("--phase") + 1] if "--phase" in command else "merge"
        calls.append(phase)
        if phase == "merge": merge_fixture(ctx, main.recover_main(ctx.output, ctx.run, cpu_fixture=True)["policy"])
        elif phase == "update": update_fixture(ctx)
        else: pytest.fail("fixture already bootstrapped")
        return 0
    def parallel(jobs, **kwargs):
        calls.append("collect")
        policy = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)["policy"]
        merge = merge_fixture(ctx, policy)
        for i, job in enumerate(jobs):
            command = job["command"]
            prompt = command[command.index("--prompt-id") + 1]
            if provider_failure and i == 1: return [0] + [1] * (len(jobs) - 1)
            group_fixture(ctx, policy, prompt, merge)
        return [0] * len(jobs)
    return single, parallel, calls


def test_first_real_boundary_fixture_pause_resume_same_run(ctx):
    ctx.args.stop_after_window = 1
    single, parallel, calls = fixture_runners(ctx)
    report = coordinator.orchestrate(ctx.args, ctx.root, dict(run=ctx.run), single, parallel, cpu_fixture=True)
    assert report["status"] == "paused_at_verified_boundary" and report["passed"] is False
    assert report["progress"]["policy"]["global_optimizer_step"] == 1
    assert calls == ["merge", "collect", "update"]
    assert not list((ctx.output / "merges").iterdir()) and not (ctx.output / "manifest.json").exists()
    # The identical semantic run resumes at the NEXT window; no recollection of p0..3.
    ctx.args.stop_after_window = 2
    again = coordinator.orchestrate(ctx.args, ctx.root, dict(run=ctx.run), single, parallel, cpu_fixture=True)
    assert again["progress"]["policy"]["global_optimizer_step"] == 2
    assert calls == ["merge", "collect", "update"] * 2
    assert retention.receipt_path(ctx.output, "compact", 1).exists()


def test_provider_failure_blocks_update_durable_whole_groups(ctx):
    single, parallel, calls = fixture_runners(ctx, provider_failure=True)
    with pytest.raises(RuntimeError, match="update forbidden"):
        coordinator.orchestrate(ctx.args, ctx.root, dict(run=ctx.run), single, parallel, cpu_fixture=True)
    assert calls == ["merge", "collect"]
    assert len(main.recover_main(ctx.output, ctx.run, cpu_fixture=True)["current_groups"]) == 1
    assert json.loads((ctx.reports / "report.json").read_text())["status"] == "interrupted"
    assert not list((ctx.output / "checkpoints").glob("policy-*"))


def test_no_overlap_permitted(ctx):
    class Active:
        active = True
        def __call__(self, *a, **k): pytest.fail("must not start another GPU phase")
    with pytest.raises(RuntimeError, match="still active"):
        coordinator.orchestrate(ctx.args, ctx.root, dict(run=ctx.run), Active(), Active(), cpu_fixture=True)


@pytest.mark.parametrize("failure", ["report", "manifest", "after_manifest_replace"])
def test_final_manifest_last_and_revoked(ctx, monkeypatch, failure):
    report = dict(passed=True, scope="runtime", run_identity={})
    writes = []
    monkeypatch.setattr(main, "final_reconstruction", lambda *a, **k: report)
    monkeypatch.setattr(cp, "fsync_directory", lambda *a, **k: None)
    def writer(path, value, **kw):
        writes.append(path.name)
        if failure == "report" and path.name == "report.json": raise OSError("report failed")
        if failure == "manifest" and path.name == "manifest.json": raise OSError("manifest failed")
        path.write_text(json.dumps(value))
        if failure == "after_manifest_replace" and path.name == "manifest.json": raise OSError("fsync failed")
    monkeypatch.setattr(cp, "durable_json", writer)
    with pytest.raises(OSError): main.publish_final(ctx.output, ctx.reports, report)
    assert writes[0] == "report.json" and not (ctx.output / "manifest.json").exists()


def test_cpu_never_runtime_pass_and_runtime_injection_forbidden(ctx):
    with pytest.raises(ValueError, match="CPU/report-only"):
        main.publish_final(ctx.output, ctx.reports, dict(passed=True, scope="cpu_fixture"), cpu_fixture=True)
    with pytest.raises(ValueError, match="CPU fixtures"):
        coordinator.orchestrate(ctx.args, ctx.root, dict(run=ctx.run), lambda *a: 0)
    with pytest.raises(ValueError, match="CPU fixture anchor"):
        main.recover_main(ctx.output, ctx.run)


def test_complete_100_step_chain_final_retention_and_historical_corruption(ctx, monkeypatch):
    checkpoints = chain_fixture(ctx, 100)
    import importlib
    sft_tool_audit = importlib.import_module("opensearch_vl_repro.sft_tool_audit")
    hashed, original_sha = [], sft_tool_audit.sha256_file
    def track(path):
        hashed.append(Path(path))
        return original_sha(path)
    monkeypatch.setattr(sft_tool_audit, "sha256_file", track)
    main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    assert not any("policy-000025" in p.parts or "policy-000050" in p.parts or "policy-000075" in p.parts for p in hashed)
    assert not any("groups" in p.parts for p in hashed)
    assert any("policy-000100" in p.parts for p in hashed)
    hashed.clear()
    report = main.final_reconstruction(ctx.output, ctx.run, cpu_fixture=True)
    assert any("policy-000025" in p.parts for p in hashed)
    assert any("groups" in p.parts and p.name.startswith("mm-") for p in hashed)
    assert report["passed"] is False and report["final_policy"]["policy_iteration"] == 100
    assert report["groups_completed"] == 400 and report["trajectories_completed"] == 1600
    assert report["optimizer_steps"] == list(range(1, 101))
    assert all(c["eligibility"] == cp.checkpoint_eligibility("main_checkpoint") for c in checkpoints)
    disk = retention.disk_accounting(ctx.output)
    assert disk["retained_full_checkpoints"] == [f"policy-{i:06d}" for i in retention.MILESTONES]
    assert len(disk["compacted_checkpoints"]) == 96
    filename = next(iter(checkpoints[0]["artifact_role_files"]["adapter"]))
    path = ctx.output / "checkpoints/policy-000001" / filename
    original_bytes = path.read_bytes()
    path.write_bytes(b"history retained corruption")
    main.recover_main(ctx.output, ctx.run, cpu_fixture=True)  # historical bytes intentionally deferred
    with pytest.raises(ValueError, match="inventory"):
        main.final_reconstruction(ctx.output, ctx.run, cpu_fixture=True)
    path.write_bytes(original_bytes)
    group = checkpoints[0]["groups"][0]
    tensor = group["members"][0]["steps"][0]["multimodal_file"]
    (ctx.output / "groups" / group["identity"]["trajectory_group_id"] / tensor).write_bytes(b"historical tensor corruption")
    main.recover_main(ctx.output, ctx.run, cpu_fixture=True)
    with pytest.raises(ValueError, match="inventory"):
        main.final_reconstruction(ctx.output, ctx.run, cpu_fixture=True)


def test_duplicate_prompt_consumption_rejected(ctx):
    chain_fixture(ctx, 1)
    policy = main.recover_main(ctx.output, ctx.run, cpu_fixture=True)["policy"]
    group_fixture(ctx, policy, "p0", merge_fixture(ctx, policy))
    with pytest.raises(ValueError, match="consumed prompt"):
        main.recover_main(ctx.output, ctx.run, cpu_fixture=True)


@pytest.mark.parametrize("case", ["historical", "unbound", "future", "foreign"])
def test_partial_forensics_never_infer_final_authority(ctx, case):
    directory = ctx.output / "groups" / (".collect-" + str(uuid.uuid4()))
    directory.mkdir()
    if case != "unbound":
        a = draft_group(ctx.run, fixture_policy(ctx.run), "p0")["identity"]
        if case == "future": a["policy_iteration"] = 100
        if case == "foreign": a["run_identity_sha256"] = "f" * 64
        a = cp.seal({k: v for k, v in a.items() if k != "trajectory_group_id"}, "trajectory_group_id")
        (directory / "attempt.json").write_text(json.dumps(a))
    if case == "historical": main.audit_private_collections(ctx.output, ctx.run)
    else:
        with pytest.raises(ValueError): main.audit_private_collections(ctx.output, ctx.run)
    assert directory.exists()  # No forensic deletion or reuse.


def test_import_safe_primitives_not_duplicated():
    for module in (main, coordinator, collection, update, retention, process):
        tree = ast.parse(inspect.getsource(module))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom): assert not (node.module or "").startswith(("torch", "vllm", "rllm", "verl", "transformers"))
            if isinstance(node, ast.Import): assert not any(a.name.startswith(("torch", "vllm", "rllm", "verl", "transformers")) for a in node.names)
    assert update.MainUpdateSession.update is update.UpdateSession.update
    for api in ("load_formal_actor", "update_formal_window", "save_formal_staging", "fresh_reload_staging"):
        assert api in inspect.getsource(update.UpdateSession)
    assert collection.collect_member is __import__("opensearch_vl_repro.rl.formal_collection", fromlist=["collect_member"]).collect_member
    source = inspect.getsource(collection.run_collection_worker)
    assert "range(4)" in source and "commit_collection(" in source
    assert "collection_failure.json" not in source
    assert "recover_formal_run(" not in inspect.getsource(main)


def test_commands_four_rank_update_and_single_gpu_collection(ctx, monkeypatch):
    monkeypatch.setenv("RANK", "9")
    command = coordinator.worker_command(ctx.args, ctx.root, "update")
    assert "--nproc_per_node=4" in command
    assert "--stop-after-window" not in command  # operational control not worker semantics
    assert coordinator.worker_environment("0")["CUDA_VISIBLE_DEVICES"] == "0"
    assert "RANK" not in coordinator.worker_environment("0,1,2,3")


def test_parallel_failure_kills_peers_and_detached_descendants(tmp_path, monkeypatch):
    """Mock Popen/proc only; real CPU reader threads, no external processes."""
    table, killed, waited = {}, [], []
    class Worker:
        def __init__(self, pid):
            self.pid, self.stdout, self.code = pid, io.StringIO("CPU MOCK\n"), None
        def poll(self):
            if self.pid == 10:
                table.pop(10, None)
                table[99] = dict(parent=1000, group=99, birth=999, state="S")
                self.code = 1
            return self.code
        def kill(self):
            killed.append(self.pid)
            table.pop(self.pid, None)
            self.code = -9
        def wait(self):
            waited.append(self.pid)
            return self.code
    def popen(command, **kwargs):
        assert kwargs["start_new_session"] and not kwargs.get("shell", False)
        pid = 10 + len(waiters)
        worker = Worker(pid)
        waiters.append(worker)
        table[pid] = dict(parent=1000, group=pid, birth=pid * 10, state="S")
        return worker
    waiters = []
    def kill(pid, sig):
        killed.append(pid)
        table[pid]["state"] = "Z"
    def reap(*a):
        for pid, row in list(table.items()):
            if row["state"] == "Z":
                table.pop(pid)
                return pid, 9
        raise ChildProcessError()
    monkeypatch.setattr(process, "os", SimpleNamespace(name="posix", getpid=lambda: 1000, kill=kill, waitpid=reap, WNOHANG=1))
    monkeypatch.setattr(process, "signal", SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(smoke, "linux_process_table", lambda *a: copy.deepcopy(table))
    monkeypatch.setattr(smoke, "set_subreaper", lambda *a: 0)
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    runner = process.ParallelProcessRunner()
    codes = runner([dict(command=["mock"], env={}, log=tmp_path / f"worker{i}.log") for i in range(4)], cwd=tmp_path)
    assert codes == [1, None, None, None]
    assert set(killed) == {11, 12, 13, 99} and set(waited) == {10, 11, 12, 13}
    assert not table and not runner.active


def test_s3_contract_stays_distinct():
    from test_rl_formal_s3_smoke import smoke_run
    smoke.require_smoke_run(smoke_run())
    with pytest.raises(ValueError): smoke.require_smoke_run(main_run())
    assert smoke.VERSION == "formal-s3-smoke20-v1"


@pytest.fixture(scope="module")
def frozen_input_fixture(tmp_path_factory):
    """Full 7992-row synthetic pinned-schema population, never formal membership."""
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from rl_quality_helpers import make_quality_bundle
    from test_rl_formal_data import overlap_file
    from opensearch_vl_repro.rl.data import prepare_formal_dataset, write_dataset_artifacts, source_sample_id
    from opensearch_vl_repro.rl.config import load_rl_config
    root = tmp_path_factory.mktemp("main-source-cpu")
    source = root / "source"
    source.mkdir()
    Image.new("RGB", (2, 2), "blue").save(source / "fixture.png")
    rows = [dict(question=f"CPU synthetic question {i}?", answer=f"CPU reference {i}",
                 images=["fixture.png"], dataset="CPU ONLY") for i in range(7992)]
    parquet = source / "rl.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    config = load_rl_config(Path(__file__).resolve().parents[1] / "configs/rl_main.yaml")
    settings = config["data"]
    (source / "source_revision.txt").write_text(settings["dataset_id"] + "\n" + settings["dataset_revision"] + "\n")
    ranked = sorted(range(len(rows)), key=lambda i: (
        hashlib.sha256(f"{settings['seed']}:{settings['dataset_id']}:{settings['dataset_revision']}:{source_sample_id(i)}".encode()).hexdigest(), i))
    quality = make_quality_bundle(root / "quality", [source_sample_id(i) for i in ranked[:420]], ["ok"] * 420, main_count=400)
    config["data"]["quality_audit_dir"] = str(quality)
    eval_file, sft_file = overlap_file(root, name="eval"), overlap_file(root, name="sft")
    kwargs = dict(source_parquet=parquet, source_root=source, dataset_id=settings["dataset_id"],
        dataset_revision=settings["dataset_revision"], seed=settings["seed"], smoke_count=20, main_count=400,
        shard_size=100, eval_overlap_manifest=eval_file, sft_overlap_manifest=sft_file, quality_audit_dir=quality)
    artifacts = prepare_formal_dataset(**kwargs)
    output = root / "data"
    write_dataset_artifacts(artifacts, output, **{k: kwargs[k] for k in (
        "source_parquet", "source_root", "eval_overlap_manifest", "sft_overlap_manifest", "quality_audit_dir")})
    return dict(root=root, output=output, config=config, kwargs=kwargs)


def load_fixture_data(f):
    return main.load_main_records(f["output"] / "main400.json", f["config"], **{k: f["kwargs"][k] for k in (
        "source_root", "source_parquet", "eval_overlap_manifest", "sft_overlap_manifest")})


def test_main_loader_complete_source_quality_shard_provenance(frozen_input_fixture):
    records, manifest = load_fixture_data(frozen_input_fixture)
    assert len(records) == 400 and manifest["selected_count"] == 400
    assert manifest["membership"] == [r["source_sample_id"] for r in records]
    assert manifest["name"] == "main" and len(manifest["shard_order"]) == 4
    assert manifest["quality_selected_main_membership_sha256"] == digest(manifest["membership"])


@pytest.mark.parametrize("case", ["membership", "shard", "quality", "question", "source_revision", "manifest_name"])
def test_main_loader_rejects_corrupt_binding(frozen_input_fixture, tmp_path, case):
    import shutil
    f = copy.deepcopy(frozen_input_fixture)
    shutil.copytree(f["root"], tmp_path / "inputs")
    replacement = tmp_path / "inputs"
    f["root"] = replacement
    f["output"] = replacement / "data"
    for k in ("source_parquet", "source_root", "eval_overlap_manifest", "sft_overlap_manifest", "quality_audit_dir"):
        f["kwargs"][k] = replacement / f["kwargs"][k].relative_to(frozen_input_fixture["root"])
    f["config"]["data"]["quality_audit_dir"] = str(f["kwargs"]["quality_audit_dir"])
    if case in {"membership", "manifest_name"}:
        p = f["output"] / "main400_manifest.json"
        value = json.loads(p.read_text())
        if case == "membership": value["membership"].reverse()
        else: value["name"] = "smoke"
        p.write_text(json.dumps(cp.seal({k: v for k, v in value.items() if k != "manifest_sha256"}, "manifest_sha256")))
    elif case in {"shard", "question"}:
        p = f["output"] / ("main400_shards/shard_000.json" if case == "shard" else "main400.json")
        value = json.loads(p.read_text())
        value[0]["question"] = "corrupted source binding"
        p.write_text(json.dumps(value))
    elif case == "quality":
        p = f["kwargs"]["quality_audit_dir"] / "rl_quality_ok_allowlist_v1.json"
        value = json.loads(p.read_text())
        value["selected_main400"].reverse()
        p.write_text(json.dumps(value))
    else:
        (f["kwargs"]["source_root"] / "source_revision.txt").write_text("foreign\nrevision\n")
    with pytest.raises(ValueError): load_fixture_data(f)
