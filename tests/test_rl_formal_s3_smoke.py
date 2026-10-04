"""CPU-only orchestration/receipt fixtures. NEVER runtime Smoke20 PASS evidence."""
import copy
import inspect
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl import formal_smoke as smoke
from opensearch_vl_repro.rl import formal_collection as collection
from opensearch_vl_repro.rl import formal_smoke_update as update
from opensearch_vl_repro.rl.formal_smoke_cli import build_parser
from opensearch_vl_repro.rl.group import publish_formal_group, validate_formal_group
from opensearch_vl_repro.rl.run_state import advance_update_attempt, persist_update_attempt
from opensearch_vl_repro.rl.training_window import build_training_window, expected_window_prompts
from test_rl_formal_contracts import (fixture_run, fixture_policy, commit_group, draft_group,
                                      prepare_checkpoint, digest)


def smoke_run():
    original = fixture_run(prompt_count=20)
    semantics = copy.deepcopy(original["semantics"])
    semantics.update(coordinator_version=smoke.VERSION,
        optimizer=dict(name="AdamW", learning_rate=1e-6, weight_decay=0.),
        ppo=dict(epochs=1, microbatch=1, clip_ratio_low=.2, clip_ratio_high=.28, entropy=0., loss_mode="vanilla"))
    semantics["base_model"]["offline_snapshot_sha256"] = digest("snapshot")
    semantics["rollout"]["config"] = copy.deepcopy(smoke.ROLLOUT)
    return cp.build_training_run_identity("cpu-contract", semantics=semantics,
        prompt_ids=original["prompt_ids"], prompt_sources=original["prompt_sources"])


@pytest.fixture
def setup(tmp_path):
    run = smoke_run()
    output, reports = smoke.smoke_paths(tmp_path, run["run_id"])
    cp.initialize_formal_run(output, run, fixture_policy(run), cpu_fixture=True)
    reports.mkdir(parents=True)
    args = SimpleNamespace(run_id=run["run_id"], config=tmp_path / "configs/rl_smoke.yaml",
        data=tmp_path / "data/rl/smoke20.json", source_root=tmp_path / "source", base_model_path=tmp_path / "base",
        sft_adapter=tmp_path / smoke.SFT_ADAPTER, judge_config=tmp_path / "judge.yaml",
        search_config=tmp_path / "search.yaml", layout_config=tmp_path / "layout.yaml",
        rollout_gpu="0", update_gpus="0,1", tool_cache_dir=None, reward_cache_dir=None)
    return tmp_path, output, reports, run, args


def current_window(output, run):
    recovered = smoke.recover_smoke(output, run, cpu_fixture=True)
    policy = recovered["policy"]
    for prompt in recovered["missing_prompts"]:
        commit_group(output, run, policy, prompt)
    recovered = smoke.recover_smoke(output, run, cpu_fixture=True)
    return recovered, build_training_window(run, policy, recovered["current_groups"],
                                           window_id=smoke.window_id(policy["policy_iteration"]))


def fixture_update(output, run, *, marker_failure=False, publication_failure=False):
    recovered, window = current_window(output, run)
    transaction = update.prepare_update_transaction(output, run, recovered, window, cpu_fixture=True)
    attempt = transaction["attempt"]
    persist_update_attempt(output, attempt, cpu_fixture=True)
    for phase in ("started", "step_may_have_run"):
        attempt = advance_update_attempt(attempt, phase)
        persist_update_attempt(output, attempt, cpu_fixture=True)
    if marker_failure:
        return attempt
    attempt = advance_update_attempt(attempt, "checkpoint_staging")
    persist_update_attempt(output, attempt, cpu_fixture=True)
    staging, manifest = prepare_checkpoint(output, run, recovered["policy"], recovered["current_groups"], window,
        attempt=attempt, kind="smoke_final" if window["expected_optimizer_step"] == 5 else "smoke_continuation")
    cp.commit_verified_checkpoint(output, staging, manifest, cpu_fixture=True)
    if not publication_failure:
        smoke.recover_smoke(output, run, cpu_fixture=True)
    return manifest


def test_frozen_smoke20_n2_k4_five_windows():
    run = smoke_run()
    smoke.require_smoke_run(run)
    assert [expected_window_prompts(run, i) for i in range(5)] == [run["prompt_ids"][i:i + 4] for i in range(0, 20, 4)]
    parser = build_parser(Path("/fixture"))
    args = parser.parse_args(["--run-id", "test", "--source-root", "/images", "--base-model-path", "/base"])
    assert args.config.name == "rl_smoke.yaml" and args.data.name == "smoke20.json"


@pytest.mark.parametrize("field,value", [("rollout_n", 4), ("groups_per_window", 2), ("world_size", 1),
    ("coordinator_version", "formal-s2-diagnostic"), ("diagnostic_version", "S2")])
def test_reject_non_smoke_identity(field, value):
    run = smoke_run()
    semantics = {**run["semantics"], field: value}
    changed = cp.build_training_run_identity(run["run_id"], semantics=semantics,
        prompt_ids=run["prompt_ids"], prompt_sources=run["prompt_sources"])
    with pytest.raises(ValueError): smoke.require_smoke_run(changed)


def test_old_unbound_run_not_resealed():
    run = fixture_run(prompt_count=20)
    before = copy.deepcopy(run)
    with pytest.raises(ValueError): smoke.require_smoke_run(run)
    assert before == run


def test_original_sft_initial_policy_and_verified_continuation(setup):
    _, output, _, run, _ = setup
    first = smoke.recover_smoke(output, run, cpu_fixture=True)["policy"]
    assert first["policy_iteration"] == 0
    assert first["adapter_fingerprint"] == run["semantics"]["source_sft"]["adapter_sha256"]
    manifest = fixture_update(output, run)
    next_policy = smoke.recover_smoke(output, run, cpu_fixture=True)["policy"]
    assert next_policy["checkpoint_identity"] == manifest["checkpoint_manifest_sha256"]
    assert next_policy["optimizer_identity"] == manifest["artifact_roles"]["optimizer"]


@pytest.mark.parametrize("field,value", [("diagnostic_mode", True), ("rollout_executed", False)])
def test_diagnostic_group_rejected(setup, field, value):
    _, output, _, run, _ = setup
    policy = fixture_policy(run)
    draft = draft_group(run, policy)
    if field == "diagnostic_mode": draft[field] = value
    else: draft["members"][0][field] = value
    commit_group(output, run, policy, draft=draft)
    with pytest.raises(ValueError, match="diagnostic"): smoke.recover_smoke(output, run, cpu_fixture=True)


def test_future_window_group_rejected(setup):
    _, output, _, run, _ = setup
    commit_group(output, run, fixture_policy(run), "p4")
    with pytest.raises(ValueError, match="future-window"): smoke.recover_smoke(output, run, cpu_fixture=True)


@pytest.mark.parametrize("field", ["pre_update_policy_fingerprint", "collection_attempt"])
def test_mixed_policy_or_attempt_members_rejected(field):
    run = smoke_run()
    draft = draft_group(run, fixture_policy(run))
    draft["members"][1]["identity"][field] = str(uuid.uuid4()) if field == "collection_attempt" else digest("stale")
    with pytest.raises(ValueError): validate_formal_group(draft, committed=False)


def test_reference_never_copied_into_model_task():
    row = dict(prompt_id="rl_000001", source_sample_id="rl_000001", question="Q?", reference_answer="SECRET_REFERENCE")
    task = collection.model_task(row, ["real image fixture"])
    assert set(task) == {"sample_id", "question", "images"}
    assert "SECRET_REFERENCE" not in repr(task)
    assert row["reference_answer"] == "SECRET_REFERENCE"


def test_incomplete_group_never_published(setup):
    _, output, _, run, _ = setup
    draft = draft_group(run, fixture_policy(run))
    stage = output / "groups" / ".partial"
    stage.mkdir()
    with pytest.raises(ValueError): collection.commit_collection(output, stage, draft["identity"], draft["members"][:1], merge_identity={}, cpu_fixture=True)
    assert not (output / "groups" / draft["identity"]["trajectory_group_id"]).exists()
    assert stage.exists()


def test_recovery_reuses_committed_current_policy_group(setup):
    _, output, _, run, _ = setup
    group = commit_group(output, run, fixture_policy(run), "p0")
    recovered = smoke.recover_smoke(output, run, cpu_fixture=True)
    assert recovered["current_groups"] == [group]
    assert recovered["missing_prompts"] == ["p1", "p2", "p3"]


def test_incomplete_attempt_gets_new_uuid_and_no_member_splicing(setup, monkeypatch):
    _, output, _, run, _ = setup
    policy = fixture_policy(run)
    row = dict(prompt_id="p0")
    monkeypatch.setattr(collection, "source_identity_for_row", lambda run, row: digest("p0"))
    real = cp.durable_json
    monkeypatch.setattr(cp, "durable_json", lambda path, value, **kwargs: real(path, value, cpu_fixture=True))
    a, first = collection.reserve_collection(output, run, policy, row)
    b, second = collection.reserve_collection(output, run, policy, row)
    assert a != b and a.exists() and b.exists()
    assert first["collection_attempt"] != second["collection_attempt"]
    assert second["collection_attempt_index"] == 1
    assert smoke.recover_smoke(output, run, cpu_fixture=True)["current_groups"] == []


def test_provider_interruption_stops_before_update_and_resumes(setup):
    root, output, reports, run, args = setup
    calls = []
    def interrupted(command, **kwargs):
        calls.append(command)
        commit_group(output, run, fixture_policy(run), "p0")
        cp.durable_json(reports / "collection_failure.json", dict(provider_interruption={"error_type": "quota_error"}), cpu_fixture=True)
        return 1
    with pytest.raises(RuntimeError): smoke.orchestrate(args, root, {"run": run}, interrupted, cpu_fixture=True)
    assert len(calls) == 1 and "collect_rl_formal_smoke.py" in calls[0][1]
    assert not list((output / "checkpoints").glob("policy-*"))
    failure = json.loads((reports / "report.json").read_text())
    assert failure["status"] == "interrupted" and failure["provider_interruption"]["error_type"] == "quota_error"
    group = smoke.recover_smoke(output, run, cpu_fixture=True)["current_groups"][0]
    result, phases = finish_fixture(setup)
    assert result["passed"] is False
    assert group in smoke.recover_smoke(output, run, cpu_fixture=True)["groups"]
    assert phases.count("update") == 5


def finish_fixture(setup):
    root, output, _, run, args = setup
    phases = []
    active = False
    def runner(command, **kwargs):
        nonlocal active
        assert not active
        active = True
        phase = command[command.index("--phase") + 1] if "--phase" in command else "collect"
        phases.append(phase)
        if phase == "collect": current_window(output, run)
        elif phase == "update": fixture_update(output, run)
        else: pytest.fail("already bootstrapped fixture")
        active = False
        return 0
    return smoke.orchestrate(args, root, {"run": run}, runner, cpu_fixture=True), phases


def test_complete_fixture_twenty_forty_five_and_lifecycle(setup):
    _, output, reports, run, _ = setup
    result, phases = finish_fixture(setup)
    assert phases == ["collect", "update"] * 5
    assert result["groups_completed"] == 20 and result["trajectories_completed"] == 40
    assert result["optimizer_steps"] == [1, 2, 3, 4, 5] and result["passed"] is False
    assert result["checkpoint_kind"] == "smoke_final" and result["eligible_for_main_init"] is False
    assert not (output / "manifest.json").exists()
    assert (reports / "report.json").exists()
    again, phases = finish_fixture(setup)
    assert phases == [] and again["final_policy"] == result["final_policy"]


def test_uncertain_update_restores_parent_and_new_uuid(setup):
    _, output, _, run, _ = setup
    prior = fixture_update(output, run, marker_failure=True)
    recovered, window = current_window(output, run)
    transaction = update.prepare_update_transaction(output, run, recovered, window, cpu_fixture=True)
    retry = transaction["retry_plan"]
    assert retry["discard_uncertain_memory"] is True
    assert retry["new_attempt"]["attempt_id"] != prior["attempt_id"]
    assert retry["reuse_group_ids"] == window["ordered_group_ids"]
    assert retry["reload_optimizer_identity"] == recovered["policy"]["optimizer_identity"]
    assert retry["reload_rng_identity"] == recovered["policy"]["rng_identity"]
    assert transaction["state"]["recovery_reload_required"] is True
    from opensearch_vl_repro.rl.run_state import transition_trainer_state
    with pytest.raises(ValueError, match="reload verification"):
        transition_trainer_state(transaction["state"], "updating")


def test_native_continuation_uncertainty_reuses_groups(setup):
    _, output, _, run, _ = setup
    first = fixture_update(output, run)
    prior = fixture_update(output, run, marker_failure=True)
    recovered, window = current_window(output, run)
    transaction = update.prepare_update_transaction(output, run, recovered, window, cpu_fixture=True)
    assert transaction["retry_plan"]["reload_checkpoint_identity"] == first["checkpoint_manifest_sha256"]
    assert prior["attempt_id"] != transaction["attempt"]["attempt_id"]


def test_published_checkpoint_crash_rolls_forward_never_duplicate_step(setup):
    _, output, _, run, _ = setup
    first = fixture_update(output, run, publication_failure=True)
    cp.durable_json(output / "latest.json", {"corrupt": True}, cpu_fixture=True)
    cp.durable_json(output / "state.json", {"corrupt": True}, cpu_fixture=True)
    result, phases = finish_fixture(setup)
    assert phases.count("update") == 4 and result["final_policy"]["policy_iteration"] == 5
    assert result["checkpoint_identities"][0] == first["checkpoint_manifest_sha256"]
    assert smoke.recover_smoke(output, run, cpu_fixture=True)["attempts"][0]["phase"] == "verified"


def test_immutable_artifact_tampering_fail_closed(setup):
    _, output, _, run, _ = setup
    fixture_update(output, run)
    filename = output / "checkpoints/policy-000001/native/model_rank_1.fixture"
    filename.write_bytes(b"corrupted fixture")
    with pytest.raises(ValueError): smoke.recover_smoke(output, run, cpu_fixture=True)


def test_partial_final_cannot_finalize(setup):
    _, output, _, run, _ = setup
    fixture_update(output, run)
    with pytest.raises(ValueError, match="incomplete"): smoke.final_reconstruction(output, run, cpu_fixture=True)


def test_checkpoint_kinds_and_main_ineligibility(setup):
    _, output, _, run, _ = setup
    finish_fixture(setup)
    checkpoints = smoke.recover_smoke(output, run, cpu_fixture=True)["checkpoints"]
    assert [c["eligibility"]["kind"] for c in checkpoints] == ["smoke_continuation"] * 4 + ["smoke_final"]
    assert all(c["eligibility"]["eligible_for_main_init"] is False for c in checkpoints)


@pytest.mark.parametrize("failure", ["report", "manifest", "after_manifest_replace"])
def test_pass_manifest_last_and_revoked_on_write_failure(setup, monkeypatch, failure):
    _, output, reports, _, _ = setup
    writes = []
    def writer(path, report, **kwargs):
        writes.append(path.name)
        if failure == "report" and path.name == "report.json": raise OSError("report publication failed")
        if failure == "manifest" and path.name == "manifest.json": raise OSError("manifest publication failed")
        path.write_text(json.dumps(report))
        if failure == "after_manifest_replace" and path.name == "manifest.json": raise OSError("fsync failed")
    monkeypatch.setattr(cp, "durable_json", writer)
    monkeypatch.setattr(cp, "fsync_directory", lambda *args, **kwargs: None)
    report = dict(passed=True, scope="runtime", checks={"test_publication_order_only": True}, run_identity={})
    # Publication I/O fault injection ONLY, not a runtime reconstruction/PASS.
    monkeypatch.setattr(smoke, "final_reconstruction", lambda *args, **kwargs: report)
    with pytest.raises(OSError): smoke.publish_final(output, reports, report)
    assert writes[0] == "report.json" and not (output / "manifest.json").exists()


def test_cpu_fixture_can_never_publish_runtime_pass(setup):
    _, output, reports, _, _ = setup
    with pytest.raises(ValueError): smoke.publish_final(output, reports, {"passed": True, "scope": "cpu_fixture"}, cpu_fixture=True)
    assert not (output / "manifest.json").exists()


def test_report_without_manifest_is_not_pass_authority(setup):
    _, output, reports, _, _ = setup
    cp.durable_json(reports / "report.json", {"passed": True, "fixture": "publication order only"}, cpu_fixture=True)
    assert not (output / "manifest.json").exists()


def test_commands_explicit_two_rank_update_and_single_gpu_collect(setup, monkeypatch):
    root, _, _, _, args = setup
    monkeypatch.setenv("DEEPSEEK_API_KEY", "SECRET_CPU_FIXTURE")
    monkeypatch.setenv("RANK", "9")
    for phase in ("bootstrap", "collect", "update"):
        command = smoke.worker_command(args, root, phase)
        env = smoke.worker_environment(args, phase)
        assert isinstance(command, list) and "SECRET_CPU_FIXTURE" not in repr(command)
        assert env["DEEPSEEK_API_KEY"] == "SECRET_CPU_FIXTURE" and "RANK" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == ("0" if phase == "collect" else "0,1")
        if phase != "collect": assert "--nproc_per_node=2" in command


@pytest.mark.parametrize("devices", ["0", "0,0", "0,1,2", "0;echo secret"])
def test_bad_gpu_ownership_rejected(setup, devices):
    *_, args = setup
    args.update_gpus = devices
    with pytest.raises(ValueError): smoke.worker_environment(args, "update")


def test_stale_merge_binding_rejected_before_backend(tmp_path):
    directory = tmp_path / "merged"
    directory.mkdir()
    (directory / "config.json").write_text("fixture")
    actor = dict(formal_binding={"version": smoke.VERSION, "policy_iteration": 0})
    identity = collection.formal_merge_identity(actor=actor, versions={}, file_hashes=cp.artifact_inventory(directory))
    manifest = dict(identity=identity, merge_complete=True, fresh_hf_forward_finite=True,
                    no_active_peft=True, merge_hf_destroyed=True, reload_hf_destroyed=True)
    (directory / "merge_manifest.json").write_text(json.dumps(manifest))
    assert collection.verify_formal_merge(directory, actor["formal_binding"]) == manifest
    with pytest.raises(ValueError, match="stale"): collection.verify_formal_merge(directory, {"policy_iteration": 1})


def test_gate_staging_diagnostic_checkpoint_cannot_handoff(setup):
    _, output, _, run, args = setup
    policy = fixture_policy(run)
    bad = {**policy, "policy_iteration": 1}
    with pytest.raises(ValueError): collection.validate_formal_handoff(output, run, bad, args.sft_adapter, cpu_fixture=True)
    with pytest.raises(ValueError): cp.read_verified_checkpoint(output / "checkpoints/.staging")
    with pytest.raises(ValueError): smoke.smoke_paths(output, "../rl_gate_c")


def test_import_safe_and_production_reuses_primitives():
    import ast
    for module in (smoke, collection, update):
        tree = ast.parse(inspect.getsource(module))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("torch", "vllm", "rllm", "verl", "transformers"))
            if isinstance(node, ast.Import):
                assert not any(n.name.startswith(("torch", "vllm", "rllm", "verl", "transformers")) for n in node.names)
    source = inspect.getsource(update)
    for api in ("load_formal_actor", "update_formal_window", "save_formal_staging", "fresh_reload_staging", "rollback_interrupted_state"):
        assert api in source
    assert "optimizer.step(" not in source and "scheduler" not in source
    collector = inspect.getsource(collection)
    for api in ("build_rllm_workflow", "live_rewards", "VLLMStaticBackend", "publish_formal_group", "identity_builder=formal_merge_identity"):
        assert api in collector


def test_runtime_disallows_injected_runner_before_launch(setup):
    root, _, _, run, args = setup
    with pytest.raises(ValueError, match="CPU orchestration"):
        smoke.orchestrate(args, root, {"run": run}, lambda *args, **kwargs: 0)


def test_runtime_cannot_use_cpu_anchor(setup):
    _, output, _, run, _ = setup
    with pytest.raises(ValueError, match="CPU fixture anchor"): smoke.recover_smoke(output, run)


def test_custom_provider_secrets_redacted(monkeypatch):
    monkeypatch.setenv("MY_JUDGE_SECRET", "secret-custom-fixture-123")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-standard-fixture-456")
    value = smoke.redact_runtime_secrets({"error": "secret-custom-fixture-123 secret-standard-fixture-456"})
    assert value == {"error": "[REDACTED] [REDACTED]"}


def test_process_overlap_is_rejected_without_launch(monkeypatch):
    runner = smoke.ProcessRunner()
    runner.active = True
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not spawn"))
    with pytest.raises(RuntimeError, match="exclusive"): runner([], env={}, cwd=Path("."), log=Path("unused"))


@pytest.mark.parametrize("state,expected", [("S", True), ("R", True), ("Z", False)])
def test_process_group_cleanup_detection(tmp_path, state, expected):
    process = tmp_path / "10"
    process.mkdir()
    (process / "stat").write_text(f"10 (worker name with spaces) {state} 1 999 " + "0 " * 16 + "123")
    assert smoke.process_group_alive(999, tmp_path) is expected
    assert smoke.process_group_alive(998, tmp_path) is False


def test_pass_publication_rejects_fabricated_checks(setup, monkeypatch):
    _, output, reports, run, _ = setup
    monkeypatch.setattr(cp, "fsync_directory", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="CPU fixture anchor"):
        smoke.publish_final(output, reports, dict(passed=True, scope="runtime", checks={"fake": True}, run_identity=run))
    assert not (output / "manifest.json").exists()


def test_editable_integration_sources_are_bound_without_import(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('MUST NOT IMPORT')")
    monkeypatch.setattr(smoke.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(package)]))
    inventory = smoke.installed_source_inventory("rllm")
    assert set(inventory) == {"0/__init__.py"}
    first = digest(inventory)
    (package / "__init__.py").write_text("raise AssertionError('changed source')")
    assert digest(smoke.installed_source_inventory("rllm")) != first


@pytest.mark.parametrize("path", ["outputs/rl_gate_c/cache", "outputs/rl_formal_s2_validation/cache",
    "outputs/sft_main_imageid_v3/cache", "reports/eval_runs/cache", "data/raw/cache",
    "outputs/rl_formal_smoke/another-run/groups/cache"])
def test_cache_locators_cannot_mutate_protected_experiments(setup, path):
    root, *_, args = setup
    args.tool_cache_dir = root / path
    with pytest.raises(ValueError): smoke.validate_cache_locators(args, root)


def test_isolated_formal_cache_locator_is_allowed(setup):
    root, output, _, _, args = setup
    args.tool_cache_dir = output / "tool_cache"
    smoke.validate_cache_locators(args, root)
    assert not args.tool_cache_dir.exists()  # Read-only validation.


@pytest.mark.parametrize("cutoff", [None, True, -1, 1])
def test_fatal_member_without_actual_integer_cutoff_cannot_commit(setup, cutoff):
    _, output, _, run, _ = setup
    draft = draft_group(run, fixture_policy(run))
    draft["members"][0].update(fatal=True, fatal_step=cutoff)
    with pytest.raises(ValueError, match="generation cutoff"):
        collection.commit_collection(output, output / "groups/.unused", draft["identity"], draft["members"],
                                     merge_identity={}, cpu_fixture=True)


def test_current_verified_adapter_handoff_with_complete_role_inventory(setup):
    from opensearch_vl_repro.inference.adapter import adapter_file_identity
    _, output, _, run, args = setup
    recovered, window = current_window(output, run)
    staging, old = prepare_checkpoint(output, run, recovered["policy"], recovered["current_groups"], window)
    directory = staging / "adapter"
    (directory / "adapter_config.json.fixture").rename(directory / "adapter_config.json")
    (directory / "adapter_model.safetensors.fixture").rename(directory / "adapter_model.safetensors")
    (directory / "README.md").write_text("CPU fixture; not actual weights or a runtime policy")
    inventory = cp.artifact_inventory(staging)
    roles = {role: {n: sha for n, sha in inventory.items() if n.startswith(role + "/")}
             for role in ("adapter", "native", "optimizer", "rng")}
    evidence = {**old["reload_evidence"], "reloaded_artifact_roles": cp.artifact_role_identities(roles, inventory)}
    manifest = cp.build_checkpoint_manifest(run, recovered["policy"], recovered["current_groups"], window,
        old["update_attempt"], old["reward_window"], artifact_role_files=roles, file_sha256=inventory,
        kind="smoke_continuation", reload_evidence=evidence, cpu_fixture=True)
    cp.commit_verified_checkpoint(output, staging, manifest, cpu_fixture=True)
    policy = smoke.recover_smoke(output, run, cpu_fixture=True)["policy"]
    adapter, actor = collection.validate_formal_handoff(output, run, policy, args.sft_adapter, cpu_fixture=True)
    assert adapter == output / "checkpoints/policy-000001/adapter"
    assert actor["actor_adapter_fingerprint"] == adapter_file_identity(adapter)["adapter_fingerprint"]
    assert "README.md" in actor["formal_binding"]["adapter_file_sha256"]
    assert actor["formal_binding"]["effective_policy_fingerprint"] == policy["effective_policy_fingerprint"]
    # Exact same historical bytes are NOT current after the next checkpoint.
    fixture_update(output, run)
    with pytest.raises(ValueError, match="CURRENT"):
        collection.validate_formal_handoff(output, run, policy, args.sft_adapter, cpu_fixture=True)


def test_detached_elastic_rank_and_engine_descendants_are_owned():
    rows = {10: dict(parent=1, group=10, birth=100, state="S"),
            20: dict(parent=10, group=20, birth=200, state="S"),
            30: dict(parent=20, group=30, birth=300, state="S"),
            99: dict(parent=1, group=99, birth=990, state="S")}
    owned = smoke.descendant_identities(rows, 10)
    assert owned == {20: 200, 30: 300}
    assert smoke.live_owned_processes(rows, owned) == [20, 30]
    # Rank 20 was adopted by the coordinator after launcher 10 died.
    rows[20]["parent"] = 1000
    assert smoke.descendant_identities(rows, 1000) == owned
    # A recycled PID may never authorize killing an unrelated user process.
    rows[20]["birth"] = 201
    assert smoke.live_owned_processes(rows, owned) == [30]


def test_process_runner_waits_for_and_reaps_detached_rank_mock(tmp_path, monkeypatch):
    """Mock Popen/proc/prctl only: no real subprocess, torchrun, CUDA or API."""
    import io
    import threading
    rows, killed, reaped = {}, [], []
    events = []
    class Worker:
        pid = 10
        stdout = io.StringIO("MOCK child log\n")
        def wait(self):
            # Launcher exited; independent session rank adopted by subreaper.
            rows.pop(10, None)
            if not killed:
                rows[20] = dict(parent=1000, group=20, birth=200, state="S")
            events.append("launcher_wait")
            return 0
        def poll(self): return 0
        def kill(self): pytest.fail("completed launcher does not need signaling")
    def popen(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        rows[10] = dict(parent=1000, group=10, birth=100, state="S")
        return Worker()
    def kill(pid, sig):
        killed.append(pid)
        rows[pid]["state"] = "Z"
    def reap(*args):
        zombies = [p for p, r in rows.items() if r["state"] == "Z"]
        if not zombies: raise ChildProcessError()
        pid = zombies[0]
        rows.pop(pid)
        reaped.append(pid)
        return pid, 9
    monkeypatch.setattr(smoke, "os", SimpleNamespace(name="posix", getpid=lambda: 1000,
        kill=kill, waitpid=reap, WNOHANG=1))
    monkeypatch.setattr(smoke, "signal", SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(smoke, "linux_process_table", lambda *args, **kwargs: copy.deepcopy(rows))
    monkeypatch.setattr(smoke, "set_subreaper", lambda value: events.append(("subreaper", value)) or 0)
    monkeypatch.setattr(smoke.subprocess, "Popen", popen)
    # Real CPU-only reader thread is fine; it sees only an in-memory fake pipe.
    assert threading.current_thread() is not None
    runner = smoke.ProcessRunner()
    assert runner(["mock-only"], env={}, cwd=tmp_path, log=tmp_path / "log.txt") == 0
    assert killed == [20] and reaped == [20] and not rows and not runner.active
    assert events[-1] == ("subreaper", 0)
