"""CPU transaction/identity tests: never publish a real GPU Gate PASS."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from threading import Barrier

import pytest

from opensearch_vl_repro.rl.group import group_identity, publish_group, read_group, run_lock, validate_group


def fixture_group():
    identity = group_identity(prompt_id="rl_000001", policy_fingerprint="a" * 64,
                              rollout_fingerprint="b" * 64, attempt="attempt1", context="context1")
    step = {"prompt_ids": [1, 2], "response_ids": [3, 4], "logprobs": [-.1, -.2],
            "model_output": {"prompt_ids": [1, 2], "completion_ids": [3, 4], "logprobs": [-.1, -.2]},
            "info": {"token_origin": "vllm.RequestOutput", "logprobs_mode": "processed_logprobs"},
            "multimodal_file": "mm.pt"}
    return {"identity": identity, "members": [
        {"identity": copy.deepcopy(identity), "rollout_index": i, "complete": True, "steps": [copy.deepcopy(step)],
         "fatal": {"fatal": False, "fatal_step": None}, "reward": {"format": 1., "accuracy": float(i), "query": .5, "total": .1 + .8 * i}}
        for i in range(2)]}


@pytest.mark.parametrize("field,value", [("prompt_id", "other"), ("policy_fingerprint", "c" * 64),
    ("rollout_fingerprint", "d" * 64), ("attempt", "attempt2"), ("context", "context2")])
def test_group_id_binds_every_runtime_identity(field, value):
    kwargs = dict(prompt_id="p", policy_fingerprint="a" * 64, rollout_fingerprint="b" * 64, attempt="one", context="c")
    first = group_identity(**kwargs)
    assert first == group_identity(**kwargs)
    assert first["trajectory_group_id"] != group_identity(**{**kwargs, field: value})["trajectory_group_id"]


@pytest.mark.parametrize("mutation", ["half", "mixed_attempt", "mixed_policy", "duplicate_index", "incomplete", "missing_logprobs"])
def test_incomplete_mixed_group_fail_closed(mutation):
    group = fixture_group()
    if mutation == "half": group["members"].pop()
    elif mutation == "mixed_attempt": group["members"][1]["identity"]["collection_attempt"] = "other"
    elif mutation == "mixed_policy": group["members"][1]["identity"]["pre_update_policy_fingerprint"] = "other"
    elif mutation == "duplicate_index": group["members"][1]["rollout_index"] = 0
    elif mutation == "incomplete": group["members"][1]["complete"] = False
    else: group["members"][1]["steps"][0]["logprobs"] = []
    with pytest.raises(ValueError): validate_group(group)


def test_atomic_commit_resume_and_no_half_reuse(tmp_path):
    staging, final = tmp_path / ".attempt", tmp_path / "group"
    staging.mkdir(); (staging / "mm.pt").write_bytes(b"CPU fixture")
    group = fixture_group()
    assert not final.exists()
    incomplete = copy.deepcopy(group); incomplete["members"].pop()
    with pytest.raises(ValueError): publish_group(staging, final, incomplete)
    assert not final.exists()
    publish_group(staging, final, group)
    assert not staging.exists() and read_group(final)["members"] == group["members"]
    with pytest.raises(FileExistsError): publish_group(tmp_path, final, group)
    (final / "mm.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"): read_group(final)


def test_failed_atomic_rename_leaves_no_committed_group(tmp_path, monkeypatch):
    from opensearch_vl_repro.rl import group as module
    staging = tmp_path / ".stage"; staging.mkdir(); (staging / "mm.pt").write_bytes(b"fixture")
    original = module.os.replace
    def replace(source, target):
        if str(target).endswith("group"): raise OSError("publication failed")
        return original(source, target)
    monkeypatch.setattr(module.os, "replace", replace)
    with pytest.raises(OSError): publish_group(staging, tmp_path / "group", fixture_group())
    assert not (tmp_path / "group").exists()


def test_lock_released_for_same_command_resume(tmp_path):
    with run_lock(tmp_path / "gate.lock"): pass
    with run_lock(tmp_path / "gate.lock"): pass


@pytest.mark.parametrize("name", [".coordinator.lock", ".formal.lock", "gate.lock"])
def test_default_run_lock_contender_fails_fast_and_releases(tmp_path, name):
    path = tmp_path / name
    with ThreadPoolExecutor(max_workers=1) as executor:
        with run_lock(path):
            def contender():
                with pytest.raises(OSError):
                    with run_lock(path): pytest.fail("duplicate authority acquired lock")
            executor.submit(contender).result(timeout=5)
        with run_lock(path): pass
    assert path.exists()  # Never delete lock files as a stale-lock bypass.


def test_blocking_lock_releases_after_exception(tmp_path):
    path = tmp_path / ".publication.lock"
    with pytest.raises(ValueError, match="CPU fixture"):
        with run_lock(path, blocking=True): raise ValueError("CPU fixture")
    with run_lock(path): pass


def test_parallel_formal_publications_wait_on_real_shared_lock(tmp_path, monkeypatch):
    from opensearch_vl_repro.rl import group as module, checkpoint as cp
    from test_rl_formal_contracts import fixture_run, fixture_policy, draft_group
    run = fixture_run(n=4, world_size=4)
    policy = fixture_policy(run)
    parent = tmp_path / "formal-cpu-fixture" / "groups"
    parent.mkdir(parents=True)
    jobs = []
    for prompt in run["prompt_ids"][:2]:
        group = draft_group(run, policy, prompt)
        staging = parent / f".stage-{prompt}"
        staging.mkdir()
        for member in group["members"]:
            (staging / member["trajectory_file"]).write_text(json.dumps(member), encoding="utf-8")
            (staging / member["steps"][0]["multimodal_file"]).write_bytes(f"CPU fixture {prompt}".encode())
        jobs.append((staging, parent / group["identity"]["trajectory_group_id"], group,
                     cp.artifact_inventory(staging)))
    path = parent / ".publication.lock"
    ready = Barrier(3)
    modes = []
    real_lock = module.run_lock
    @contextmanager
    def observed_lock(actual, **kwargs):
        assert actual == path
        modes.append(kwargs.get("blocking", False))
        ready.wait(timeout=5)
        with real_lock(actual, **kwargs): yield  # REAL OS advisory lock, not mocked.
    monkeypatch.setattr(module, "run_lock", observed_lock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        with real_lock(path):
            futures = [executor.submit(module.publish_formal_group, staging, destination, group, cpu_fixture=True)
                       for staging, destination, group, inventory in jobs]
            ready.wait(timeout=5)
            for future in futures:
                # A held OS lock makes completion impossible, independent of speed.
                with pytest.raises(TimeoutError): future.result(timeout=.1)
            assert modes == [True, True]
            for staging, destination, group, inventory in jobs:
                assert staging.is_dir() and not destination.exists()
                assert cp.artifact_inventory(staging) == inventory
        values = [future.result(timeout=10) for future in futures]
    for (staging, destination, group, inventory), value in zip(jobs, values, strict=True):
        assert not staging.exists()
        assert module.read_formal_group(destination) == value
        assert value["identity"] == group["identity"]
        assert value["file_sha256"] == inventory
        assert value["committed"] is True and value["evidence_scope"] == "cpu_fixture"
    assert values[0]["identity"] != values[1]["identity"] and path.exists()
