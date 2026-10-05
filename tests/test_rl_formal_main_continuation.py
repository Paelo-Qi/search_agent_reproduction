"""Existing authority barriers, NOT evidence that continuation is implemented.

The current single-run schema cannot represent an inherited nonzero anchor.
Keep these barriers fail-closed rather than silently copying/resealing a parent.
"""
import copy

import pytest

from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl.run_state import reconstruct_consumed_ledger
from test_rl_formal_s4_main import main_run
from test_rl_formal_contracts import fixture_policy, digest


@pytest.mark.parametrize("iteration", [2, 7])
def test_nonzero_parent_policy_cannot_be_bootstrapped_or_rebound(tmp_path, iteration):
    parent = main_run()
    child = cp.build_training_run_identity("child-cpu-fixture", semantics=copy.deepcopy(parent["semantics"]),
        prompt_ids=parent["prompt_ids"], prompt_sources=parent["prompt_sources"])
    initial = fixture_policy(parent)
    policy = cp.seal({**{k: v for k, v in initial.items() if k != "effective_policy_fingerprint"},
        "policy_iteration": iteration, "global_optimizer_step": iteration,
        "parent_checkpoint_identity": digest("parent-checkpoint"),
        "cumulative_consumed_group_ids": [f"fixture-group-{i}" for i in range(4 * iteration)]},
        "effective_policy_fingerprint")
    cp.validate_policy(policy)
    output = tmp_path / "child"
    with pytest.raises(ValueError, match="source SFT iteration zero"):
        cp.initialize_formal_run(output, child, policy, cpu_fixture=True)
    assert not output.exists()
    with pytest.raises(ValueError, match="iteration-zero anchor"):
        reconstruct_consumed_ledger(child, policy, [], [])
    # Merely substituting the child hash still must NOT legalize inherited state.
    rebound = cp.seal({**{k: v for k, v in policy.items() if k != "effective_policy_fingerprint"},
        "run_identity_sha256": child["run_identity_sha256"]}, "effective_policy_fingerprint")
    with pytest.raises(ValueError, match="iteration-zero anchor"):
        reconstruct_consumed_ledger(child, rebound, [], [])
    assert cp.checkpoint_eligibility("main_checkpoint")["eligible_for_main_init"] is False


def test_parent_anchor_remains_immutable_when_child_identity_is_rejected(tmp_path):
    parent = main_run()
    root = tmp_path / "parent"
    cp.initialize_formal_run(root, parent, fixture_policy(parent), cpu_fixture=True)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    child = cp.build_training_run_identity("child-cpu-fixture", semantics=parent["semantics"],
        prompt_ids=parent["prompt_ids"], prompt_sources=parent["prompt_sources"])
    with pytest.raises(ValueError, match="resume semantics differ"):
        cp._load_anchor(root, child)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
