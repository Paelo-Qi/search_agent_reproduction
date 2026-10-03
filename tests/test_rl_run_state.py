"""Strict formal FSM tests; the legacy RLRunState remains backward compatible."""
import pytest

from opensearch_vl_repro.rl import checkpoint as cp
from opensearch_vl_repro.rl.run_state import (
    advance_update_attempt, new_trainer_state, new_update_attempt,
    persist_update_attempt, transition_trainer_state,
    rollback_interrupted_state,
)
from test_rl_formal_contracts import context, prepare_checkpoint


def test_state_machine_verified_step_only(context):
    root, run, policy, groups, window = context
    state = new_trainer_state(run, policy)
    for status in ("collecting", "ready_to_update", "updating", "checkpointing"):
        state = transition_trainer_state(state, status)
        assert state["policy"]["policy_iteration"] == 0
    staging, manifest = prepare_checkpoint(root, run, policy, groups, window)
    with pytest.raises(ValueError, match="publication"):
        transition_trainer_state(state, "iteration_verified", checkpoint=manifest)
    cp.commit_verified_checkpoint(root, staging, manifest, cpu_fixture=True)
    state = transition_trainer_state(state, "iteration_verified", checkpoint=manifest,
                                     checkpoint_directory=root / "checkpoints" / "policy-000001")
    assert state["policy"]["policy_iteration"] == 1
    assert transition_trainer_state(state, "collecting")["policy"] == state["policy"]
    assert transition_trainer_state(state, "completed")["status"] == "completed"


@pytest.mark.parametrize("statuses,next_status", [
    (["collecting"], "completed"), (["collecting", "ready_to_update", "updating"], "collecting"),
    (["interrupted"], "collecting"), (["failed"], "collecting"), ([], "iteration_verified"),
])
def test_illegal_formal_transitions(context, statuses, next_status):
    _, run, policy, _, _ = context
    state = new_trainer_state(run, policy)
    for status in statuses: state = transition_trainer_state(state, status)
    with pytest.raises(ValueError): transition_trainer_state(state, next_status)


def test_attempt_cannot_skip_or_overwrite(context):
    root, _, _, _, window = context
    attempt = new_update_attempt(window)
    persist_update_attempt(root, attempt, cpu_fixture=True)
    with pytest.raises(ValueError): advance_update_attempt(attempt, "checkpoint_staging")
    with pytest.raises((ValueError, FileExistsError)): persist_update_attempt(root, attempt, cpu_fixture=True)
    failed = advance_update_attempt(attempt, "failed", failure_reason="CPU fixture interruption")
    persist_update_attempt(root, failed, cpu_fixture=True)
    with pytest.raises(ValueError): advance_update_attempt(failed, "started")


def test_explicit_rollback_is_not_an_illegal_forward_transition(context):
    root, run, policy, _, window = context
    state = new_trainer_state(run, policy)
    for status in ("collecting", "ready_to_update", "updating", "interrupted"):
        state = transition_trainer_state(state, status)
    attempt = new_update_attempt(window)
    persist_update_attempt(root, attempt, cpu_fixture=True)
    for phase in ("started", "step_may_have_run"):
        attempt = advance_update_attempt(attempt, phase)
        persist_update_attempt(root, attempt, cpu_fixture=True)
    restored = rollback_interrupted_state(state, root, run, attempt, window, cpu_fixture=True)
    assert restored["state"]["policy"] == policy
    assert restored["state"]["status"] == "ready_to_update"
    assert restored["state"]["recovery_reload_required"] is True
    assert restored["retry_plan"]["new_attempt"]["attempt_id"] != attempt["attempt_id"]
    with pytest.raises(ValueError, match="reload verification"):
        transition_trainer_state(restored["state"], "updating")
