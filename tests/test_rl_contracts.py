from dataclasses import replace

import pytest

from opensearch_vl_repro.rl.cache_contract import tool_cache_key
from opensearch_vl_repro.rl.run_state import (INTERRUPT_REASONS, RLProgressSnapshot,
    RLRunState, group_ready_for_update, interrupt_reason_for)


def test_cache_key_is_canonical_and_versioned():
    base = dict(tool_name="text_search", provider="serper", tool_behavior_version="4",
                protocol_version="v3")
    first = tool_cache_key(**base, arguments={"q": "hello", "options": {"a": 1, "b": 2}})
    assert first == tool_cache_key(**base, arguments={"options": {"b": 2, "a": 1}, "q": "hello"})
    assert first != tool_cache_key(**{**base, "protocol_version": "v4"}, arguments={"q": "hello", "options": {"a": 1, "b": 2}})
    assert first != tool_cache_key(**{**base, "tool_behavior_version": "5"}, arguments={"q": "hello", "options": {"a": 1, "b": 2}})
    with pytest.raises(ValueError):
        tool_cache_key(**base, arguments={"q": float("nan")})


def run_state():
    return RLRunState("run", "interrupted", 1, 4, 10, 10, "rl_000010", 110, 440, 8,
                      "actor", "rollout", "reward", "cache-v1", "ckpt",
                      "quota_exhausted", "serper", "budget exhausted")


def test_run_state_roundtrip_and_interrupt_validation():
    state = run_state()
    assert RLRunState.from_json(state.to_json()) == state
    assert "manual_interrupt" in INTERRUPT_REASONS
    with pytest.raises(ValueError):
        replace(state, interrupt_reason="unknown").to_dict()
    with pytest.raises(ValueError):
        replace(state, status="running").to_dict()


def test_progress_roundtrip_and_validation():
    progress = RLProgressSnapshot(1, 4, 10, 100, 110, 400, 440, 1600, 8,
                                  .7, 12, 12 / 440, 2, 150, 50, 50 / 150, 600, 900)
    assert RLProgressSnapshot.from_dict(progress.to_dict()) == progress
    with pytest.raises(ValueError):
        replace(progress, completed_groups=401).to_dict()
    with pytest.raises(ValueError):
        replace(progress, cache_hit_rate=float("nan")).to_dict()


def test_provider_interrupt_is_distinct_from_model_error():
    assert interrupt_reason_for("quota_error") == "quota_exhausted"
    assert interrupt_reason_for("provider_error", judge=True) == "judge_unavailable"
    assert interrupt_reason_for("invalid_response") == "malformed_provider_response"
    with pytest.raises(ValueError):
        interrupt_reason_for("invalid_argument")


def test_group_atomicity_rejects_partial_or_mixed_attempts():
    assert group_ready_for_update([("attempt-2", i) for i in range(4)], rollout_n=4)
    assert not group_ready_for_update([("attempt-2", i) for i in range(2)], rollout_n=4)
    assert not group_ready_for_update([("old", 0), ("old", 1), ("new", 2), ("new", 3)], rollout_n=4)
    assert not group_ready_for_update([("same", 0), ("same", 0), ("same", 2), ("same", 3)], rollout_n=4)
