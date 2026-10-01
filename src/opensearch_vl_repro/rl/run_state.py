"""Future trainer's group-atomic resume, interrupt and progress schemas."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

RUN_STATUSES = frozenset({"running", "interrupted", "completed", "failed"})
INTERRUPT_REASONS = frozenset({
    "quota_exhausted", "auth_failed", "rate_limit_persistent", "provider_unavailable",
    "network_unavailable", "judge_unavailable", "malformed_provider_response",
    "manual_interrupt", "provider_misconfigured",
})

PROVIDER_ERROR_INTERRUPTS = {
    "quota_error": "quota_exhausted",
    "authentication_error": "auth_failed",
    "rate_limit_persistent": "rate_limit_persistent",
    "provider_error": "provider_unavailable",
    "network_error": "network_unavailable",
    "timeout": "network_unavailable",
    "invalid_response": "malformed_provider_response",
    "configuration_error": "provider_misconfigured",
}


def interrupt_reason_for(error_type: str, *, judge: bool = False) -> str:
    """Classify only known infrastructure failures; never map model mistakes."""
    if error_type not in PROVIDER_ERROR_INTERRUPTS:
        raise ValueError(f"not a recoverable provider error: {error_type!r}")
    if judge and error_type in {"provider_error", "timeout", "network_error"}:
        return "judge_unavailable"
    return PROVIDER_ERROR_INTERRUPTS[error_type]


def group_ready_for_update(rollouts: list[tuple[str, int]], *, rollout_n: int) -> bool:
    """A group commits only a complete set from one attempt, never mixed resume."""
    if rollout_n < 1:
        raise ValueError("rollout_n must be positive")
    return (len(rollouts) == rollout_n
            and len({attempt_id for attempt_id, _ in rollouts}) == 1
            and {index for _, index in rollouts} == set(range(rollout_n)))


@dataclass(frozen=True)
class RLRunState:
    run_id: str
    status: str
    current_shard: int
    total_shards: int
    shard_group_index: int
    next_group_index: int
    last_completed_group_id: str | None
    completed_groups: int
    completed_trajectories: int
    optimizer_step: int
    actor_fingerprint: str
    rollout_config_hash: str
    reward_config_hash: str
    tool_cache_version: str
    checkpoint_path: str | None = None
    interrupt_reason: str | None = None
    interrupt_provider: str | None = None
    interrupt_message: str | None = None
    rng_state_metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.run_id or self.status not in RUN_STATUSES or self.total_shards < 1:
            raise ValueError("RL run state identity/status is invalid")
        if any(value < 0 for value in (self.current_shard, self.shard_group_index,
                                       self.next_group_index, self.completed_groups,
                                       self.completed_trajectories, self.optimizer_step)):
            raise ValueError("RL run state counters must be nonnegative")
        if self.current_shard >= self.total_shards:
            raise ValueError("current_shard exceeds total_shards")
        if not all((self.actor_fingerprint, self.rollout_config_hash,
                    self.reward_config_hash, self.tool_cache_version)):
            raise ValueError("RL run state fingerprints/version are required")
        if self.interrupt_reason is not None and self.interrupt_reason not in INTERRUPT_REASONS:
            raise ValueError("invalid RL interrupt reason")
        if self.status == "interrupted" and self.interrupt_reason is None:
            raise ValueError("interrupted RL run requires interrupt reason")
        if self.status != "interrupted" and self.interrupt_reason is not None:
            raise ValueError("interrupt reason requires interrupted status")
        if self.rng_state_metadata is not None and not isinstance(self.rng_state_metadata, dict):
            raise ValueError("rng_state_metadata must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": 1, **asdict(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RLRunState":
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("RL run state schema mismatch")
        instance = cls(**{key: item for key, item in value.items() if key != "schema_version"})
        instance.validate()
        return instance

    @classmethod
    def from_json(cls, raw: str) -> "RLRunState":
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class RLProgressSnapshot:
    current_shard: int
    total_shards: int
    shard_group_index: int
    shard_group_total: int
    completed_groups: int
    total_groups: int
    completed_trajectories: int
    total_trajectories: int
    optimizer_step: int
    mean_reward: float | None
    fatal_count: int
    fatal_rate: float
    tool_error_count: int
    tool_call_count: int
    cache_hit_count: int
    cache_hit_rate: float
    elapsed_seconds: float
    eta_seconds: float | None

    def validate(self) -> None:
        if self.total_shards < 1 or not 0 <= self.current_shard < self.total_shards:
            raise ValueError("progress shard index is invalid")
        if any(value < 0 for value in (
                self.shard_group_index, self.shard_group_total, self.completed_groups,
                self.total_groups, self.completed_trajectories, self.total_trajectories,
                self.optimizer_step, self.fatal_count, self.tool_error_count,
                self.tool_call_count, self.cache_hit_count)):
            raise ValueError("progress counters must be nonnegative")
        if (self.shard_group_index > self.shard_group_total
                or self.completed_groups > self.total_groups
                or self.completed_trajectories > self.total_trajectories
                or self.fatal_count > self.completed_trajectories
                or self.cache_hit_count > self.tool_call_count):
            raise ValueError("progress counters exceed totals")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
               for value in (self.fatal_rate, self.cache_hit_rate)):
            raise ValueError("progress rates must be finite fractions")
        if (self.mean_reward is not None and (not math.isfinite(self.mean_reward) or not 0 <= self.mean_reward <= 1)
                or not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0
                or self.eta_seconds is not None and (not math.isfinite(self.eta_seconds) or self.eta_seconds < 0)):
            raise ValueError("progress time/reward values are invalid")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RLProgressSnapshot":
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("RL progress schema mismatch")
        instance = cls(**{key: item for key, item in value.items() if key != "schema_version"})
        instance.validate()
        return instance
