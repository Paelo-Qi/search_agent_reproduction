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


FORMAL_STATE_TRANSITIONS = {
    "initialized": {"collecting", "interrupted", "failed"},
    "collecting": {"ready_to_update", "interrupted", "failed"},
    "ready_to_update": {"updating", "interrupted", "failed"},
    "updating": {"checkpointing", "interrupted", "failed"},
    "checkpointing": {"iteration_verified", "interrupted", "failed"},
    "iteration_verified": {"collecting", "completed", "interrupted", "failed"},
    "interrupted": set(),  # recovery is a separately validated rollback, not a forward transition
    "failed": set(), "completed": set(),
}
ATTEMPT_TRANSITIONS = {
    "prepared": {"started", "failed"}, "started": {"step_may_have_run", "failed"},
    "step_may_have_run": {"checkpoint_staging", "failed"},
    "checkpoint_staging": {"verified", "failed"}, "verified": set(), "failed": set(),
}


def new_update_attempt(window, *, attempt_id=None):
    import uuid
    from .checkpoint import check_seal, require_uuid, seal
    check_seal(window, "window_sha256")
    attempt_id = attempt_id or str(uuid.uuid4())
    require_uuid(attempt_id)
    return seal({"schema_version": 1, "attempt_id": attempt_id, "sequence": 0,
                 "phase": "prepared", "window_sha256": window["window_sha256"],
                 "parent_checkpoint_identity": window["parent_checkpoint_identity"],
                 "parent_policy_fingerprint": window["parent_policy_fingerprint"],
                 "expected_optimizer_step": window["expected_optimizer_step"],
                 "run_identity_sha256": window["run_identity_sha256"],
                 "previous_event_sha256": None, "failure_reason": None,
                 "verified_checkpoint_identity": None}, "attempt_event_sha256")


def validate_update_attempt(attempt):
    from .checkpoint import check_seal, require_counter, require_digest, require_uuid
    check_seal(attempt, "attempt_event_sha256")
    require_uuid(attempt["attempt_id"])
    require_counter(attempt["schema_version"], 1)
    require_counter(attempt["sequence"])
    require_counter(attempt["expected_optimizer_step"], 1)
    if attempt["schema_version"] != 1 or attempt["phase"] not in ATTEMPT_TRANSITIONS:
        raise ValueError("invalid update attempt schema/phase")
    for field in ("window_sha256", "parent_checkpoint_identity", "parent_policy_fingerprint", "run_identity_sha256"):
        require_digest(attempt[field])
    if attempt["sequence"] == 0:
        if attempt["phase"] != "prepared" or attempt["previous_event_sha256"] is not None:
            raise ValueError("invalid initial attempt")
    else:
        require_digest(attempt["previous_event_sha256"])
    if attempt["phase"] == "failed":
        if not isinstance(attempt["failure_reason"], str) or not attempt["failure_reason"]:
            raise ValueError("failure reason required")
    elif attempt["failure_reason"] is not None:
        raise ValueError("failure reason on nonfailed attempt")
    if attempt["phase"] == "verified":
        require_digest(attempt["verified_checkpoint_identity"])
    elif attempt["verified_checkpoint_identity"] is not None:
        raise ValueError("unverified attempt cannot claim checkpoint")


def advance_update_attempt(attempt, phase, *, failure_reason=None, checkpoint=None, checkpoint_directory=None):
    from .checkpoint import read_verified_checkpoint, seal, validate_checkpoint_manifest
    validate_update_attempt(attempt)
    if phase not in ATTEMPT_TRANSITIONS[attempt["phase"]]:
        raise ValueError("illegal update attempt transition")
    checkpoint_id = None
    if phase == "verified":
        if checkpoint is None:
            raise ValueError("verified checkpoint evidence required")
        validate_checkpoint_manifest(checkpoint)
        if checkpoint_directory is None or read_verified_checkpoint(checkpoint_directory) != checkpoint:
            raise ValueError("verified attempt requires immutable checkpoint publication")
        if checkpoint["update_attempt"]["attempt_event_sha256"] != attempt["attempt_event_sha256"]:
            raise ValueError("checkpoint belongs to a different update attempt")
        checkpoint_id = checkpoint["checkpoint_manifest_sha256"]
    result = seal({**{k: v for k, v in attempt.items() if k != "attempt_event_sha256"},
                   "sequence": attempt["sequence"] + 1, "phase": phase,
                   "previous_event_sha256": attempt["attempt_event_sha256"],
                   "failure_reason": failure_reason, "verified_checkpoint_identity": checkpoint_id},
                  "attempt_event_sha256")
    validate_update_attempt(result)
    return result


def read_update_attempt(root, attempt_id):
    """Validate the complete immutable event chain, not just its final phase."""
    from pathlib import Path
    from .checkpoint import require_uuid
    require_uuid(attempt_id)
    events = sorted((Path(root) / "attempts" / attempt_id).glob("*.json"))
    if not events:
        raise ValueError("missing durable attempt history")
    previous = None
    for index, path in enumerate(events):
        event = json.loads(path.read_text(encoding="utf-8"))
        if path.is_symlink():
            raise ValueError("attempt event symlinks forbidden")
        validate_update_attempt(event)
        if (event["sequence"] != index or event["attempt_id"] != attempt_id
                or path.name != f"{index:06d}-{event['phase']}.json"):
            raise ValueError("attempt history index/identity mismatch")
        if previous is not None and (
                event["phase"] not in ATTEMPT_TRANSITIONS[previous["phase"]]
                or event["previous_event_sha256"] != previous["attempt_event_sha256"]
                or any(event[k] != previous[k] for k in (
                    "window_sha256", "parent_checkpoint_identity", "parent_policy_fingerprint",
                    "expected_optimizer_step", "run_identity_sha256"))):
            raise ValueError("attempt history chain conflict")
        previous = event
    return previous


def persist_update_attempt(root, attempt, *, cpu_fixture=False):
    """Append-only event history. Even failed attempts are never overwritten."""
    from .checkpoint import _formal_root, durable_json, fsync_directory, read_verified_checkpoint
    from .group import run_lock
    validate_update_attempt(attempt)
    root = _formal_root(root)
    with run_lock(root / ".formal.lock"):
        directory = root / "attempts" / attempt["attempt_id"]
        directory.mkdir(parents=True, exist_ok=True)
        fsync_directory(directory.parent, cpu_fixture=cpu_fixture)
        events = sorted(directory.glob("*.json"))
        if events:
            previous = read_update_attempt(root, attempt["attempt_id"])
            if (attempt["sequence"] != previous["sequence"] + 1
                    or attempt["previous_event_sha256"] != previous["attempt_event_sha256"]
                    or attempt["phase"] not in ATTEMPT_TRANSITIONS[previous["phase"]]
                    or any(attempt[k] != previous[k] for k in (
                        "attempt_id", "window_sha256", "parent_checkpoint_identity",
                        "parent_policy_fingerprint", "expected_optimizer_step", "run_identity_sha256"))):
                raise ValueError("attempt history conflict")
        elif attempt["sequence"] != 0:
            raise ValueError("missing attempt history")
        if attempt["phase"] == "verified":
            checkpoint = read_verified_checkpoint(root / "checkpoints" / f"policy-{attempt['expected_optimizer_step']:06d}")
            if (checkpoint["checkpoint_manifest_sha256"] != attempt["verified_checkpoint_identity"]
                    or checkpoint["update_attempt"]["attempt_event_sha256"] != attempt["previous_event_sha256"]):
                raise ValueError("verified attempt must reference this immutable checkpoint")
        path = directory / f"{attempt['sequence']:06d}-{attempt['phase']}.json"
        if path.exists():
            raise FileExistsError("attempt event immutable")
        durable_json(path, attempt, cpu_fixture=cpu_fixture)


def new_trainer_state(run, policy):
    from .checkpoint import seal, validate_policy, validate_training_run_identity
    validate_training_run_identity(run)
    validate_policy(policy)
    if policy["run_identity_sha256"] != run["run_identity_sha256"]:
        raise ValueError("state policy/run mismatch")
    return seal({"schema_version": 1, "status": "initialized", "policy": policy,
                 "run_identity_sha256": run["run_identity_sha256"],
                 "training_behavior_fingerprint": run["training_behavior_fingerprint"]}, "state_sha256")


def transition_trainer_state(state, status, *, checkpoint=None, checkpoint_directory=None,
                             reload_receipt=None, actor=None):
    from .checkpoint import check_seal, read_verified_checkpoint, require_counter, seal, validate_checkpoint_manifest, validate_policy
    check_seal(state, "state_sha256")
    require_counter(state["schema_version"], 1)
    if state["schema_version"] != 1 or state["run_identity_sha256"] != state["policy"]["run_identity_sha256"]:
        raise ValueError("formal state schema/run mismatch")
    validate_policy(state["policy"])
    if status not in FORMAL_STATE_TRANSITIONS.get(state["status"], set()):
        raise ValueError("illegal formal trainer transition")
    if status == "updating" and state.get("recovery_reload_required") is True:
        from .formal_policy_update import require_formal_reload
        if reload_receipt is None or actor is None:
            raise ValueError("S2 model/optimizer/RNG reload verification required before retry update")
        require_formal_reload(actor, state["policy"], reload_receipt)
        # Preserve the recovery flag/history. Only a live capability authorizes
        # this transition; changing a JSON flag cannot authorize S2 updates.
    policy = state["policy"]
    if status == "iteration_verified":
        if checkpoint is None:
            raise ValueError("cannot advance policy without verified checkpoint")
        validate_checkpoint_manifest(checkpoint)
        if checkpoint_directory is None or read_verified_checkpoint(checkpoint_directory) != checkpoint:
            raise ValueError("policy advance requires immutable checkpoint publication")
        if (checkpoint["parent_policy"] != policy
                or checkpoint["training_behavior_fingerprint"] != state["training_behavior_fingerprint"]
                or checkpoint["run"]["run_identity_sha256"] != state["run_identity_sha256"]):
            raise ValueError("state/checkpoint lineage mismatch")
        policy = checkpoint_policy(checkpoint)
    elif checkpoint is not None:
        raise ValueError("checkpoint only applies to verified transition")
    return seal({**{k: v for k, v in state.items() if k != "state_sha256"},
                 "status": status, "policy": policy}, "state_sha256")


def checkpoint_policy(checkpoint):
    from .checkpoint import seal, validate_checkpoint_manifest
    validate_checkpoint_manifest(checkpoint)
    parent = checkpoint["parent_policy"]
    return seal({"policy_iteration": checkpoint["policy_iteration"],
                 "global_optimizer_step": checkpoint["global_optimizer_step"],
                 "run_identity_sha256": checkpoint["run"]["run_identity_sha256"],
                 "parent_checkpoint_identity": parent["checkpoint_identity"],
                 "checkpoint_identity": checkpoint["checkpoint_manifest_sha256"],
                 "adapter_fingerprint": checkpoint["artifact_roles"]["adapter"],
                 "native_identity": checkpoint["artifact_roles"]["native"],
                 "execution_contract": checkpoint["execution_contract"],
                 "optimizer_identity": checkpoint["artifact_roles"]["optimizer"],
                 "rng_identity": checkpoint["artifact_roles"]["rng"],
                 "cumulative_consumed_group_ids": parent["cumulative_consumed_group_ids"]
                 + checkpoint["consumed_group_ids"]}, "effective_policy_fingerprint")


def retry_update_plan(attempt, window, policy, *, verified_checkpoints):
    """Schema-only rollback plan: no in-memory state survives an ambiguous attempt."""
    from .checkpoint import check_seal, validate_policy, validate_checkpoint_manifest
    validate_update_attempt(attempt)
    validate_policy(policy)
    check_seal(window, "window_sha256")
    for checkpoint in verified_checkpoints:
        validate_checkpoint_manifest(checkpoint)
        if (checkpoint["parent_policy"]["checkpoint_identity"] == policy["checkpoint_identity"]
                or set(checkpoint["consumed_group_ids"]) & set(window["ordered_group_ids"])):
            raise ValueError("already verified successor: roll forward, never repeat update")
    if (attempt["phase"] == "verified" or attempt["window_sha256"] != window["window_sha256"]
            or attempt["parent_policy_fingerprint"] != policy["effective_policy_fingerprint"]
            or window["parent_policy_fingerprint"] != policy["effective_policy_fingerprint"]
            or window["parent_checkpoint_identity"] != policy["checkpoint_identity"]
            or set(window["ordered_group_ids"]) & set(policy["cumulative_consumed_group_ids"])):
        raise ValueError("retry must restore the unconsumed parent policy")
    fresh = new_update_attempt(window)
    return {"reload_checkpoint_identity": policy["checkpoint_identity"],
            "reload_adapter_identity": policy["adapter_fingerprint"],
            "reload_native_identity": policy["native_identity"],
            "reload_optimizer_identity": policy["optimizer_identity"], "reload_rng_identity": policy["rng_identity"],
            "discard_uncertain_memory": True, "reuse_group_ids": list(window["ordered_group_ids"]),
            "new_attempt": fresh, "previous_attempt_id": attempt["attempt_id"]}


def reconstruct_consumed_ledger(run, initial, groups, verified_checkpoints, *, window=None, checkpoint_root=None):
    from pathlib import Path
    from .checkpoint import read_verified_checkpoint, validate_checkpoint_manifest, validate_policy, validate_training_run_identity
    from .group import validate_formal_group
    from .training_window import validate_training_window
    validate_training_run_identity(run)
    validate_policy(initial)
    if initial["policy_iteration"] != 0 or initial["run_identity_sha256"] != run["run_identity_sha256"]:
        raise ValueError("ledger requires this run's iteration-zero anchor")
    ledger = {p: {"status": "pending", "group_id": None, "checkpoint_identity": None} for p in run["prompt_ids"]}
    consumed, current = set(), initial
    for checkpoint in verified_checkpoints:
        validate_checkpoint_manifest(checkpoint)
        if (checkpoint_root is None or read_verified_checkpoint(
                Path(checkpoint_root) / f"policy-{checkpoint['policy_iteration']:06d}") != checkpoint):
            raise ValueError("consumption requires immutable published checkpoint, not an attempt/index")
        if checkpoint["parent_policy"] != current or checkpoint["run"]["run_identity_sha256"] != run["run_identity_sha256"]:
            raise ValueError("conflicting/out-of-order checkpoint chain")
        for group in checkpoint["groups"]:
            validate_formal_group(group, committed=True, run=run)
            identity = group["identity"]
            gid, prompt = identity["trajectory_group_id"], identity["prompt_id"]
            if gid in consumed or ledger[prompt]["status"] == "consumed_by_verified_checkpoint":
                raise ValueError("group/prompt consumed twice")
            consumed.add(gid)
            ledger[prompt] = {"status": "consumed_by_verified_checkpoint", "group_id": gid,
                              "checkpoint_identity": checkpoint["checkpoint_manifest_sha256"]}
        current = checkpoint_policy(checkpoint)
    seen, group_states = set(), {}
    for group in groups:
        validate_formal_group(group, committed=True, run=run)
        identity = group["identity"]
        gid, prompt = identity["trajectory_group_id"], identity["prompt_id"]
        if (gid in seen or prompt not in ledger or identity["run_identity_sha256"] != run["run_identity_sha256"]):
            raise ValueError("invalid ledger group")
        seen.add(gid)
        group_states[gid] = {
            "status": "consumed_by_verified_checkpoint" if gid in consumed else "group_committed",
            "prompt_id": prompt, "policy_fingerprint": identity["pre_update_policy_fingerprint"],
            "eligible_for_current_policy": gid not in consumed
            and identity["pre_update_policy_fingerprint"] == current["effective_policy_fingerprint"],
        }
        if gid in consumed:
            if ledger[prompt]["group_id"] != gid:
                raise ValueError("consumption conflict")
            continue
        if ledger[prompt]["status"] == "consumed_by_verified_checkpoint":
            raise ValueError("new attempt for a consumed prompt")
        # Old unconsumed groups stay auditable but cannot update a different policy.
        if identity["pre_update_policy_fingerprint"] == current["effective_policy_fingerprint"]:
            if ledger[prompt]["group_id"] is not None:
                raise ValueError("multiple committed groups for one current prompt")
            ledger[prompt] = {"status": "group_committed", "group_id": gid, "checkpoint_identity": None}
    if window is not None:
        selected = [next(g for g in groups if g["identity"]["trajectory_group_id"] == gid)
                    for gid in window["ordered_group_ids"]]
        validate_training_window(window, run, current, selected)
        for group in selected:
            prompt = group["identity"]["prompt_id"]
            ledger[prompt]["status"] = "assigned_to_window"
            group_states[group["identity"]["trajectory_group_id"]]["status"] = "assigned_to_window"
    return {"policy": current, "prompts": ledger, "groups": group_states, "consumed_group_ids": sorted(consumed)}


def rollback_interrupted_state(state, root, run, attempt, window, *, cpu_fixture=False):
    """Explicit CPU rollback plan, NOT an ordinary updating -> collecting transition.

    S2 must execute the returned model/optimizer/RNG reload before running an update.
    """
    from pathlib import Path
    from .checkpoint import check_seal, recover_formal_run, seal
    from .group import read_formal_group
    from .training_window import validate_training_window
    check_seal(state, "state_sha256")
    if (state["status"] not in {"updating", "checkpointing", "interrupted", "failed"}
            or state["run_identity_sha256"] != run["run_identity_sha256"]
            or state["training_behavior_fingerprint"] != run["training_behavior_fingerprint"]
            or state["policy"]["effective_policy_fingerprint"] != attempt["parent_policy_fingerprint"]
            or read_update_attempt(root, attempt["attempt_id"]) != attempt):
        raise ValueError("rollback requires this run's durable interrupted attempt")
    recovered = recover_formal_run(root, run, cpu_fixture=cpu_fixture)
    groups = [read_formal_group(Path(root) / "groups" / gid) for gid in window["ordered_group_ids"]]
    plan = retry_update_plan(attempt, window, recovered["policy"], verified_checkpoints=recovered["checkpoints"])
    validate_training_window(window, run, recovered["policy"], groups)
    restored = seal({**{k: v for k, v in state.items() if k != "state_sha256"},
                     "status": "ready_to_update", "policy": recovered["policy"],
                     "recovery_reload_required": True,
                     "recovery_parent_checkpoint": recovered["policy"]["checkpoint_identity"]}, "state_sha256")
    return {"state": restored, "retry_plan": plan}
