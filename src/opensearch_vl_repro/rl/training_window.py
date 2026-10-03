"""Formal multi-group window and rank-assignment metadata only (no tensor materialization)."""
from __future__ import annotations

import math

from opensearch_vl_repro.eval_subset import canonical_json_sha256

from .checkpoint import (FORMAL_WEIGHTING, RL_WINDOW_SCHEMA_VERSION, check_seal,
                         require_counter, seal, validate_policy, validate_training_run_identity)
from .group import validate_formal_group


def build_training_window(run, policy, groups, *, window_id):
    validate_training_run_identity(run)
    validate_policy(policy)
    if (not isinstance(window_id, str) or not window_id or not groups
            or policy["run_identity_sha256"] != run["run_identity_sha256"]):
        raise ValueError("window identity/groups required")
    ids, prompts, members = [], [], []
    for group in groups:
        validate_formal_group(group, committed=True)
        identity = group["identity"]
        if (identity["policy_iteration"] != policy["policy_iteration"]
                or identity["pre_update_policy_fingerprint"] != policy["effective_policy_fingerprint"]
                or identity["parent_checkpoint_identity"] != policy["checkpoint_identity"]
                or identity["run_identity_sha256"] != run["run_identity_sha256"]
                or identity["expected_n"] != run["semantics"]["rollout_n"]
                or identity["rollout_config_fingerprint"] != canonical_json_sha256(run["semantics"]["rollout"])):
            raise ValueError("mixed pre-update policy/iteration/rollout group")
        ids.append(identity["trajectory_group_id"])
        prompts.append(identity["prompt_id"])
        members.extend(m["member_id"] for m in group["members"])
    if (len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts)
            or len(set(members)) != len(members)
            or set(ids) & set(policy["cumulative_consumed_group_ids"])
            or not set(prompts) <= set(run["prompt_ids"])):
        raise ValueError("duplicate/consumed/out-of-run window group/member/prompt")
    return seal({"schema_version": RL_WINDOW_SCHEMA_VERSION, "window_id": window_id,
                 "policy_iteration": policy["policy_iteration"],
                 "parent_policy_fingerprint": policy["effective_policy_fingerprint"],
                 "parent_checkpoint_identity": policy["checkpoint_identity"],
                 "run_identity_sha256": run["run_identity_sha256"],
                 "ordered_group_ids": ids,
                 "group_hashes": [g["group_payload_sha256"] for g in groups],
                 "rollout_n": run["semantics"]["rollout_n"], "weighting": FORMAL_WEIGHTING,
                 "expected_optimizer_step": policy["global_optimizer_step"] + 1,
                 "training_behavior_fingerprint": run["training_behavior_fingerprint"]}, "window_sha256")


def validate_training_window(window, run, policy, groups):
    require_counter(window["schema_version"], 1)
    check_seal(window, "window_sha256")
    if build_training_window(run, policy, groups, window_id=window["window_id"]) != window:
        raise ValueError("window content/schema mismatch")


def deterministic_rank_plan(logical_row_ids, world_size):
    require_counter(world_size, 1)
    if (not logical_row_ids or any(not isinstance(r, str) or not r for r in logical_row_ids)
            or len(set(logical_row_ids)) != len(logical_row_ids)):
        raise ValueError("nonempty unique logical rows required")
    count = len(logical_row_ids)
    factor = math.lcm(count, world_size) // count
    physical = list(logical_row_ids) * factor
    return {"logical_count": count, "physical_count": len(physical),
            "world_size": world_size, "replication_factor": factor,
            "local_row_count": len(physical) // world_size,
            "multiplicity": {row: factor for row in logical_row_ids},
            "rank_assignments": [physical[rank::world_size] for rank in range(world_size)]}
