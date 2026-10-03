"""Runtime-only, single-writer, whole-group publication. No prepared-data writes."""
from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl.actor_gate import atomic_json
from opensearch_vl_repro.sft_tool_audit import sha256_file


def group_identity(*, prompt_id, policy_fingerprint, rollout_fingerprint, attempt, context):
    if not all(isinstance(x, str) and x for x in (prompt_id, policy_fingerprint, rollout_fingerprint, attempt)):
        raise ValueError("group identity requires prompt/policy/config/attempt")
    value = dict(prompt_id=prompt_id, pre_update_policy_fingerprint=policy_fingerprint,
                 rollout_config_fingerprint=rollout_fingerprint, collection_attempt=attempt, context=context)
    return {**value, "trajectory_group_id": canonical_json_sha256(value)}


def validate_group(group):
    _validate_group_members(group, 2)


def _validate_group_members(group, expected_n):
    identity = group["identity"]
    if identity["trajectory_group_id"] != canonical_json_sha256({k: v for k, v in identity.items() if k != "trajectory_group_id"}):
        raise ValueError("group identity hash mismatch")
    members = group["members"]
    if len(members) != expected_n or {m["rollout_index"] for m in members} != set(range(expected_n)):
        raise ValueError(f"exactly one complete n={expected_n} group required")
    for member in members:
        if (member["identity"] != identity or member.get("complete") is not True
                or not member.get("steps") or not member.get("reward")):
            raise ValueError("incomplete/mixed-policy/mixed-attempt group")
        reward = member["reward"]
        if any(isinstance(reward.get(k), bool) or not isinstance(reward.get(k), (int, float))
               or not math.isfinite(reward[k]) or not 0 <= reward[k] <= 1
               for k in ("format", "accuracy", "query", "total")):
            raise ValueError("finite composed group rewards required")
        if not math.isclose(reward["total"], reward["format"] * (.8 * reward["accuracy"] + .2 * reward["query"]), abs_tol=1e-12):
            raise ValueError("group reward formula changed")
        for step in member["steps"]:
            if (not step["prompt_ids"] or not step["response_ids"]
                    or len(step["response_ids"]) != len(step["logprobs"])
                    or step["info"].get("token_origin") != "vllm.RequestOutput"
                    or step["info"].get("logprobs_mode") != "processed_logprobs"
                    or step["model_output"].get("logprobs") != step["logprobs"]
                    or step["model_output"]["completion_ids"] != step["response_ids"]
                    or step["model_output"]["prompt_ids"] != step["prompt_ids"]):
                raise ValueError("invalid actual rLLM token fields")
            if (any(type(token) is not int or token < 0 for token in step["prompt_ids"] + step["response_ids"])
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                           or value > 1e-6 for value in step["logprobs"])):
                raise ValueError("invalid actual token IDs/logprobs")


def publish_group(staging: Path, destination: Path, group):
    validate_group(group)
    if destination.exists():
        raise FileExistsError("committed group cannot be overwritten")
    files = {p.relative_to(staging).as_posix(): sha256_file(p) for p in sorted(staging.rglob("*")) if p.is_file()}
    if not files:
        raise ValueError("group multimodal artifacts missing")
    if any(step["multimodal_file"] not in files for m in group["members"] for step in m["steps"]):
        raise ValueError("group step multimodal file missing")
    payload = {**group, "file_sha256": files, "formal_rl_initialization_allowed": False}
    payload["group_payload_sha256"] = canonical_json_sha256(payload)
    atomic_json(staging / "group.json", payload)
    os.replace(staging, destination)  # same filesystem: ONE publication point
    return payload


def read_group(directory: Path):
    value = json.loads((directory / "group.json").read_text(encoding="utf-8"))
    if value.get("group_payload_sha256") != canonical_json_sha256({k: v for k, v in value.items() if k != "group_payload_sha256"}):
        raise ValueError("committed group payload changed")
    validate_group(value)
    if value.get("formal_rl_initialization_allowed") is not False:
        raise ValueError("Gate group cannot initialize formal RL")
    if any(step["multimodal_file"] not in value["file_sha256"] for m in value["members"] for step in m["steps"]):
        raise ValueError("group step multimodal file not checksum-bound")
    for name, digest in value["file_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != digest:
            raise ValueError("committed group artifact changed")
    return value


@contextmanager
def run_lock(path: Path):
    """OS advisory lock, released on process death; no stale-lock bypass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def formal_group_identity(run, policy, *, prompt_id, source_identity, attempt_id, attempt_index):
    from .checkpoint import (RL_GROUP_SCHEMA_VERSION, require_counter, require_digest,
                             require_uuid, source_identity_for_prompt, validate_policy, validate_training_run_identity)
    validate_training_run_identity(run)
    validate_policy(policy)
    require_uuid(attempt_id)
    require_counter(attempt_index)
    require_digest(source_identity)
    if prompt_id not in run["prompt_ids"] or policy["run_identity_sha256"] != run["run_identity_sha256"]:
        raise ValueError("prompt/policy outside formal run")
    if source_identity != source_identity_for_prompt(run, prompt_id):
        raise ValueError("source identity differs from authoritative run membership")
    identity = {"schema_version": RL_GROUP_SCHEMA_VERSION, "prompt_id": prompt_id,
                "source_identity": source_identity, "run_identity_sha256": run["run_identity_sha256"],
                "policy_iteration": policy["policy_iteration"],
                "parent_checkpoint_identity": policy["checkpoint_identity"],
                "pre_update_policy_fingerprint": policy["effective_policy_fingerprint"],
                "rollout_config_fingerprint": canonical_json_sha256(run["semantics"]["rollout"]),
                "collection_attempt": attempt_id, "collection_attempt_index": attempt_index,
                "expected_n": run["semantics"]["rollout_n"]}
    return {**identity, "trajectory_group_id": canonical_json_sha256(identity)}


def validate_formal_group(group, *, committed=False, run=None):
    from .checkpoint import (RL_GROUP_SCHEMA_VERSION, check_seal, require_counter,
                             require_digest, require_uuid, source_identity_for_prompt)
    identity = group["identity"]
    require_counter(identity["schema_version"], 1)
    if identity["schema_version"] != RL_GROUP_SCHEMA_VERSION:
        raise ValueError("formal group schema mismatch")
    require_counter(identity["expected_n"], 2)
    require_counter(identity["policy_iteration"])
    require_counter(identity["collection_attempt_index"])
    require_uuid(identity["collection_attempt"])
    for key in ("source_identity", "run_identity_sha256", "parent_checkpoint_identity",
                "pre_update_policy_fingerprint", "rollout_config_fingerprint"):
        require_digest(identity[key])
    if not isinstance(identity["prompt_id"], str) or not identity["prompt_id"]:
        raise ValueError("missing prompt identity")
    if run is not None and (
            identity["source_identity"] != source_identity_for_prompt(run, identity["prompt_id"])
            or identity["run_identity_sha256"] != run["run_identity_sha256"]):
        raise ValueError("group source identity differs from authoritative run membership")
    _validate_group_members(group, identity["expected_n"])
    members = group["members"]
    if any(type(m["rollout_index"]) is not int for m in members):
        raise ValueError("formal rollout indices must be integers")
    ids = [m.get("member_id") for m in members]
    trajectories = [m.get("trajectory_file") for m in members]
    if (any(not isinstance(m, str) or not m for m in ids) or len(set(ids)) != len(ids)
            or len(set(trajectories)) != len(trajectories)
            or any(type(m.get("fatal")) is not bool or not isinstance(m.get("trajectory_file"), str)
                   or not m["trajectory_file"] for m in members)):
        raise ValueError("unique complete trajectory identities/fatal flags required")
    if committed:
        check_seal(group, "group_payload_sha256")
        if group.get("committed") is not True or not group.get("file_sha256"):
            raise ValueError("uncommitted formal group")
        required = {m["trajectory_file"] for m in members}
        required.update(s["multimodal_file"] for m in members for s in m["steps"])
        if not required <= group["file_sha256"].keys():
            raise ValueError("member artifact hashes missing")
        for digest in group["file_sha256"].values():
            require_digest(digest)


def publish_formal_group(staging, destination, group, *, cpu_fixture=False):
    from .checkpoint import _formal_root, artifact_inventory, publish_directory, seal
    staging, destination = Path(staging), Path(destination)
    _formal_root(destination)
    validate_formal_group(group)
    if destination.name != group["identity"]["trajectory_group_id"]:
        raise ValueError("formal group destination must match immutable identity")
    with run_lock(destination.parent / ".publication.lock"):
        payload = seal({**group, "committed": True, "file_sha256": artifact_inventory(staging),
                        "evidence_scope": "cpu_fixture" if cpu_fixture else "runtime"}, "group_payload_sha256")
        validate_formal_group(payload, committed=True)
        publish_directory(staging, destination, "group.json", payload, cpu_fixture=cpu_fixture)
    return payload


def read_formal_group(directory):
    from .checkpoint import verify_artifacts
    directory = Path(directory)
    if directory.name.startswith(".") or directory.is_symlink():
        raise ValueError("staging is not a committed group")
    value = json.loads((directory / "group.json").read_text(encoding="utf-8"))
    validate_formal_group(value, committed=True)
    if directory.name != value["identity"]["trajectory_group_id"]:
        raise ValueError("committed group directory identity mismatch")
    verify_artifacts(directory, value["file_sha256"], exclude=("group.json",))
    return value
