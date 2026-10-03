"""RL lineage built from the existing verified SFT adapter identity."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
from opensearch_vl_repro.inference.adapter import adapter_identity
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.sft_train_plan import STAGE_NAMES


@dataclass(frozen=True)
class RLLineage:
    schema_version: int
    base_model: str
    base_revision: str
    sft_adapter_fingerprint: str
    sft_adapter_config_fingerprint: str
    sft_checkpoint_metadata_fingerprint: str
    sft_stage: str
    sft_lineage: tuple[str, ...]
    runtime_tool_protocol_version: str
    rl_run_id: str
    rl_adapter_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sft_lineage": list(self.sft_lineage)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RLLineage":
        result = cls(**{**value, "sft_lineage": tuple(value["sft_lineage"])})
        result.validate()
        return result

    def validate(self) -> None:
        if (self.schema_version != 1 or not self.base_model or not self.base_revision
                or not self.rl_run_id
                or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in (
                    self.sft_adapter_fingerprint, self.sft_adapter_config_fingerprint,
                    self.sft_checkpoint_metadata_fingerprint))
                or self.runtime_tool_protocol_version != RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
                or self.sft_stage not in STAGE_NAMES
                or not self.sft_lineage or self.sft_lineage[-1] != self.sft_stage):
            raise ValueError("invalid RL/SFT checkpoint lineage")
        if self.rl_adapter_fingerprint is not None and re.fullmatch(r"[0-9a-f]{64}", self.rl_adapter_fingerprint) is None:
            raise ValueError("invalid RL adapter fingerprint")


def validate_sft_overlap_scope(lineage: RLLineage, sft_shards: list[str]) -> None:
    """The overlap audit must cover exactly the shards seen by this adapter."""
    lineage.validate()
    if sft_shards != list(lineage.sft_lineage):
        raise ValueError("SFT overlap shard scope differs from RL initialization adapter lineage")


def build_rl_run_manifest(lineage: RLLineage, *, config: dict[str, Any],
                          data_manifest_sha256: str | None) -> dict[str, Any]:
    lineage.validate()
    if data_manifest_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", data_manifest_sha256) is None:
        raise ValueError("invalid RL data manifest fingerprint")
    payload = {"schema_version": 1, "lineage": lineage.to_dict(),
               "config_sha256": canonical_json_sha256(config),
               "data_manifest_sha256": data_manifest_sha256}
    payload["run_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def build_rl_lineage(*, config: dict[str, Any], sft_config: dict[str, Any],
                     adapter_path: str | Path, run_id: str) -> RLLineage:
    model = sft_config["model"]
    lora = sft_config["lora"]
    if model["freeze_vision_tower"] is not True or model["freeze_multimodal_projector"] is not True:
        raise ValueError("RL requires SFT frozen vision tower and projector")
    if config["model"]["continue_from_sft_adapter"] is not True:
        raise ValueError("RL cannot start from an empty LoRA")
    identity = adapter_identity(adapter_path, base_model=model["name_or_path"],
                                base_revision=model["revision"])
    adapter_config = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
    if (adapter_config.get("peft_type") != "LORA"
            or adapter_config.get("r") != lora["rank"]
            or adapter_config.get("lora_alpha") != lora["alpha"]
            or float(adapter_config.get("lora_dropout", -1)) != float(lora["dropout"])
            or set(adapter_config.get("target_modules", [])) != set(lora["target_modules"])):
        raise ValueError("RL LoRA parameters differ from SFT adapter")
    metadata = json.loads((Path(adapter_path).parent / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("stage_complete") is not True:
        raise ValueError("RL requires a complete SFT stage checkpoint")
    stage = identity["training_cumulative_stage"]
    if stage != config["model"]["sft_stage"]:
        raise ValueError("RL SFT stage differs from plan")
    lineage = identity["source_checkpoint_lineage"]
    result = RLLineage(1, model["name_or_path"], model["revision"],
                       identity["adapter_fingerprint"], identity["adapter_config_fingerprint"],
                       identity["checkpoint_metadata_fingerprint"], stage, tuple(lineage),
                       RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION, run_id)
    result.validate()
    return result


# Formal control-plane versions are independent of all Gate schemas/versions.
RL_TRAINING_BEHAVIOR_VERSION = 1
RL_RUN_SCHEMA_VERSION = 1
RL_CHECKPOINT_SCHEMA_VERSION = 1
RL_GROUP_SCHEMA_VERSION = 1
RL_WINDOW_SCHEMA_VERSION = 1
FORMAL_WEIGHTING = "per_generation_row_mean_v1"


def require_digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("SHA256 identity required")


def require_counter(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid integer counter")


def require_uuid(value):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError("canonical attempt UUID required")


def seal(value, field):
    """Detached JSON value: callers cannot mutate nested inputs after construction."""
    result = json.loads(json.dumps(value, allow_nan=False))
    result[field] = canonical_json_sha256(result)
    return result


def check_seal(value, field):
    require_digest(value.get(field))
    if value[field] != canonical_json_sha256({k: v for k, v in value.items() if k != field}):
        raise ValueError(f"{field} mismatch")


def build_training_run_identity(run_id, *, semantics, prompt_ids, locators=None):
    required = {"dataset", "base_model", "source_sft", "execution_contract", "rollout",
                "rollout_n", "weighting", "optimizer", "ppo", "world_size", "reward",
                "tool_protocol_version", "image_protocol_version", "integration_source_hashes"}
    if not isinstance(run_id, str) or not run_id or not required <= semantics.keys():
        raise ValueError("incomplete formal run semantics")
    if (not prompt_ids or any(not isinstance(p, str) or not p for p in prompt_ids)
            or len(set(prompt_ids)) != len(prompt_ids)):
        raise ValueError("ordered unique prompt membership required")
    require_digest(semantics["dataset"]["sha256"])
    require_digest(semantics["source_sft"]["adapter_sha256"])
    require_digest(semantics["source_sft"]["metadata_sha256"])
    require_counter(semantics["rollout_n"], 2)
    require_counter(semantics["world_size"], 1)
    if (not isinstance(semantics["reward"], dict)
            or not semantics["reward"].get("version") or not semantics["reward"].get("semantics")
            or not semantics["source_sft"].get("stage") or not semantics["source_sft"].get("lineage")):
        raise ValueError("versioned reward semantics and SFT lineage required")
    if (semantics["weighting"] != FORMAL_WEIGHTING or not semantics["dataset"].get("split")
            or not all(semantics["base_model"].get(k) for k in ("name", "revision"))
            or not all(semantics[k] for k in ("execution_contract", "optimizer", "ppo", "reward",
                                             "tool_protocol_version", "image_protocol_version"))
            or not semantics["rollout"].get("behavior_version")
            or not isinstance(semantics["rollout"].get("config"), dict)
            or not semantics["integration_source_hashes"]):
        raise ValueError("invalid formal training semantics")
    for digest in semantics["integration_source_hashes"].values():
        require_digest(digest)
    # Paths belong ONLY in locators. Avoid accidental machine-dependent identities.
    def reject_paths(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"path", "directory", "locator"} or key.endswith("_path"):
                    raise ValueError("paths must be separated from semantic identity")
                reject_paths(child)
        elif isinstance(item, list):
            for child in item:
                reject_paths(child)
    reject_paths(semantics)
    behavior = {"schema_version": RL_RUN_SCHEMA_VERSION,
                "training_behavior_version": RL_TRAINING_BEHAVIOR_VERSION,
                "semantics": semantics, "prompt_ids": prompt_ids,
                "prompt_membership_sha256": canonical_json_sha256(prompt_ids)}
    result = seal({**behavior, "training_behavior_fingerprint": canonical_json_sha256(behavior),
                   "run_id": run_id}, "run_identity_sha256")
    result["locators"] = json.loads(json.dumps(locators or {}))
    return result


def validate_training_run_identity(run):
    require_counter(run["schema_version"], 1)
    require_counter(run["training_behavior_version"], 1)
    expected = build_training_run_identity(run["run_id"], semantics=run["semantics"],
                                          prompt_ids=run["prompt_ids"], locators=run.get("locators"))
    if expected != run:
        raise ValueError("formal run identity/schema mismatch")


def require_same_training_run(expected, resumed):
    validate_training_run_identity(expected)
    validate_training_run_identity(resumed)
    if expected["run_identity_sha256"] != resumed["run_identity_sha256"]:
        raise ValueError("resume semantics differ")


def initial_policy(run, *, optimizer_identity, rng_identity):
    validate_training_run_identity(run)
    for value in (optimizer_identity, rng_identity):
        require_digest(value)
    return seal({"policy_iteration": 0, "global_optimizer_step": 0,
                 "run_identity_sha256": run["run_identity_sha256"],
                 "parent_checkpoint_identity": None,
                 "checkpoint_identity": canonical_json_sha256({"sft": run["semantics"]["source_sft"],
                                                               "run": run["run_identity_sha256"]}),
                 "adapter_fingerprint": run["semantics"]["source_sft"]["adapter_sha256"],
                 "native_identity": canonical_json_sha256({"base": run["semantics"]["base_model"],
                                                            "sft": run["semantics"]["source_sft"]}),
                 "execution_contract": run["semantics"]["execution_contract"],
                 "optimizer_identity": optimizer_identity, "rng_identity": rng_identity,
                 "cumulative_consumed_group_ids": []}, "effective_policy_fingerprint")


def validate_policy(policy):
    check_seal(policy, "effective_policy_fingerprint")
    for key in ("policy_iteration", "global_optimizer_step"):
        require_counter(policy[key])
    if policy["policy_iteration"] != policy["global_optimizer_step"]:
        raise ValueError("one verified optimizer step per policy iteration required")
    for key in ("run_identity_sha256", "checkpoint_identity", "adapter_fingerprint",
                "native_identity", "optimizer_identity", "rng_identity"):
        require_digest(policy[key])
    ids = policy["cumulative_consumed_group_ids"]
    if len(ids) != len(set(ids)) or any(not isinstance(g, str) or not g for g in ids):
        raise ValueError("duplicate/invalid consumed group")
    if (not policy["execution_contract"] or policy["policy_iteration"] == 0
            and (policy["parent_checkpoint_identity"] is not None or ids)):
        raise ValueError("invalid initial policy")
    if policy["policy_iteration"]:
        require_digest(policy["parent_checkpoint_identity"])
        if not ids:
            raise ValueError("verified updated policy requires consumed groups")


def fsync_directory(path, *, cpu_fixture=False):
    """Linux production requires directory fsync; Windows fixtures make NO such claim."""
    if os.name == "nt":
        if not cpu_fixture:
            raise OSError("production directory fsync requires POSIX; Windows supports CPU fixtures only")
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value, *, cpu_fixture=False):
    """Replace a derived index atomically; immutable records use exclusive destinations."""
    from opensearch_vl_repro.rl.actor_gate import atomic_json
    path = Path(path)
    atomic_json(path, value)
    fsync_directory(path.parent, cpu_fixture=cpu_fixture)


def artifact_inventory(directory, *, exclude=()):
    from opensearch_vl_repro.sft_tool_audit import sha256_file
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("artifact symlinks forbidden")
        if path.is_file() and path.relative_to(directory).as_posix() not in exclude:
            files[path.relative_to(directory).as_posix()] = sha256_file(path)
    return files


def verify_artifacts(directory, inventory, *, exclude=()):
    if not inventory or artifact_inventory(directory, exclude=exclude) != inventory:
        raise ValueError("artifact inventory mismatch")
    for digest in inventory.values():
        require_digest(digest)


def publish_directory(staging, destination, manifest_name, manifest, *, cpu_fixture=False):
    """Manifest is written last IN staging; rename publishes the whole immutable record.

    Caller must hold the run lock. A post-rename crash is recovered from manifests,
    never by retrying the optimizer based on a stale latest.json.
    """
    staging, destination = Path(staging), Path(destination)
    if (staging.parent.resolve() != destination.parent.resolve() or not staging.name.startswith(".")
            or staging.is_symlink() or destination.exists() or not staging.is_dir()
            or (staging / manifest_name).exists()):
        raise ValueError("exclusive same-parent staging/publication required")
    # Detect unsupported durability BEFORE publication on Windows.
    fsync_directory(staging.parent, cpu_fixture=cpu_fixture)
    artifact_inventory(staging)
    for path in staging.rglob("*"):
        if path.is_file():
            with path.open("rb+") as stream:
                os.fsync(stream.fileno())
    durable_json(staging / manifest_name, manifest, cpu_fixture=cpu_fixture)
    for path in sorted((p for p in staging.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        fsync_directory(path, cpu_fixture=cpu_fixture)
    fsync_directory(staging, cpu_fixture=cpu_fixture)
    os.rename(staging, destination)
    fsync_directory(destination.parent, cpu_fixture=cpu_fixture)


def checkpoint_eligibility(kind):
    if kind not in {"gate_artifact", "smoke_continuation", "smoke_final", "main_checkpoint"}:
        raise ValueError("unknown checkpoint eligibility kind")
    return {"kind": kind, "same_run_resume": kind != "gate_artifact",
            "eligible_for_main_init": False}


def build_checkpoint_manifest(run, parent_policy, groups, window, attempt, reward_window, *,
                              artifact_roles, file_sha256, kind, reload_evidence, cpu_fixture=False):
    from .run_state import validate_update_attempt
    from .training_window import validate_training_window
    validate_training_window(window, run, parent_policy, groups)
    validate_update_attempt(attempt)
    check_seal(reward_window, "reward_window_sha256")
    if (attempt["phase"] != "checkpoint_staging" or attempt["window_sha256"] != window["window_sha256"]
            or attempt["parent_checkpoint_identity"] != parent_policy["checkpoint_identity"]
            or attempt["parent_policy_fingerprint"] != parent_policy["effective_policy_fingerprint"]
            or attempt["run_identity_sha256"] != run["run_identity_sha256"]
            or attempt["expected_optimizer_step"] != window["expected_optimizer_step"]):
        raise ValueError("checkpoint update attempt mismatch")
    if reward_window["window_sha256"] != window["window_sha256"] or reward_window["status"] != "signal":
        raise ValueError("zero-signal/foreign window cannot authorize an optimizer checkpoint")
    if kind == "gate_artifact":
        raise ValueError("Gate artifacts cannot enter formal checkpoint chain")
    if set(artifact_roles) != {"adapter", "native", "optimizer", "rng"} or not file_sha256:
        raise ValueError("complete checkpoint artifact roles required")
    for digest in list(artifact_roles.values()) + list(file_sha256.values()):
        require_digest(digest)
    if not set(artifact_roles.values()) <= set(file_sha256.values()):
        raise ValueError("checkpoint role hashes not bound to files")
    scope = "cpu_fixture" if cpu_fixture else "runtime"
    if (reload_evidence.get("scope") != scope
            or reload_evidence.get("reloaded_artifact_roles") != artifact_roles
            or any(reload_evidence.get(k) is not True for k in (
                "adapter_reloaded", "native_reloaded", "optimizer_reloaded", "rng_reloaded",
                "execution_contract_verified"))):
        raise ValueError("complete matching reload verification evidence required")
    if not cpu_fixture and (reward_window.get("test_estimator_injected") is not False
                            or any(g.get("evidence_scope") != "runtime" for g in groups)):
        raise ValueError("CPU fixture evidence cannot authorize runtime checkpoint")
    # Recheck raw/final reward rows: a resealed foreign/forged assembly is not sufficient.
    import math
    from .reward import clamp_fatal_advantages
    expected_rows = [(g["identity"]["trajectory_group_id"], m) for g in groups
                     for m in sorted(g["members"], key=lambda member: member["rollout_index"])]
    rows = reward_window["rows"]
    if len(rows) != len(expected_rows) or reward_window.get("estimator") != "official_verl_rloo":
        raise ValueError("reward assembly identity mismatch")
    for row, (gid, member) in zip(rows, expected_rows):
        same_group = [m["reward"]["total"] for g_id, m in expected_rows if g_id == gid]
        reward = float(member["reward"]["total"])
        raw = reward - (math.fsum(same_group) - reward) / (len(same_group) - 1)
        final = clamp_fatal_advantages([raw], [member["fatal"]])[0]
        if (row["group_id"] != gid or row["member_id"] != member["member_id"]
                or row["rollout_index"] != member["rollout_index"] or row["fatal"] is not member["fatal"]
                or row["reward"] != reward
                or any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) for k in (
                    "raw_advantage", "final_advantage"))
                or not math.isclose(row["raw_advantage"], raw, abs_tol=1e-12)
                or not math.isclose(row["final_advantage"], final, abs_tol=1e-12)):
            raise ValueError("reward/fatal/group assembly mismatch")
    if all(row["final_advantage"] == 0 for row in rows):
        raise ValueError("zero signal cannot claim a verified update")
    return seal({"schema_version": RL_CHECKPOINT_SCHEMA_VERSION, "verified": True,
                 "evidence_scope": scope, "run": run, "parent_policy": parent_policy,
                 "parent_checkpoint_identity": parent_policy["checkpoint_identity"],
                 "policy_iteration": parent_policy["policy_iteration"] + 1,
                 "global_optimizer_step": window["expected_optimizer_step"],
                 "source_lineage": run["semantics"]["source_sft"],
                 "execution_contract": run["semantics"]["execution_contract"],
                 "training_behavior_fingerprint": run["training_behavior_fingerprint"],
                 "artifact_roles": artifact_roles, "file_sha256": file_sha256,
                 "groups": groups, "consumed_group_ids": window["ordered_group_ids"],
                 "consumed_group_hashes": window["group_hashes"], "window": window,
                 "update_attempt": attempt, "reward_window": reward_window,
                 "eligibility": checkpoint_eligibility(kind), "reload_evidence": reload_evidence},
                "checkpoint_manifest_sha256")


def validate_checkpoint_manifest(manifest):
    require_counter(manifest["schema_version"], 1)
    check_seal(manifest, "checkpoint_manifest_sha256")
    expected = build_checkpoint_manifest(
        manifest["run"], manifest["parent_policy"], manifest["groups"], manifest["window"],
        manifest["update_attempt"], manifest["reward_window"], artifact_roles=manifest["artifact_roles"],
        file_sha256=manifest["file_sha256"], kind=manifest["eligibility"]["kind"],
        reload_evidence=manifest["reload_evidence"], cpu_fixture=manifest["evidence_scope"] == "cpu_fixture")
    if expected != manifest:
        raise ValueError("verified checkpoint schema mismatch")


def _formal_root(root):
    root = Path(root).resolve()
    if any(part.lower().startswith("rl_gate") for part in root.parts):
        raise ValueError("formal outputs must be physically isolated from Gate outputs")
    return root


def initialize_formal_run(root, run, policy, *, cpu_fixture=False):
    """CPU metadata bootstrap only; this does not create any trainer or real weights."""
    from .group import run_lock
    validate_training_run_identity(run)
    validate_policy(policy)
    if policy != initial_policy(run, optimizer_identity=policy["optimizer_identity"], rng_identity=policy["rng_identity"]):
        raise ValueError("formal run must start from source SFT iteration zero")
    root = _formal_root(root)
    with run_lock(root / ".formal.lock"):
        staging = root / f".identity-{uuid.uuid4()}"
        staging.mkdir()
        payload = seal({"run": run, "initial_policy": policy,
                        "evidence_scope": "cpu_fixture" if cpu_fixture else "runtime"}, "anchor_sha256")
        publish_directory(staging, root / "identity", "run.json", payload, cpu_fixture=cpu_fixture)
        for name in ("checkpoints", "groups", "attempts"):
            (root / name).mkdir(exist_ok=True)
        fsync_directory(root, cpu_fixture=cpu_fixture)
    return payload


def _load_anchor(root, run):
    value = json.loads((root / "identity" / "run.json").read_text(encoding="utf-8"))
    check_seal(value, "anchor_sha256")
    require_same_training_run(run, value["run"])
    policy = value["initial_policy"]
    if policy != initial_policy(run, optimizer_identity=policy["optimizer_identity"], rng_identity=policy["rng_identity"]):
        raise ValueError("invalid SFT anchor")
    return value


def read_verified_checkpoint(directory):
    directory = Path(directory)
    if directory.name.startswith(".") or directory.is_symlink():
        raise ValueError("staging is not a verified checkpoint")
    value = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    validate_checkpoint_manifest(value)
    if directory.name != f"policy-{value['policy_iteration']:06d}":
        raise ValueError("checkpoint directory/iteration mismatch")
    verify_artifacts(directory, value["file_sha256"], exclude=("checkpoint.json",))
    return value


def _recover_formal_run(root, run):
    from .group import read_formal_group
    from .run_state import reconstruct_consumed_ledger
    anchor = _load_anchor(root, run)
    checkpoints = [read_verified_checkpoint(p) for p in sorted((root / "checkpoints").glob("*"))
                   if not p.name.startswith(".")]
    groups = [read_formal_group(p) for p in sorted((root / "groups").glob("*"))
              if not p.name.startswith(".")]
    group_hashes = {g["identity"]["trajectory_group_id"]: g["group_payload_sha256"] for g in groups}
    for checkpoint in checkpoints:
        if checkpoint["evidence_scope"] != anchor["evidence_scope"]:
            raise ValueError("fixture/runtime evidence scopes mixed")
        for gid, digest in zip(checkpoint["consumed_group_ids"], checkpoint["consumed_group_hashes"]):
            if group_hashes.get(gid) != digest:
                raise ValueError("checkpoint references missing/changed committed group")
    ledger = reconstruct_consumed_ledger(run, anchor["initial_policy"], groups, checkpoints,
                                         checkpoint_root=root / "checkpoints")
    return {"checkpoints": checkpoints, "ledger": ledger, "policy": ledger["policy"]}


def recover_formal_run(root, run, *, cpu_fixture=False):
    """Ignore stale/missing latest/state, reconstruct only from immutable receipts."""
    from .group import run_lock
    root = _formal_root(root)
    with run_lock(root / ".formal.lock"):
        recovered = _recover_formal_run(root, run)
        durable_json(root / "latest.json", {"policy": recovered["policy"]}, cpu_fixture=cpu_fixture)
        durable_json(root / "state.json", recovered["ledger"], cpu_fixture=cpu_fixture)
    return recovered


def commit_verified_checkpoint(root, staging, manifest, *, cpu_fixture=False):
    """Commit point is directory rename, NOT an attempt flag or latest/state write.

    Failures before rename leave only hidden staging. Failures after rename must
    roll forward through recover_formal_run, never rerun the already-committed step.
    """
    from .group import read_formal_group, run_lock
    from .run_state import checkpoint_policy, read_update_attempt
    root, staging = _formal_root(root), Path(staging)
    validate_checkpoint_manifest(manifest)
    if manifest["evidence_scope"] != ("cpu_fixture" if cpu_fixture else "runtime"):
        raise ValueError("checkpoint publication evidence scope mismatch")
    with run_lock(root / ".formal.lock"):
        recovered = _recover_formal_run(root, manifest["run"])
        if (recovered["policy"] != manifest["parent_policy"]
                or _load_anchor(root, manifest["run"])["evidence_scope"] != manifest["evidence_scope"]):
            raise ValueError("duplicate/conflicting/stale-parent checkpoint successor")
        consumed_prompts = {p for p, entry in recovered["ledger"]["prompts"].items()
                            if entry["status"] == "consumed_by_verified_checkpoint"}
        if consumed_prompts & {g["identity"]["prompt_id"] for g in manifest["groups"]}:
            raise ValueError("prompt already consumed by verified checkpoint")
        # Validate the next policy before any immutable publication.
        validate_policy(checkpoint_policy(manifest))
        for group in manifest["groups"]:
            gid = group["identity"]["trajectory_group_id"]
            if read_formal_group(root / "groups" / gid) != group:
                raise ValueError("checkpoint group is not durably committed")
        attempt = manifest["update_attempt"]
        if read_update_attempt(root, attempt["attempt_id"]) != attempt:
            raise ValueError("checkpoint requires current durable checkpoint-staging attempt")
        verify_artifacts(staging, manifest["file_sha256"])
        destination = root / "checkpoints" / f"policy-{manifest['policy_iteration']:06d}"
        publish_directory(staging, destination, "checkpoint.json", manifest, cpu_fixture=cpu_fixture)
        recovered = _recover_formal_run(root, manifest["run"])
        durable_json(root / "latest.json", {"policy": recovered["policy"]}, cpu_fixture=cpu_fixture)
        durable_json(root / "state.json", recovered["ledger"], cpu_fixture=cpu_fixture)
    return manifest
