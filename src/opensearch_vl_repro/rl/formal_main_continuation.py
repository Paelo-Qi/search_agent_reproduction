"""Narrow v4/v5 Main -> v7 handoff authority; never rewrite a parent identity.

Only bootstrap weights/state are materialized. Historical manifests form a
sealed prefix proof, not pretend local checkpoints. Final audit reopens parent
evidence read-only; routine recovery uses the child-owned frozen authority.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from pathlib import Path

from . import checkpoint as cp
from . import formal_main_retention as retention
from .run_state import checkpoint_policy
from .group import read_formal_group
from .training_window import expected_window_prompts
from opensearch_vl_repro.eval_subset import canonical_json_sha256 as digest
from opensearch_vl_repro.sft_tool_audit import sha256_file
from opensearch_vl_repro.agent.reliability import SEARCH_BEHAVIOR_VERSION, provider_reliability_semantics

VERSION = "formal-main-continuation-v1"
PARENT_VERSIONS = {"formal-s4-main400-v4", "formal-s4-main400-v5"}
# Exact source delta whitelist, not directories/globs or arbitrary package changes.
SOURCE_DELTA = frozenset("src/opensearch_vl_repro/" + p for p in (
    "agent/reliability.py", "agent/search_providers.py", "agent/search_tools.py",
    "rl/formal_main.py", "rl/formal_main_cli.py", "rl/formal_main_coordinator.py",
    "rl/formal_main_collection.py", "rl/formal_main_update.py", "rl/formal_main_retention.py",
    "rl/formal_main_continuation.py", "rl/run_state.py", "rl/formal_policy_update.py"))
_CAPABILITY = object()


def read_json(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
                                for p in path.parents):
        raise ValueError("continuation authority redirects reads")
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def readonly_parent_lock(root, *, cpu_fixture=False):
    """Existing lock files ONLY, opened rb; never create/write a parent file."""
    with ExitStack() as stack:
        for name in (".coordinator.lock", ".formal.lock", "groups/.publication.lock"):
            path = Path(root) / name
            if not path.exists() and cpu_fixture:
                continue
            if path.is_symlink():
                raise ValueError("parent lock redirects authority")
            stream = stack.enter_context(path.open("rb"))
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes
                class Overlapped(ctypes.Structure):
                    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                for method, args in (("LockFileEx", [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                        wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]),
                        ("UnlockFileEx", [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                        wintypes.DWORD, ctypes.POINTER(Overlapped)])):
                    getattr(kernel, method).argtypes = args
                    getattr(kernel, method).restype = wintypes.BOOL
                handle, offset = msvcrt.get_osfhandle(stream.fileno()), Overlapped()
                if not kernel.LockFileEx(handle, 3, 0, 1, 0, ctypes.byref(offset)):
                    raise OSError("parent active/locked; stop and reconcile parent before handoff")
                stack.callback(kernel.UnlockFileEx, handle, 0, 1, 0, ctypes.byref(offset))
            else:
                import fcntl
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    raise OSError("parent active/locked; stop parent before handoff") from None
                stack.callback(fcntl.flock, stream.fileno(), fcntl.LOCK_UN)
        yield


def semantic_delta(parent, child):
    """Equality everywhere except the enumerated infrastructure boundary."""
    from .formal_main import require_main_run
    require_main_run(parent, _historical=True)
    require_main_run(child)
    if parent["run_id"] == child["run_id"]:
        raise ValueError("continuation must use a NEW child run-id")
    if parent["prompt_ids"] != child["prompt_ids"] or parent["prompt_sources"] != child["prompt_sources"]:
        raise ValueError("continuation membership/order/source changed")
    old, new = copy.deepcopy(parent["semantics"]), copy.deepcopy(child["semantics"])
    if old.get("continuation"):
        raise ValueError("nested continuation is not part of this v1 contract")
    for s in (old, new):
        for k in ("coordinator_version", "search_behavior_version", "provider_reliability", "continuation"):
            s.pop(k, None)
        s["rollout"].pop("behavior_version")
    a, b = old.pop("integration_source_hashes"), new.pop("integration_source_hashes")
    changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    if any(k not in SOURCE_DELTA for k in changed) or old != new:
        raise ValueError("illegal continuation training/software/source semantic delta")
    return changed


def prefix_proof(parent, anchor, checkpoints):
    """Validate exact contiguous consumption from immutable parent manifests."""
    cp.check_seal(anchor, "anchor_sha256")
    cp.require_same_training_run(parent, anchor["run"])
    policy = anchor["initial_policy"]
    if policy != cp.initial_policy(parent, optimizer_identity=policy["optimizer_identity"], rng_identity=policy["rng_identity"]):
        raise ValueError("parent requires original SFT iteration-zero anchor")
    rows, gids, identities = [], [], []
    for step, c in enumerate(checkpoints, 1):
        cp.validate_checkpoint_manifest(c)
        cp.require_same_training_run(parent, c["run"])
        if (c["policy_iteration"] != step or c["global_optimizer_step"] != step
                or c["parent_policy"] != policy or c["eligibility"] != cp.checkpoint_eligibility("main_checkpoint")
                or c["window"]["window_id"] != f"main-window-{step - 1:06d}"):
            raise ValueError("parent prefix chain gap/splice/kind/step mismatch")
        expected = expected_window_prompts(parent, step - 1)
        if [g["identity"]["prompt_id"] for g in c["groups"]] != expected:
            raise ValueError("parent prefix prompt order mismatch")
        for g in c["groups"]:
            gid = g["identity"]["trajectory_group_id"]
            rows.append(dict(prompt_id=g["identity"]["prompt_id"], group_id=gid,
                checkpoint_identity=c["checkpoint_manifest_sha256"], group_payload_sha256=g["group_payload_sha256"]))
            gids.append(gid)
        identities.append(c["checkpoint_manifest_sha256"])
        policy = checkpoint_policy(c)
    n = policy["policy_iteration"]
    if not 1 <= n <= 100 or len(set(gids)) != 4 * n or policy["cumulative_consumed_group_ids"] != gids:
        raise ValueError("invalid inherited consumed group prefix")
    if [r["prompt_id"] for r in rows] != parent["prompt_ids"][:4 * n]:
        raise ValueError("inherited prefix has skipped/duplicated prompts")
    return dict(policy=policy, rows=rows, checkpoint_identities=identities)


def resolve_parent(project, parent_id, *, cpu_fixture=False):
    """Read-only metadata recovery, then FULL latest + heavyweight consumed prefix."""
    from .formal_main import main_paths
    output, _ = main_paths(project, parent_id)
    with readonly_parent_lock(output, cpu_fixture=cpu_fixture):
        return _resolve_locked(project, parent_id, cpu_fixture=cpu_fixture)


def _resolve_locked(project, parent_id, *, cpu_fixture=False):
    from .formal_main import main_paths, recover_main, require_main_update_evidence
    output, _ = main_paths(project, parent_id)
    # Caller holds all parent authority locks across verification/materialization.
    anchor = read_json(output / "identity/run.json")
    parent = anchor["run"]
    if parent["run_id"] != parent_id:
        raise ValueError("parent namespace/run-id mismatch")
    if any((output / "checkpoints").glob(".update-*")):
        raise ValueError("ambiguous unpublished/staging parent update; reconcile old run first")
    value = recover_main(output, parent, cpu_fixture=cpu_fixture, reconcile=False, cleanup=False, _historical=True)
    if any(a["phase"] not in {"verified", "failed"} for a in value["attempts"]):
        raise ValueError("parent has ambiguous/unresolved update")
    proof = prefix_proof(parent, anchor, value["checkpoints"])
    n = proof["policy"]["policy_iteration"]
    directory = output / "checkpoints" / f"policy-{n:06d}"
    selected = cp.read_verified_checkpoint(directory)
    metadata_sha256 = {}
    for c in value["checkpoints"]:
        metadata_sha256[f"checkpoints/policy-{c['policy_iteration']:06d}/checkpoint.json"] = sha256_file(
            output / "checkpoints" / f"policy-{c['policy_iteration']:06d}" / "checkpoint.json")
        retention.check_checkpoint_files(output, parent, c, value["checkpoints"], heavy=True, cpu_fixture=cpu_fixture)
        if not cpu_fixture:
            require_main_update_evidence(output / "checkpoints" / f"policy-{c['policy_iteration']:06d}", c)
        for g in c["groups"]:
            if read_formal_group(output / "groups" / g["identity"]["trajectory_group_id"]) != g:
                raise ValueError("inherited group payload changed")
            gid = g["identity"]["trajectory_group_id"]
            metadata_sha256[f"groups/{gid}/group.json"] = sha256_file(output / "groups" / gid / "group.json")
    return dict(parent_run=parent, parent_anchor=anchor, checkpoints=value["checkpoints"],
        proof=proof, selected=selected, selected_file_sha256=sha256_file(directory / "checkpoint.json"),
        prefix_metadata_sha256=metadata_sha256)


def binding(plan):
    parent, selected, proof = plan["parent_run"], plan["selected"], plan["proof"]
    return dict(version=VERSION, parent_run_id=parent["run_id"], parent_run_identity_sha256=parent["run_identity_sha256"],
        parent_coordinator_version=parent["semantics"]["coordinator_version"],
        parent_policy_iteration=selected["policy_iteration"], global_optimizer_step=selected["global_optimizer_step"],
        parent_checkpoint_identity=selected["checkpoint_manifest_sha256"], parent_checkpoint_file_sha256=plan["selected_file_sha256"],
        parent_effective_policy_fingerprint=proof["policy"]["effective_policy_fingerprint"],
        parent_artifact_roles=selected["artifact_roles"], parent_inventory_sha256=digest(selected["file_sha256"]),
        inherited_prefix_sha256=digest(proof), parent_evidence_sha256=digest(plan))


def inherited_policy(run, parent_policy):
    b = run["semantics"]["continuation"]
    # New child anchor identity explicitly records the cross-run edge. Parent
    # manifest/policy remain untouched, never resealed with a substituted hash.
    return cp.seal(dict(policy_iteration=parent_policy["policy_iteration"],
        global_optimizer_step=parent_policy["global_optimizer_step"], run_identity_sha256=run["run_identity_sha256"],
        parent_checkpoint_identity=parent_policy["checkpoint_identity"],
        checkpoint_identity=digest(dict(continuation=b, child_run_identity=run["run_identity_sha256"])),
        adapter_fingerprint=parent_policy["adapter_fingerprint"], native_identity=parent_policy["native_identity"],
        optimizer_identity=parent_policy["optimizer_identity"], rng_identity=parent_policy["rng_identity"],
        execution_contract=parent_policy["execution_contract"],
        cumulative_consumed_group_ids=parent_policy["cumulative_consumed_group_ids"],
        continuation_version=VERSION, parent_effective_policy_fingerprint=parent_policy["effective_policy_fingerprint"],
        continuation_binding_sha256=digest(b)), "effective_policy_fingerprint")


def validate_receipt(receipt, run, *, cpu_fixture=False):
    cp.check_seal(receipt, "continuation_receipt_sha256")
    cp.validate_training_run_identity(run)
    if (receipt["version"] != VERSION or receipt["child_run_identity_sha256"] != run["run_identity_sha256"]
            or receipt["child_run_id"] != run["run_id"]
            or receipt["evidence_scope"] != ("cpu_fixture" if cpu_fixture else "runtime")):
        raise ValueError("continuation child identity/scope/version mismatch")
    if receipt.get("handoff_stage") != "materialized_requires_live_reload" or receipt["child_semantics"] != run["semantics"]:
        raise ValueError("continuation stage/child semantics differ")
    plan = receipt["parent_evidence"]
    proof = prefix_proof(plan["parent_run"], plan["parent_anchor"], plan["checkpoints"])
    if proof != plan["proof"] or plan["selected"] != plan["checkpoints"][-1]:
        raise ValueError("continuation prefix proof changed")
    if run["semantics"].get("continuation") != binding(plan):
        raise ValueError("frozen continuation binding differs")
    if receipt["allowed_source_delta"] != semantic_delta(plan["parent_run"], run):
        raise ValueError("continuation source delta receipt changed")
    expected = inherited_policy(run, proof["policy"])
    cp.validate_policy(expected)
    if receipt["inherited_policy"] != expected:
        raise ValueError("inherited policy authority changed")
    inventory = {**plan["selected"]["file_sha256"], "checkpoint.json": plan["selected_file_sha256"]}
    if receipt["bootstrap_inventory"] != inventory:
        raise ValueError("continuation bootstrap inventory changed")
    return receipt


def read_authority(output, run, *, cpu_fixture=False, full=False):
    output = Path(output)
    receipt = validate_receipt(read_json(output / "continuation/receipt.json"), run, cpu_fixture=cpu_fixture)
    n = receipt["inherited_policy"]["policy_iteration"]
    directory = output / "continuation/bootstrap" / f"policy-{n:06d}"
    if retention.physical_files(directory) != set(receipt["bootstrap_inventory"]):
        raise ValueError("incomplete continuation bootstrap")
    if cp.read_checkpoint_manifest_only(directory) != receipt["parent_evidence"]["selected"]:
        raise ValueError("child-owned parent checkpoint metadata changed")
    if sha256_file(directory / "checkpoint.json") != receipt["parent_evidence"]["selected_file_sha256"]:
        raise ValueError("parent checkpoint bytes changed")
    if full:
        cp.verify_artifacts(directory, receipt["bootstrap_inventory"])
    return receipt


def load_anchor(output, run, *, cpu_fixture=False):
    receipt = read_authority(output, run, cpu_fixture=cpu_fixture)
    anchor = read_json(Path(output) / "identity/run.json")
    cp.check_seal(anchor, "anchor_sha256")
    cp.require_same_training_run(run, anchor["run"])
    if (anchor["initial_policy"] != receipt["inherited_policy"]
            or anchor.get("continuation_receipt_sha256") != receipt["continuation_receipt_sha256"]
            or anchor["evidence_scope"] != receipt["evidence_scope"]):
        raise ValueError("continuation anchor/receipt mismatch")
    return anchor, receipt


def attach_binding(args, project, semantics, *, cpu_fixture=False):
    """Read-only context creation; resume reads frozen authority, never parent latest."""
    from .formal_main import main_paths
    output, _ = main_paths(project, args.run_id)
    parent_id = getattr(args, "continue_from_run", None)
    if (output / "continuation").exists():
        receipt = read_json(output / "continuation/receipt.json")
        cp.check_seal(receipt, "continuation_receipt_sha256")
        b = binding(receipt["parent_evidence"])
        if parent_id != b["parent_run_id"]:
            raise ValueError("existing child requires the SAME --continue-from-run")
        semantics["continuation"] = b
    elif parent_id:
        if (output / "identity").exists() or parent_id == args.run_id:
            raise ValueError("cannot continue into an existing/unrelated/self Main run")
        semantics["continuation"] = binding(resolve_parent(project, parent_id, cpu_fixture=cpu_fixture))
    elif any(output.glob(".continuation-*")):
        raise ValueError("partial continuation requires explicit parent, never SFT fallback")


def materialize(project, output, run, *, cpu_fixture=False, guard=None):
    """Caller owns CHILD coordinator lock. Atomic receipt publishes after all bytes."""
    from .formal_main import main_paths
    output = Path(output)
    if (output / "continuation").exists():
        receipt = read_authority(output, run, cpu_fixture=cpu_fixture, full=True)
    else:
        if (output / "identity").exists():
            raise ValueError("cannot retrofit continuation into existing Main")
        parent_id = run["semantics"]["continuation"]["parent_run_id"]
        parent_output, _ = main_paths(project, parent_id)
        with readonly_parent_lock(parent_output, cpu_fixture=cpu_fixture):
            # Locks are already held: resolve without nested lock acquisition.
            plan = _resolve_locked(project, parent_id, cpu_fixture=cpu_fixture)
            if binding(plan) != run["semantics"]["continuation"]:
                raise ValueError("parent changed before publication; no child authority published")
            changed = semantic_delta(plan["parent_run"], run)
            n = plan["selected"]["policy_iteration"]
            source = parent_output / "checkpoints" / f"policy-{n:06d}"
            inventory = {**plan["selected"]["file_sha256"], "checkpoint.json": plan["selected_file_sha256"]}
            receipt = cp.seal(dict(version=VERSION, evidence_scope="cpu_fixture" if cpu_fixture else "runtime",
                handoff_stage="materialized_requires_live_reload", child_semantics=run["semantics"],
                child_run_id=run["run_id"], child_run_identity_sha256=run["run_identity_sha256"],
                parent_evidence=plan, inherited_policy=inherited_policy(run, plan["proof"]["policy"]),
                bootstrap_inventory=inventory, allowed_source_delta=changed), "continuation_receipt_sha256")
            validate_receipt(receipt, run, cpu_fixture=cpu_fixture)
            anchor_preview = cp.seal(dict(run=run, initial_policy=receipt["inherited_policy"],
                evidence_scope=receipt["evidence_scope"], continuation_receipt_sha256=receipt["continuation_receipt_sha256"]), "anchor_sha256")
            needed = sum((source / name).stat().st_size for name in inventory) + sum(
                len(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
                for value in (receipt, anchor_preview))
            if guard is None:
                if not cpu_fixture:
                    raise ValueError("production continuation requires existing disk guard")
            else:
                guard(needed)
            staging = output / (".continuation-" + str(uuid.uuid4()))
            target = staging / "bootstrap" / source.name
            target.mkdir(parents=True)
            for name in inventory:
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, destination)
            cp.verify_artifacts(target, inventory)
            cp.publish_directory(staging, output / "continuation", "receipt.json", receipt, cpu_fixture=cpu_fixture)
    # A crash after receipt publication but before anchor publication is idempotent:
    # immutable receipt is authority; never resolve parent again.
    if not (output / "identity").exists():
        staging = output / (".identity-" + str(uuid.uuid4()))
        staging.mkdir()
        anchor = cp.seal(dict(run=run, initial_policy=receipt["inherited_policy"],
            evidence_scope=receipt["evidence_scope"], continuation_receipt_sha256=receipt["continuation_receipt_sha256"]), "anchor_sha256")
        cp.publish_directory(staging, output / "identity", "run.json", anchor, cpu_fixture=cpu_fixture)
    # Also recover a crash immediately AFTER anchor rename, before directory setup.
    for name in ("checkpoints", "groups", "attempts", "merges"):
        (output / name).mkdir(exist_ok=True)
    cp.fsync_directory(output, cpu_fixture=cpu_fixture)
    load_anchor(output, run, cpu_fixture=cpu_fixture)
    return receipt


@dataclass(frozen=True)
class PrefixCapability:
    receipt: dict
    token: object


def prefix_capability(receipt):
    return PrefixCapability(receipt, _CAPABILITY)


def inherited_ledger(capability, run, initial):
    if not isinstance(capability, PrefixCapability) or capability.token is not _CAPABILITY:
        raise ValueError("verified inherited-prefix authority required")
    r = validate_receipt(capability.receipt, run, cpu_fixture=capability.receipt["evidence_scope"] == "cpu_fixture")
    if initial != r["inherited_policy"]:
        raise ValueError("prefix authority does not match initial policy")
    ledger = {p: dict(status="pending", group_id=None, checkpoint_identity=None) for p in run["prompt_ids"]}
    for row in r["parent_evidence"]["proof"]["rows"]:
        ledger[row["prompt_id"]] = dict(status="consumed_by_verified_checkpoint", group_id=row["group_id"],
                                       checkpoint_identity=row["checkpoint_identity"])
    return ledger, set(initial["cumulative_consumed_group_ids"])


@dataclass(frozen=True)
class BootstrapReloadCapability:
    output: Path
    receipt: dict
    token: object


def bootstrap_reload_capability(output, run, policy, *, cpu_fixture=False):
    _, r = load_anchor(output, run, cpu_fixture=cpu_fixture)
    read_authority(output, run, cpu_fixture=cpu_fixture, full=True)
    if policy != r["inherited_policy"]:
        raise ValueError("cross-run reload allowed ONLY for inherited initial policy")
    return BootstrapReloadCapability(Path(output), r, _CAPABILITY)


def authorize_actor_reload(capability, run, policy, directory, *, cpu_fixture=False):
    if not isinstance(capability, BootstrapReloadCapability) or capability.token is not _CAPABILITY:
        raise ValueError("verified continuation bootstrap reload capability required")
    _, r = load_anchor(capability.output, run, cpu_fixture=cpu_fixture)
    if r != capability.receipt or policy != r["inherited_policy"]:
        raise ValueError("stale/foreign continuation reload capability")
    expected = capability.output / "continuation/bootstrap" / f"policy-{policy['policy_iteration']:06d}"
    if Path(directory).resolve() != expected.resolve():
        raise ValueError("continuation reload must use child-owned bootstrap")
    value = cp.read_verified_checkpoint(directory)
    if value != r["parent_evidence"]["selected"]:
        raise ValueError("continuation parent native checkpoint changed")
    return value, r["parent_evidence"]["proof"]["policy"]


def report_fields(receipt, local_windows=0):
    b = binding(receipt["parent_evidence"])
    parent = receipt["parent_evidence"]["parent_run"]["semantics"]
    return dict(continuation=True, continuation_version=VERSION,
        continuation_receipt_sha256=receipt["continuation_receipt_sha256"],
        parent_run_id=b["parent_run_id"], parent_run_identity=b["parent_run_identity_sha256"],
        parent_policy_iteration=b["parent_policy_iteration"], inherited_windows=b["parent_policy_iteration"],
        inherited_prompts=4 * b["parent_policy_iteration"], local_windows=local_windows,
        provider_reliability_transition=dict(old=parent.get("provider_reliability", {
            "serpapi_google_lens": dict(max_attempts=3, backoff_seconds=[1, 2])}),
            old_search_behavior_version=parent.get("search_behavior_version", 2),
            new_search_behavior_version=SEARCH_BEHAVIOR_VERSION,
            new=provider_reliability_semantics()))


def bootstrap_directory(output, receipt):
    return Path(output) / "continuation/bootstrap" / f"policy-{receipt['inherited_policy']['policy_iteration']:06d}"


def publish_live_reload(output, run, rows):
    """Rank0 only, after ALL ranks issue actual S2 reload capabilities + teardown."""
    from .formal_main import require_main_update_evidence
    r = read_authority(output, run, full=True)
    directory = bootstrap_directory(output, r)
    require_main_update_evidence(directory, r["parent_evidence"]["selected"])
    verify_live_rows(directory, r, rows)
    record = cp.seal(dict(version=VERSION, scope="runtime", child_run_identity_sha256=run["run_identity_sha256"],
        continuation_receipt_sha256=r["continuation_receipt_sha256"], per_rank=rows), "activation_sha256")
    destination = Path(output) / "continuation_reload"
    if destination.exists():
        require_active(output, run)
        return
    staging = Path(output) / (".continuation-reload-" + str(uuid.uuid4()))
    staging.mkdir()
    cp.publish_directory(staging, destination, "receipt.json", record)


def verify_live_rows(directory, r, rows):
    if len(rows) != 4 or {row["rank"] for row in rows} != set(range(4)):
        raise ValueError("continuation needs four actual native reload ranks")
    for row in rows:
        cp.check_seal(row, "reload_receipt_sha256")
        name = f"runtime_state_rank_{row['rank']}.json"
        if sha256_file(directory / name) != r["parent_evidence"]["selected"]["file_sha256"][name]:
            raise ValueError("bootstrap bound rank state changed")
        expected = read_json(directory / f"runtime_state_rank_{row['rank']}.json")
        cp.check_seal(expected, "runtime_state_sha256")
        if (row["scope"] != "runtime" or row["world_size"] != 4 or row["policy"] != r["inherited_policy"]
                or any(row.get(k) != expected.get(k) for k in (
                    "parameter_sha256", "optimizer_state_sha256", "native_rng_sha256", "global_optimizer_step"))
                or any(row.get(k) is not True for k in ("model_loaded", "adapter_identity_match",
                    "native_identity_match", "optimizer_identity_match", "rng_identity_match",
                    "global_step_match", "execution_contract_match", "runtime_dropout_match"))):
            raise ValueError("continuation live native/AdamW/RNG/global step differs")


def require_active(output, run):
    if not run["semantics"].get("continuation"):
        return
    r = read_authority(output, run)
    a = read_json(Path(output) / "continuation_reload/receipt.json")
    cp.check_seal(a, "activation_sha256")
    if (a["version"] != VERSION or a["scope"] != "runtime"
            or a["child_run_identity_sha256"] != run["run_identity_sha256"]
            or a["continuation_receipt_sha256"] != r["continuation_receipt_sha256"]
            or len(a["per_rank"]) != 4 or {row["rank"] for row in a["per_rank"]} != set(range(4))):
        raise ValueError("continuation activation identity/ranks mismatch")
    for row in a["per_rank"]:
        cp.check_seal(row, "reload_receipt_sha256")
        if row["policy"] != r["inherited_policy"] or row["scope"] != "runtime":
            raise ValueError("continuation activation inherited policy mismatch")
    verify_live_rows(bootstrap_directory(output, r), r, a["per_rank"])


def final_prefix_audit(project, receipt, run, *, cpu_fixture=False):
    """FULL frozen prefix evidence, not parent latest; later parent progress cannot drift N."""
    from .formal_main import main_paths, require_main_update_evidence
    validate_receipt(receipt, run, cpu_fixture=cpu_fixture)
    plan = receipt["parent_evidence"]
    parent, checkpoints = plan["parent_run"], plan["checkpoints"]
    output, _ = main_paths(project, parent["run_id"])
    with readonly_parent_lock(output, cpu_fixture=cpu_fixture):
        if read_json(output / "identity/run.json") != plan["parent_anchor"]:
            raise ValueError("parent anchor changed")
        for name, sha in plan["prefix_metadata_sha256"].items():
            if sha256_file(output / name) != sha:
                raise ValueError("frozen parent prefix metadata bytes changed")
        for c in checkpoints:
            directory = output / "checkpoints" / f"policy-{c['policy_iteration']:06d}"
            if cp.read_checkpoint_manifest_only(directory) != c:
                raise ValueError("frozen parent checkpoint changed")
            # Selected boundary is FULL in child-owned bootstrap even if a later
            # parent successor legally compacts its original native files.
            if c != checkpoints[-1]:
                retention.check_checkpoint_files(output, parent, c, checkpoints, heavy=True, cpu_fixture=cpu_fixture)
            else:
                compact = retention.receipt_path(output, "compact", c["policy_iteration"])
                if compact.exists():
                    successor = cp.read_checkpoint_manifest_only(output / "checkpoints" /
                        f"policy-{c['policy_iteration'] + 1:06d}")
                    cp.require_same_training_run(parent, successor["run"])
                    retention.check_checkpoint_files(output, parent, c, checkpoints + [successor],
                                                    heavy=True, cpu_fixture=cpu_fixture)
                    if not cpu_fixture:
                        require_main_update_evidence(output / "checkpoints" /
                            f"policy-{successor['policy_iteration']:06d}", successor)
                else:
                    cp.read_verified_checkpoint(directory)
            retention.verified_successor(output, c)
            if not cpu_fixture:
                require_main_update_evidence(directory, c)
            for g in c["groups"]:
                if read_formal_group(output / "groups" / g["identity"]["trajectory_group_id"]) != g:
                    raise ValueError("parent prefix group evidence changed")
    return dict(windows=len(checkpoints), groups=4 * len(checkpoints), members=16 * len(checkpoints),
                prompt_ids=[r["prompt_id"] for r in plan["proof"]["rows"]])
