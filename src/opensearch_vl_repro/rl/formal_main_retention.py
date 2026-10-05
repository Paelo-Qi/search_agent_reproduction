"""S4 role-driven deletion, durable authorization, bounded metadata recovery.

Receipts live outside immutable checkpoint inventories. No broad delete glob,
no deletion of source SFT, and no receipt can authorize deleting a latest policy.
"""
from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path

from . import checkpoint as cp
from .run_state import read_update_attempt
from .training_window import build_training_window

VERSION = "formal-s4-retention-v1"
MILESTONES = [25, 50, 75, 100]
CONTRACT = dict(version=VERSION, full_milestones=MILESTONES, keep_latest_full=True,
                removable_roles=["native", "optimizer", "rng"])
GIB = 1024 ** 3


def receipt_path(root, kind, iteration):
    return Path(root) / "retention" / f"{kind}-{iteration:06d}" / "receipt.json"


def read_receipt(path):
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("retention receipt redirects authority")
    value = json.loads(path.read_text(encoding="utf-8"))
    cp.check_seal(value, "receipt_sha256")
    if value.get("retention_version") != VERSION:
        raise ValueError("retention version mismatch")
    return value


def authorize(root, kind, iteration, value, *, cpu_fixture=False):
    path = receipt_path(root, kind, iteration)
    value = cp.seal(value, "receipt_sha256")
    if path.exists():
        if read_receipt(path) != value:
            raise ValueError("conflicting immutable deletion authorization")
        return value
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent.parent / (".authorization-" + str(uuid.uuid4()))
    staging.mkdir()
    cp.publish_directory(staging, path.parent, "receipt.json", value, cpu_fixture=cpu_fixture)
    return value


def physical_files(directory, *, exclude=()):
    """Names only: intentionally no historical multi-GB byte hashing."""
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("artifact directory redirects deletion")
    result = set()
    for p in directory.rglob("*"):
        if p.is_symlink() or not p.resolve().is_relative_to(directory.resolve()):
            raise ValueError("artifact symlinks forbidden")
        if p.is_file():
            name = p.relative_to(directory).as_posix()
            if name not in exclude:
                result.add(name)
    return result


def delete_authorized(directory, files, *, remove_directory=False, cpu_fixture=False):
    """Exact previously validated inventory, never a glob or user input locator."""
    directory = Path(directory)
    for name in files:
        target = directory / name
        if (target.is_symlink() or not target.resolve().is_relative_to(directory.resolve())
                or target == directory):
            raise ValueError("unsafe authorized deletion target")
        if target.exists():
            target.unlink()
            cp.fsync_directory(target.parent, cpu_fixture=cpu_fixture)
    # Only prune now-empty directories inside the exact authorized artifact tree.
    for sub in sorted((p for p in directory.rglob("*") if p.is_dir()),
                      key=lambda p: len(p.parts), reverse=True):
        if not any(sub.iterdir()):
            sub.rmdir()
            cp.fsync_directory(sub.parent, cpu_fixture=cpu_fixture)
    if remove_directory:
        directory.rmdir()  # fails if ANY unlisted content remains
    cp.fsync_directory(directory.parent, cpu_fixture=cpu_fixture)


def compaction_value(run, checkpoint, successor, latest):
    step = checkpoint["policy_iteration"]
    if (step >= latest or step in MILESTONES
            or successor["policy_iteration"] != step + 1
            or successor["parent_checkpoint_identity"] != checkpoint["checkpoint_manifest_sha256"]
            or checkpoint["eligibility"] != cp.checkpoint_eligibility("main_checkpoint")
            or successor["run"]["run_identity_sha256"] != run["run_identity_sha256"]
            or checkpoint["run"]["run_identity_sha256"] != run["run_identity_sha256"]):
        raise ValueError("compaction requires exact verified successor; current/milestone protected")
    roles = checkpoint["artifact_role_files"]
    removable = {r: roles[r] for r in CONTRACT["removable_roles"]}
    retained = {r: files for r, files in roles.items() if r not in removable}
    return dict(kind="checkpoint-compaction", retention_version=VERSION,
        run_identity_sha256=run["run_identity_sha256"], policy_iteration=step,
        checkpoint_manifest_sha256=checkpoint["checkpoint_manifest_sha256"],
        successor_manifest_sha256=successor["checkpoint_manifest_sha256"],
        original_file_sha256=checkpoint["file_sha256"], removable_role_files=removable,
        retained_role_files=retained)


def verified_successor(root, successor):
    attempt = read_update_attempt(root, successor["update_attempt"]["attempt_id"])
    if (attempt["phase"] != "verified"
            or attempt["verified_checkpoint_identity"] != successor["checkpoint_manifest_sha256"]
            or attempt["previous_event_sha256"] != successor["update_attempt"]["attempt_event_sha256"]):
        raise ValueError("compaction requires durable verified successor attempt")


def check_checkpoint_files(root, run, checkpoint, checkpoints, *, heavy=False,
                           finish_cleanup=False, cpu_fixture=False):
    step = checkpoint["policy_iteration"]
    directory = Path(root) / "checkpoints" / f"policy-{step:06d}"
    path = receipt_path(root, "compact", step)
    files = checkpoint["file_sha256"]
    if path.exists():
        latest = max(c["policy_iteration"] for c in checkpoints)
        if step >= latest:
            raise ValueError("latest checkpoint cannot have compaction authorization")
        successors = [c for c in checkpoints if c["policy_iteration"] == step + 1]
        if len(successors) != 1:
            raise ValueError("compaction requires unique absolute-step successor")
        successor = successors[0]
        verified_successor(root, successor)
        value = read_receipt(path)
        expected = cp.seal(compaction_value(run, checkpoint, successor, latest), "receipt_sha256")
        if value != expected:
            raise ValueError("compaction authorization mismatch")
        removable = {n: sha for role in value["removable_role_files"].values() for n, sha in role.items()}
        files = {n: sha for n, sha in checkpoint["file_sha256"].items() if n not in removable}
        actual = physical_files(directory, exclude=("checkpoint.json",))
        if not set(files) <= actual or not actual <= set(files) | set(removable):
            raise ValueError("unauthorized missing/extra historical artifact")
        if finish_cleanup:
            delete_authorized(directory, removable, cpu_fixture=cpu_fixture)
        elif heavy and actual != set(files):
            raise ValueError("authorized compaction cleanup incomplete")
    elif physical_files(directory, exclude=("checkpoint.json",)) != set(files):
        raise ValueError("missing historical artifact without compaction authorization")
    if heavy:
        cp.verify_artifacts(directory, files, exclude=("checkpoint.json",))


def compact_history(root, run, checkpoints, *, cpu_fixture=False):
    for checkpoint, successor in zip(checkpoints, checkpoints[1:]):
        step = checkpoint["policy_iteration"]
        if step in MILESTONES:
            continue
        verified_successor(root, successor)
        check_checkpoint_files(root, run, checkpoint, checkpoints, cpu_fixture=cpu_fixture)
        authorize(root, "compact", step, compaction_value(run, checkpoint, successor, checkpoints[-1]["policy_iteration"]),
                  cpu_fixture=cpu_fixture)
        check_checkpoint_files(root, run, checkpoint, checkpoints, finish_cleanup=True, cpu_fixture=cpu_fixture)


def retirement_value(run, policy, groups, merge, *, merge_manifest_sha256):
    cp.require_digest(merge_manifest_sha256)
    window = build_training_window(run, policy, groups, window_id=f"main-window-{policy['policy_iteration']:06d}")
    cp.check_seal(merge, "merged_checkpoint_fingerprint")
    if (merge.get("run_identity_sha256") != run["run_identity_sha256"]
            or merge.get("effective_policy_fingerprint") != policy["effective_policy_fingerprint"]
            or merge.get("parent_checkpoint_identity") != policy["checkpoint_identity"]
            or merge.get("policy_iteration") != policy["policy_iteration"]
            or any(g.get("static_merge") != merge for g in groups)):
        raise ValueError("exact K4 groups must reference the same current static merge")
    return dict(kind="merge-retirement", retention_version=VERSION,
        run_identity_sha256=run["run_identity_sha256"], policy_iteration=policy["policy_iteration"],
        effective_policy_fingerprint=policy["effective_policy_fingerprint"],
        merged_checkpoint_fingerprint=merge["merged_checkpoint_fingerprint"],
        merge_identity=merge, merged_file_sha256=merge["merged_file_sha256"],
        merge_manifest_sha256=merge_manifest_sha256,
        group_ids=window["ordered_group_ids"], group_payload_hashes=window["group_hashes"])


def check_retirement(root, run, policy, groups, *, finish_cleanup=False, cpu_fixture=False):
    step = policy["policy_iteration"]
    directory = Path(root) / "merges" / f"policy-{step:06d}"
    path = receipt_path(root, "merge", step)
    if not path.exists():
        if not directory.exists() and groups:
            raise ValueError("merge absent without retirement authorization")
        return None
    value = read_receipt(path)
    if value != cp.seal(retirement_value(run, policy, groups, value["merge_identity"],
                        merge_manifest_sha256=value["merge_manifest_sha256"]), "receipt_sha256"):
        raise ValueError("merge retirement authorization mismatch")
    if directory.exists():
        # Retirement is crash-idempotent even after partial authorized deletion.
        actual = physical_files(directory)
        authorized = set(value["merged_file_sha256"]) | {"merge_manifest.json"}
        if not actual <= authorized:
            raise ValueError("unauthorized merge files")
        manifest = directory / "merge_manifest.json"
        if manifest.exists():
            from opensearch_vl_repro.sft_tool_audit import sha256_file
            if sha256_file(manifest) != value["merge_manifest_sha256"]:
                raise ValueError("retired merge manifest changed")
        if finish_cleanup:
            delete_authorized(directory, authorized, remove_directory=True, cpu_fixture=cpu_fixture)
    return value


def retire_merge(root, run, policy, groups, *, workers_dead, cpu_fixture=False):
    if workers_dead is not True:
        raise ValueError("merge retirement requires collection worker teardown")
    step = policy["policy_iteration"]
    path = receipt_path(root, "merge", step)
    if not path.exists():
        from .formal_collection import verify_formal_merge
        merge = groups[0]["static_merge"] if groups else {}
        from opensearch_vl_repro.sft_tool_audit import sha256_file
        directory = Path(root) / "merges" / f"policy-{step:06d}"
        value = retirement_value(run, policy, groups, merge,
                                 merge_manifest_sha256=sha256_file(directory / "merge_manifest.json"))
        verify_formal_merge(directory, merge)
        authorize(root, "merge", step, value, cpu_fixture=cpu_fixture)
    return check_retirement(root, run, policy, groups, finish_cleanup=True, cpu_fixture=cpu_fixture)


def disk_accounting(root, *, previous_peak=0, report_root=None):
    root = Path(root)
    def size(path):
        return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    report_bytes = size(report_root) if report_root is not None else 0
    current = size(root) + report_bytes
    checkpoints = sorted(p for p in (root / "checkpoints").glob("policy-*") if p.is_dir())
    compacted = [p.name for p in checkpoints if receipt_path(root, "compact", int(p.name[7:])).exists()]
    return dict(current_run_bytes=current, report_bytes=report_bytes, free_filesystem_bytes=shutil.disk_usage(root).free,
        retained_full_checkpoints=[p.name for p in checkpoints if p.name not in compacted],
        compacted_checkpoints=compacted, active_merge_bytes=size(root / "merges"),
        groups_bytes=size(root / "groups"), peak_observed_run_bytes=max(previous_peak, current))


def disk_guard(accounting, *, needed_bytes, max_run_bytes=250 * GIB, min_free_bytes=30 * GIB):
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (needed_bytes, max_run_bytes, min_free_bytes))
            or needed_bytes < 0 or max_run_bytes <= 0 or min_free_bytes < 0
            or accounting["current_run_bytes"] + needed_bytes > max_run_bytes
            or accounting["free_filesystem_bytes"] - needed_bytes < min_free_bytes):
        raise OSError("Main disk headroom guard: staging forbidden before ENOSPC")
