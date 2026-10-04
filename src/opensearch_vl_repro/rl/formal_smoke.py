"""Formal Smoke20 control plane. Importing this module owns no GPU or process.

Immutable S1 receipts are authority; subprocess reports and counters are not.
Explicit CPU orchestration fixtures can never publish a runtime PASS.
"""
from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256 as digest
from . import checkpoint as cp
from .group import read_formal_group, run_lock
from .run_state import (advance_update_attempt, checkpoint_policy, read_update_attempt,
                        persist_update_attempt)
from .training_window import build_training_window, expected_window_prompts

VERSION = "formal-s3-smoke20-v1"
SFT_ADAPTER = "outputs/sft_main_imageid_v3/checkpoint-3k/adapter"
ROLLOUT = dict(temperature=.7, top_p=1., top_k=-1, max_new_tokens=512,
               max_model_len=8192, tensor_parallel_size=1, gpu_memory_utilization=.6,
               max_turns=16, logprobs_mode="processed_logprobs")
PINNED = {"transformers": "4.57.1", "peft": "0.21.1", "verl": "0.6.1",
          "vllm": "0.11.0", "rllm": "0.2.1"}


def redact_runtime_secrets(value):
    """Include custom provider api_key_env names, without recording env values."""
    from opensearch_vl_repro.agent.reliability import redact_secrets
    secrets = [v for k, v in os.environ.items() if v and re.search(r"KEY|TOKEN|SECRET|PASSWORD", k, re.I)]
    def visit(item):
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
            return item
        if isinstance(item, dict):
            return {visit(k): visit(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(v) for v in item]
        return item
    return visit(redact_secrets(value))


def source_identity(row):
    return digest({k: row[k] for k in (
        "source_sample_id", "prompt_id", "question_hash", "image_hashes", "image_relpaths")})


def require_smoke_run(run):
    cp.validate_training_run_identity(run)
    s = run["semantics"]
    if (s.get("coordinator_version") != VERSION or len(run["prompt_ids"]) != 20
            or s["rollout_n"] != 2 or s["groups_per_window"] != 4
            or s["require_complete_windows"] is not True or s["world_size"] != 2
            or s["weighting"] != cp.FORMAL_WEIGHTING or s["ppo"]["epochs"] != 1
            or s.get("diagnostic_version") or not s["base_model"].get("offline_snapshot_sha256")):
        raise ValueError("Formal S3 requires bound Smoke20/n2/K4/W2, not Gate/S2 diagnostic evidence")
    cp.require_digest(s["base_model"]["offline_snapshot_sha256"])
    if (s["rollout"]["config"] != ROLLOUT
            or s["optimizer"] != dict(name="AdamW", learning_rate=1e-6, weight_decay=0.)
            or any(s["ppo"].get(k) != v for k, v in dict(epochs=1, microbatch=1,
                clip_ratio_low=.2, clip_ratio_high=.28, entropy=0., loss_mode="vanilla").items())
            or s["image_protocol_version"] != "runtime-image-id-grounding-v3"):
        raise ValueError("Formal Smoke frozen rollout/AdamW/PPO/image semantics mismatch")


def smoke_paths(root, run_id, protected=()):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("safe nonempty run-id required")
    root = Path(root).resolve()
    paths = (root / "outputs/rl_formal_smoke" / run_id, root / "reports/rl_formal_smoke" / run_id)
    for target in paths:
        for ancestor in (target, *target.parents):
            if ancestor == root:
                break
            if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
                raise ValueError("Formal output ancestor redirects writes")
        if not target.resolve().is_relative_to(root):
            raise ValueError("Formal output escapes workspace")
        for source in protected:
            source = Path(source).resolve()
            if target.is_relative_to(source) or source.is_relative_to(target):
                raise ValueError("Formal output overlaps protected source")
    return paths


def validate_cache_locators(args, root):
    """Optional cache sharing never authorizes writes into protected experiments."""
    root = Path(root).resolve()
    protected = [root / name for name in ("data", "configs", "reports", ".eval-runtime",
        "outputs/sft_main", "outputs/sft_main_imageid_v3", "outputs/rl_formal_s2_validation",
        "outputs/rl_smoke", "outputs/rl_main")]
    protected += [args.source_root.resolve(), args.base_model_path.resolve(), args.sft_adapter.parent.resolve()]
    for field in ("tool_cache_dir", "reward_cache_dir"):
        locator = getattr(args, field, None)
        if locator is None:
            continue
        target = Path(locator).resolve()
        if any(target.is_relative_to(p) or p.is_relative_to(target) for p in protected):
            raise ValueError("Formal cache locator overlaps protected inputs/experiment outputs")
        if target.is_relative_to(root / "outputs"):
            parts = target.relative_to(root / "outputs").parts
            if parts[0].startswith("rl_gate"):
                raise ValueError("Formal cache must not write Gate outputs")
            if parts[0] == "rl_formal_smoke" and (len(parts) < 3 or parts[2] not in {"tool_cache", "reward_cache"}):
                raise ValueError("Formal cache cannot write immutable run artifacts")


def installed_source_inventory(name):
    """Resolve actual package source without importing it (also editable installs)."""
    from opensearch_vl_repro.sft_tool_audit import sha256_file
    spec = importlib.util.find_spec(name)  # TOP-LEVEL only; no parent import.
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    files = {f"{index}/{p.relative_to(directory).as_posix()}": sha256_file(p)
             for index, location in enumerate(locations) for directory in [Path(location)]
             for p in sorted(directory.rglob("*.py")) if p.is_file()}
    if not files:
        raise ValueError(f"installed {name} source inventory missing")
    return files


def software_binding(root):
    """Read installed source bytes without importing/initializing the frameworks."""
    from opensearch_vl_repro.sft_tool_audit import sha256_file
    versions = {name: importlib.metadata.version(name) for name in (*PINNED, "torch", "flash-attn")}
    if any(versions[name] != value for name, value in PINNED.items()):
        raise ValueError(f"Formal S3 pinned software mismatch: {versions}")
    if not versions["torch"].startswith("2.8."):
        raise ValueError("Formal S3 requires the S2-verified torch 2.8 stack")
    hashes = {p.relative_to(root).as_posix(): sha256_file(p)
              for directory in (root / "src", root / "scripts") for p in sorted(directory.rglob("*.py"))}
    for name in ("verl", "rllm", "vllm"):
        hashes["installed_" + name] = digest(installed_source_inventory(name))
    return versions, hashes


def prepare_context(args, root):
    """Read-only identity construction. No model, API, output or cache mutation."""
    from .config import load_rl_config
    from .actor_gate import BASE_MODEL, BASE_REVISION, load_smoke_records
    from .offline_snapshot import offline_snapshot_files
    from .rl_actor_semantics import execution_contract, require_saved_source_dropout
    from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
    from opensearch_vl_repro.sft_train_plan import load_main_config
    from opensearch_vl_repro.sft_tool_audit import sha256_file
    validate_cache_locators(args, root)
    config = load_rl_config(args.config)
    if config["rollout_n"] != 2 or config["data"]["smoke_count"] != 20:
        raise ValueError("Formal Smoke20 must use rl_smoke.yaml n=2, not rl_main n=4")
    if (args.sft_adapter.resolve() != (root / SFT_ADAPTER).resolve()
            or (root / config["model"]["sft_adapter"]).resolve() != (root / SFT_ADAPTER).resolve()
            or args.data.resolve() != (root / "data/rl/smoke20.json").resolve()):
        raise ValueError("Formal Smoke20 requires frozen data and original checkpoint-3k")
    canonical = load_main_config(root / config["model"]["sft_config"], base_eval_config=root / "configs/eval_base_300.yaml")
    config["data"]["quality_audit_dir"] = str(root / config["data"]["quality_audit_dir"])
    records, manifest = load_smoke_records(args.data, config)
    files = offline_snapshot_files(args.base_model_path, revision=BASE_REVISION, strict=True)
    # Same pinned content identity already verified in S2, not just a folder name.
    from .formal_s2_validation import require_snapshot_sha
    snapshot_sha = require_snapshot_sha(files)
    lineage = cp.build_rl_lineage(config=config, sft_config=canonical, adapter_path=args.sft_adapter, run_id=args.run_id)
    require_saved_source_dropout(args.sft_adapter)
    versions, hashes = software_binding(root)
    semantics = dict(coordinator_version=VERSION,
        dataset=dict(sha256=manifest["samples_sha256"], split="smoke", manifest_sha256=manifest["manifest_sha256"],
                     config_sha256=sha256_file(args.config)),
        base_model=dict(name=BASE_MODEL, revision=BASE_REVISION, offline_snapshot_sha256=snapshot_sha),
        source_sft=dict(adapter_sha256=lineage.sft_adapter_fingerprint, metadata_sha256=lineage.sft_checkpoint_metadata_fingerprint,
                        stage=lineage.sft_stage, lineage=list(lineage.sft_lineage)),
        execution_contract=execution_contract(canonical), rollout=dict(behavior_version=VERSION, config=copy.deepcopy(ROLLOUT)),
        rollout_n=2, groups_per_window=4, require_complete_windows=True, weighting=cp.FORMAL_WEIGHTING,
        optimizer=dict(name="AdamW", learning_rate=1e-6, weight_decay=0.),
        ppo=dict(epochs=1, microbatch=1, clip_ratio_low=.2, clip_ratio_high=.28, entropy=0., loss_mode="vanilla"),
        world_size=2, reward=dict(version="live-accuracy-query-v1", semantics="format*(.8*accuracy+.2*query)",
                                 judge_config_sha256=sha256_file(args.judge_config)),
        tool_protocol_version=RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION, image_protocol_version="runtime-image-id-grounding-v3",
        search_config_sha256=sha256_file(args.search_config), layout_config_sha256=sha256_file(args.layout_config),
        integration_source_hashes=hashes, software_versions=versions, initial_seed=config["data"]["seed"])
    run = cp.build_training_run_identity(args.run_id, semantics=semantics,
        prompt_ids=[r["prompt_id"] for r in records],
        prompt_sources=[dict(prompt_id=r["prompt_id"], source_identity=source_identity(r)) for r in records],
        locators=dict(base_snapshot=str(args.base_model_path.resolve()), source_root=str(args.source_root.resolve())))
    require_smoke_run(run)
    runtime = copy.deepcopy(canonical)
    runtime["model"]["name_or_path"] = str(args.base_model_path.resolve())
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return dict(run=run, records=records, canonical=canonical, runtime=runtime, versions=versions, git_commit=git_commit)


def window_id(iteration):
    return f"smoke-window-{iteration:06d}"


def recover_smoke(root, run, *, cpu_fixture=False):
    """Reconstruct all S1 receipts, reconcile a post-publication attempt crash.

    Never mark an unpublished update successful or promote hidden artifacts.
    Caller holds coordinator lock; workers are already dead before this call.
    """
    require_smoke_run(run)
    root = Path(root)
    anchor = cp._load_anchor(root, run)
    scope = "cpu_fixture" if cpu_fixture else "runtime"
    if anchor["evidence_scope"] != scope:
        raise ValueError("CPU fixture anchor cannot enter runtime coordinator")
    recovered = cp.recover_formal_run(root, run, cpu_fixture=cpu_fixture)
    groups = [read_formal_group(p) for p in sorted((root / "groups").iterdir()) if not p.name.startswith(".")]
    checkpoints = recovered["checkpoints"]
    policies = [anchor["initial_policy"]] + [checkpoint_policy(c) for c in checkpoints]
    current = recovered["policy"]["policy_iteration"]
    if current > 5:
        raise ValueError("Smoke20 cannot advance beyond five updates")
    for group in groups:
        ident = group["identity"]
        iteration = ident["policy_iteration"]
        if (group["evidence_scope"] != scope or group.get("diagnostic_mode")
                or iteration >= 5 or iteration > current
                or ident["prompt_id"] not in expected_window_prompts(run, iteration)
                or ident["pre_update_policy_fingerprint"] != policies[iteration]["effective_policy_fingerprint"]
                or any(m.get("diagnostic_fixture") or m.get("rollout_executed") is False for m in group["members"])
                or any(s["info"].get("diagnostic_fixture") for m in group["members"] for s in m["steps"])):
            raise ValueError("foreign/diagnostic/future-window/stale-policy Formal group")
        if not cpu_fixture:
            binding = group.get("static_merge", {})
            cp.check_seal(binding, "merged_checkpoint_fingerprint")
            if (binding.get("run_identity_sha256") != run["run_identity_sha256"]
                    or binding.get("effective_policy_fingerprint") != policies[iteration]["effective_policy_fingerprint"]
                    or binding.get("parent_checkpoint_identity") != policies[iteration]["checkpoint_identity"]
                    or binding.get("policy_iteration") != iteration
                    or binding.get("base_snapshot_sha256") != run["semantics"]["base_model"]["offline_snapshot_sha256"]
                    or any(m.get("rollout_executed") is not True
                           or m.get("rllm_provenance", {}).get("generation_loop") != "rllm.workflows.multi_turn_workflow.MultiTurnWorkflow.run"
                           or any(m["reward"].get(k, {}).get("status") != "success" for k in ("accuracy_judge", "query_judge"))
                           for m in group["members"])):
                raise ValueError("runtime Formal group lacks actual policy/rLLM/live reward provenance")
    by_step = {c["global_optimizer_step"]: c for c in checkpoints}
    for step, checkpoint in by_step.items():
        if not cpu_fixture:
            require_runtime_update_evidence(root / "checkpoints" / f"policy-{step:06d}", checkpoint)
        if checkpoint["eligibility"]["kind"] != ("smoke_final" if step == 5 else "smoke_continuation"):
            raise ValueError("Smoke checkpoint kind mismatch")
        attempt = read_update_attempt(root, checkpoint["update_attempt"]["attempt_id"])
        if attempt["phase"] == "checkpoint_staging" and attempt == checkpoint["update_attempt"]:
            attempt = advance_update_attempt(attempt, "verified", checkpoint=checkpoint,
                                             checkpoint_directory=root / "checkpoints" / f"policy-{step:06d}")
            persist_update_attempt(root, attempt, cpu_fixture=cpu_fixture)
        if attempt["phase"] != "verified" or attempt["verified_checkpoint_identity"] != checkpoint["checkpoint_manifest_sha256"]:
            raise ValueError("published checkpoint/attempt history conflict")
    attempts = []
    for directory in sorted((root / "attempts").iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("invalid attempt directory")
        attempt = read_update_attempt(root, directory.name)
        step = attempt["expected_optimizer_step"]
        if attempt["run_identity_sha256"] != run["run_identity_sha256"] or not 1 <= step <= min(current + 1, 5):
            raise ValueError("foreign/future update attempt")
        matching = [g for p in expected_window_prompts(run, step - 1) for g in groups
                    if g["identity"]["prompt_id"] == p and g["identity"]["policy_iteration"] == step - 1]
        window = build_training_window(run, policies[step - 1], matching, window_id=window_id(step - 1))
        if (attempt["window_sha256"] != window["window_sha256"]
                or attempt["parent_policy_fingerprint"] != policies[step - 1]["effective_policy_fingerprint"]
                or attempt["parent_checkpoint_identity"] != policies[step - 1]["checkpoint_identity"]):
            raise ValueError("attempt parent/window binding conflict")
        if step <= current:
            if attempt["phase"] not in {"failed", "verified"}:
                raise ValueError("unresolved historical update attempt")
            if attempt["phase"] == "verified" and attempt["attempt_id"] != by_step[step]["update_attempt"]["attempt_id"]:
                raise ValueError("duplicate verified attempt")
        elif attempt["phase"] == "verified":
            raise ValueError("verified attempt without immutable checkpoint")
        attempts.append(attempt)
    active = [a for a in attempts if a["expected_optimizer_step"] == current + 1 and a["phase"] != "failed"]
    if len(active) > 1:
        raise ValueError("multiple unresolved update attempts")
    current_groups = {g["identity"]["prompt_id"]: g for g in groups if g["identity"]["policy_iteration"] == current}
    expected = expected_window_prompts(run, current) if current < 5 else []
    return {**recovered, "groups": groups, "attempts": attempts,
            "current_groups": [current_groups[p] for p in expected if p in current_groups],
            "missing_prompts": [p for p in expected if p not in current_groups]}


def require_runtime_update_evidence(directory, checkpoint):
    """Rank evidence is itself hashed inside the verified native checkpoint."""
    from .formal_s2_validation import strict_oc_summary
    from .training_window import deterministic_rank_plan
    import math
    rows = [json.loads((directory / f"formal_update_rank_{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    step = checkpoint["global_optimizer_step"]
    for rank, evidence in enumerate(rows):
        update = evidence["update"]
        losses = update["metrics"].get("actor/pg_loss", [])
        if (evidence.get("scope") != "runtime" or evidence["rank"] != rank
                or update["before_step"] != step - 1 or update["after_step"] != step
                or update["update_audit"]["optimizer_step_count"] != 1
                or update["attempt"] != checkpoint["update_attempt"]
                or update["window_sha256"] != checkpoint["window"]["window_sha256"]
                or not strict_oc_summary(update["alignment"])
                or not losses or not all(math.isfinite(float(loss)) for loss in losses)
                or evidence["loaded_parent"]["policy"] != checkpoint["parent_policy"]
                or evidence["actor_contract"].get("passed") is not True
                or evidence["full_lora_sha256"] == evidence["previous_full_lora_sha256"]
                or any(evidence["fresh_reload"].get(k) is not True for k in (
                    "original_actor_destroyed", "fresh_multimodal_forward_finite", "adapter_reloaded",
                    "native_reloaded", "optimizer_reloaded", "rng_reloaded", "execution_contract_verified"))):
            raise ValueError("incomplete/foreign native one-step rank update evidence")
        plan = evidence["deterministic_plan"]
        if plan != deterministic_rank_plan(list(plan["multiplicity"]), 2):
            raise ValueError("non-deterministic runtime rank plan")
        saved = json.loads((directory / f"runtime_state_rank_{rank}.json").read_text(encoding="utf-8"))
        reloaded = [v for v in evidence["fresh_reload"]["per_rank"] if v["rank"] == rank]
        if len(reloaded) != 1 or reloaded[0]["state"] != {k: v for k, v in saved.items()
                if k not in {"runtime_state_sha256", "window_sha256", "execution_contract"}}:
            raise ValueError("native saved/fresh-reloaded model/optimizer/RNG mismatch")
    if rows[0]["full_lora_sha256"] != rows[1]["full_lora_sha256"]:
        raise ValueError("post-update full LoRA differs across ranks")


def final_reconstruction(root, run, *, cpu_fixture=False):
    value = recover_smoke(root, run, cpu_fixture=cpu_fixture)
    checkpoints, groups, policy = value["checkpoints"], value["groups"], value["policy"]
    if (len(checkpoints) != 5 or len(groups) != 20 or sum(len(g["members"]) for g in groups) != 40
            or [c["global_optimizer_step"] for c in checkpoints] != list(range(1, 6))
            or [c["policy_iteration"] for c in checkpoints] != list(range(1, 6))
            or any(e["status"] != "consumed_by_verified_checkpoint" for e in value["ledger"]["prompts"].values())
            or len(value["ledger"]["consumed_group_ids"]) != 20
            or policy != checkpoint_policy(checkpoints[-1])
            or any(c["eligibility"]["eligible_for_main_init"] is not False for c in checkpoints)
            or any(a["phase"] not in {"verified", "failed"} for a in value["attempts"])):
        raise ValueError("incomplete Smoke20 immutable reconstruction")
    return dict(version=VERSION, scope="cpu_fixture" if cpu_fixture else "runtime",
        passed=not cpu_fixture, checks=dict(immutable_chain=True, prompts20=True, groups20=True,
        members40=True, ordered_windows5=True, native_steps_1_to_5=True, attempt_chains=True,
        final_policy_verified=True, eligible_for_main_init_false=True), run_identity=run,
        final_policy=policy, checkpoint_identities=[c["checkpoint_manifest_sha256"] for c in checkpoints],
        optimizer_steps=list(range(1, 6)), eligible_for_main_init=False,
        checkpoint_kind="smoke_final", groups_completed=20, trajectories_completed=40)


def revoke_pass(output, *, cpu_fixture=False):
    path = Path(output) / "manifest.json"
    if path.exists():
        path.unlink()
        cp.fsync_directory(Path(output), cpu_fixture=cpu_fixture)


def publish_final(output, reports, report, *, cpu_fixture=False):
    if report.get("passed") is not True or report.get("scope") != "runtime" or cpu_fixture:
        raise ValueError("CPU fixtures/report-only evidence cannot publish Formal Smoke20 PASS")
    if not report.get("checks") or not all(v is True for v in report["checks"].values()):
        raise ValueError("incomplete reconstruction cannot publish PASS")
    revoke_pass(output)
    try:
        # Public helper also fails closed: caller-supplied truthy JSON checks
        # are not authority. Revalidate immutable bytes at publication time.
        derived = final_reconstruction(output, report["run_identity"])
        if any(report.get(k) != v for k, v in derived.items()):
            raise ValueError("final report differs from immutable Smoke20 reconstruction")
        cp.durable_json(Path(reports) / "report.json", report)
        cp.durable_json(Path(output) / "manifest.json", report)  # LAST durable PASS artifact
    except BaseException:
        revoke_pass(output)
        raise


def worker_command(args, root, phase):
    if phase not in {"bootstrap", "collect", "update"}:
        raise ValueError("unknown Formal worker phase")
    common = ["--run-id", args.run_id, "--config", str(args.config), "--data", str(args.data),
        "--source-root", str(args.source_root), "--base-model-path", str(args.base_model_path),
        "--sft-adapter", str(args.sft_adapter), "--judge-config", str(args.judge_config),
        "--search-config", str(args.search_config), "--layout-config", str(args.layout_config)]
    for flag in ("tool_cache_dir", "reward_cache_dir"):
        if getattr(args, flag, None) is not None:
            common += ["--" + flag.replace("_", "-"), str(getattr(args, flag))]
    if phase == "collect":
        command = [sys.executable, str(root / "scripts/collect_rl_formal_smoke.py"), *common]
    else:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                   str(root / "scripts/update_rl_formal_smoke.py"), "--phase", phase, *common]
    return command


def worker_environment(args, phase):
    env = os.environ.copy()  # Credentials are inherited ONLY, never serialized.
    devices = args.rollout_gpu if phase == "collect" else args.update_gpus
    if (not re.fullmatch(r"\d+", args.rollout_gpu) or not re.fullmatch(r"\d+,\d+", args.update_gpus)
            or len(set(args.update_gpus.split(","))) != 2):
        raise ValueError("one rollout GPU and exactly two distinct update GPUs required")
    # Coordinator may itself be launched from a torchrun shell; workers must not
    # inherit stale launcher identity or allocator ownership.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=devices, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    return env


def linux_process_table(proc_root=Path("/proc")):
    """Read PID birth identity too; never signal a recycled, unrelated PID."""
    if not proc_root.is_dir():
        raise RuntimeError("Formal worker cleanup requires Linux /proc process verification")
    rows = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            rows[int(entry.name)] = dict(state=fields[0], parent=int(fields[1]),
                group=int(fields[2]), birth=int(fields[19]))
        except (FileNotFoundError, ProcessLookupError):
            pass  # A concurrent exit is expected, not a live owner.
    return rows


def descendant_identities(rows, parent):
    """Include children in new sessions (torch elastic does setsid per rank)."""
    result, parents = {}, {parent}
    while True:
        children = {pid: row["birth"] for pid, row in rows.items()
                    if row["parent"] in parents and pid not in result and pid != parent}
        if not children:
            return result
        result.update(children)
        parents.update(children)


def live_owned_processes(rows, identities):
    return [pid for pid, birth in identities.items()
            if pid in rows and rows[pid]["birth"] == birth and rows[pid]["state"] != "Z"]


def process_group_alive(group_id, proc_root=Path("/proc")):
    return any(r["group"] == group_id and r["state"] != "Z" for r in linux_process_table(proc_root).values())


def set_subreaper(value=None):
    """Linux-only, lazy. Orphaned rank/engine sessions reparent to coordinator."""
    import ctypes
    library = ctypes.CDLL(None, use_errno=True)
    prior = ctypes.c_int()
    if library.prctl(37, ctypes.byref(prior), 0, 0, 0) != 0:  # PR_GET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot query child subreaper")
    if value is not None and library.prctl(36, int(value), 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot set child subreaper")
    return prior.value


class ProcessRunner:
    """Blocking process-group lifetime. No next phase before cleanup/reaping.

    Linux subreaper + birth-bound descendants cover detached torch elastic rank
    sessions too. No next phase until all owned descendants are dead/reaped.
    No shell, no credential-bearing command report.
    """
    def __init__(self):
        self.active = False

    def __call__(self, command, *, env, cwd, log):
        from opensearch_vl_repro.agent.reliability import redact_secrets
        if self.active or os.name != "posix":
            raise RuntimeError("exclusive POSIX worker process lifetime required")
        self.active = True
        process, reader, errors, prior_subreaper = None, None, [], None
        secrets = [v for k, v in env.items() if v and re.search(r"KEY|TOKEN|SECRET|PASSWORD", k, re.I)]
        try:
            if descendant_identities(linux_process_table(), os.getpid()):
                raise RuntimeError("coordinator already owns child processes; exclusive worker lifetime required")
            prior_subreaper = set_subreaper(True)
            process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            def read_log():
                try:
                    with Path(log).open("w", encoding="utf-8") as stream:
                        for line in process.stdout:
                            for secret in secrets:
                                line = line.replace(secret, "[REDACTED]")
                            line = redact_secrets(line)
                            stream.write(line)
                            stream.flush()
                            print(line, end="", flush=True)
                except BaseException as exc:
                    errors.append(exc)
                    # A broken log sink must not strand a worker behind a full
                    # stdout pipe while the coordinator waits for its exit.
                    process.kill()  # Popen protects its still-unreaped child PID.
            # Wait for the worker, not EOF: orphaned framework children can
            # retain stdout after an abrupt worker death. Cleanup kills them
            # before joining the reader and before a subsequent phase starts.
            reader = threading.Thread(target=read_log, daemon=True)
            reader.start()
            code = process.wait()
        finally:
            try:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    deadline, owned = time.monotonic() + 30, {}
                    while True:
                        rows = linux_process_table()
                        # Subreaper adoption also covers children spawned just
                        # before an abrupt parent death, without a polling gap.
                        owned.update(descendant_identities(rows, os.getpid()))
                        live = live_owned_processes(rows, owned)
                        for pid in live:
                            # Re-check birth immediately before signaling.
                            if pid in live_owned_processes(linux_process_table(), {pid: owned[pid]}):
                                try:
                                    os.kill(pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                        # Coordinator had no unrelated children at launch.
                        while True:
                            try:
                                pid, _ = os.waitpid(-1, os.WNOHANG)
                                if pid == 0:
                                    break
                            except ChildProcessError:
                                break
                        remaining = descendant_identities(linux_process_table(), os.getpid())
                        if not live and not remaining and not process_group_alive(process.pid):
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError("framework descendants still alive; next GPU phase forbidden")
                        time.sleep(.05)
                    if reader is not None:
                        reader.join(timeout=30)
                        if reader.is_alive():
                            raise RuntimeError("worker descendants did not release the log pipe")
                    if process.stdout is not None:
                        process.stdout.close()
            finally:
                if prior_subreaper is not None:
                    set_subreaper(prior_subreaper)
                self.active = False
        if errors:
            raise errors[0]
        return code


def progress(value):
    groups = value["groups"]
    members = [m for g in groups for m in g["members"]]
    return dict(policy=value["policy"], committed_groups=len(groups), completed_trajectories=len(members),
        fatal_count=sum(m["fatal"] for m in members), rewards=[m["reward"]["total"] for m in members],
        reward_cache_hits=sum(bool(m["reward"].get(key, {}).get("cache_hit")) for m in members
                              for key in ("accuracy_judge", "query_judge")),
        tool_errors=sum(t.get("status") != "success" for m in members
                        for t in m.get("trajectory", {}).get("turns", []) if t.get("role") == "tool"))


def orchestrate(args, root, ctx, runner=None, *, cpu_fixture=False):
    """Same run-id resumes automatically, with strict unchanged semantic binding."""
    redact_secrets = redact_runtime_secrets
    if runner is not None and not cpu_fixture:
        raise ValueError("injected subprocess runners are CPU orchestration fixtures only")
    runner = runner or ProcessRunner()
    run = ctx["run"]
    require_smoke_run(run)
    output, reports = smoke_paths(root, args.run_id, [args.source_root, args.base_model_path, args.sft_adapter.parent])
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    started, stage, events = time.monotonic(), "recovery", []
    with run_lock(output / ".coordinator.lock"):
        def write(status, **extra):
            report = dict(version=VERSION, passed=False, status=status, stage=stage,
                run_identity_sha256=run["run_identity_sha256"], elapsed_seconds=time.monotonic() - started,
                subprocesses=events, eligible_for_main_init=False, software_versions=ctx.get("versions"),
                git_commit=ctx.get("git_commit"), **extra)
            cp.durable_json(reports / "report.json", redact_secrets(report), cpu_fixture=cpu_fixture)
        def launch(phase):
            nonlocal stage
            stage = phase
            print(f"[Smoke20] {phase} isolated worker", flush=True)
            write("running")
            before = time.monotonic()
            code = runner(worker_command(args, root, phase), env=worker_environment(args, phase), cwd=root,
                          log=reports / f"{len(events):04d}-{phase}.log")
            events.append(dict(phase=phase, exit_code=code, elapsed_seconds=time.monotonic() - before))
            if code != 0:
                raise RuntimeError(f"{phase} worker exited {code}; immutable receipts retained; resume same run-id")
        try:
            revoke_pass(output, cpu_fixture=cpu_fixture)
            print("[Smoke20] recovery from immutable Formal receipts", flush=True)
            write("running")
            if not (output / "identity").exists():
                # bootstrap is idempotent after anchor publication; never reseal
                # existing unbound/old run identities to make resume pass.
                launch("bootstrap")
            recovered = recover_smoke(output, run, cpu_fixture=cpu_fixture)
            while recovered["policy"]["policy_iteration"] < 5:
                iteration = recovered["policy"]["policy_iteration"]
                print(f"[Smoke20] window {iteration + 1}/5 policy={iteration}", flush=True)
                write("running", progress=progress(recovered))
                if recovered["missing_prompts"]:
                    launch("collect")
                    recovered = recover_smoke(output, run, cpu_fixture=cpu_fixture)
                    if recovered["policy"]["policy_iteration"] != iteration or recovered["missing_prompts"]:
                        raise ValueError("collection returned without complete same-policy window")
                print(f"[Smoke20] window {iteration + 1}/5 rollout process fully exited", flush=True)
                launch("update")
                recovered = recover_smoke(output, run, cpu_fixture=cpu_fixture)
                if recovered["policy"]["policy_iteration"] != iteration + 1:
                    raise ValueError("update returned without exactly one immutable successor")
                print(f"[Smoke20] checkpoint {iteration + 1}/5 verified", flush=True)
            stage = "final_reconstruction"
            report = final_reconstruction(output, run, cpu_fixture=cpu_fixture)
            report.update(subprocesses=events, elapsed_seconds=time.monotonic() - started,
                          progress=progress(recovered), software_versions=ctx.get("versions"), git_commit=ctx.get("git_commit"))
            if cpu_fixture:
                cp.durable_json(reports / "report.json", report, cpu_fixture=True)
            else:
                publish_final(output, reports, report)
                print("FORMAL RL SMOKE20 PASS (NOT eligible for Main400 initialization)", flush=True)
            return report
        except BaseException as exc:
            revoke_pass(output, cpu_fixture=cpu_fixture)
            provider = None
            # Worker failure receipt is informational, never permits advancing.
            failure = reports / "collection_failure.json"
            if failure.is_file():
                provider = json.loads(failure.read_text(encoding="utf-8")).get("provider_interruption")
            write("interrupted", error=f"{type(exc).__name__}: {exc}", provider_interruption=provider)
            raise


def run_smoke(args, root):
    root = Path(root).resolve()
    # Construct/compare semantics before any GPU/model worker is launched.
    ctx = prepare_context(args, root)
    return orchestrate(args, root, ctx)
