"""Formal Main400 control plane. Heavy frameworks are imported only in workers."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256 as digest
from . import checkpoint as cp
from . import formal_main_retention as retention
from .formal_smoke import ROLLOUT, SFT_ADAPTER, source_identity, software_binding
from .group import read_formal_group, read_formal_group_manifest_only, run_lock
from .run_state import (advance_update_attempt, checkpoint_policy, persist_update_attempt,
                        read_update_attempt, reconstruct_consumed_ledger)
from .training_window import build_training_window, expected_window_prompts
from .context_budget import CONTEXT_BUDGET_POLICY
from opensearch_vl_repro.agent.reliability import SEARCH_BEHAVIOR_VERSION, provider_reliability_semantics

VERSION = "formal-s4-main400-v7"
MAIN_ROLLOUT = copy.deepcopy(ROLLOUT)
MAIN_ROLLOUT["max_model_len"] = 16384
SEED_SCHEME = "initial_seed+n*global_prompt_position+rollout_index-v1"


def require_main_run(run, *, _historical=False):
    from .actor_gate import BASE_MODEL, BASE_REVISION
    from .rl_actor_semantics import contract_sft_config
    from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
    cp.validate_training_run_identity(run)
    s = run["semantics"]
    version = s.get("coordinator_version")
    if _historical:
        from .formal_main_continuation import PARENT_VERSIONS
        if version not in PARENT_VERSIONS or s.get("continuation"):
            raise ValueError("continuation parent must be original single-run Main v4/v5")
        if version.endswith("-v4"):
            if "search_behavior_version" in s or "provider_reliability" in s:
                raise ValueError("historical Main v4 search identity must remain unchanged")
        # Historical v5 stays bound to Search v3, independent of current behavior.
        elif (s.get("search_behavior_version") != 3
                or s.get("provider_reliability") != provider_reliability_semantics()):
            raise ValueError("historical Main v5 provider identity differs")
    elif (version != VERSION or s.get("search_behavior_version") != SEARCH_BEHAVIOR_VERSION
          or s.get("provider_reliability") != provider_reliability_semantics()):
        raise ValueError("Formal Main requires current v7/search-v4 identity")
    if (len(run["prompt_ids"]) != 400
            or s["rollout_n"] != 4 or s["groups_per_window"] != 4 or s["world_size"] != 4
            or s["require_complete_windows"] is not True or s["weighting"] != cp.FORMAL_WEIGHTING
            or s["dataset"]["split"] != "main" or s.get("diagnostic_version")
            or s.get("retention") != retention.CONTRACT or s.get("rollout_seed_scheme") != SEED_SCHEME
            or s["rollout"] != dict(behavior_version=version, context_budget_policy=CONTEXT_BUDGET_POLICY, config=MAIN_ROLLOUT)
            or s["optimizer"] != dict(name="AdamW", learning_rate=1e-6, weight_decay=0.)
            or s["ppo"] != dict(epochs=1, microbatch=1, clip_ratio_low=.2, clip_ratio_high=.28,
                                entropy=0., loss_mode="vanilla")
            or s["image_protocol_version"] != "runtime-image-id-grounding-v3"
            or s["tool_protocol_version"] != RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
            or s["base_model"].get("name") != BASE_MODEL or s["base_model"].get("revision") != BASE_REVISION
            or s["source_sft"].get("stage") != "main_b_2k"
            or s["source_sft"].get("lineage") != ["main_a_1k", "main_b_2k"]
            or s["reward"].get("semantics") != "format*(.8*accuracy+.2*query)"
            or type(s.get("initial_seed")) is not int or s["initial_seed"] != 20260506):
        raise ValueError("Formal S4 requires exact Main400/n4/K4/W4/100 windows and frozen semantics")
    cp.require_digest(s["base_model"].get("offline_snapshot_sha256"))
    contract_sft_config(s["execution_contract"])


def main_paths(root, run_id, protected=()):
    # Same safety checks as S3, with a separate fixed physical namespace.
    from .formal_smoke import smoke_paths
    smoke_paths(root, run_id, protected)
    root = Path(root).resolve()
    paths = (root / "outputs/rl_formal_main" / run_id, root / "reports/rl_formal_main" / run_id)
    for target in paths:
        for ancestor in (target, *target.parents):
            if ancestor == root:
                break
            if ancestor.is_symlink() or (hasattr(ancestor, "is_junction") and ancestor.is_junction()):
                raise ValueError("Main output redirects writes")
        if not target.resolve().is_relative_to(root) or any(
                target.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(target)
                for p in protected):
            raise ValueError("Main output overlaps protected source")
    return paths


def member_seed(run, prompt, index):
    if type(index) is not int or index not in range(4):
        raise ValueError("Main member index must be 0..3")
    return run["semantics"]["initial_seed"] + 4 * run["prompt_ids"].index(prompt) + index


def window_id(iteration):
    return f"main-window-{iteration:06d}"


def load_main_records(data, config, *, source_root, source_parquet, eval_overlap_manifest,
                      sft_overlap_manifest):
    """Reuse complete pinned-source/quality/shard/image/deterministic preflight."""
    from .data import preflight_dataset
    settings = config["data"]
    result = preflight_dataset(Path(data).parent, source_parquet=source_parquet,
        source_root=source_root, eval_overlap_manifest=eval_overlap_manifest,
        sft_overlap_manifest=sft_overlap_manifest, quality_audit_dir=settings["quality_audit_dir"])
    records = json.loads(Path(data).read_text(encoding="utf-8"))
    manifest = json.loads(Path(data).with_name("main400_manifest.json").read_text(encoding="utf-8"))
    if (config["rollout_n"] != 4 or len(records) != 400 or manifest.get("name") != "main"
            or manifest["selected_count"] != 400 or manifest["samples_sha256"] != digest(records)
            or manifest["membership"] != [r["source_sample_id"] for r in records]
            or any(result[k] != settings[v] for k, v in (
                ("dataset_id", "dataset_id"), ("dataset_revision", "dataset_revision"),
                ("selection_seed", "seed"), ("selection_version", "selection_version"),
                ("source_rows", "source_rows"), ("main_count", "main_count"),
                ("shard_size", "shard_size"))) or result["shard_count"] != 4
            or result["main_count"] != 400 or result["shard_size"] != 100):
        raise ValueError("Main frozen population/membership/provenance mismatch")
    return records, manifest


def prepare_context(args, root):
    from .config import load_rl_config
    from .actor_gate import BASE_MODEL, BASE_REVISION
    from .offline_snapshot import offline_snapshot_files
    from .formal_s2_validation import require_snapshot_sha
    from .rl_actor_semantics import execution_contract, require_saved_source_dropout
    from .formal_smoke import validate_cache_locators
    from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
    from opensearch_vl_repro.sft_train_plan import load_main_config
    from opensearch_vl_repro.sft_tool_audit import sha256_file
    root = Path(root).resolve()
    validate_cache_locators(args, root)
    output, _ = main_paths(root, args.run_id)
    if getattr(args, "continue_from_run", None):
        parent_output, parent_reports = main_paths(root, args.continue_from_run)
        for field in ("tool_cache_dir", "reward_cache_dir"):
            target = getattr(args, field, None)
            if target is not None and any(Path(target).resolve().is_relative_to(p) for p in (parent_output, parent_reports)):
                raise ValueError("continuation cache cannot write parent namespace")
    # Additional namespace restrictions for shared cache locators.
    for field in ("tool_cache_dir", "reward_cache_dir"):
        target = getattr(args, field, None)
        if target is not None and Path(target).resolve().is_relative_to(root / "outputs"):
            target = Path(target).resolve()
            if not any(target.is_relative_to(output / name) for name in ("tool_cache", "reward_cache")):
                raise ValueError("Main cache must not write another experiment or immutable artifacts")
    if args.sft_adapter.resolve() != (root / SFT_ADAPTER).resolve():
        raise ValueError("Main requires ORIGINAL SFT checkpoint-3k; Smoke/Gate/staging forbidden")
    config = load_rl_config(args.config)
    if (args.config.resolve() != (root / "configs/rl_main.yaml").resolve()
            or args.data.resolve() != (root / "data/rl/main400.json").resolve()
            or args.sft_adapter.resolve() != (root / SFT_ADAPTER).resolve()
            or (root / config["model"]["sft_adapter"]).resolve() != (root / SFT_ADAPTER).resolve()):
        raise ValueError("Main requires frozen rl_main/main400 and ORIGINAL SFT, never Smoke/Gate/staging")
    config["data"]["quality_audit_dir"] = str(root / config["data"]["quality_audit_dir"])
    records, manifest = load_main_records(args.data, config, source_root=args.source_root,
        source_parquet=args.source_parquet, eval_overlap_manifest=args.eval_overlap_manifest,
        sft_overlap_manifest=args.sft_overlap_manifest)
    canonical = load_main_config(root / config["model"]["sft_config"], base_eval_config=root / "configs/eval_base_300.yaml")
    snapshot_sha = require_snapshot_sha(offline_snapshot_files(args.base_model_path, revision=BASE_REVISION, strict=True))
    lineage = cp.build_rl_lineage(config=config, sft_config=canonical, adapter_path=args.sft_adapter, run_id=args.run_id)
    require_saved_source_dropout(args.sft_adapter)
    versions, hashes = software_binding(root)
    semantics = dict(coordinator_version=VERSION,
        dataset=dict(sha256=manifest["samples_sha256"], split="main", manifest_sha256=manifest["manifest_sha256"],
                     config_sha256=sha256_file(args.config)),
        base_model=dict(name=BASE_MODEL, revision=BASE_REVISION, offline_snapshot_sha256=snapshot_sha),
        source_sft=dict(adapter_sha256=lineage.sft_adapter_fingerprint, metadata_sha256=lineage.sft_checkpoint_metadata_fingerprint,
                        stage=lineage.sft_stage, lineage=list(lineage.sft_lineage)),
        execution_contract=execution_contract(canonical), rollout=dict(behavior_version=VERSION,
            context_budget_policy=CONTEXT_BUDGET_POLICY, config=copy.deepcopy(MAIN_ROLLOUT)),
        rollout_n=4, groups_per_window=4, require_complete_windows=True, weighting=cp.FORMAL_WEIGHTING,
        optimizer=dict(name="AdamW", learning_rate=1e-6, weight_decay=0.),
        ppo=dict(epochs=1, microbatch=1, clip_ratio_low=.2, clip_ratio_high=.28, entropy=0., loss_mode="vanilla"),
        world_size=4, reward=dict(version="live-accuracy-query-v1", semantics="format*(.8*accuracy+.2*query)",
                                 judge_config_sha256=sha256_file(args.judge_config)),
        tool_protocol_version=RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION, image_protocol_version="runtime-image-id-grounding-v3",
        search_config_sha256=sha256_file(args.search_config), layout_config_sha256=sha256_file(args.layout_config),
        search_behavior_version=SEARCH_BEHAVIOR_VERSION, provider_reliability=provider_reliability_semantics(),
        integration_source_hashes=hashes, software_versions=versions, initial_seed=config["data"]["seed"],
        retention=copy.deepcopy(retention.CONTRACT), rollout_seed_scheme=SEED_SCHEME)
    from .formal_main_continuation import attach_binding, read_authority
    attach_binding(args, root, semantics)
    run = cp.build_training_run_identity(args.run_id, semantics=semantics,
        prompt_ids=[r["prompt_id"] for r in records],
        prompt_sources=[dict(prompt_id=r["prompt_id"], source_identity=source_identity(r)) for r in records],
        locators=dict(base_snapshot=str(args.base_model_path.resolve()), source_root=str(args.source_root.resolve())))
    require_main_run(run)
    if (output / "continuation").exists():
        read_authority(output, run)
    runtime = copy.deepcopy(canonical)
    runtime["model"]["name_or_path"] = str(args.base_model_path.resolve())
    return dict(run=run, records=records, canonical=canonical, runtime=runtime, versions=versions,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())


def recover_main(root, run, *, cpu_fixture=False, reconcile=True, cleanup=True, _historical=False):
    """Metadata history + FULL current parent/current unconsumed groups only.

    Caller owns coordinator lifetime/lock; publication calls this under formal
    lock with reconcile=False. No S1 full-history recovery is used indirectly.
    """
    if _historical and (reconcile or cleanup):
        raise ValueError("historical parent recovery is strictly read-only")
    require_main_run(run, _historical=_historical)
    root = Path(root)
    scope = "cpu_fixture" if cpu_fixture else "runtime"
    authority = None
    if run["semantics"].get("continuation"):
        from .formal_main_continuation import load_anchor, prefix_capability, read_authority
        anchor, authority = load_anchor(root, run, cpu_fixture=cpu_fixture)
    else:
        anchor = cp._load_anchor(root, run)
    if anchor["evidence_scope"] != scope:
        raise ValueError("CPU fixture anchor cannot enter runtime Main")
    checkpoints = [cp.read_checkpoint_manifest_only(p) for p in sorted((root / "checkpoints").iterdir())
                   if not p.name.startswith(".")]
    groups = [read_formal_group_manifest_only(p) for p in sorted((root / "groups").iterdir())
              if not p.name.startswith(".")]
    ledger = reconstruct_consumed_ledger(run, anchor["initial_policy"], groups, checkpoints,
        checkpoint_root=root / "checkpoints", checkpoint_reader=cp.read_checkpoint_manifest_only,
        inherited_prefix=prefix_capability(authority) if authority else None)
    current = ledger["policy"]["policy_iteration"]
    if current > 100:
        raise ValueError("Main cannot exceed 100 optimizer steps")
    policies = {anchor["initial_policy"]["policy_iteration"]: anchor["initial_policy"],
                **{c["policy_iteration"]: checkpoint_policy(c) for c in checkpoints}}
    by_step = {c["policy_iteration"]: c for c in checkpoints}
    by_gid = {g["identity"]["trajectory_group_id"]: g for g in groups}
    for group in groups:
        ident = group["identity"]
        iteration = ident["policy_iteration"]
        if retention.physical_files(root / "groups" / ident["trajectory_group_id"], exclude=("group.json",)) != set(group["file_sha256"]):
            raise ValueError("missing/extra historical group artifact")
        if (iteration not in policies or iteration >= 100 or iteration > current or group["evidence_scope"] != scope
                or group.get("diagnostic_mode") or ident["expected_n"] != 4
                or ident["prompt_id"] not in expected_window_prompts(run, iteration)
                or ident["pre_update_policy_fingerprint"] != policies[iteration]["effective_policy_fingerprint"]
                or ident["parent_checkpoint_identity"] != policies[iteration]["checkpoint_identity"]
                or ident["rollout_config_fingerprint"] != digest(run["semantics"]["rollout"])):
            raise ValueError("foreign/stale/diagnostic Main group")
        merge = group.get("static_merge", {})
        cp.check_seal(merge, "merged_checkpoint_fingerprint")
        if (merge.get("version") != run["semantics"]["coordinator_version"] or merge.get("run_identity_sha256") != run["run_identity_sha256"]
                or merge.get("policy_iteration") != iteration
                or merge.get("effective_policy_fingerprint") != policies[iteration]["effective_policy_fingerprint"]
                or merge.get("parent_checkpoint_identity") != policies[iteration]["checkpoint_identity"]
                or merge.get("base_snapshot_sha256") != run["semantics"]["base_model"]["offline_snapshot_sha256"]):
            raise ValueError("Main static merge binding mismatch")
        for member in group["members"]:
            if (member.get("member_seed") != member_seed(run, ident["prompt_id"], member["rollout_index"])
                    or member.get("diagnostic_fixture") or member.get("rollout_executed") is False
                    or any(s["info"].get("diagnostic_fixture") for s in member["steps"])):
                raise ValueError("Main frozen member seed/actual rollout mismatch")
            if not cpu_fixture and (member.get("rollout_executed") is not True
                    or member.get("rllm_provenance", {}).get("generation_loop") != "rllm.workflows.multi_turn_workflow.MultiTurnWorkflow.run"
                    or any(member["reward"].get(k, {}).get("status") != "success" for k in ("accuracy_judge", "query_judge"))):
                raise ValueError("Main requires real rLLM/live reward evidence")
    if checkpoints:
        cp.read_verified_checkpoint(root / "checkpoints" / f"policy-{current:06d}")
    elif authority:
        read_authority(root, run, cpu_fixture=cpu_fixture, full=True)
    for c in checkpoints:
        step = c["policy_iteration"]
        if c["evidence_scope"] != scope or c["eligibility"] != cp.checkpoint_eligibility("main_checkpoint"):
            raise ValueError("Main checkpoint scope/eligibility mismatch")
        if any(by_gid.get(g["identity"]["trajectory_group_id"]) != g for g in c["groups"]):
            raise ValueError("checkpoint consumed missing/changed group metadata")
        if c["window"]["window_id"] != window_id(step - 1):
            raise ValueError("Main window identity mismatch")
        directory = root / "checkpoints" / f"policy-{step:06d}"
        attempt = read_update_attempt(root, c["update_attempt"]["attempt_id"])
        if attempt["phase"] == "checkpoint_staging" and attempt == c["update_attempt"]:
            if reconcile:
                # New verified event ALWAYS full verifies bytes, even recovery.
                attempt = advance_update_attempt(attempt, "verified", checkpoint=c, checkpoint_directory=directory)
                persist_update_attempt(root, attempt, cpu_fixture=cpu_fixture)
            elif step != current:
                raise ValueError("unresolved historical publication")
        elif (attempt["phase"] != "verified" or attempt["verified_checkpoint_identity"] != c["checkpoint_manifest_sha256"]
              or attempt["previous_event_sha256"] != c["update_attempt"]["attempt_event_sha256"]):
            raise ValueError("Main checkpoint/attempt history conflict")
        retention.check_checkpoint_files(root, run, c, checkpoints, finish_cleanup=cleanup, cpu_fixture=cpu_fixture)
        if not cpu_fixture:
            require_main_update_evidence(directory, c)
        if retention.check_retirement(root, run, c["parent_policy"], c["groups"],
                finish_cleanup=cleanup, cpu_fixture=cpu_fixture) is None:
            raise ValueError("consumed window lacks merge retirement authorization")
    attempts = []
    for p in sorted((root / "attempts").iterdir()):
        if p.is_symlink() or not p.is_dir():
            raise ValueError("invalid attempt directory")
        a = read_update_attempt(root, p.name)
        step = a["expected_optimizer_step"]
        if a["run_identity_sha256"] != run["run_identity_sha256"] or step - 1 not in policies or not 1 <= step <= min(current + 1, 100):
            raise ValueError("foreign/future update attempt")
        matching = [g for prompt in expected_window_prompts(run, step - 1) for g in groups
                    if g["identity"]["prompt_id"] == prompt and g["identity"]["policy_iteration"] == step - 1]
        window = build_training_window(run, policies[step - 1], matching, window_id=window_id(step - 1))
        if (a["window_sha256"] != window["window_sha256"]
                or a["parent_checkpoint_identity"] != policies[step - 1]["checkpoint_identity"]
                or a["parent_policy_fingerprint"] != policies[step - 1]["effective_policy_fingerprint"]):
            raise ValueError("attempt Main window/parent mismatch")
        if step <= current and a["phase"] not in {"failed", "verified"}:
            if not (not reconcile and step == current and a == checkpoints[-1]["update_attempt"]):
                raise ValueError("unresolved historical attempt")
        if a["phase"] == "verified" and (step not in by_step or a["attempt_id"] != by_step[step]["update_attempt"]["attempt_id"]):
            raise ValueError("duplicate/foreign verified attempt")
        attempts.append(a)
    if len([a for a in attempts if a["expected_optimizer_step"] == current + 1 and a["phase"] != "failed"]) > 1:
        raise ValueError("multiple active Main attempts")
    expected = expected_window_prompts(run, current) if current < 100 else []
    current_groups = [g for prompt in expected for g in groups if g["identity"]["prompt_id"] == prompt
                      and g["identity"]["policy_iteration"] == current]
    for g in current_groups:
        read_formal_group(root / "groups" / g["identity"]["trajectory_group_id"])
    if current < 100:
        retention.check_retirement(root, run, policies[current], current_groups,
                                  finish_cleanup=cleanup, cpu_fixture=cpu_fixture)
    # Orphan receipts must never silently authorize a future/current deletion.
    expected_receipts = {receipt.parent.name for c in checkpoints for receipt in (
        retention.receipt_path(root, "merge", c["policy_iteration"] - 1),)}
    expected_receipts |= {retention.receipt_path(root, "merge", current).parent.name} if len(current_groups) == 4 else set()
    expected_receipts |= {retention.receipt_path(root, "compact", c["policy_iteration"]).parent.name
                         for c in checkpoints[:-1] if c["policy_iteration"] not in retention.MILESTONES}
    if any(p.name not in expected_receipts for p in (root / "retention").glob("*") if not p.name.startswith(".")):
        raise ValueError("orphan/current/milestone retention authorization")
    allowed_merge = f"policy-{current:06d}" if current < 100 else None
    if any(p.name != allowed_merge and not p.name.startswith(".") for p in (root / "merges").glob("*")):
        raise ValueError("unretired historical/foreign static merge")
    return dict(checkpoints=checkpoints, groups=groups, policy=ledger["policy"], ledger=ledger, attempts=attempts,
        continuation_authority=authority,
        current_groups=current_groups, missing_prompts=[p for p in expected if p not in {g["identity"]["prompt_id"] for g in current_groups}])


def require_main_update_evidence(directory, checkpoint):
    """W4 specialization of existing O/C/one-step/LoRA/fresh-native evidence."""
    from .formal_smoke import require_runtime_update_evidence
    from opensearch_vl_repro.sft_tool_audit import sha256_file
    require_runtime_update_evidence(directory, checkpoint, world_size=4)
    required = {f"formal_update_rank_{rank}.json" for rank in range(4)} | {f"runtime_state_rank_{rank}.json" for rank in range(4)}
    if not required <= checkpoint["artifact_role_files"].get("metadata", {}).keys():
        raise ValueError("Main requires four immutable bound rank evidence/runtime state files")
    for prefix, role in (("model", "native"), ("optim", "optimizer"), ("extra_state", "rng")):
        if set(checkpoint["artifact_role_files"][role]) != {
                f"distributed/{prefix}_world_size_4_rank_{rank}.pt" for rank in range(4)}:
            raise ValueError("Main requires complete four-rank native/optimizer/RNG role layout")
    for name in required:
        if sha256_file(Path(directory) / name) != checkpoint["file_sha256"][name]:
            raise ValueError("Main historical rank metadata checksum mismatch")
    for rank in range(4):
        row = json.loads((Path(directory) / f"formal_update_rank_{rank}.json").read_text(encoding="utf-8"))
        actor = row["actor_contract"]
        if (row["loaded_parent"].get("world_size") != 4 or row["loaded_parent"].get("rank") != rank
                or row["rank_plan"].get("world_size") != 4 or row["rank_plan"].get("rank") != rank
                or row["rank_plan"].get("window_sha256") != checkpoint["window"]["window_sha256"]
                or any(actor.get(k) != v for k, v in dict(fsdp2=True, bf16=True, model_training=True,
                    language_model_training=True, decoder_layers=36, decoder_layers_training=36,
                    decoder_layers_gradient_checkpointing=36, effective_attention_implementation="flash_attention_2").items())
                or actor["dropout"].get("source_adapter_lora_dropout") != .05
                or actor["dropout"].get("runtime_effective_lora_dropout") != 0.
                or {r["rank"] for r in row["fresh_reload"]["per_rank"]} != set(range(4))):
            raise ValueError("Main W4 actor/parent/rank/fresh-reload evidence mismatch")


def commit_main_checkpoint(root, staging, manifest, *, cpu_fixture=False):
    """S1 publication invariants, with S4 bounded recovery instead of S1 history I/O."""
    root, staging = Path(root), Path(staging)
    cp.validate_checkpoint_manifest(manifest)
    with run_lock(root / ".formal.lock"):
        recovered = recover_main(root, manifest["run"], cpu_fixture=cpu_fixture, reconcile=False, cleanup=False)
        if recovered["policy"] != manifest["parent_policy"] or recovered["missing_prompts"]:
            raise ValueError("duplicate/stale/incomplete Main checkpoint successor")
        if (manifest["evidence_scope"] != ("cpu_fixture" if cpu_fixture else "runtime")
                or manifest["eligibility"] != cp.checkpoint_eligibility("main_checkpoint")
                or manifest["groups"] != recovered["current_groups"]):
            raise ValueError("Main publication scope/group/kind mismatch")
        if retention.check_retirement(root, manifest["run"], manifest["parent_policy"], manifest["groups"]) is None:
            raise ValueError("Main update publication requires retired merge")
        if read_update_attempt(root, manifest["update_attempt"]["attempt_id"]) != manifest["update_attempt"]:
            raise ValueError("durable checkpoint-staging attempt required")
        if not cpu_fixture:
            require_main_update_evidence(staging, manifest)
        cp.validate_policy(checkpoint_policy(manifest))
        cp.verify_artifacts(staging, manifest["file_sha256"])
        cp.publish_directory(staging, root / "checkpoints" / f"policy-{manifest['policy_iteration']:06d}",
                             "checkpoint.json", manifest, cpu_fixture=cpu_fixture)
    return manifest


def final_reconstruction(root, run, *, cpu_fixture=False):
    if not cpu_fixture and run["semantics"].get("continuation"):
        from .formal_main_continuation import require_active
        require_active(root, run)
    value = recover_main(root, run, cpu_fixture=cpu_fixture)
    checkpoints, groups = value["checkpoints"], value["groups"]
    inherited = dict(windows=0, groups=0, members=0, prompt_ids=[])
    authority = value["continuation_authority"]
    if authority:
        from .formal_main_continuation import final_prefix_audit, read_authority
        read_authority(root, run, cpu_fixture=cpu_fixture, full=True)
        inherited = final_prefix_audit(Path(root).parents[2], authority, run, cpu_fixture=cpu_fixture)
    ordered_prompts = inherited["prompt_ids"] + [g["identity"]["prompt_id"] for c in checkpoints for g in c["groups"]]
    if ordered_prompts != run["prompt_ids"]:
        raise ValueError("final inherited prefix + local suffix membership/order/gap mismatch")
    if (len(checkpoints) + inherited["windows"] != 100 or len(groups) + inherited["groups"] != 400
            or sum(len(g["members"]) for g in groups) + inherited["members"] != 1600
            or value["policy"]["global_optimizer_step"] != 100
            or len(value["ledger"]["consumed_group_ids"]) != 400
            or any(e["status"] != "consumed_by_verified_checkpoint" for e in value["ledger"]["prompts"].values())
            or any(a["phase"] not in {"verified", "failed"} for a in value["attempts"])
            or any((Path(root) / "merges").iterdir())):
        raise ValueError("incomplete Main400 final immutable reconstruction")
    for g in groups:
        read_formal_group(Path(root) / "groups" / g["identity"]["trajectory_group_id"])
    for c in checkpoints:
        retention.check_checkpoint_files(root, run, c, checkpoints, heavy=True, cpu_fixture=cpu_fixture)
        step = c["policy_iteration"]
        if step not in retention.MILESTONES and not retention.receipt_path(root, "compact", step).exists():
            raise ValueError("historical non-milestone checkpoint not compacted")
    # Private historical failed attempts stay forensic, but none may belong to
    # an unconsumed prompt (the final ordered membership is fully consumed).
    audit_private_collections(root, run)
    result = dict(version=VERSION, scope="cpu_fixture" if cpu_fixture else "runtime", passed=not cpu_fixture,
        checks=dict(immutable_chain=True, ordered_groups400=True, members1600=True, windows100=True,
                    verified_steps100=True, retention_verified=True, no_active_merge=True),
        run_identity=run, final_policy=value["policy"], eligible_for_main_init=False,
        checkpoint_kind="main_checkpoint", groups_completed=400, trajectories_completed=1600,
        optimizer_steps=list(range(1, 101)))
    if authority:
        from .formal_main_continuation import report_fields
        result.update(report_fields(authority, len(checkpoints)))
    return result


def audit_private_collections(root, run):
    """Preserve historical failure forensics, but never guess their authority."""
    for directory in (Path(root) / "groups").glob(".collect-*"):
        p = directory / "attempt.json"
        if directory.is_symlink() or not directory.is_dir() or p.is_symlink() or not p.is_file():
            raise ValueError("unbound partial collection remains; inspect forensic directory")
        a = json.loads(p.read_text(encoding="utf-8"))
        cp.check_seal(a, "trajectory_group_id")
        if (a["run_identity_sha256"] != run["run_identity_sha256"]
                or type(a["policy_iteration"]) is not int or not 0 <= a["policy_iteration"] < 100
                or a["prompt_id"] not in expected_window_prompts(run, a["policy_iteration"])
                or a["source_identity"] != cp.source_identity_for_prompt(run, a["prompt_id"])):
            raise ValueError("partial current/foreign collection remains")


def publish_final(output, reports, report, *, cpu_fixture=False):
    from .formal_smoke import revoke_pass
    if cpu_fixture or report.get("passed") is not True or report.get("scope") != "runtime":
        raise ValueError("CPU/report-only evidence cannot publish Main runtime PASS")
    revoke_pass(output)
    try:
        derived = final_reconstruction(output, report["run_identity"])
        if any(report.get(k) != v for k, v in derived.items()):
            raise ValueError("Main report differs from heavyweight reconstruction")
        cp.durable_json(Path(reports) / "report.json", report)
        cp.durable_json(Path(output) / "manifest.json", report)  # LAST PASS artifact
    except BaseException:
        revoke_pass(output)
        raise
