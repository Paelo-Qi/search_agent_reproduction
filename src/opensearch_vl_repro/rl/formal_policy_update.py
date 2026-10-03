"""S2 single-window dataplane. No collector, scheduler, coordinator or PASS publication.

Only production defaults construct the existing verl FSDP2 actor/checkpoint
manager. Explicit CPU fixtures cannot issue runtime checkpoint evidence.
"""
from __future__ import annotations

import gc
import json
import math
import random
import weakref
from dataclasses import dataclass, field
from pathlib import Path

from opensearch_vl_repro.eval_subset import canonical_json_sha256
from .checkpoint import (
    artifact_inventory, artifact_role_identities, check_seal, durable_json, initial_policy,
    read_verified_checkpoint, seal, validate_policy, validate_training_run_identity, verify_artifacts,
)
from .old_logprob import fingerprint, parameter_fingerprint, rng_fingerprint
from .rl_actor_semantics import (
    configure_rl_lora_dropout_runtime, contract_sft_config, execution_contract,
    require_rl_lora_dropout_runtime, require_saved_source_dropout,
)

_RELOAD_SEAL = object()


@dataclass(frozen=True)
class FormalReloadReceipt:
    artifact: dict
    actor_id: int
    source_files: dict = field(repr=False)
    seal: object = field(repr=False)


@dataclass
class LoadedFormalActor:
    """Owns the actor so staging verification can destroy it BEFORE constructing fresh."""
    actor: object
    policy: dict
    reload_receipt: FormalReloadReceipt
    rank: int
    world_size: int
    cpu_fixture: bool = False


def _collect(value, world_size, *, cpu_fixture=False):
    import torch.distributed as dist
    if cpu_fixture:
        if world_size != 1:
            raise ValueError("live CPU actor fixtures use W=1; generic plans are tested separately")
        return [value]
    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise ValueError("formal primitives require the configured distributed process group")
    rows = [None] * world_size
    dist.all_gather_object(rows, value)
    return rows


def _rank(world_size, cpu_fixture):
    import torch.distributed as dist
    if cpu_fixture:
        if world_size != 1:
            raise ValueError("CPU fixture world size must be 1")
        return 0
    if not dist.is_initialized() or dist.get_world_size() != world_size:
        raise ValueError("configured world size differs from actual FSDP group")
    return dist.get_rank()


def _boundary(operation, world_size, cpu_fixture):
    """Sync local guard failures before peers enter an FSDP operation.

    Failures inside FSDP itself still require the launcher to tear down the group.
    This is not a provider retry or distributed orchestration loop.
    """
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = _collect(error, world_size, cpu_fixture=cpu_fixture)
    if any(e is not None for e in errors):
        raise ValueError(f"formal collective guard failed: {errors}")
    return result


def _rank_identity(rows, key):
    if {r["rank"] for r in rows} != set(range(len(rows))):
        raise ValueError("complete distinct runtime state ranks required")
    return canonical_json_sha256({str(r["rank"]): r[key] for r in sorted(rows, key=lambda r: r["rank"])})


def optimizer_step_counter(actor, expected):
    """Global step plus actual AdamW moments, not an adapter-only counter claim."""
    import torch
    if type(expected) is not int or expected < 0 or not isinstance(actor.actor_optimizer, torch.optim.AdamW):
        raise ValueError("formal actor requires AdamW and nonnegative global step")
    steps = [float(state["step"].item()) if isinstance(state.get("step"), torch.Tensor) else state.get("step")
             for state in actor.actor_optimizer.state.values()]
    if expected == 0:
        if steps or actor.actor_optimizer.state:
            raise ValueError("initial policy requires fresh empty AdamW")
    elif not steps or any(type(s) not in (int, float) or not 0 < s <= expected or int(s) != s for s in steps) or max(steps) != expected:
        raise ValueError("AdamW state/global optimizer step mismatch")
    return expected


def _snapshot(actor, manager, rank, global_step):
    from .actor_gate import trainable_policy
    trainable_policy(actor.actor_module, actor.actor_optimizer)
    optimizer_step_counter(actor, global_step)
    return dict(rank=rank, global_optimizer_step=global_step,
                parameter_sha256=parameter_fingerprint(actor),
                optimizer_state_sha256=fingerprint(actor.actor_optimizer.state_dict()),
                # Use the EXACT RNG representation owned by official native extra_state.
                native_rng_sha256=fingerprint(manager.get_rng_state()))


def _options(run):
    opt, ppo = run["semantics"]["optimizer"], run["semantics"]["ppo"]
    lr = opt.get("learning_rate", opt.get("lr"))
    epochs = ppo.get("epochs", ppo.get("ppo_epochs"))
    if (opt.get("name", opt.get("optimizer")) != "AdamW" or type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0
            or ("lr" in opt and "learning_rate" in opt and opt["lr"] != lr)
            or type(opt.get("weight_decay")) not in (int, float) or not math.isfinite(opt["weight_decay"]) or opt["weight_decay"] < 0
            or type(epochs) is not int or epochs != 1
            or ppo.get("clip_ratio_low") != .2 or ppo.get("clip_ratio_high") != .28):
        raise ValueError("formal optimizer/PPO semantics must bind AdamW, one epoch and frozen clipping")
    return dict(optimizer=dict(learning_rate=lr, weight_decay=opt["weight_decay"]),
                clip_ratio_low=.2, clip_ratio_high=.28)


def _require_update_config(actor, local_row_count):
    expected = dict(ppo_mini_batch_size=local_row_count, ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=1, shuffle=False, clip_ratio_low=.2, clip_ratio_high=.28, entropy_coeff=0.,
        use_kl_loss=False, use_rollout_log_probs=True, loss_agg_mode="seq-mean-token-mean", use_dynamic_bsz=False)
    if any(getattr(actor.config, k, None) != v for k, v in expected.items()):
        raise ValueError("formal one-window actor reduction/update configuration changed")


def _defaults(construct, manager_factory, cpu_fixture):
    from .verl_actor_gate import construct_rl_actor, checkpoint_manager
    if not cpu_fixture and (construct is not None or manager_factory is not None):
        raise ValueError("injected actor/checkpoint backends are CPU fixture evidence only")
    if not cpu_fixture:
        import importlib.metadata
        if importlib.metadata.version("verl") != "0.6.1":
            raise ValueError("S2 loss/checkpoint contract requires pinned verl 0.6.1")
    return construct or construct_rl_actor, manager_factory or checkpoint_manager


def _issue_reload(actor, policy, snapshot, runtime, source_files, *, run, rank, world_size, scope):
    artifact = seal(dict(schema_version=1, scope=scope, rank=rank, world_size=world_size,
        policy=policy, base_model=run["semantics"]["base_model"], actor_options=_options(run),
        source_files_sha256=fingerprint(source_files),
        **{k: v for k, v in snapshot.items() if k != "rank"},
        live_rng_sha256=rng_fingerprint(), lora_dropout_runtime_sha256=runtime["lora_dropout_runtime_sha256"],
        model_loaded=True, adapter_identity_match=True, native_identity_match=True,
        optimizer_identity_match=True, rng_identity_match=True, global_step_match=True,
        execution_contract_match=True, runtime_dropout_match=True), "reload_receipt_sha256")
    return FormalReloadReceipt(artifact, id(actor), source_files, _RELOAD_SEAL)


def require_reload_capability(receipt, policy):
    """No JSON boolean (including recovery_reload_required=False) is authority."""
    if not isinstance(receipt, FormalReloadReceipt) or receipt.seal is not _RELOAD_SEAL:
        raise ValueError("actual S2 model/optimizer/RNG reload receipt required")
    check_seal(receipt.artifact, "reload_receipt_sha256")
    if receipt.artifact["policy"] != policy:
        raise ValueError("reload receipt policy identity mismatch")
    return receipt.artifact


def require_formal_reload(actor, policy, receipt, *, window=None):
    from .actor_gate import trainable_policy
    from opensearch_vl_repro.sft_tool_audit import sha256_file
    validate_policy(policy)
    a = require_reload_capability(receipt, policy)
    if receipt.actor_id != id(actor):
        raise ValueError("stale/foreign loaded actor receipt")
    if not receipt.source_files or fingerprint(receipt.source_files) != a["source_files_sha256"]:
        raise ValueError("reload source inventory changed")
    if window is not None:
        check_seal(window, "window_sha256")
        if (window["parent_checkpoint_identity"] != policy["checkpoint_identity"]
                or window["parent_policy_fingerprint"] != policy["effective_policy_fingerprint"]
                or window["policy_iteration"] != policy["policy_iteration"]
                or window["run_identity_sha256"] != policy["run_identity_sha256"]
                or window["expected_optimizer_step"] != policy["global_optimizer_step"] + 1):
            raise ValueError("loaded actor/window lineage mismatch")
    trainable_policy(actor.actor_module, actor.actor_optimizer)
    runtime = require_rl_lora_dropout_runtime(actor.actor_module, contract_sft_config(policy["execution_contract"]))
    if (parameter_fingerprint(actor) != a["parameter_sha256"]
            or fingerprint(actor.actor_optimizer.state_dict()) != a["optimizer_state_sha256"]
            or rng_fingerprint() != a["live_rng_sha256"]
            or getattr(actor, "_formal_global_optimizer_step", None) != policy["global_optimizer_step"]
            or runtime["lora_dropout_runtime_sha256"] != a["lora_dropout_runtime_sha256"]
            or any(not Path(path).is_file() or sha256_file(Path(path)) != sha for path, sha in receipt.source_files.items())):
        raise ValueError("loaded model/optimizer/RNG/files/execution identity changed")
    optimizer_step_counter(actor, policy["global_optimizer_step"])
    return a


def load_formal_actor(run, *, config, source_adapter, processor, mesh, policy=None,
                      checkpoint_directory=None, initial_seed=None, cpu_fixture=False,
                      construct=None, manager_factory=None):
    """Iteration zero binds original SFT; N>0 requires S1 immutable native checkpoint.

    Returns the actual initial PolicyIdentity if policy is omitted. S3 must use it
    to initialize S1's run anchor, never invent empty optimizer/RNG identities.
    """
    import numpy as np
    import torch
    from .checkpoint import build_rl_lineage
    from .run_state import checkpoint_policy
    validate_training_run_identity(run)
    construct, manager_factory = _defaults(construct, manager_factory, cpu_fixture)
    world = run["semantics"]["world_size"]
    rank = _rank(world, cpu_fixture)
    options = _options(run)
    base, source = run["semantics"]["base_model"], run["semantics"]["source_sft"]
    if (config["model"]["name_or_path"] != base["name"] or config["model"]["revision"] != base["revision"]
            or execution_contract(config) != run["semantics"]["execution_contract"]):
        raise ValueError("formal base/revision/execution config identity mismatch")
    files, expected, manifest = {}, None, None
    if checkpoint_directory is None:
        if policy is not None and policy["policy_iteration"] != 0:
            raise ValueError("updated policy requires a verified native checkpoint, not just an adapter")
        adapter = Path(source_adapter).resolve()
        def source_checks():
            actual = build_rl_lineage(config={"model": {"continue_from_sft_adapter": True, "sft_stage": source["stage"]}},
                                     sft_config=config, adapter_path=adapter, run_id=run["run_id"])
            if (actual.sft_adapter_fingerprint != source["adapter_sha256"]
                    or actual.sft_checkpoint_metadata_fingerprint != source["metadata_sha256"]
                    or actual.sft_stage != source["stage"] or adapter.parent.name != "checkpoint-3k"
                    or list(actual.sft_lineage) != source["lineage"]):
                raise ValueError("original checkpoint-3k SFT lineage mismatch (Gate artifacts forbidden)")
            require_saved_source_dropout(adapter)
            from opensearch_vl_repro.inference.adapter import adapter_file_identity
            return {str(adapter / name): sha for name, sha in adapter_file_identity(adapter)["file_sha256"].items()}
        files = _boundary(source_checks, world, cpu_fixture)
        files[str(adapter.parent / "metadata.json")] = source["metadata_sha256"]
        if type(initial_seed) is not int or initial_seed < 0:
            raise ValueError("explicit frozen initial RNG seed required")
    else:
        directory = Path(checkpoint_directory).resolve()
        def native_checks():
            value = read_verified_checkpoint(directory)
            if value["run"] != run or value["evidence_scope"] != ("cpu_fixture" if cpu_fixture else "runtime"):
                raise ValueError("foreign run/scope native checkpoint")
            roles, inventory = checkpoint_staging_roles(directory, world, exclude=("checkpoint.json",))
            if roles != value["artifact_role_files"] or inventory != value["file_sha256"]:
                raise ValueError("native layout does not match verified S1.1 roles")
            if policy is None or checkpoint_policy(value) != policy:
                raise ValueError("checkpoint does not match requested PolicyIdentity")
            return value
        manifest = _boundary(native_checks, world, cpu_fixture)
        adapter = directory / "adapter"
        require_saved_source_dropout(adapter)
        files = {str(directory / name): sha for name, sha in manifest["file_sha256"].items()}
        from opensearch_vl_repro.sft_tool_audit import sha256_file
        files[str(directory / "checkpoint.json")] = sha256_file(directory / "checkpoint.json")
        expected = json.loads((directory / f"runtime_state_rank_{rank}.json").read_text(encoding="utf-8"))
        check_seal(expected, "runtime_state_sha256")
        if (expected["global_optimizer_step"] != policy["global_optimizer_step"]
                or expected["rank"] != rank or expected["execution_contract"] != policy["execution_contract"]
                or expected["window_sha256"] != manifest["window"]["window_sha256"]):
            raise ValueError("native extra metadata/global step mismatch")
    actor, _ = construct(config=config, gate=options, adapter=adapter, mesh=mesh)
    manager = manager_factory(actor, processor)
    if manifest is None:
        random.seed(initial_seed)
        np.random.seed(initial_seed)
        torch.manual_seed(initial_seed)
        if not cpu_fixture:
            torch.cuda.manual_seed(initial_seed)
        snap = _snapshot(actor, manager, rank, 0)
        ranks = _collect(snap, world, cpu_fixture=cpu_fixture)
        actual_policy = initial_policy(run, optimizer_identity=_rank_identity(ranks, "optimizer_state_sha256"),
                                      rng_identity=_rank_identity(ranks, "native_rng_sha256"))
        if policy is not None and policy != actual_policy:
            raise ValueError("initial optimizer/RNG PolicyIdentity mismatch")
        policy = actual_policy
    else:
        from .actor_gate import lora_snapshot, reload_matches
        adapter_weights = lora_snapshot(actor.actor_module)
        manager.load_checkpoint(str(directory / "distributed"), del_local_after_load=False)
        configure_rl_lora_dropout_runtime(actor.actor_module, config)
        snap = _snapshot(actor, manager, rank, policy["global_optimizer_step"])
        if (not reload_matches(adapter_weights, lora_snapshot(actor.actor_module))
                or any(snap[k] != expected.get(k) for k in snap)):
            raise ValueError("actual native model/adapter/AdamW/RNG reload mismatch")
    actor._formal_global_optimizer_step = policy["global_optimizer_step"]
    if any(g["lr"] != options["optimizer"]["learning_rate"] or g["weight_decay"] != options["optimizer"]["weight_decay"]
           for g in actor.actor_optimizer.param_groups):
        raise ValueError("loaded optimizer hyperparameters differ from frozen run")
    runtime = require_rl_lora_dropout_runtime(actor.actor_module, config)
    receipt = _issue_reload(actor, policy, snap, runtime, files, run=run, rank=rank, world_size=world,
                            scope="cpu_fixture" if cpu_fixture else "runtime")
    del manager
    return LoadedFormalActor(actor, policy, receipt, rank, world, cpu_fixture)


def checkpoint_staging_roles(directory, world_size, *, exclude=()):
    """Physical native layout -> exact, complete S1.1 multi-file role maps."""
    inventory = artifact_inventory(Path(directory), exclude=exclude)
    roles = {role: {} for role in ("adapter", "native", "optimizer", "rng", "metadata")}
    expected = {f"distributed/{prefix}_world_size_{world_size}_rank_{rank}.pt": role
                for prefix, role in (("model", "native"), ("optim", "optimizer"), ("extra_state", "rng"))
                for rank in range(world_size)}
    if not expected.keys() <= inventory.keys():
        raise ValueError("native checkpoint missing a model/optimizer/extra shard")
    for name, sha in inventory.items():
        if name in expected:
            role = expected[name]
        elif name.startswith("adapter/"):
            role = "adapter"
        else:
            if name.startswith("distributed/") and Path(name).name.startswith(("model_world_size_", "optim_world_size_", "extra_state_world_size_")):
                raise ValueError("unexpected native checkpoint rank/world size")
            role = "metadata"
        roles[role][name] = sha
    roles = {role: files for role, files in roles.items() if files}
    artifact_role_identities(roles, inventory)
    return roles, inventory


def update_formal_window(loaded, data, batch_receipt, *, root, run, groups, window, reward_window,
                         attempt, trainer_state=None):
    """One official PPO update; durable uncertainty marker precedes update_policy.

    Returns staging-phase attempt/evidence; it does NOT commit a checkpoint/run.
    """
    from .training_window import validate_training_window
    from .training_batch import formal_training_rows, materialize_actor_microbatches, require_formal_batch
    from .old_logprob import prepare_formal_old_log_probs, verify_formal_old_receipt
    from .policy_alignment import formal_alignment_artifact, require_policy_alignment
    from .run_state import (advance_update_attempt, persist_update_attempt, read_update_attempt,
                            transition_trainer_state, validate_update_attempt)
    from .verl_policy_update import configure_one_update, audited_policy_update
    policy, actor = loaded.policy, loaded.actor
    def guards():
        validate_training_window(window, run, policy, groups)
        formal_training_rows(window, run, policy, groups, reward_window)
        require_formal_reload(actor, policy, loaded.reload_receipt, window=window)
        batch = require_formal_batch(data, batch_receipt, window)
        if (batch["rank"] != loaded.rank or batch["world_size"] != loaded.world_size
                or reward_window["status"] != "signal"
                or (not loaded.cpu_fixture and (reward_window["test_estimator_injected"] is not False
                    or any(g["evidence_scope"] != "runtime" for g in groups)))):
            raise ValueError("rank/scope/signal mismatch")
        validate_update_attempt(attempt)
        if (attempt["phase"] != "prepared" or attempt["window_sha256"] != window["window_sha256"]
                or attempt["parent_checkpoint_identity"] != policy["checkpoint_identity"]
                or attempt["parent_policy_fingerprint"] != policy["effective_policy_fingerprint"]
                or attempt["run_identity_sha256"] != run["run_identity_sha256"]
                or attempt["expected_optimizer_step"] != policy["global_optimizer_step"] + 1):
            raise ValueError("fresh parent-bound prepared attempt required")
        if trainer_state is not None:
            transition_trainer_state(trainer_state, "updating", reload_receipt=loaded.reload_receipt, actor=actor)
        return batch
    batch = _boundary(guards, loaded.world_size, loaded.cpu_fixture)
    configure_one_update(actor, batch["local_row_count"], _options(run))
    _require_update_config(actor, batch["local_row_count"])
    prepared = prepare_formal_old_log_probs(actor, data, window=window, policy=policy,
        reload_receipt=loaded.reload_receipt, batch_receipt=batch_receipt)
    alignment = formal_alignment_artifact(_collect(prepared["alignment"], loaded.world_size,
        cpu_fixture=loaded.cpu_fixture), world_size=loaded.world_size, window_sha256=window["window_sha256"])
    require_policy_alignment(alignment)
    _boundary(lambda: verify_formal_old_receipt(actor, data, prepared["receipt"], alignment, window=window,
        policy=policy, reload_receipt=loaded.reload_receipt, batch_receipt=batch_receipt), loaded.world_size, loaded.cpu_fixture)
    _boundary(lambda: _require_update_config(actor, batch["local_row_count"]), loaded.world_size, loaded.cpu_fixture)
    def persist_boundary():
        nonlocal attempt
        if loaded.rank == 0:
            persist_update_attempt(root, attempt, cpu_fixture=loaded.cpu_fixture)
            for phase in ("started", "step_may_have_run"):
                attempt = advance_update_attempt(attempt, phase)
                persist_update_attempt(root, attempt, cpu_fixture=loaded.cpu_fixture)
        return True
    _boundary(persist_boundary, loaded.world_size, loaded.cpu_fixture)
    attempt = read_update_attempt(root, attempt["attempt_id"])
    before = optimizer_step_counter(actor, policy["global_optimizer_step"])
    try:
        # Nothing touches optimizer before the durable marker, including zero_grad.
        with materialize_actor_microbatches(actor):
            metrics, audit = audited_policy_update(actor, data, runtime_receipt=prepared["receipt"].artifact)
        after = optimizer_step_counter(actor, before + 1)
        actor._formal_global_optimizer_step = after
        actor.actor_optimizer.zero_grad(set_to_none=True)
        def staging_boundary():
            nonlocal attempt
            if loaded.rank == 0:
                attempt = advance_update_attempt(attempt, "checkpoint_staging")
                persist_update_attempt(root, attempt, cpu_fixture=loaded.cpu_fixture)
        _boundary(staging_boundary, loaded.world_size, loaded.cpu_fixture)
        attempt = read_update_attempt(root, attempt["attempt_id"])
        return dict(attempt=attempt, metrics=metrics, update_audit=audit, before_step=before, after_step=after,
            expected_optimizer_step=window["expected_optimizer_step"], alignment=alignment,
            old_logprob_receipt=prepared["receipt"].artifact, rollout_actor_handoff=prepared["handoff"],
            window_sha256=window["window_sha256"])
    except BaseException as exc:
        # No consumption or policy advancement. S1 rollback MUST destroy uncertain memory.
        if loaded.rank == 0:
            failed = advance_update_attempt(attempt, "failed", failure_reason=f"{type(exc).__name__}: {exc}")
            persist_update_attempt(root, failed, cpu_fixture=loaded.cpu_fixture)
        raise


def save_formal_staging(loaded, processor, staging, *, window, update_evidence, manager_factory=None, save=None):
    """Official native save + PEFT export; NOT immutable publication or PASS."""
    from .verl_actor_gate import checkpoint_manager, save_checkpoint
    from .run_state import validate_update_attempt
    from .checkpoint import _formal_root
    from .rl_actor_semantics import require_update_forward_audit
    actor, staging = loaded.actor, _formal_root(Path(staging))
    if not loaded.cpu_fixture and (manager_factory is not None or save is not None):
        raise ValueError("injected save backend is CPU fixture evidence only")
    validate_update_attempt(update_evidence["attempt"])
    if (update_evidence["attempt"]["phase"] != "checkpoint_staging"
            or update_evidence["window_sha256"] != window["window_sha256"]
            or update_evidence["attempt"]["window_sha256"] != window["window_sha256"]
            or window["parent_policy_fingerprint"] != loaded.policy["effective_policy_fingerprint"]
            or window["parent_checkpoint_identity"] != loaded.policy["checkpoint_identity"]
            or update_evidence["before_step"] != loaded.policy["global_optimizer_step"]
            or update_evidence["after_step"] != window["expected_optimizer_step"]
            or getattr(actor, "_formal_global_optimizer_step", None) != update_evidence["after_step"]
            or update_evidence["update_audit"]["optimizer_step_count"] != 1
            or not staging.name.startswith(".")):
        raise ValueError("exclusive staging and exactly one bound update required")
    runtime = require_rl_lora_dropout_runtime(actor.actor_module, contract_sft_config(loaded.policy["execution_contract"]))
    require_update_forward_audit(update_evidence["update_audit"]["rl_dropout_execution_audit"],
        loaded.policy["execution_contract"], runtime["lora_dropout_runtime_sha256"])
    # The shared directory is reserved by rank0 ONLY; all ranks rendezvous before
    # native save collectives, so mkdir races cannot strand peers in FSDP save.
    def reserve():
        if loaded.rank == 0:
            staging.mkdir(parents=True, exist_ok=False)
    _boundary(reserve, loaded.world_size, loaded.cpu_fixture)
    manager_factory, save = manager_factory or checkpoint_manager, save or save_checkpoint
    manager = manager_factory(actor, processor)
    snap = _snapshot(actor, manager, loaded.rank, update_evidence["after_step"])
    snap = seal({**snap, "window_sha256": window["window_sha256"],
                 "execution_contract": loaded.policy["execution_contract"]}, "runtime_state_sha256")
    save(actor, processor, staging, global_step=update_evidence["after_step"])
    require_saved_source_dropout(staging / "adapter")
    durable_json(staging / f"runtime_state_rank_{loaded.rank}.json", snap, cpu_fixture=loaded.cpu_fixture)
    # A real finite actor save must not advance RNG/optimizer/weights.
    if any(_snapshot(actor, manager, loaded.rank, snap["global_optimizer_step"])[k] != snap[k]
           for k in ("parameter_sha256", "optimizer_state_sha256", "native_rng_sha256")):
        raise ValueError("checkpoint save changed live state")
    _collect(True, loaded.world_size, cpu_fixture=loaded.cpu_fixture)
    roles, files = checkpoint_staging_roles(staging, loaded.world_size)
    return seal(dict(artifact_role_files=roles, file_sha256=files,
                     artifact_roles=artifact_role_identities(roles, files),
                     window_sha256=window["window_sha256"], global_optimizer_step=snap["global_optimizer_step"],
                     scope="cpu_fixture" if loaded.cpu_fixture else "runtime"), "staging_sha256")


def fresh_reload_staging(loaded, data, staging, description, *, config, processor, mesh,
                         construct=None, manager_factory=None):
    """Destroy old actor -> fresh PEFT/native/AdamW/RNG -> finite MM forward.

    Caller must not retain actor/module/optimizer refs. Missing destruction or
    reload evidence fails closed; final S1 commit remains caller-owned (S3).
    """
    import torch
    from .actor_gate import lora_snapshot, reload_matches
    from .training_batch import materialize_actor_microbatches
    construct, manager_factory = _defaults(construct, manager_factory, loaded.cpu_fixture)
    check_seal(description, "staging_sha256")
    staging = Path(staging)
    verify_artifacts(staging, description["file_sha256"])
    if (description["scope"] != ("cpu_fixture" if loaded.cpu_fixture else "runtime")
            or artifact_role_identities(description["artifact_role_files"], description["file_sha256"]) != description["artifact_roles"]
            or execution_contract(config) != loaded.policy["execution_contract"]
            or config["model"]["name_or_path"] != loaded.reload_receipt.artifact["base_model"]["name"]
            or config["model"]["revision"] != loaded.reload_receipt.artifact["base_model"]["revision"]
            or description["global_optimizer_step"] != loaded.policy["global_optimizer_step"] + 1):
        raise ValueError("fresh reload staging identity mismatch")
    expected = json.loads((staging / f"runtime_state_rank_{loaded.rank}.json").read_text(encoding="utf-8"))
    check_seal(expected, "runtime_state_sha256")
    if (expected["window_sha256"] != description["window_sha256"]
            or expected["global_optimizer_step"] != description["global_optimizer_step"]
            or expected["execution_contract"] != loaded.policy["execution_contract"]):
        raise ValueError("staged runtime state/window/step/contract mismatch")
    module_ref, optimizer_ref = weakref.ref(loaded.actor.actor_module), weakref.ref(loaded.actor.actor_optimizer)
    loaded.actor = None
    gc.collect()
    if not loaded.cpu_fixture:
        torch.cuda.empty_cache()
    _boundary(lambda: _require_destroyed(module_ref, optimizer_ref), loaded.world_size, loaded.cpu_fixture)
    require_saved_source_dropout(staging / "adapter")
    # Bind constructor options to run semantics AND the saved actual param_groups.
    optimizer_state = torch.load(staging / "distributed" / f"optim_world_size_{loaded.world_size}_rank_{loaded.rank}.pt",
                                 map_location="cpu", weights_only=False)
    opt = optimizer_state["param_groups"][0]
    options = loaded.reload_receipt.artifact["actor_options"]
    if (opt["lr"] != options["optimizer"]["learning_rate"] or opt["weight_decay"] != options["optimizer"]["weight_decay"]):
        raise ValueError("staged optimizer hyperparameters differ from frozen run")
    actor, _ = construct(config=config, gate=options,
                         adapter=staging / "adapter", mesh=mesh)
    manager = manager_factory(actor, processor)
    adapter_weights = lora_snapshot(actor.actor_module)
    manager.load_checkpoint(str(staging / "distributed"), del_local_after_load=False)
    configure_rl_lora_dropout_runtime(actor.actor_module, config)
    actor._formal_global_optimizer_step = expected["global_optimizer_step"]
    snap = _snapshot(actor, manager, loaded.rank, expected["global_optimizer_step"])
    _boundary(lambda: _require_reloaded(snap, expected, reload_matches(adapter_weights, lora_snapshot(actor.actor_module))),
              loaded.world_size, loaded.cpu_fixture)
    runtime = require_rl_lora_dropout_runtime(actor.actor_module, config)
    # Exactly one active microbatch per rank; no old-policy denominator comparison after update.
    micro = data.split(1)[0]
    rng = rng_fingerprint()
    with materialize_actor_microbatches(actor), torch.no_grad():
        values, _ = actor.compute_log_prob(micro, calculate_entropy=False)
    if not bool(torch.isfinite(values).all()) or values.shape != micro.batch["responses"].shape or rng_fingerprint() != rng:
        raise ValueError("fresh native multimodal forward failed/changed RNG")
    if any(_snapshot(actor, manager, loaded.rank, expected["global_optimizer_step"])[k] != expected[k]
           for k in ("parameter_sha256", "optimizer_state_sha256", "native_rng_sha256")):
        raise ValueError("fresh verification forward changed saved state")
    _collect(True, loaded.world_size, cpu_fixture=loaded.cpu_fixture)
    evidence = dict(scope=description["scope"], reloaded_artifact_roles=description["artifact_roles"],
        adapter_reloaded=True, native_reloaded=True, optimizer_reloaded=True, rng_reloaded=True,
        execution_contract_verified=True, original_actor_destroyed=True, fresh_multimodal_forward_finite=True,
        global_optimizer_step=expected["global_optimizer_step"],
        per_rank=_collect(dict(rank=loaded.rank, state=snap, lora_dropout_runtime_sha256=runtime["lora_dropout_runtime_sha256"]),
                          loaded.world_size, cpu_fixture=loaded.cpu_fixture))
    del manager
    return actor, evidence


def _require_destroyed(module_ref, optimizer_ref):
    if module_ref() is not None or optimizer_ref() is not None:
        raise ValueError("original actor/optimizer still alive before fresh construction")


def _require_reloaded(snapshot, expected, adapter_matches):
    if not adapter_matches or any(snapshot[k] != expected.get(k) for k in snapshot):
        raise ValueError("fresh native model/adapter/optimizer/RNG state mismatch")
