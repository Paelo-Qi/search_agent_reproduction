"""One isolated two-rank production S2 transaction, never a resident trainer."""
from __future__ import annotations

import gc
import json
import os
from datetime import timedelta
from pathlib import Path

from . import checkpoint as cp
from .formal_smoke import prepare_context, recover_smoke, smoke_paths, window_id
from .run_state import (advance_update_attempt, new_update_attempt, persist_update_attempt,
                        new_trainer_state, transition_trainer_state, rollback_interrupted_state)
from .training_window import build_training_window


def prepare_update_transaction(root, run, recovered, window, *, cpu_fixture=False):
    """No actor reuse: construct a parent-bound fresh UUID and S1 rollback state."""
    policy = recovered["policy"]
    state = new_trainer_state(run, policy)
    state = transition_trainer_state(state, "collecting")
    state = transition_trainer_state(state, "ready_to_update")
    previous = [a for a in recovered["attempts"] if a["expected_optimizer_step"] == window["expected_optimizer_step"]]
    if not previous:
        return dict(state=state, attempt=new_update_attempt(window), retry_plan=None)
    active = [a for a in previous if a["phase"] != "failed"]
    prior = active[0] if active else previous[-1]
    # This process owns NO predecessor actor. The parent is always constructed
    # below from real native model/AdamW/RNG (or actual seeded SFT iteration0).
    interrupted = transition_trainer_state(state, "interrupted")
    rollback = rollback_interrupted_state(interrupted, root, run, prior, window, cpu_fixture=cpu_fixture)
    if prior["phase"] != "failed":
        failed = advance_update_attempt(prior, "failed", failure_reason="worker exited without immutable successor; restore verified parent")
        persist_update_attempt(root, failed, cpu_fixture=cpu_fixture)
    return dict(state=rollback["state"], attempt=rollback["retry_plan"]["new_attempt"], retry_plan=rollback["retry_plan"])


class UpdateSession:
    # Main only specializes control-plane bindings; S2 dataplane remains shared.
    world_size = 2
    window_count = 5
    paths = staticmethod(smoke_paths)
    context = staticmethod(prepare_context)
    recover = staticmethod(recover_smoke)
    window_identity = staticmethod(window_id)
    transaction = staticmethod(prepare_update_transaction)
    commit = staticmethod(cp.commit_verified_checkpoint)

    def checkpoint_kind(self, step):
        return "smoke_final" if step == 5 else "smoke_continuation"

    def __init__(self, args, root, torch, rank, device, mesh):
        self.args, self.root, self.torch = args, root, torch
        self.rank, self.device, self.mesh = rank, device, mesh
        self.output, self.reports = self.paths(root, args.run_id)
        self.stage, self.loaded = "inputs", None

    def collect(self, value):
        rows = [None] * self.world_size
        self.torch.distributed.all_gather_object(rows, value)
        return rows

    def boundary(self, stage, operation):
        from .formal_smoke import redact_runtime_secrets as redact_secrets
        self.stage = stage
        result, error = None, None
        try:
            cp.durable_json(self.reports / f"update-rank-{self.rank}.json",
                dict(passed=False, status="running", phase=self.args.phase, rank=self.rank, stage=stage))
            result = operation()
            self.torch.cuda.synchronize(self.device)
        except Exception as exc:
            error = redact_secrets(dict(rank=self.rank, stage=stage, error=f"{type(exc).__name__}: {exc}"))
        errors = self.collect(error)
        if any(e is not None for e in errors):
            raise RuntimeError(f"Formal update boundary failed: {errors}")
        return result

    def prepare(self):
        from opensearch_vl_repro.model import load_processor
        self.ctx = self.boundary("prepare_inputs", lambda: self.context(self.args, self.root))
        self.run = self.ctx["run"]
        if len(set(self.collect(self.run["run_identity_sha256"]))) != 1:
            raise ValueError("rank Formal run identities differ")
        self.processor = self.boundary("processor", lambda: load_processor(self.ctx["runtime"], local_files_only=True))

    def load(self, policy=None):
        from .formal_policy_update import load_formal_actor
        directory = (self.output / "checkpoints" / f"policy-{policy['policy_iteration']:06d}"
                     if policy is not None and policy["policy_iteration"] > 0 else None)
        self.loaded = self.boundary("native_parent_load", lambda: load_formal_actor(self.run,
            canonical_config=self.ctx["canonical"], runtime_config=self.ctx["runtime"],
            source_adapter=self.args.sft_adapter, processor=self.processor, mesh=self.mesh, policy=policy,
            checkpoint_directory=directory, initial_seed=self.run["semantics"]["initial_seed"]))
        from .formal_s2_validation import actor_contract
        return self.boundary("execution_contract", lambda: actor_contract(self.loaded.actor, self.ctx["canonical"]))

    def bootstrap(self):
        self.prepare()
        if (self.output / "identity").exists():
            self.boundary("existing_anchor", lambda: self.recover(self.output, self.run) if self.rank == 0 else None)
            return
        contract = self.load()
        self.boundary("publish_initial_anchor", lambda: cp.initialize_formal_run(self.output, self.run, self.loaded.policy)
                      if self.rank == 0 else None)
        self.boundary("bootstrap_receipt", lambda: cp.durable_json(self.reports / f"bootstrap-rank-{self.rank}.json",
            dict(passed=False, scope="runtime", initial_policy=self.loaded.policy,
                 reload=self.loaded.reload_receipt.artifact, actor_contract=contract)))
        self.loaded = None
        gc.collect()
        self.torch.cuda.empty_cache()

    def update(self):
        from .formal_policy_update import (update_formal_window, save_formal_staging,
                                          fresh_reload_staging, checkpoint_staging_roles)
        from .training_batch import formal_training_rows, build_rank_local_dataproto
        from .training_window import deterministic_rank_plan
        from .rloo import assemble_window_rloo
        from .formal_s2_validation import full_lora_fingerprint
        self.prepare()
        recovered = self.collect(self.boundary("recover", lambda: self.recover(self.output, self.run)
                                               if self.rank == 0 else None))[0]
        if recovered["missing_prompts"] or recovered["policy"]["policy_iteration"] >= self.window_count:
            raise ValueError("one update requires exactly K committed current-policy groups")
        policy, groups = recovered["policy"], recovered["current_groups"]
        step = policy["global_optimizer_step"] + 1
        window = self.boundary("window", lambda: build_training_window(self.run, policy, groups,
                                              window_id=self.window_identity(policy["policy_iteration"])))
        transaction = self.collect(self.boundary("rollback_plan", lambda: self.transaction(
            self.output, self.run, recovered, window) if self.rank == 0 else None))[0]
        contract = self.load(policy)  # Actual model/native AdamW/RNG continuation; NEVER uncertain memory.
        reward = self.boundary("official_rloo", lambda: assemble_window_rloo(window, self.run, policy, groups))
        if reward["status"] != "signal":
            raise ValueError("zero-signal window cannot authorize a Smoke optimizer step; no fabricated reward")
        rows, _ = formal_training_rows(window, self.run, policy, groups, reward)
        plan = deterministic_rank_plan([r["logical_row_id"] for r in rows], self.world_size)
        directories = {g["identity"]["trajectory_group_id"]: self.output / "groups" / g["identity"]["trajectory_group_id"] for g in groups}
        data, receipt = self.boundary("rank_local_batch", lambda: build_rank_local_dataproto(window,
            self.run, policy, groups, reward, group_directories=directories, rank=self.rank,
            model=self.loaded.actor.actor_module, pad_id=self.processor.tokenizer.pad_token_id, temperature=.7))
        before = self.boundary("pre_update_lora", lambda: full_lora_fingerprint(self.loaded.actor))
        updating_state = self.boundary("trainer_updating", lambda: transition_trainer_state(transaction["state"],
            "updating", actor=self.loaded.actor, reload_receipt=self.loaded.reload_receipt))
        updated = self.boundary("OC_one_PPO_step", lambda: update_formal_window(self.loaded, data, receipt,
            root=self.output, run=self.run, groups=groups, window=window, reward_window=reward,
            attempt=transaction["attempt"], trainer_state=transaction["state"]))
        after = self.boundary("post_update_lora", lambda: full_lora_fingerprint(self.loaded.actor))
        if after == before or len(set(self.collect(after))) != 1:
            raise ValueError("full LoRA did not change consistently across ranks")
        staging = self.output / "checkpoints" / (".update-" + updated["attempt"]["attempt_id"])
        description = self.boundary("native_save", lambda: save_formal_staging(self.loaded, self.processor,
            staging, window=window, update_evidence=updated))
        fresh, reload = self.boundary("fresh_native_reload", lambda: fresh_reload_staging(self.loaded, data,
            staging, description, canonical_config=self.ctx["canonical"], runtime_config=self.ctx["runtime"],
            processor=self.processor, mesh=self.mesh))
        del fresh
        gc.collect()
        self.torch.cuda.empty_cache()
        # Persist complete rank update evidence IN the immutable checkpoint, not
        # just stdout/mutable progress. It is added AFTER staging verification;
        # rebuild the exact metadata role inventory before S1 publication.
        evidence = dict(scope="runtime", rank=self.rank, update=updated, actor_contract=contract,
            loaded_parent=self.loaded.reload_receipt.artifact, retry_plan=transaction["retry_plan"],
            rank_plan=receipt.artifact, deterministic_plan=plan, previous_full_lora_sha256=before,
            full_lora_sha256=after, fresh_reload=reload)
        self.boundary("immutable_update_evidence", lambda: cp.durable_json(staging / f"formal_update_rank_{self.rank}.json", evidence))
        roles, files = self.boundary("final_staging_inventory", lambda: checkpoint_staging_roles(staging, self.world_size))
        reload["reloaded_artifact_roles"] = cp.artifact_role_identities(roles, files)
        # Native/optimizer/RNG/adapter bytes remain exactly those actually
        # reloaded; ONLY newly added evidence metadata changes its role hash.
        if any(reload["reloaded_artifact_roles"][k] != description["artifact_roles"][k]
               for k in ("adapter", "native", "optimizer", "rng")):
            raise ValueError("native training artifacts changed after fresh reload")
        manifest = self.boundary("checkpoint_manifest", lambda: cp.build_checkpoint_manifest(self.run,
            policy, groups, window, updated["attempt"], reward, artifact_role_files=roles, file_sha256=files,
            kind=self.checkpoint_kind(step), reload_evidence=reload))
        self.boundary("immutable_checkpoint_publication", lambda: self.commit(self.output, staging, manifest)
                      if self.rank == 0 else None)
        checkpoint_directory = self.output / "checkpoints" / f"policy-{step:06d}"
        self.boundary("verified_attempt", lambda: persist_update_attempt(self.output,
            advance_update_attempt(updated["attempt"], "verified", checkpoint=manifest,
                                   checkpoint_directory=checkpoint_directory)) if self.rank == 0 else None)
        # The trainer FSM is disposable progress, reconstructed from receipts.
        state = transition_trainer_state(updating_state, "checkpointing")
        state = transition_trainer_state(state, "iteration_verified", checkpoint=manifest, checkpoint_directory=checkpoint_directory)
        self.boundary("progress", lambda: cp.durable_json(self.reports / f"update-rank-{self.rank}.json",
            dict(passed=False, status="iteration_verified", stage=self.stage, scope="runtime", state=state)))
        self.loaded = None
        del data
        gc.collect()
        self.torch.cuda.empty_cache()


def run_update_worker(args, root):
    from .formal_s2_validation import require_launcher
    rank, device = require_launcher(os.environ)
    root = Path(root).resolve()
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2 or not torch.cuda.is_bf16_supported():
        raise ValueError("Formal update requires exactly two real CUDA/BF16 GPUs")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    session = UpdateSession(args, root, torch, rank, device, init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",)))
    try:
        if args.phase == "bootstrap":
            session.bootstrap()
        elif args.phase == "update":
            session.update()
        else:
            raise ValueError("unknown Formal update phase")
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except BaseException as exc:
        from .formal_smoke import redact_runtime_secrets as redact_secrets
        cp.durable_json(session.reports / f"update_failure_rank_{rank}.json", redact_secrets(dict(
            passed=False, status="interrupted", rank=rank, stage=session.stage, error=f"{type(exc).__name__}: {exc}")))
        # NO failure-path collectives. torchrun/coordinator terminate all peers.
        raise
