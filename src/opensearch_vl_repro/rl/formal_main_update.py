"""Four-rank launcher specialization; the existing S3/S2 update is not duplicated."""
from __future__ import annotations

import gc
import os
from datetime import timedelta
from pathlib import Path

from . import checkpoint as cp
from .formal_main import prepare_context, recover_main, main_paths, window_id, commit_main_checkpoint
from .formal_smoke_update import UpdateSession
from .run_state import (new_trainer_state, transition_trainer_state, new_update_attempt,
                        retry_update_plan, advance_update_attempt, persist_update_attempt)


def prepare_update_transaction(root, run, recovered, window, *, cpu_fixture=False):
    """Already bounded-recovered parent; require live S2 reload on every retry."""
    policy = recovered["policy"]
    state = new_trainer_state(run, policy)
    state = transition_trainer_state(transition_trainer_state(state, "collecting"), "ready_to_update")
    previous = [a for a in recovered["attempts"] if a["expected_optimizer_step"] == window["expected_optimizer_step"]]
    if not previous:
        return dict(state=state, attempt=new_update_attempt(window), retry_plan=None)
    active = [a for a in previous if a["phase"] != "failed"]
    prior = active[0] if active else previous[-1]
    plan = retry_update_plan(prior, window, policy, verified_checkpoints=recovered["checkpoints"])
    if prior["phase"] != "failed":
        persist_update_attempt(root, advance_update_attempt(prior, "failed",
            failure_reason="Main worker exited without successor; restore native verified parent"), cpu_fixture=cpu_fixture)
    state = cp.seal({**{k: v for k, v in state.items() if k != "state_sha256"},
        "recovery_reload_required": True, "recovery_parent_checkpoint": policy["checkpoint_identity"]}, "state_sha256")
    return dict(state=state, attempt=plan["new_attempt"], retry_plan=plan)


class MainUpdateSession(UpdateSession):
    world_size = 4
    window_count = 100
    paths = staticmethod(main_paths)
    context = staticmethod(prepare_context)
    recover = staticmethod(recover_main)
    window_identity = staticmethod(window_id)
    transaction = staticmethod(prepare_update_transaction)
    commit = staticmethod(commit_main_checkpoint)

    def checkpoint_kind(self, step):
        return "main_checkpoint"

    def load(self, policy=None):
        if not self.run["semantics"].get("continuation"):
            return super().load(policy)
        from .formal_main_continuation import (load_anchor, bootstrap_reload_capability,
                                               bootstrap_directory, require_active)
        _, receipt = load_anchor(self.output, self.run)
        if self.args.phase != "bootstrap":
            require_active(self.output, self.run)
        if policy != receipt["inherited_policy"]:
            return super().load(policy)  # suffix uses UNCHANGED same-run native path
        from .formal_policy_update import load_formal_actor
        capability = self.boundary("continuation_authority_full", lambda:
            bootstrap_reload_capability(self.output, self.run, policy))
        self.loaded = self.boundary("continuation_native_parent_load", lambda: load_formal_actor(self.run,
            canonical_config=self.ctx["canonical"], runtime_config=self.ctx["runtime"],
            source_adapter=self.args.sft_adapter, processor=self.processor, mesh=self.mesh, policy=policy,
            checkpoint_directory=bootstrap_directory(self.output, receipt), continuation_capability=capability))
        from .formal_s2_validation import actor_contract
        return self.boundary("execution_contract", lambda: actor_contract(self.loaded.actor, self.ctx["canonical"]))

    def bootstrap(self):
        if not getattr(self.args, "continue_from_run", None):
            return super().bootstrap()
        self.prepare()
        from .formal_main_continuation import load_anchor, publish_live_reload
        from .formal_policy_update import require_reload_capability
        anchor, _ = self.boundary("continuation_anchor", lambda: load_anchor(self.output, self.run))
        policy = anchor["initial_policy"]
        self.load(policy)
        actual = self.boundary("continuation_actual_reload_capability", lambda:
            require_reload_capability(self.loaded.reload_receipt, policy))
        rows = self.collect(actual)
        self.loaded = None
        gc.collect()
        self.torch.cuda.empty_cache()
        self.boundary("continuation_live_reload_publication", lambda:
            publish_live_reload(self.output, self.run, rows) if self.rank == 0 else None)


def require_launcher(environ):
    world, rank, local = (int(environ.get(k, "-1")) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if world != 4 or rank not in range(4) or local not in range(4) or rank != local:
        raise ValueError("Main requires standalone single-node WORLD_SIZE=4/local ranks 0..3")
    return rank, local


def run_update_worker(args, root):
    rank, device = require_launcher(os.environ)
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4 or not torch.cuda.is_bf16_supported():
        raise ValueError("Main update requires exactly four visible CUDA/BF16 GPUs")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    session = MainUpdateSession(args, Path(root).resolve(), torch, rank, device,
                                init_device_mesh("cuda", (4,), mesh_dim_names=("fsdp",)))
    try:
        if args.phase == "bootstrap":
            session.bootstrap()
        elif args.phase == "update":
            session.update()
        else:
            raise ValueError("invalid Main update phase")
        session.loaded = None
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except BaseException as exc:
        from .formal_smoke import redact_runtime_secrets
        cp.durable_json(session.reports / f"update_failure_rank_{rank}.json", redact_runtime_secrets(dict(
            passed=False, status="interrupted", rank=rank, stage=session.stage, error=f"{type(exc).__name__}: {exc}")))
        raise  # no failure-path collective; supervisor tears down every rank
