"""Main400 process orchestration only; no second PPO or resident GPU actor."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from . import checkpoint as cp
from . import formal_main_retention as retention
from .formal_main import VERSION, prepare_context, require_main_run, main_paths, recover_main, final_reconstruction, publish_final
from .formal_smoke import ProcessRunner, revoke_pass, redact_runtime_secrets, progress
from .formal_main_process import ParallelProcessRunner
from .group import run_lock


def gpu_list(value, *, count=None):
    if not isinstance(value, str) or not re.fullmatch(r"\d+(,\d+)*", value):
        raise ValueError("explicit numeric GPU list required")
    devices = value.split(",")
    if len(set(devices)) != len(devices) or not 1 <= len(devices) <= 4 or count is not None and len(devices) != count:
        raise ValueError("distinct GPUs required (update exactly four)")
    return devices


def worker_command(args, root, phase, *, prompt=None):
    fields = ("run_id", "config", "data", "source_root", "source_parquet", "base_model_path", "sft_adapter",
              "judge_config", "search_config", "layout_config", "eval_overlap_manifest", "sft_overlap_manifest")
    common = [part for k in fields for part in ("--" + k.replace("_", "-"), str(getattr(args, k)))]
    for k in ("tool_cache_dir", "reward_cache_dir"):
        if getattr(args, k, None) is not None:
            common += ["--" + k.replace("_", "-"), str(getattr(args, k))]
    if phase == "collect":
        if prompt is None:
            raise ValueError("each collection worker requires one prompt")
        return [sys.executable, str(root / "scripts/collect_rl_formal_main.py"), *common, "--prompt-id", prompt]
    if phase == "merge":
        return [sys.executable, str(root / "scripts/prepare_rl_formal_main_merge.py"), *common]
    if phase not in {"bootstrap", "update"}:
        raise ValueError("unknown Main phase")
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
            str(root / "scripts/update_rl_formal_main.py"), "--phase", phase, *common]


def worker_environment(devices):
    gpu_list(devices)
    env = os.environ.copy()
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK"):
        env.pop(k, None)
    env.update(CUDA_VISIBLE_DEVICES=devices, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    return env


def collection_waves(missing, devices, parallelism):
    if type(parallelism) is not int or not 1 <= parallelism <= min(4, len(devices)) or len(set(missing)) != len(missing):
        raise ValueError("unique prompt owners and 1..4 collection parallelism required")
    return [[(prompt, devices[i]) for i, prompt in enumerate(missing[begin:begin + parallelism])]
            for begin in range(0, len(missing), parallelism)]


def orchestrate(args, root, ctx, single=None, parallel=None, *, cpu_fixture=False):
    if not cpu_fixture and (single is not None or parallel is not None):
        raise ValueError("injected worker runners are CPU fixtures only")
    single, parallel = single or ProcessRunner(), parallel or ParallelProcessRunner()
    run = ctx["run"]
    require_main_run(run)
    devices = gpu_list(args.collection_gpus)
    gpu_list(args.update_gpus, count=4)
    collection_waves([], devices, args.collection_parallelism)
    stop = args.stop_after_window
    if stop is not None and (type(stop) is not int or not 1 <= stop <= 100):
        raise ValueError("stop-after-window must be 1..100")
    output, reports = main_paths(root, args.run_id, [args.source_root, args.base_model_path, args.sft_adapter.parent])
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    started, stage, events, peak = time.monotonic(), "recovery", [], 0
    with run_lock(output / ".coordinator.lock"):
        def write(status, **extra):
            nonlocal peak
            disk = retention.disk_accounting(output, previous_peak=peak, report_root=reports)
            peak = disk["peak_observed_run_bytes"]
            report = dict(version=VERSION, passed=False, status=status, stage=stage,
                scope="cpu_fixture" if cpu_fixture else "runtime", run_identity_sha256=run["run_identity_sha256"],
                elapsed_seconds=time.monotonic() - started, subprocesses=events, disk=disk,
                eligible_for_main_init=False, software_versions=ctx.get("versions"), git_commit=ctx.get("git_commit"), **extra)
            cp.durable_json(reports / "report.json", redact_runtime_secrets(report), cpu_fixture=cpu_fixture)
            return report
        def guard(phase):
            disk = retention.disk_accounting(output, previous_peak=peak, report_root=reports)
            needed = (args.merge_headroom_gib if phase == "merge" else args.update_headroom_gib) * retention.GIB
            retention.disk_guard(disk, needed_bytes=needed, max_run_bytes=args.max_run_gib * retention.GIB,
                                 min_free_bytes=args.min_free_gib * retention.GIB)
        def launch(phase):
            nonlocal stage
            if getattr(single, "active", False) or getattr(parallel, "active", False):
                raise RuntimeError("owned worker still active; next GPU phase forbidden")
            stage = phase
            write("running")
            before = time.monotonic()
            env = worker_environment(devices[0] if phase == "merge" else args.update_gpus)
            code = single(worker_command(args, root, phase), env=env, cwd=root, log=reports / f"{len(events):04d}-{phase}.log")
            events.append(dict(phase=phase, exit_code=code, elapsed_seconds=time.monotonic() - before))
            if code != 0 or getattr(single, "active", False):
                raise RuntimeError(f"Main {phase} worker failed; no next GPU phase; resume same run-id")
            # Observe coexistence of predecessor/successor BEFORE retention.
            write("running")
        try:
            revoke_pass(output, cpu_fixture=cpu_fixture)
            # Persist peak history across same-run invocations; never identity authority.
            old_report = reports / "report.json"
            if old_report.exists():
                peak = json.loads(old_report.read_text(encoding="utf-8")).get("disk", {}).get("peak_observed_run_bytes", 0)
            write("running")
            if not (output / "identity").exists():
                guard("update")
                launch("bootstrap")
            recovered = recover_main(output, run, cpu_fixture=cpu_fixture)
            retention.compact_history(output, run, recovered["checkpoints"], cpu_fixture=cpu_fixture)
            while recovered["policy"]["policy_iteration"] < 100:
                iteration = recovered["policy"]["policy_iteration"]
                if stop is not None and iteration >= stop:
                    return write("paused_at_verified_boundary", progress=progress(recovered))
                print(f"[Main400] window {iteration + 1}/100 policy={iteration}", flush=True)
                write("running", progress=progress(recovered))
                if recovered["missing_prompts"]:
                    guard("merge")
                    launch("merge")  # exactly ONE shared merge, full verified once
                    for wave in collection_waves(recovered["missing_prompts"], devices, args.collection_parallelism):
                        if getattr(single, "active", False) or getattr(parallel, "active", False):
                            raise RuntimeError("GPU lifetime overlap forbidden")
                        stage = "collect"
                        write("running")
                        jobs = [dict(command=worker_command(args, root, "collect", prompt=p), env=worker_environment(gpu),
                                     log=reports / f"{len(events):04d}-collect-{run['prompt_ids'].index(p):04d}.log") for p, gpu in wave]
                        before = time.monotonic()
                        codes = parallel(jobs, cwd=root)
                        events.append(dict(phase="collect", prompts=[p for p, _ in wave], exit_codes=codes,
                                           elapsed_seconds=time.monotonic() - before))
                        if len(codes) != len(jobs) or any(code != 0 for code in codes) or getattr(parallel, "active", False):
                            raise RuntimeError("Main collection wave failed; peers reaped; update forbidden")
                    recovered = recover_main(output, run, cpu_fixture=cpu_fixture)
                if recovered["missing_prompts"] or recovered["policy"]["policy_iteration"] != iteration:
                    raise ValueError("collection must publish exact K4 same-policy whole groups")
                stage = "merge_retirement"
                retention.retire_merge(output, run, recovered["policy"], recovered["current_groups"],
                    workers_dead=not getattr(single, "active", False) and not getattr(parallel, "active", False), cpu_fixture=cpu_fixture)
                guard("update")
                launch("update")
                recovered = recover_main(output, run, cpu_fixture=cpu_fixture)
                if recovered["policy"]["policy_iteration"] != iteration + 1:
                    raise ValueError("Main update requires exactly one verified successor")
                stage = "checkpoint_retention"
                retention.compact_history(output, run, recovered["checkpoints"], cpu_fixture=cpu_fixture)
                write("iteration_verified", progress=progress(recovered))
                if stop is not None and iteration + 1 >= stop:
                    return write("paused_at_verified_boundary", progress=progress(recovered))
            stage = "final_heavy_audit"
            report = final_reconstruction(output, run, cpu_fixture=cpu_fixture)
            report.update(disk=retention.disk_accounting(output, previous_peak=peak, report_root=reports), subprocesses=events,
                          elapsed_seconds=time.monotonic() - started)
            if cpu_fixture:
                cp.durable_json(reports / "report.json", report, cpu_fixture=True)
            else:
                publish_final(output, reports, report)
            return report
        except BaseException as exc:
            revoke_pass(output, cpu_fixture=cpu_fixture)
            failures = [json.loads(p.read_text(encoding="utf-8")) for p in reports.glob("collection_failure-*.json")]
            write("interrupted", error=f"{type(exc).__name__}: {exc}", collection_failures=failures)
            raise


def run_main(args, root):
    root = Path(root).resolve()
    return orchestrate(args, root, prepare_context(args, root))
