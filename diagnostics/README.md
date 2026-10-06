# Formal Main coordinator stdout timing

`diagnose_formal_main_timing.py` is an opt-in diagnostic launcher. It delegates
to the original `run_main()` with the original parser/arguments. It does not
change worker commands, environment, scheduling, recovery, publication,
retention, exception handling, identity validation or any artifact schema.
No timer fields or additional artifacts are written. Runtime wrappers are
restored on return or exception, and are not installed in worker subprocesses.

## Why this launcher is outside src AND scripts

`formal_smoke.software_binding()` hashes every Python file in **both** `src/`
and `scripts/`, plus the existing installed-framework inventories.
`formal_main.prepare_context()` puts that inventory into the run semantics.
For an existing child, `read_authority()` / `validate_receipt()` checks the
child identity and exact semantics. Recovery's `load_anchor()` additionally
uses `require_same_training_run()`. Activation/capability checks remain intact.
Editing the coordinator, or adding a Python launcher under `scripts/`, would
therefore break the existing child's same-run identity. This diagnostic is
intentionally under `diagnostics/`: **do not move it into src or scripts, or
change the frozen source files / hashes / allowlist to enable it**.

Provided all other frozen inputs, installed software and src/scripts bytes
still match the child, this launcher preserves its same-run compatibility.
It does not bypass a mismatch, read a newer parent boundary, reseal a receipt
or create a new continuation. AutoDL artifacts are not present locally; the
reported verified policy25 is a user-provided runtime fact, not a local audit.

## AutoDL: diagnose the existing child for Window26 only

Use exactly the same frozen/operational arguments as the existing invocation
(`MAIN_ARGS` below means that existing argument array, not a new configuration).
Replace only the coordinator entrypoint, keeping its run/parent IDs:

```bash
set -o pipefail
PYTHONPATH=src python -u diagnostics/diagnose_formal_main_timing.py \
  "${MAIN_ARGS[@]}" \
  --run-id formal-main400-s4-continuation-v7-attempt1 \
  --continue-from-run formal-main400-s4-attempt4 \
  --stop-after-window 26 2>&1 | tee /tmp/main400-window26-timing.log
```

Do not duplicate run-id/parent/stop flags in `MAIN_ARGS`. Run this only on
AutoDL after stopping any other coordinator for this child. The original
absolute stop and verified recovery checks determine what runs; policy25 and
Window26 are **not hardcoded** in the diagnostic.

## Timing interpretation

All elapsed times use `time.monotonic()`, print with `flush=True`, and exist
only in stdout/stderr, which the shell may save with `tee`:

```text
[MAIN-TIMING] window=26 phase=recover_after_collect elapsed_seconds=...
[MAIN-TIMING] window=26 phase=retire_merge elapsed_seconds=...
[MAIN-TIMING] window=26 phase=recover_after_update elapsed_seconds=...
[MAIN-TIMING] window=26 phase=compact_history elapsed_seconds=...
[MAIN-TIMING] window=26 phase=window_total elapsed_seconds=...
[MAIN-TIMING] window=26 phase=subprocess_total elapsed_seconds=...
[MAIN-TIMING] window=26 phase=coordinator_overhead elapsed_seconds=...
```

Also emitted: `disk_accounting_write`, `disk_accounting_guard`,
`disk_accounting_continuation_guard`, `subprocess_merge`, `subprocess_collect`,
`subprocess_update` (bootstrap if needed), and startup recovery/compaction.
`window=startup` separates pre-loop work from per-window work.

Window numbering is the original coordinator's `iteration + 1`. A window
starts immediately before its original `[Main400]` header and ends before the
next header, or when orchestration returns/raises. Thus the final window
includes the existing stop-boundary report write; at Window100 it also includes
the existing final heavy audit. `subprocess_total` is the sum of actual runner
call wall times; a parallel wave is counted **once**, not once per GPU.
`coordinator_overhead = window_total - subprocess_total`, so it includes report
writes, guards, recovery, retention and diagnostic stdout overhead. Phase times
can be nested: **do not sum disk timings into overhead again**. There is no
separate full `write()` or `guard()` timer; their disk calls are measured, while
their remaining work is included in coordinator overhead.

`outcome=interrupted` means an original call/window raised. Partial timings
are diagnostic only; the original exception and peer cleanup propagate, with
no PASS claim, retry or added update gating. This launcher has not run locally
against a model/GPU/API or any real Window26 artifacts.

## Targeted CPU acceptance only

```bash
PYTHONPATH=src python -m pytest \
  tests/test_rl_formal_main_timing.py \
  tests/test_rl_formal_s4_main.py \
  tests/test_rl_formal_main_continuation.py \
  -k 'timing or exact_main_identity or main_v7 or first_real_boundary or provider_failure_blocks_update or no_overlap or coordinator_absolute_stop or only_enumerated_source_changes or continued_or_v6 or goal_a1' \
  -o addopts= -q -p no:cacheprovider
git diff --check
```

On Windows use a fresh short `--basetemp` under the writable workspace to avoid
sandbox temp permissions / MAX_PATH limits. No full pytest is required here.
