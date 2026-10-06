"""Opt-in stdout timing; deliberately OUTSIDE the hashed src/scripts trees."""
from __future__ import annotations

import builtins
from contextlib import contextmanager, ExitStack
from functools import wraps
from pathlib import Path
import sys
import time
from types import CodeType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl import formal_main_coordinator as coordinator
from opensearch_vl_repro.rl.formal_main_cli import build_parser


class CoordinatorTiming:
    """Observe coordinator call boundaries, not recovery's internal operations."""

    def __init__(self, orchestrate):
        self.code = orchestrate.__code__
        self.local_codes = {self.code, *(c for c in self.code.co_consts if isinstance(c, CodeType))}
        self.window = self.started = None
        self.subprocess_seconds = 0.0

    def context(self, caller):
        if caller.f_code not in self.local_codes:
            return None
        while caller is not None and caller.f_code is not self.code:
            caller = caller.f_back
        return caller

    @staticmethod
    def emit(window, phase, elapsed, *, outcome="returned"):
        builtins.print(f"[MAIN-TIMING] window={window} phase={phase} "
                       f"elapsed_seconds={elapsed:.6f} outcome={outcome}", flush=True)

    def finish_window(self, *, outcome="returned"):
        if self.started is not None:
            elapsed = time.monotonic() - self.started
            self.emit(self.window, "window_total", elapsed, outcome=outcome)
            self.emit(self.window, "subprocess_total", self.subprocess_seconds, outcome=outcome)
            self.emit(self.window, "coordinator_overhead", elapsed - self.subprocess_seconds, outcome=outcome)
            self.window = self.started = None
            self.subprocess_seconds = 0.0

    def print(self, *args, **kwargs):
        caller = sys._getframe(1)
        # The original window header is emitted before the window's first write.
        if caller.f_code is self.code and "iteration" in caller.f_locals:
            window = caller.f_locals["iteration"] + 1
            if window != self.window:
                self.finish_window()
                self.window, self.started = window, time.monotonic()
        return builtins.print(*args, **kwargs)

    def wrap(self, operation, phase, *, subprocess=False):
        @wraps(operation)
        def timed(*args, **kwargs):
            caller = sys._getframe(1)
            frame = self.context(caller)
            if frame is None:
                return operation(*args, **kwargs)
            window = frame.f_locals.get("iteration")
            window = "startup" if window is None else window + 1
            label = phase(caller, frame) if callable(phase) else phase
            before, outcome = time.monotonic(), "interrupted"
            try:
                result = operation(*args, **kwargs)
                outcome = "returned"
                return result
            finally:
                elapsed = time.monotonic() - before
                if subprocess and window == self.window:
                    self.subprocess_seconds += elapsed
                self.emit(window, label, elapsed, outcome=outcome)
        return timed


def recovery_phase(caller, frame):
    if "iteration" not in frame.f_locals:
        return "recover_initial"
    return {"collect": "recover_after_collect", "update": "recover_after_update"}.get(
        frame.f_locals["stage"], "recover_other")


@contextmanager
def timing_instrumentation():
    """All original calls/args/results/exceptions are delegated and then restored."""
    original = coordinator.orchestrate
    timing = CoordinatorTiming(original)

    @wraps(original)
    def orchestrate(*args, **kwargs):
        outcome = "interrupted"
        try:
            result = original(*args, **kwargs)
            outcome = "returned"
            return result
        finally:
            timing.finish_window(outcome=outcome)

    with ExitStack() as stack:
        replacements = [
            (coordinator, "orchestrate", orchestrate),
            (coordinator, "print", timing.print),
            (coordinator, "recover_main", timing.wrap(coordinator.recover_main, recovery_phase)),
            (coordinator.retention, "retire_merge", timing.wrap(coordinator.retention.retire_merge, "retire_merge")),
            (coordinator.retention, "compact_history", timing.wrap(coordinator.retention.compact_history, "compact_history")),
            (coordinator.retention, "disk_accounting", timing.wrap(coordinator.retention.disk_accounting,
                lambda caller, frame: "disk_accounting_" + caller.f_code.co_name)),
            (coordinator.ProcessRunner, "__call__", timing.wrap(coordinator.ProcessRunner.__call__,
                lambda caller, frame: "subprocess_" + caller.f_locals["phase"], subprocess=True)),
            (coordinator.ParallelProcessRunner, "__call__", timing.wrap(coordinator.ParallelProcessRunner.__call__,
                "subprocess_collect", subprocess=True)),
        ]
        for owner, name, value in replacements:
            stack.enter_context(patch.object(owner, name, value, create=name == "print"))
        yield timing


def run_with_timing(args, root):
    with timing_instrumentation():
        return coordinator.run_main(args, root)


if __name__ == "__main__":
    run_with_timing(build_parser(ROOT).parse_args(), ROOT)
