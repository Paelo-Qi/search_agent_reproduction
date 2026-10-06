"""Isolated stdout instrumentation; no real processes, models, GPU or API."""
import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import re

import pytest

from opensearch_vl_repro.rl import formal_main as main
from opensearch_vl_repro.rl import formal_main_continuation as handoff
from opensearch_vl_repro.rl import formal_main_coordinator as coordinator
from opensearch_vl_repro.rl import formal_smoke as smoke
from test_rl_formal_s4_main import ctx, fixture_runners  # reuse the existing CPU fixture

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("formal_main_timing_diagnostic",
    ROOT / "diagnostics/diagnose_formal_main_timing.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_launcher_delegates_to_original_run_main_and_restores_on_error(monkeypatch):
    original = (coordinator.orchestrate, coordinator.recover_main,
        coordinator.retention.retire_merge, coordinator.retention.compact_history,
        coordinator.retention.disk_accounting, coordinator.ProcessRunner.__call__,
        coordinator.ParallelProcessRunner.__call__)
    error = RuntimeError("original exception")
    calls = []
    def run_main(args, root):
        calls.append((args, root))
        assert coordinator.orchestrate is not original[0]
        raise error
    monkeypatch.setattr(coordinator, "run_main", run_main)
    args = object()
    with pytest.raises(RuntimeError) as caught:
        diagnostic.run_with_timing(args, ROOT)
    assert caught.value is error and calls == [(args, ROOT)]
    assert original == (coordinator.orchestrate, coordinator.recover_main,
        coordinator.retention.retire_merge, coordinator.retention.compact_history,
        coordinator.retention.disk_accounting, coordinator.ProcessRunner.__call__,
        coordinator.ParallelProcessRunner.__call__)
    assert "print" not in vars(coordinator)


def test_source_inventory_identity_and_continuation_allowlist_unchanged(monkeypatch):
    monkeypatch.setattr(smoke.importlib.metadata, "version",
        lambda name: smoke.PINNED.get(name, "2.8.0" if name == "torch" else "2.8.3"))
    monkeypatch.setattr(smoke, "installed_source_inventory", lambda name: {"fixture.py": "a" * 64})
    before = smoke.software_binding(ROOT)
    frozen = (main.VERSION, copy.deepcopy(main.MAIN_ROLLOUT), handoff.VERSION,
              handoff.SOURCE_DELTA, handoff.PARENT_VERSIONS.copy())
    run_main = coordinator.run_main
    with diagnostic.timing_instrumentation():
        assert smoke.software_binding(ROOT) == before
        assert coordinator.run_main is run_main
        assert frozen == (main.VERSION, main.MAIN_ROLLOUT, handoff.VERSION,
                          handoff.SOURCE_DELTA, handoff.PARENT_VERSIONS)
    assert smoke.software_binding(ROOT) == before
    assert not any(p.startswith("diagnostics/") for p in before[1])


@pytest.mark.parametrize("failure", [False, True])
def test_cpu_window_preserves_phase_order_artifacts_and_error_gating(ctx, monkeypatch, capsys, failure):
    single, parallel, calls = fixture_runners(ctx, provider_failure=failure)
    class Single:
        active = False
        def __call__(self, *args, **kwargs): return single(*args, **kwargs)
    class Parallel:
        active = False
        def __call__(self, *args, **kwargs): return parallel(*args, **kwargs)
    monkeypatch.setattr(coordinator, "ProcessRunner", Single)
    monkeypatch.setattr(coordinator, "ParallelProcessRunner", Parallel)
    original_orchestrate = coordinator.orchestrate
    identity = copy.deepcopy(ctx.run)
    ctx.args.stop_after_window = 1
    before = {p.relative_to(ctx.output / "identity").as_posix(): p.read_bytes()
              for p in (ctx.output / "identity").rglob("*") if p.is_file()}
    with diagnostic.timing_instrumentation():
        with pytest.raises(RuntimeError, match="update forbidden") if failure else nullcontext():
            coordinator.orchestrate(ctx.args, ctx.root, dict(run=ctx.run), cpu_fixture=True)
    assert ctx.run == identity
    assert before == {p.relative_to(ctx.output / "identity").as_posix(): p.read_bytes()
                      for p in (ctx.output / "identity").rglob("*") if p.is_file()}
    assert calls == (["merge", "collect"] if failure else ["merge", "collect", "update"])
    text = capsys.readouterr().out
    rows = re.findall(r"\[MAIN-TIMING\] window=(\S+) phase=(\S+) elapsed_seconds=([\d.]+) outcome=(\S+)", text)
    assert rows and all(float(elapsed) >= 0 for _, _, elapsed, _ in rows)
    window_phases = [phase for window, phase, _, _ in rows if window == "1"]
    ordered = ["subprocess_merge", "subprocess_collect", "recover_after_collect", "retire_merge",
               "subprocess_update", "recover_after_update", "compact_history"]
    assert [phase for phase in window_phases if phase in ordered] == (ordered[:2] if failure else ordered)
    required = ["window_total", "subprocess_total", "coordinator_overhead"]
    if not failure:
        required += ["recover_after_collect", "retire_merge", "recover_after_update", "compact_history"]
        values = {phase: float(elapsed) for window, phase, elapsed, _ in rows if window == "1"}
        assert abs(values["window_total"] - values["subprocess_total"] - values["coordinator_overhead"]) < 0.00001
    assert set(required) <= set(window_phases)
    if failure:
        assert "subprocess_update" not in window_phases and "recover_after_update" not in window_phases
        assert any(phase == "window_total" and outcome == "interrupted" for _, phase, _, outcome in rows)
    # Persisted reports/checkpoints contain ONLY the original data/schema.
    for base in (ctx.output, ctx.reports):
        for p in base.rglob("*.json"):
            raw = p.read_text(encoding="utf-8")
            assert "MAIN-TIMING" not in raw and "coordinator_overhead" not in raw
            assert "recover_after_collect" not in raw and "subprocess_total" not in raw
    report = json.loads((ctx.reports / "report.json").read_text())
    assert report["status"] == ("interrupted" if failure else "paused_at_verified_boundary")
    assert not (ctx.output / "manifest.json").exists()
    assert coordinator.orchestrate is original_orchestrate


def test_absolute_window_numbers_and_total_do_not_use_policy25_or_a_counter(capsys, monkeypatch):
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: 15.)
    timing = diagnostic.CoordinatorTiming(coordinator.orchestrate)
    timing.window, timing.started, timing.subprocess_seconds = 26, 10., 2.
    timing.finish_window()
    text = capsys.readouterr().out
    assert "window=26 phase=window_total elapsed_seconds=5.000000" in text
    assert "window=26 phase=coordinator_overhead elapsed_seconds=3.000000" in text
    # No wall-clock duration assertions: caller supplies absolute iteration;
    # source test ensures that no separate counter/hardcoded boundary exists.
    source = (ROOT / "diagnostics/diagnose_formal_main_timing.py").read_text()
    assert 'caller.f_locals["iteration"] + 1' in source
    assert "policy25" not in source and "Window26" not in source
    assert "time.monotonic()" in source and "settrace" not in source and "setprofile" not in source
