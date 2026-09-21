from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from opensearch_vl_repro.inference.smoke_support import (
    CudaSmokeContext,
    StagedInferenceError,
    exception_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(filename: str):
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCuda:
    def __init__(self, events: list[str], available: bool) -> None:
        self.events = events
        self.available = available

    def is_available(self) -> bool:
        self.events.append("is_available")
        return self.available

    def set_device(self, device) -> None:
        self.events.append(f"set_device:{device}")

    def init(self) -> None:
        self.events.append("init")

    def reset_peak_memory_stats(self, device) -> None:
        self.events.append(f"reset:{device}")

    def max_memory_allocated(self, device) -> int:
        self.events.append(f"read:{device}")
        return 64 * 1024 * 1024


class FakeTorch:
    def __init__(self, available: bool) -> None:
        self.events: list[str] = []
        self.cuda = FakeCuda(self.events, available)

    def device(self, value: str) -> str:
        self.events.append(f"device:{value}")
        return f"resolved:{value}"


def test_cuda_memory_context_initializes_device_before_first_reset() -> None:
    torch = FakeTorch(available=True)
    context = CudaSmokeContext.initialize(torch, "cuda:0")
    context.reset_peak_memory_stats()
    assert torch.events == [
        "is_available",
        "device:cuda:0",
        "set_device:resolved:cuda:0",
        "init",
        "reset:resolved:cuda:0",
    ]


def test_cuda_unavailable_never_initializes_or_resets() -> None:
    torch = FakeTorch(available=False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        CudaSmokeContext.initialize(torch, "cuda:0")
    assert torch.events == ["is_available"]


def test_staged_exception_report_preserves_original_error_and_traceback() -> None:
    try:
        try:
            raise ValueError("processor fixture failed")
        except ValueError as original:
            raise StagedInferenceError("preprocessing", original) from original
    except StagedInferenceError as exc:
        report = exception_report(exc, "generation")
    assert report["type"] == "ValueError"
    assert report["message"] == "processor fixture failed"
    assert report["stage"] == "preprocessing"
    assert "ValueError: processor fixture failed" in report["traceback"]


@pytest.mark.parametrize("script_name", ["run_4b_smoke.py", "run_4b_agent_smoke.py"])
def test_both_smoke_wrappers_report_cpu_failure_with_stage(
    script_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script(script_name)
    torch = FakeTorch(available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    report_path = tmp_path / f"{Path(script_name).stem}.json"
    result = module.main(
        [
            "--config",
            str(PROJECT_ROOT / "configs" / "eval_4b.yaml"),
            "--report",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 1
    assert report["passed"] is False
    assert report["error"]["type"] == "RuntimeError"
    assert report["error"]["stage"] == "cuda_init"
    assert "CUDA is not available" in report["error"]["message"]
    assert "RuntimeError" in report["error"]["traceback"]
    assert torch.events == ["is_available"]


def test_success_report_keeps_null_error_and_cuda_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script("run_4b_smoke.py")
    torch = FakeTorch(available=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    fake_bundle = SimpleNamespace(
        environment={
            "model": "fixture-model",
            "model_revision": "fixture-revision",
            "dtype": "bfloat16",
            "device": "cuda:0",
        },
        model=SimpleNamespace(training=False),
    )
    monkeypatch.setattr(module, "load_inference_bundle", lambda config: fake_bundle)
    report_path = tmp_path / "success.json"
    result = module.main(
        [
            "--config",
            str(PROJECT_ROOT / "configs" / "eval_4b.yaml"),
            "--load-only",
            "--report",
            str(report_path),
        ]
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["passed"] is True
    assert report["error"] is None
    assert torch.events[:5] == [
        "is_available",
        "device:cuda:0",
        "set_device:resolved:cuda:0",
        "init",
        "reset:resolved:cuda:0",
    ]


def test_wrappers_use_shared_cuda_context_for_all_memory_stats() -> None:
    for script_name in ("run_4b_smoke.py", "run_4b_agent_smoke.py"):
        source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "CudaSmokeContext.initialize" in source
        assert "cuda_memory.reset_peak_memory_stats()" in source
        assert "torch.cuda.reset_peak_memory_stats" not in source
