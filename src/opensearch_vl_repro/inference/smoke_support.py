from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any


class StagedInferenceError(RuntimeError):
    def __init__(self, stage: str, original: Exception) -> None:
        super().__init__(str(original))
        self.stage = stage
        self.original = original


def exception_report(exc: Exception, stage: str) -> dict[str, Any]:
    original = exc.original if isinstance(exc, StagedInferenceError) else exc
    failure_stage = exc.stage if isinstance(exc, StagedInferenceError) else stage
    return {
        "type": type(original).__name__,
        "message": str(original),
        "stage": failure_stage,
        "traceback": traceback.format_exc(),
    }


def error_report(
    *, type_name: str, message: str, stage: str, traceback_text: str | None = None
) -> dict[str, Any]:
    return {
        "type": type_name,
        "message": message,
        "stage": stage,
        "traceback": traceback_text,
    }


@dataclass(frozen=True)
class CudaSmokeContext:
    torch_module: Any
    device: Any

    @classmethod
    def initialize(cls, torch_module: Any, device_spec: str) -> "CudaSmokeContext":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA is not available; 4B smoke was not run")
        device = torch_module.device(device_spec)
        torch_module.cuda.set_device(device)
        torch_module.cuda.init()
        return cls(torch_module=torch_module, device=device)

    def reset_peak_memory_stats(self) -> None:
        self.torch_module.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_mb(self) -> float:
        return self.torch_module.cuda.max_memory_allocated(self.device) / (1024**2)

