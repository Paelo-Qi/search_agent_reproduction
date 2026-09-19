from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "accelerate",
    "huggingface-hub",
    "Pillow",
    "PyYAML",
    "ijson",
)


def environment_report() -> dict[str, Any]:
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": False,
        "cuda_runtime": None,
        "cudnn": None,
        "gpus": [],
    }
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_runtime"] = torch.version.cuda
        report["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            report["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        report["nvidia_smi"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        report["nvidia_smi"] = []
    return report


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    return output

