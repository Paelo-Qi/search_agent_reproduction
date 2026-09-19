from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    required = {
        "project": ("seed", "output_dir", "report_dir"),
        "model": ("name_or_path", "dtype"),
        "data": ("path", "max_length"),
        "lora": ("rank", "alpha", "target_modules"),
        "training": (
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "learning_rate",
        ),
    }
    for section, keys in required.items():
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing config section: {section}")
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise ValueError(f"Missing {section} keys: {', '.join(missing)}")
    return config

