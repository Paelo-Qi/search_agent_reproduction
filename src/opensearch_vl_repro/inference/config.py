from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InferenceConfig:
    model_name_or_path: str
    revision: str
    dtype: str
    device: str
    attn_implementation: str
    trust_remote_code: bool
    image_max_pixels: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float
    seed: int
    max_agent_turns: int
    data_path: Path

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        # Transformers 5.x warns when sampling-only knobs are supplied to
        # deterministic generation, so omit them instead of passing inert values.
        if self.do_sample:
            kwargs.update(temperature=self.temperature, top_p=self.top_p)
        return kwargs


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing inference config section: {name}")
    return value


def load_inference_config(path: str | Path) -> InferenceConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("inference config must be a mapping")
    model = _section(raw, "model")
    generation = _section(raw, "generation")
    runtime = _section(raw, "runtime")
    data = _section(raw, "data")

    required = {
        "model": ("model_name_or_path", "revision", "dtype", "device"),
        "generation": ("max_new_tokens", "temperature", "do_sample", "top_p"),
        "runtime": ("seed", "max_agent_turns"),
        "data": ("path",),
    }
    sections = {"model": model, "generation": generation, "runtime": runtime, "data": data}
    for name, keys in required.items():
        missing = [key for key in keys if key not in sections[name]]
        if missing:
            raise ValueError(f"missing {name} keys: {', '.join(missing)}")

    project_root = config_path.parent.parent
    data_path = Path(str(data["path"]))
    if not data_path.is_absolute():
        data_path = (project_root / data_path).resolve()
    config = InferenceConfig(
        model_name_or_path=str(model["model_name_or_path"]),
        revision=str(model["revision"]),
        dtype=str(model["dtype"]),
        device=str(model["device"]),
        attn_implementation=str(model.get("attn_implementation", "sdpa")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        image_max_pixels=int(model.get("image_max_pixels", 1048576)),
        max_new_tokens=int(generation["max_new_tokens"]),
        temperature=float(generation["temperature"]),
        do_sample=bool(generation["do_sample"]),
        top_p=float(generation["top_p"]),
        seed=int(runtime["seed"]),
        max_agent_turns=int(runtime["max_agent_turns"]),
        data_path=data_path,
    )
    if not config.model_name_or_path or not config.revision:
        raise ValueError("model name and pinned revision must not be empty")
    if config.max_new_tokens < 1 or config.max_agent_turns < 1:
        raise ValueError("max_new_tokens and max_agent_turns must be positive")
    if config.temperature < 0 or not 0 < config.top_p <= 1:
        raise ValueError("temperature must be non-negative and top_p must be in (0, 1]")
    return config
