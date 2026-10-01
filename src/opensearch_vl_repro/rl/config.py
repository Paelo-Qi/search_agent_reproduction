"""Strict, static RL-0 configuration; hardware topology is intentionally absent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from opensearch_vl_repro.agent.tool_contracts import RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION

from .reward import unit_reward


def load_rl_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("RL config must be a mapping")
    if config.get("algorithm", {}).get("advantage_estimator") != "rloo":
        raise ValueError("RL advantage_estimator must be rloo")
    if config["algorithm"].get("fatal_consecutive_errors") != 3:
        raise ValueError("RL fatal threshold must be 3")
    if config.get("rollout_n") not in {2, 4}:
        raise ValueError("RL logical rollout_n must be 2 or 4")
    if config.get("agent", {}).get("max_agent_turns") != 16:
        raise ValueError("RL max_agent_turns must be 16")
    reward = config.get("reward", {})
    a = unit_reward(reward.get("accuracy_weight"), "accuracy_weight")
    q = unit_reward(reward.get("query_weight"), "query_weight")
    if abs(a + q - 1) > 1e-12 or reward.get("format_multiplicative") is not True:
        raise ValueError("RL reward policy must be 0.8/0.2 multiplicative")
    if (a, q) != (.8, .2):
        raise ValueError("RL reward weights must be 0.8/0.2")
    model = config.get("model", {})
    if model.get("continue_from_sft_adapter") is not True or not model.get("sft_adapter"):
        raise ValueError("RL must continue an existing SFT adapter")
    if config.get("tool", {}).get("runtime_protocol") != "current":
        raise ValueError("RL runtime image/tool protocol mismatch")
    config["tool"]["resolved_runtime_protocol"] = RUNTIME_IMAGE_SEARCH_PROTOCOL_VERSION
    if any(key in config for key in ("fsdp", "vllm", "distributed", "train_prompt_batch_size", "ppo_mini_batch_size")):
        raise ValueError("untested physical/distributed settings are forbidden in RL-0")
    return config
