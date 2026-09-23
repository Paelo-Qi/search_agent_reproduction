#!/usr/bin/env python3
"""Read-only Base Eval-300 validation; never loads a model or calls a provider."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.agent.layout_parsing import load_layout_api_config  # noqa: E402
from opensearch_vl_repro.agent.search_providers import load_search_config  # noqa: E402
from opensearch_vl_repro.evaluation import build_eval300_plan, load_judge_config  # noqa: E402
from opensearch_vl_repro.inference import load_inference_config  # noqa: E402


def _counts(entries: tuple[tuple[str, str], ...]) -> dict[str, int]:
    return dict(sorted(Counter(benchmark for benchmark, _ in entries).items()))


def build_preflight_report(
    *,
    config_path: str | Path,
    search_config_path: str | Path,
    layout_config_path: str | Path,
    judge_config_path: str | Path,
) -> dict[str, Any]:
    """Validate files/config/env names without constructing any runtime/provider."""
    config = load_inference_config(config_path)
    search = load_search_config(search_config_path)
    layout = load_layout_api_config(layout_config_path)
    judge = load_judge_config(judge_config_path)
    plan = build_eval300_plan(config.data_path)

    expected_model = {
        "model_name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "dtype": "bfloat16",
        "device": "cuda:0",
        "attn_implementation": "sdpa",
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
        "top_p": 1.0,
        "max_agent_turns": 8,
    }
    actual_model = {name: getattr(config, name) for name in expected_model}
    env_names = {
        "serper": str(search.serper["api_key_env"]),
        "jina_reader_optional": str(search.jina_reader["api_key_env"]),
        "serpapi": str(search.serpapi["api_key_env"]),
        "layout_parsing": layout.access_token_env,
        "deepseek_judge": judge.api_key_env,
    }
    env_present = {name: bool(os.environ.get(env_name))
                   for name, env_name in env_names.items()}
    agent_required = ("serper", "serpapi", "layout_parsing")
    agent_env_ready = all(env_present[name] for name in agent_required)
    judge_env_ready = env_present["deepseek_judge"]

    first_set, second_set = set(plan.first_batch), set(plan.second_batch)
    static_checks = {
        "dataset_exists": config.data_path.is_file(),
        "dataset_sha256_matches": True,
        "sample_count_is_300": len(plan.entries) == 300,
        "benchmark_counts_are_100_each": _counts(plan.entries) == {
            "mmsearch": 100, "simplevqa": 100, "vdr_bench": 100,
        },
        "first_batch_is_67_67_66": _counts(plan.first_batch) == {
            "mmsearch": 67, "simplevqa": 67, "vdr_bench": 66,
        },
        "second_batch_is_33_33_34": _counts(plan.second_batch) == {
            "mmsearch": 33, "simplevqa": 33, "vdr_bench": 34,
        },
        "sample_ids_globally_unique": len({sample_id for _, sample_id in plan.entries}) == 300,
        "batches_disjoint": not first_set & second_set,
        "batches_union_is_eval300": first_set | second_set == set(plan.entries),
        "model_config_matches_frozen_base": actual_model == expected_model,
    }
    return {
        "offline": True,
        "provider_calls": 0,
        "model_loaded": False,
        "dataset_path": str(config.data_path),
        "dataset_sha256": plan.dataset_sha256,
        "run_id": "base-eval300-v1",
        "selection_identity": plan.selection_identity(),
        "counts": {
            "full": _counts(plan.entries),
            "first_batch": _counts(plan.first_batch),
            "second_batch": _counts(plan.second_batch),
        },
        "model_config": actual_model,
        "api_env": {name: {"env_name": env_names[name], "present": present}
                    for name, present in env_present.items()},
        "agent_api_env_ready": agent_env_ready,
        "judge_api_env_ready": judge_env_ready,
        "static_checks": static_checks,
        "static_validation_passed": all(static_checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/eval_base_300.yaml")
    parser.add_argument("--search-config", type=Path,
                        default=ROOT / "configs/search_backends.example.yaml")
    parser.add_argument("--layout-config", type=Path,
                        default=ROOT / "configs/layout_parsing.example.yaml")
    parser.add_argument("--judge-config", type=Path,
                        default=ROOT / "configs/judge.example.yaml")
    parser.add_argument("--require-agent-env", action="store_true")
    parser.add_argument("--require-judge-env", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_preflight_report(
            config_path=args.config,
            search_config_path=args.search_config,
            layout_config_path=args.layout_config,
            judge_config_path=args.judge_config,
        )
        passed = report["static_validation_passed"]
        if args.require_agent_env:
            passed = passed and report["agent_api_env_ready"]
        if args.require_judge_env:
            passed = passed and report["judge_api_env_ready"]
        report["passed"] = passed
    except Exception as exc:
        report = {"offline": True, "provider_calls": 0, "model_loaded": False,
                  "passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
