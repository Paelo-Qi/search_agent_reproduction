#!/usr/bin/env python3
"""Explicit opt-in single-pair live DeepSeek check; no network call at import."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opensearch_vl_repro.evaluation import (  # noqa: E402
    DeepSeekJudge, JudgeSample, load_judge_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OPT-IN live DeepSeek judge smoke.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/judge.example.yaml")
    parser.add_argument("--question", required=True)
    parser.add_argument("--reference-answer", required=True)
    parser.add_argument("--model-answer", required=True)
    parser.add_argument("--expected-verdict", choices=("correct", "incorrect"),
                        default="correct")
    args = parser.parse_args(argv)
    result = DeepSeekJudge(load_judge_config(args.config)).judge(JudgeSample(
        "live-smoke", "synthetic", args.question, args.reference_answer, args.model_answer,
    ))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if (result.status == "success"
                 and result.verdict == args.expected_verdict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
