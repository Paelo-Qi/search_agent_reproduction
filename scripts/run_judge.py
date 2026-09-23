#!/usr/bin/env python3
"""Judge an existing Agent run; never invokes the Agent itself."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.evaluation import (  # noqa: E402
    DeepSeekJudge, JudgeRunner, build_judge_manifest, load_judge_config,
    load_judge_samples,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable DeepSeek correctness judge.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/judge.example.yaml")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/eval/combined_eval_300.parquet")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        parser.error("--run-id must be a safe 1-80 character identifier")
    run_dir = ROOT / "reports/eval_runs" / args.run_id
    parent = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    samples = load_judge_samples(run_dir / "trajectories.jsonl", args.dataset)
    config = load_judge_config(args.config)
    manifest = build_judge_manifest(parent_manifest=parent, config=config, samples=samples)
    summary = JudgeRunner(DeepSeekJudge(config), run_dir / "judge",
                          judge_manifest=manifest).run(samples, retry_failed=args.retry_failed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 and summary["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
