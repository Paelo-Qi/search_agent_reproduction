"""Local input only; dry-run by default. Does not fetch SearchVL-RL-8K."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl.data import prepare_prompts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="local JSON list of source_sample_id/question/image_paths")
    parser.add_argument("--dataset-id", default="local-dry-run")
    parser.add_argument("--dataset-revision", default="unversioned")
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--validation-count", type=int, default=0)
    args = parser.parse_args()
    if args.input is None:
        print(json.dumps({"dry_run": True, "prepared": False,
                          "required_record_fields": ["source_sample_id", "question", "image_paths"],
                          "note": "No data downloaded or written."}, indent=2))
        return 0
    try:
        records = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("RL source JSON must be a list")
        samples, manifest = prepare_prompts(records, dataset_id=args.dataset_id,
            dataset_revision=args.dataset_revision, seed=args.seed, limit=args.limit,
            validation_count=args.validation_count)
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"dry_run": True, "passed": False, "error": str(exc)}))
        return 1
    print(json.dumps({"dry_run": True, "passed": True, "manifest": manifest,
                      "sample_ids": [item.sample_id for item in samples]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
