#!/usr/bin/env python3
"""Build image-verified dev50 IDs from clean pinned source; no downloads or model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.sft_tool_audit import write_json_atomic  # noqa: E402
from opensearch_vl_repro.tool_protocol_dev import (  # noqa: E402
    DEV_SEED, build_dev_manifest, require_unchanged_dev_ids, validate_dev_manifest,
)


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    cli.add_argument("--pool-manifest", type=Path, default=ROOT / "data/sft_main_imageid_v3/manifest.json")
    cli.add_argument("--eval", type=Path, default=ROOT / "data/eval/combined_eval_300.parquet")
    cli.add_argument("--output-dir", type=Path, default=ROOT / "data/eval/tool_protocol_dev50_imageid_v3")
    cli.add_argument("--expected-ids", type=Path, required=True,
                     help="Historical Dev50 ids.json; v3 must preserve ordered membership")
    cli.add_argument("--seed", type=int, default=DEV_SEED)
    args = cli.parse_args()
    if args.output_dir.resolve() == (ROOT / "data/eval/tool_protocol_dev50").resolve():
        raise ValueError("v3 Dev50 must not overwrite the historical Dev50 directory")
    ids, manifest = build_dev_manifest(
        args.raw_dir, args.pool_manifest, args.eval, seed=args.seed)
    validate_dev_manifest(ids, manifest, pool_manifest_path=args.pool_manifest)
    expected = json.loads(args.expected_ids.read_text(encoding="utf-8"))
    require_unchanged_dev_ids(ids, expected)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"v3 Dev50 output must be a new empty directory: {args.output_dir}")
    # Neither file is written until all image/question/protocol checks pass.
    write_json_atomic(args.output_dir / "ids.json", ids)
    write_json_atomic(args.output_dir / "tool_protocol_dev50_manifest.json", manifest)
    print(f"tool-protocol dev set: {len(ids)} records; {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
