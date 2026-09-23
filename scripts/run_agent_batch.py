#!/usr/bin/env python3
"""Sequential Agent batch execution only; deliberately performs no scoring."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.runtime import AgentRuntime  # noqa: E402
from opensearch_vl_repro.evaluation import (  # noqa: E402
    BatchRunner, BatchSample, build_run_manifest, load_selection_manifest,
)
from opensearch_vl_repro.inference import (  # noqa: E402
    QwenAgentModel, load_inference_bundle, load_inference_config, read_eval_sample,
    read_eval_samples_by_ids,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sequential resumable Agent execution (no judge or benchmark scoring)."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "eval_4b.yaml")
    parser.add_argument("--search-config", type=Path,
                        default=PROJECT_ROOT / "configs" / "search_backends.example.yaml")
    parser.add_argument("--layout-config", type=Path,
                        default=PROJECT_ROOT / "configs" / "layout_parsing.example.yaml")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".eval-runtime" / "cache")
    parser.add_argument("--start", type=int, default=0)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--limit", type=int,
                        help="Explicit batch size; this command does not default to a formal eval run.")
    selection.add_argument("--selection-manifest", type=Path,
                           help="ID-based Dev selection manifest; independent of parquet row order.")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        parser.error("--run-id must be a safe 1-80 character identifier")
    if args.start < 0 or (args.limit is not None and args.limit < 1):
        parser.error("--start must be non-negative and --limit must be positive")
    if args.selection_manifest is not None and args.start != 0:
        parser.error("--start is only valid with continuous --limit selection")

    config = load_inference_config(args.config)
    entries = selection_manifest = None
    if args.selection_manifest is not None:
        entries, selection_manifest = load_selection_manifest(args.selection_manifest)
        selection_identity = {
            "selection_mode": "id_manifest",
            "selection_manifest_checksum": selection_manifest["selection_manifest_checksum"],
            "selected_ids_checksum": selection_manifest["combined"]["canonical_id_checksum"],
            "sample_count": len(entries),
        }
    else:
        selection_identity = None
    run_manifest = build_run_manifest(
        run_id=args.run_id,
        model_name_or_path=config.model_name_or_path,
        model_revision=config.revision,
        inference_config_path=args.config,
        dataset_path=config.data_path,
        eval_manifest_path=config.data_path.parent / "manifest.json",
        start=args.start,
        limit=args.limit,
        sample_selection=selection_identity,
        max_agent_turns=config.max_agent_turns,
        search_config_path=args.search_config,
        layout_config_path=args.layout_config,
    )
    bundle = load_inference_bundle(config)
    runtime = AgentRuntime(
        model=QwenAgentModel(bundle),
        tool_registry=create_phase3_tool_registry(
            search_config=args.search_config, layout_config=args.layout_config,
            cache_dir=args.cache_dir,
        ),
        max_agent_turns=config.max_agent_turns,
    )
    samples = []
    eval_samples = (read_eval_samples_by_ids(config.data_path, entries)
                    if entries is not None else
                    [read_eval_sample(config.data_path, index)
                     for index in range(args.start, args.start + args.limit)])
    for sample in eval_samples:
        samples.append(BatchSample(
            sample_id=sample.sample_id, benchmark=sample.benchmark,
            question=sample.question, images=sample.images,
        ))
    output_dir = PROJECT_ROOT / "reports" / "eval_runs" / args.run_id
    summary = BatchRunner(runtime, output_dir, run_manifest=run_manifest).run(
        samples, retry_failed=args.retry_failed,
    )
    print(json.dumps({"run_id": args.run_id, "output_dir": str(output_dir.resolve()),
                      **summary}, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 and summary["pending"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
