#!/usr/bin/env python3
"""Run fixed dev50 with the production Agent runtime, isolated from Eval-300."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.agent.phase3_registry import create_phase3_tool_registry  # noqa: E402
from opensearch_vl_repro.agent.runtime import AgentRuntime  # noqa: E402
from opensearch_vl_repro.evaluation import BatchRunner, BatchSample, build_run_manifest  # noqa: E402
from opensearch_vl_repro.inference import (QwenAgentModel, load_inference_bundle,  # noqa: E402
                                           load_inference_config)
from opensearch_vl_repro.tool_protocol_dev import (load_dev_source_samples,  # noqa: E402
                                                   validate_dev_manifest)
from opensearch_vl_repro.sft_protocol_diagnostics import CORRECTED_POOL_MANIFEST_SHA256  # noqa: E402
from opensearch_vl_repro.sft_tool_audit import sha256_file  # noqa: E402


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-id", required=True)
    cli.add_argument("--dev-dir", type=Path, default=ROOT / "data/eval/tool_protocol_dev50")
    cli.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    cli.add_argument("--config", type=Path, default=ROOT / "configs/eval_4b.yaml")
    cli.add_argument("--search-config", type=Path, default=ROOT / "configs/search_backends.example.yaml")
    cli.add_argument("--layout-config", type=Path, default=ROOT / "configs/layout_parsing.example.yaml")
    cli.add_argument("--cache-dir", type=Path, default=ROOT / ".eval-runtime/cache")
    cli.add_argument("--adapter", type=Path)
    cli.add_argument("--max-samples", type=int)
    cli.add_argument("--retry-failed", action="store_true")
    args = cli.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        cli.error("--run-id must be a safe 1-80 character identifier")
    if args.max_samples is not None and args.max_samples < 1:
        cli.error("--max-samples must be positive")
    ids = json.loads((args.dev_dir / "ids.json").read_text(encoding="utf-8"))
    manifest_path = args.dev_dir / "tool_protocol_dev50_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_dev_manifest(ids, manifest)
    pool_path = ROOT / "data/sft_main/manifest.json"
    if sha256_file(pool_path) != CORRECTED_POOL_MANIFEST_SHA256:
        raise ValueError("run requires the pinned corrected SFT 8k manifest")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    if manifest["source_file_sha256"] != {
            source: row["sha256"] for source, row in pool["source_files"].items()}:
        raise ValueError("dev source hashes differ from corrected SFT pool provenance")
    config = load_inference_config(args.config)
    if args.adapter:
        config = replace(config, adapter_path=args.adapter.expanduser().resolve())
    run_manifest = build_run_manifest(
        run_id=args.run_id, model_name_or_path=config.model_name_or_path,
        model_revision=config.revision, inference_config_path=args.config,
        dataset_path=manifest_path, eval_manifest_path=None, start=None, limit=None,
        sample_selection={"selection_mode": "tool_protocol_dev50",
                          "ids_sha256": manifest["ids_sha256"], "sample_count": len(ids)},
        max_agent_turns=config.max_agent_turns,
        search_config_path=args.search_config, layout_config_path=args.layout_config,
        adapter_path=config.adapter_path)
    inputs = load_dev_source_samples(ids, manifest, args.raw_dir)
    samples = [BatchSample(sample_id=sample_id, benchmark="tool_protocol_dev50",
                           question=question, images=images)
               for sample_id, question, images in inputs]
    bundle = load_inference_bundle(config)
    runtime = AgentRuntime(
        model=QwenAgentModel(bundle),
        tool_registry=create_phase3_tool_registry(
            search_config=args.search_config, layout_config=args.layout_config,
            cache_dir=args.cache_dir), max_agent_turns=config.max_agent_turns)
    output = ROOT / "reports/tool_protocol_dev_runs" / args.run_id
    summary = BatchRunner(runtime, output, run_manifest=run_manifest).run(
        samples, retry_failed=args.retry_failed, max_samples=args.max_samples)
    print(json.dumps({"run_id": args.run_id, "output_dir": str(output), **summary},
                     ensure_ascii=False, indent=2))
    if summary.get("interruption") is not None:
        return 1
    if args.max_samples is not None and summary.get("pending", 0):
        return 0
    return 0 if summary.get("pending", 0) == 0 and summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
