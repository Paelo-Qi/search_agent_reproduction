"""Read-only RL-0 lineage and protocol preflight. No GPU or API use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl.checkpoint import (build_rl_lineage,
    build_rl_run_manifest, validate_sft_overlap_scope)
from opensearch_vl_repro.rl.config import load_rl_config
from opensearch_vl_repro.rl.data import preflight_dataset, prepare_formal_dataset
from opensearch_vl_repro.sft_train_plan import load_main_config


def preflight(config_path: Path, *, run_id: str = "rl-preflight",
              data_dir: Path | None = None, eval_overlap_manifest: Path | None = None,
              sft_overlap_manifest: Path | None = None,
              source_parquet: Path | None = None, source_root: Path | None = None,
              quality_audit_dir: Path | None = None, quality_only: bool = False,
              data_only: bool = False,
              development_fixture: bool = False) -> dict:
    config = load_rl_config(config_path)
    if development_fixture and (not data_only or quality_only):
        raise ValueError("development fixture preflight must be data-only")
    if quality_only and data_only:
        raise ValueError("quality-only and data-only modes are mutually exclusive")
    data_config = config.get("data", {})
    if data_config.get("manifest") is not None:
        target = data_dir or ROOT / data_config["output_dir"]
        quality_dir = quality_audit_dir or ROOT / data_config["quality_audit_dir"]
        if any(item is None for item in (eval_overlap_manifest, sft_overlap_manifest,
                                         source_parquet, source_root)):
            raise ValueError("RL data preflight requires source parquet/root and both overlap manifests")
        if quality_only:
            artifacts = prepare_formal_dataset(
                source_parquet=source_parquet, source_root=source_root,
                dataset_id=data_config["dataset_id"], dataset_revision=data_config["dataset_revision"],
                seed=data_config["seed"], smoke_count=data_config["smoke_count"],
                main_count=data_config["main_count"], shard_size=data_config["shard_size"],
                eval_overlap_manifest=eval_overlap_manifest,
                sft_overlap_manifest=sft_overlap_manifest, quality_audit_dir=quality_dir)
            manifest = artifacts["main_manifest"]
            if manifest["source_rows"] != data_config["source_rows"]:
                raise ValueError("RL quality preflight source population count mismatch")
            return {"passed": True, "scope": "quality_only", "adapter_checked": False,
                    "selected_count": len(artifacts["main"]),
                    "audited_count": manifest["audited_count"],
                    "quality_ok_count": manifest["quality_ok_count"],
                    "quality_audit_sha256": manifest["quality_audit_sha256"],
                    "main_manifest_sha256": manifest["manifest_sha256"]}
        data_result = preflight_dataset(target, source_parquet=source_parquet,
            source_root=source_root, eval_overlap_manifest=eval_overlap_manifest,
            sft_overlap_manifest=sft_overlap_manifest, quality_audit_dir=quality_dir,
            allow_incomplete_sft=development_fixture)
        if (data_result["dataset_id"] != data_config["dataset_id"]
                or data_result["dataset_revision"] != data_config["dataset_revision"]
                or data_result["selection_seed"] != data_config["seed"]
                or data_result["selection_version"] != data_config["selection_version"]):
            raise ValueError("RL data source identity differs from config")
        if not development_fixture and (data_result["main_count"] != data_config["main_count"]
                or data_result["smoke_count"] != data_config["smoke_count"]
                or data_result["shard_size"] != data_config["shard_size"]
                or data_result["source_rows"] != data_config["source_rows"]
                or not data_result["sft_image_audit_complete"]):
            raise ValueError("formal RL data counts/source population or SFT audit mismatch")
        manifest_sha = data_result["main_manifest_sha256"]
    else:
        if data_only:
            raise ValueError("RL config has no data manifest for data-only preflight")
        data_result, manifest_sha = None, None
    if data_only:
        return {"passed": True, "scope": "development_fixture_data_only" if development_fixture else "data_only",
                "adapter_checked": False, "data": data_result}
    model = config["model"]
    sft = load_main_config(ROOT / model["sft_config"],
                           base_eval_config=ROOT / "configs/eval_base_300.yaml")
    identity = build_rl_lineage(config=config, sft_config=sft,
                                adapter_path=ROOT / model["sft_adapter"], run_id=run_id)
    if data_result is None:
        raise ValueError("full RL preflight requires prepared data and SFT overlap scope")
    validate_sft_overlap_scope(identity, data_result["sft_shards"])
    run_manifest = build_rl_run_manifest(identity, config=config, data_manifest_sha256=manifest_sha)
    return {"passed": True, "scope": "full", "adapter_checked": True,
            "rl_config": str(config_path), "data": data_result,
            "runtime_tool_protocol_version": config["tool"]["resolved_runtime_protocol"],
            "lineage": identity.to_dict(), "run_manifest": run_manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl_smoke.yaml")
    parser.add_argument("--run-id", default="rl-preflight")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--source-parquet", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--quality-audit-dir", type=Path)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--eval-overlap-manifest", type=Path)
    parser.add_argument("--sft-overlap-manifest", type=Path)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--development-fixture", action="store_true")
    args = parser.parse_args()
    try:
        result = preflight(args.config, run_id=args.run_id, data_dir=args.data_dir,
            eval_overlap_manifest=args.eval_overlap_manifest,
            sft_overlap_manifest=args.sft_overlap_manifest,
            source_parquet=args.source_parquet, source_root=args.source_root,
            quality_audit_dir=args.quality_audit_dir, quality_only=args.quality_only,
            data_only=args.data_only, development_fixture=args.development_fixture)
    except (ValueError, KeyError, FileNotFoundError, TypeError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
