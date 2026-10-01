"""Read-only RL-0 lineage and protocol preflight. No GPU or API use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.rl.checkpoint import build_rl_lineage, build_rl_run_manifest
from opensearch_vl_repro.rl.config import load_rl_config
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.sft_train_plan import load_main_config


def preflight(config_path: Path, *, run_id: str = "rl-preflight") -> dict:
    config = load_rl_config(config_path)
    model = config["model"]
    sft = load_main_config(ROOT / model["sft_config"],
                           base_eval_config=ROOT / "configs/eval_base_300.yaml")
    identity = build_rl_lineage(config=config, sft_config=sft,
                                adapter_path=ROOT / model["sft_adapter"], run_id=run_id)
    manifest_path = config.get("data", {}).get("manifest")
    manifest = None
    manifest_sha = None
    if manifest_path is not None:
        path = ROOT / manifest_path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("RL data manifest is invalid")
        expected = manifest.get("manifest_sha256")
        payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if expected != canonical_json_sha256(payload):
            raise ValueError("RL data manifest checksum mismatch")
        manifest_sha = expected
    run_manifest = build_rl_run_manifest(identity, config=config, data_manifest_sha256=manifest_sha)
    return {"passed": True, "rl_config": str(config_path),
            "runtime_tool_protocol_version": config["tool"]["resolved_runtime_protocol"],
            "lineage": identity.to_dict(), "data_manifest": manifest_path,
            "run_manifest": run_manifest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rl_smoke.yaml")
    parser.add_argument("--run-id", default="rl-preflight")
    args = parser.parse_args()
    try:
        result = preflight(args.config, run_id=args.run_id)
    except (ValueError, KeyError, FileNotFoundError, TypeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
