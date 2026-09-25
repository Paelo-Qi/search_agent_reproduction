#!/usr/bin/env python3
"""Fresh-process Base-4B + formal LoRA adapter generation on an SFT smoke image."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.data import build_messages, load_json_records  # noqa: E402
from opensearch_vl_repro.inference import (  # noqa: E402
    generate_chat, load_inference_bundle, load_inference_config,
)
from opensearch_vl_repro.reporting import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval_base_300.yaml")
    parser.add_argument("--sample-data", type=Path, default=ROOT / "data/sft_4b_smoke_100.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/sft_4b_smoke/adapter_reload.json")
    args = parser.parse_args()
    config = replace(load_inference_config(args.config), adapter_path=args.adapter.expanduser().resolve())
    bundle = load_inference_bundle(config)
    sample = load_json_records(args.sample_data)[0]
    messages, _, _ = build_messages(sample, args.sample_data)
    first_user = next(message for message in messages if message["role"] == "user")
    output = generate_chat(bundle, [first_user])
    if not output:
        raise RuntimeError("fresh-process adapter generation returned empty text")
    report = {"passed": True, "fresh_process": True,
              "base_model": config.model_name_or_path, "base_revision": config.revision,
              "adapter_identity": bundle.environment["adapter_identity"],
              "sample_data": str(args.sample_data.resolve()),
              "generated_text": output}
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
