#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opensearch_vl_repro.config import load_config  # noqa: E402
from opensearch_vl_repro.data import OpenSearchVLCollator, load_json_records, tensor_shapes  # noqa: E402
from opensearch_vl_repro.model import load_base_model, load_processor, move_batch  # noqa: E402
from opensearch_vl_repro.reporting import write_json  # noqa: E402


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Require CUDA and run one multimodal forward pass.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sft_smoke.yaml")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "model_load.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 0 model-load gate")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support BF16")

    config = load_config(args.config)
    dataset_path = (PROJECT_ROOT / config["data"]["path"]).resolve()
    sample = load_json_records(dataset_path)[0]
    processor = load_processor(config)
    collator = OpenSearchVLCollator(processor, dataset_path, int(config["data"]["max_length"]))
    batch = collator([sample])

    device = torch.device("cuda:0")
    model = load_base_model(config, for_training=False).to(device).eval()
    batch = move_batch(batch, device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        outputs = model(**batch)
    loss = float(outputs.loss.detach().float().cpu())
    if not math.isfinite(loss):
        raise FloatingPointError(f"non-finite model-load test loss: {loss}")
    report = {
        "passed": True,
        "model": config["model"]["name_or_path"],
        "model_revision": config["model"].get("revision"),
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "loss": loss,
        "tensor_shapes": tensor_shapes(batch),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
