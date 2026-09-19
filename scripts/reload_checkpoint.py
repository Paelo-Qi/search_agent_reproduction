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
from opensearch_vl_repro.model import load_base_model, move_batch  # noqa: E402
from opensearch_vl_repro.reporting import write_json  # noqa: E402


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description="Reload the saved adapter in a fresh process.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "sft_smoke.yaml")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / "checkpoint_reload.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint reload validation")

    config = load_config(args.config)
    adapter_dir = (PROJECT_ROOT / config["project"]["output_dir"] / "adapter").resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter checkpoint is missing: {adapter_dir}")
    processor = AutoProcessor.from_pretrained(adapter_dir)
    base_model = load_base_model(config, for_training=False)
    model = PeftModel.from_pretrained(base_model, adapter_dir).to("cuda:0").eval()

    dataset_path = (PROJECT_ROOT / config["data"]["path"]).resolve()
    sample = load_json_records(dataset_path)[0]
    batch = OpenSearchVLCollator(
        processor, dataset_path, int(config["data"]["max_length"])
    )([sample])
    batch = move_batch(batch, torch.device("cuda:0"))
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        outputs = model(**batch)
    loss = float(outputs.loss.detach().float().cpu())
    if not math.isfinite(loss):
        raise FloatingPointError(f"non-finite reload loss: {loss}")
    report = {
        "passed": True,
        "fresh_process": True,
        "adapter_path": str(adapter_dir),
        "base_model": config["model"]["name_or_path"],
        "forward_loss": loss,
        "tensor_shapes": tensor_shapes(batch),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

