"""Offline adapter identity validation only; no model, CUDA, dataset or API."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensearch_vl_repro.inference.adapter import adapter_identity
from opensearch_vl_repro.inference.config import load_inference_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval_base_300.yaml")
    args = parser.parse_args(argv)
    config = load_inference_config(args.config)
    identity = adapter_identity(args.adapter, base_model=config.model_name_or_path,
                                base_revision=config.revision)
    print("adapter_kind=" + identity.get("training_origin", "formal_sft"))
    for name in ("policy_iteration", "global_optimizer_step", "checkpoint_identity", "adapter_fingerprint"):
        if name in identity:
            print(f"{name}={identity[name]}")
    print("PASS — adapter identity only; no model/GPU/Eval execution")
    return identity


if __name__ == "__main__":
    main()
