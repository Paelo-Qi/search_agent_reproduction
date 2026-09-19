#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / name
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gib(value: int | float | None) -> str:
    return "n/a" if value is None else f"{float(value) / 2**30:.2f} GiB"


def main() -> None:
    env = read("reports/environment.json")
    dataset = read("data/sft_smoke_100.meta.json")
    inspect = read("reports/sample_inspection.json")
    model_load = read("reports/model_load.json")
    training = read("reports/training.json")
    reload = read("reports/checkpoint_reload.json")
    status = read("reports/phase0_status.json")
    packages = env.get("packages", {})
    gpu_names = ", ".join(gpu["name"] for gpu in env.get("gpus", []))
    audit = training["parameter_audit"]
    source_counts = ", ".join(
        f"{key}={value}" for key, value in dataset["source_counts"].items()
    )
    checks = "\n".join(
        f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}"
        for name, passed in status["checks"].items()
    )
    content = f"""# Phase 0 reproduction notes

Status: **{'PASSED' if status['passed'] else 'FAILED'}**

## Provenance

- OpenSearch-VL: `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`
- Search-VL-SFT-36K: `2c1c460af4fa15bd63210cbf426a96664b959944`
- Base model: `{training['model']}`
- Base model revision: `{training['model_revision']}`
- No OpenSearch-VL source was copied; Transformers and PEFT are dependencies.

## Environment

- Python: `{env['python'].splitlines()[0]}`
- PyTorch: `{packages.get('torch')}`
- CUDA runtime: `{env.get('cuda_runtime')}`
- Transformers: `{packages.get('transformers')}`
- PEFT: `{packages.get('peft')}`
- GPUs ({len(env.get('gpus', []))}): `{gpu_names}`

## Data and preprocessing

- Samples: {dataset['sample_count']} ({source_counts})
- Seed: {dataset['seed']}
- Dataset SHA-256: `{dataset['output_sha256']}`
- Active tokens in inspected sample: {inspect['label_masking']['active_tokens']}
- Supervised assistant tokens: {inspect['label_masking']['supervised_assistant_tokens']}
- Masked prompt/observation tokens: {inspect['label_masking']['masked_tokens']}
- Vision fields: `{', '.join(inspect['vision_fields'])}`

## LoRA and training

- Targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Resolved language-model linear modules: {training['resolved_lora_target_count']}
- Trainable parameters: {audit['trainable_parameters']:,} / {audit['total_parameters']:,} ({audit['trainable_percentage']:.4f}%)
- Vision/projector tensors frozen: {training['frozen_vision_tensor_count']}
- Per-device batch: {training['per_device_train_batch_size']}
- Gradient accumulation: {training['gradient_accumulation_steps']}
- Effective global batch: {training['effective_global_batch_size']}
- Optimizer steps: {training['optimizer_steps']}
- Loss: {training['first_loss']:.6f} -> {training['final_loss']:.6f}
- LoRA update proof: `{training['lora_probe_parameter']}`, max |delta| = {training['lora_probe_max_abs_delta']:.8g}
- Peak allocated/reserved memory: {gib(training['peak_allocated_bytes'])} / {gib(training['peak_reserved_bytes'])}
- Adapter: `{training['checkpoint_path']}`

## Checkpoint validation

- Model-load forward loss: {model_load['loss']:.6f}
- Fresh-process adapter reload: {'PASS' if reload['passed'] else 'FAIL'}
- Reload forward loss: {reload['forward_loss']:.6f}

## Acceptance checklist

{checks}

## Issues and decision

No Phase 0 gate remains open if the status above is PASSED. Enter Phase 1 only
after manually reviewing `reports/sample_inspection.json`, the per-step loss/GPU
history in `reports/training.json`, and this checklist.
"""
    output = PROJECT_ROOT / "docs" / "reproduction_notes.md"
    output.write_text(content, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
