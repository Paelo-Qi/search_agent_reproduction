#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python scripts/collect_env.py
python scripts/prepare_sft_smoke.py --count 100 --seed 20260506
python scripts/inspect_sft_sample.py --config configs/sft_smoke.yaml --index 0
python scripts/model_load_test.py --config configs/sft_smoke.yaml
bash scripts/train_sft_smoke.sh

# This is intentionally a separate Python process from training.
python scripts/reload_checkpoint.py --config configs/sft_smoke.yaml
python scripts/verify_phase0.py --require-complete
python scripts/render_reproduction_notes.py

