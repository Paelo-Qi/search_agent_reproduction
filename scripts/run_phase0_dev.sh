#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# Prevent stale evidence from an earlier Dev run from satisfying this run.
rm -f \
  "$ROOT_DIR/reports/dev_environment.json" \
  "$ROOT_DIR/reports/dev_sample_inspection.json" \
  "$ROOT_DIR/reports/dev_model_load.json" \
  "$ROOT_DIR/reports/dev_training.json" \
  "$ROOT_DIR/reports/dev_checkpoint_reload.json" \
  "$ROOT_DIR/reports/phase0_dev_status.json"

# Always leave a truthful Dev-only status file if a CUDA/model step fails.
write_incomplete_status() {
  python scripts/verify_phase0_dev.py || true
}
trap write_incomplete_status EXIT

python scripts/collect_env.py --output reports/dev_environment.json
python scripts/prepare_sft_dev.py
python scripts/inspect_sft_sample.py \
  --config configs/sft_dev.yaml \
  --index 1 \
  --report reports/dev_sample_inspection.json
python scripts/model_load_test.py \
  --config configs/sft_dev.yaml \
  --report reports/dev_model_load.json
bash scripts/train_sft_dev.sh

# This must remain a fresh process, separate from the training process.
python scripts/reload_checkpoint.py \
  --config configs/sft_dev.yaml \
  --report reports/dev_checkpoint_reload.json
python scripts/verify_phase0_dev.py --require-complete
trap - EXIT
