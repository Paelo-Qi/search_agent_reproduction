# OpenSearch-VL small-scale reproduction

This repository provides two real-model execution paths for validating the
OpenSearch-VL SFT engineering chain with `Qwen/Qwen3-VL-2B-Instruct`.

- **Canonical/main Phase 0 target:** exactly 2 CUDA GPUs with BF16 support. The
  planned formal environment is 2 x A800 80GB.
- **Phase 0 Dev Smoke:** exactly 1 CUDA GPU with BF16 support, four fully local
  synthetic multimodal samples, and two optimizer steps.

A800 is planned hardware, not a GPU-model check in the code. Dev Smoke passing
does **not** mean Formal Phase 0 passed. Neither path claims a benchmark gain,
and this repository contains no RL implementation.

The common chain is:

```text
multimodal trajectory -> Qwen3-VL processor -> input_ids / labels / vision tensors
-> forward -> backward -> LoRA optimizer update -> adapter save
-> fresh-process base model + adapter reload
```

The original project is a reference, not a vendored dependency. See
`docs/upstream_provenance.md` for the pinned commits and consulted files.

Phase 2 adds an opt-in real local visual-tool loop and a PaddleOCR AI Studio
asynchronous layout API adapter without changing either Phase 0 gate. See
[Phase 2 visual tools](docs/phase2_visual_tools.md); its CPU smoke is
`python scripts/run_local_visual_smoke.py`.

Phase 3 adds a separate [real-search registry](docs/phase3_search_backends.md):
Serper for `web_search`, Serper plus Jina Reader for `text_search`, and SerpApi
image upload plus Google Lens for `image_search`. The Phase 2 registry remains
available for mock-search regression. No LLM summarization is used. The layout
smoke now defaults to a deterministic synthetic document; its timeout defaults
are 120 s/request, 5 s polling interval, and 600 s maximum polling duration.

The three real-provider smokes and both Qwen-Agent integration smokes were
reported by the user as passed on AutoDL. The integration commands remain
available for regression checks:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_4b_agent_smoke.py \
  --phase3-search-tools --phase3-tool text_search \
  --report reports/4b_phase3_text_search_smoke.json
CUDA_VISIBLE_DEVICES=0 python scripts/run_4b_agent_smoke.py \
  --phase3-search-tools --phase3-tool image_search \
  --report reports/4b_phase3_image_search_smoke.json
```

Both reports showed a successful real tool call, its observation in the next
Qwen turn, a nonempty final answer, and `trajectory.status=success`; Phase 3 is
therefore complete based on that external acceptance. Offline `pytest` alone
would not establish this live-model result.

Phase 4 adds bounded transient retry, a shared success-only filesystem cache,
sample-level resume, and full text/tool trajectory persistence without adding
Agent capabilities or scoring. See [Phase 4 reliability](docs/phase4_reliability.md).
Each batch run now has a deterministic `run_manifest.json`; the same `run_id`
can resume only the same model/checkpoint, frozen dataset, selection, inference
behavior, search/layout configuration, and tool contracts. Jina auth/quota/
configuration errors are explicit tool failures, while ordinary page failures
retain snippet fallback. PaddleOCR result downloads retry transient failures
without resubmitting the OCR job. Summary cache misses count real tool
executions, not HTTP/API calls.
Its completely offline engineering smoke is:

```bash
python scripts/run_phase4_reliability_smoke.py
```

## Three distinct validation states

1. **Local/static tests — `pytest`**

   These validate data format, deterministic synthetic generation, configuration
   separation, and assistant/observation masking. They do not prove that the
   model can load or train on CUDA.

2. **Phase 0 Dev Smoke — 1 GPU + synthetic data**

   This proves the minimum real-model training path can execute. Its isolated
   result is `reports/phase0_dev_status.json`.

3. **Formal Phase 0 — 2 GPUs + official Search-VL-SFT data**

   This remains the formal engineering gate: 100 samples from all seven sources,
   target 20 optimizer steps (strict minimum 10), and fresh-process reload. Its
   independent result is `reports/phase0_status.json`.

Recommended order:

```bash
# 1. Local/static tests
pytest

# 2. Cheap single-GPU development smoke
bash scripts/run_phase0_dev.sh

# 3. Formal Phase 0 later
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_phase0.sh
```

Dev Smoke is recommended but optional. `run_phase0.sh` neither reads nor
requires the Dev status and can run independently.

## Environment (AutoDL/Linux)

Use Python 3.11 and a CUDA environment compatible with the selected PyTorch
wheel:

```bash
cd OpenSearch-VL-Reproduction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Official PyTorch CUDA 12.6 wheels.
pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126

pip install -e ".[test]"
pytest
```

If the machine uses another CUDA runtime, install the matching official PyTorch
wheel. Both GPU paths use PyTorch SDPA; FlashAttention is not required.

## Phase 0 Dev Smoke

The Dev configuration is `configs/sft_dev.yaml`:

```text
1 visible BF16 CUDA GPU
Qwen/Qwen3-VL-2B-Instruct at the same pinned revision as Formal Phase 0
BF16 LoRA; vision tower and multimodal projector frozen
4 local synthetic multimodal samples; max_length=4096
batch=1; gradient accumulation=2; optimizer steps=2
```

Run the full path:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase0_dev.sh
```

The runner generates colored geometric images and trajectories locally. It
does not access a Hugging Face dataset and does not download the official image
archives. Two trajectories include `human -> gpt -> observation -> gpt`, which
exercises the rule that assistant bodies are supervised while observations are
excluded from loss.

Dev outputs are isolated from Formal Phase 0:

```text
data/sft_dev_4.json
data/dev_media/*.png
outputs/phase0_dev/qwen3_vl_2b_lora/adapter/
reports/dev_environment.json
reports/dev_sample_inspection.json
reports/dev_model_load.json
reports/dev_training.json
reports/dev_checkpoint_reload.json
reports/phase0_dev_status.json
```

Dev Smoke is intended to find processor/template mismatches, CUDA/dtype issues,
LoRA injection or freezing mistakes, forward/backward failures, non-finite
losses, and adapter save/reload failures as early and cheaply as possible. On a
host without CUDA it must remain `READY FOR SINGLE-GPU EXECUTION - NOT YET
PASSED`; no GPU evidence is synthesized.

Individual Dev stages can also be run directly:

```bash
python scripts/prepare_sft_dev.py
python scripts/inspect_sft_sample.py \
  --config configs/sft_dev.yaml --index 1 \
  --report reports/dev_sample_inspection.json
python scripts/model_load_test.py \
  --config configs/sft_dev.yaml --report reports/dev_model_load.json
bash scripts/train_sft_dev.sh
python scripts/reload_checkpoint.py \
  --config configs/sft_dev.yaml --report reports/dev_checkpoint_reload.json
python scripts/verify_phase0_dev.py --require-complete
```

## Formal Phase 0 gate

The formal path remains fixed to:

- OpenSearch-VL commit: `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`
- Search-VL-SFT-36K revision: `2c1c460af4fa15bd63210cbf426a96664b959944`
- Model revision: `89644892e4d85e24eaac8bacfd4f463576704203`
- BF16 LoRA, no quantization and no full fine-tuning
- frozen vision tower and multimodal projector
- 100 official, source-stratified samples from all seven sources
- exactly 2 visible CUDA GPUs with BF16 support
- target 20 optimizer steps; formal verifier requires at least 10
- global batch = `2 GPUs x 1 sample x 4 accumulation = 8`
- fresh-process adapter reload

Prepare the formal data:

```bash
python scripts/prepare_sft_smoke.py --count 100 --seed 20260506
```

The official dataset publishes roughly 10 GB of source image ZIP files. The
preparation script samples the large JSON files and extracts only selected
images, but must still obtain the published archives. This behavior is exclusive
to the formal preparation path.

Run or inspect formal stages:

```bash
python scripts/inspect_sft_sample.py \
  --config configs/sft_smoke.yaml --index 0
CUDA_VISIBLE_DEVICES=0 python scripts/model_load_test.py \
  --config configs/sft_smoke.yaml
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_sft_smoke.sh
CUDA_VISIBLE_DEVICES=0 python scripts/reload_checkpoint.py \
  --config configs/sft_smoke.yaml
python scripts/verify_phase0.py --require-complete
python scripts/render_reproduction_notes.py
```

Or execute the complete, independent formal path:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_phase0.sh
```

Only assistant message bodies receive labels. User/system content and
environment observations are excluded from SFT likelihood. Training audits that
all trainable tensors are LoRA tensors, vision/projector tensors remain frozen,
losses are finite, and at least one LoRA tensor changes. The adapter is saved
once after training finishes; there is no misleading intermediate `save_steps`
setting.

## Repository layout

```text
configs/sft_dev.yaml                single-GPU synthetic Dev configuration
configs/sft_smoke.yaml              strict two-GPU formal configuration
scripts/prepare_eval_subset.py      frozen Phase 1 evaluation-set builder
scripts/prepare_sft_dev.py          deterministic local data/image generation
scripts/run_phase0_dev.sh           complete independent Dev path
scripts/verify_phase0_dev.py        Dev-only evidence evaluator
scripts/prepare_sft_smoke.py        official formal data preparation
scripts/run_phase0.sh               complete independent formal path
scripts/verify_phase0.py            strict formal evidence evaluator
scripts/run_phase4_reliability_smoke.py offline cache/retry/resume smoke
scripts/run_agent_batch.py           resumable sequential Agent runner, no scoring
src/opensearch_vl_repro/agent/reliability.py retry/cache/image-hash primitives
src/opensearch_vl_repro/evaluation/  batch state and trajectory persistence
src/opensearch_vl_repro/training.py shared LoRA/audit/evidence implementation
tests/                              CPU-only data, masking, and config tests
```

## Phase 1 fixed evaluation subsets

All later Base, SFT, and SFT+RL comparisons must reuse the same fixed 300
questions:

```text
SimpleVQA-100
MMSearch-100
VDR-Bench-100
```

They come from the three corresponding parquet files in
`Osilly/Vision-DeepResearch-Eval`, pinned at revision
`deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9`. Sampling is unstratified fixed
random sampling with seed `20260506`, performed after stable ID sorting. Once
the ID manifests exist, reruns reuse those IDs and do not resample.

```bash
python scripts/prepare_eval_subset.py
```

The command downloads only the three pinned source parquet files, validates
required values and unique IDs, decodes every selected packed image with PIL,
and writes:

```text
data/eval/simplevqa_100.parquet
data/eval/mmsearch_100.parquet
data/eval/vdr_bench_100.parquet
data/eval/combined_eval_300.parquet
data/eval/*_100_ids.json
data/eval/manifest.json
reports/eval_subset_report.json
```

These are fixed evaluation subsets for this reduced reproduction and do not
represent complete official benchmark scores. For MMSearch, later experiments
will use final-answer accuracy only; they will not report the official end2end,
requery, rerank, or summarization composite score. Dataset construction itself
did not implement inference or tools; later phases added 4B inference and the
opt-in Agent/search backends described above.

See `docs/evaluation_subset.md` for the frozen artifact checksums and actual
source schemas.

## Scope boundary

Do not treat `reports/phase0_dev_status.json` as the formal Phase 0 gate.
Phase 4 includes reliability and batch-execution infrastructure but does not
run formal baseline scoring, the 300-item benchmark, judging, 3K SFT, RL, full
fine-tuning, or QLoRA. A local/mock test pass is not a live provider or model
acceptance.
