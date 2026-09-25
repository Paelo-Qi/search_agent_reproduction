# SFT-0: fixed data, resumable 4B LoRA, and adapter evaluation

This is an engineering plan, **not evidence that 4B GPU training has passed**. The
paper's full-parameter, large-scale SFT is intentionally replaced by a small
BF16 LoRA experiment to isolate the contribution of SFT and later RL under the
same fixed Eval-300 protocol. No QLoRA or full fine-tuning is used.

## Pinned identities and scope

- Base: `Qwen/Qwen3-VL-4B-Instruct` at
  `ebb281ec70b05090aa6165b016eac8ec08e71b17`, exactly the revision in
  `configs/eval_base_300.yaml`.
- Data: `OpenSearch-VL/Search-VL-SFT-36K` at
  `2c1c460af4fa15bd63210cbf426a96664b959944`.
- Training: 2 BF16 CUDA GPUs; no quantization; LoRA rank 16/alpha 32/dropout
  0.05 on language `q/k/v/o/gate/up/down_proj`; vision tower and multimodal
  projector frozen; `image_max_pixels=262144`, `max_length=32000`, gradient
  checkpointing on. The default micro-batch 1 × accumulation 4 may be changed
  to micro-batch 2 × accumulation 2 **before** the formal run; both are global
  batch 8. Do not change micro-batch halfway through a lineage.
- Base Eval inference retains `image_max_pixels=1048576`, 16 Agent turns,
  `max_new_tokens=512`, temperature 0, deterministic decoding, and the same
  prompt/tools/backends/Judge. Only the optional adapter changes.

## Deterministic 8k pool

`scripts/prepare_sft_main.py` reads the seven pinned source JSON files and
uses largest-remainder allocation (alphabetical tie-break) against the **raw**
36,592-source population: fvqa 965, livevqa 2913, palace 650, webqa 832,
wiki_art 1113, wiki_en 766, wiki_zh 761. Within each source, only samples
accepted by the existing Phase 0 collator format are eligible; seed-qualified
SHA256 ranks select once from that eligible list and then partition without
replacement. This is stable across local and AutoDL Python versions.
The current raw files contain 41 invalid-format records, reported by source
in the manifest; they are neither rewritten nor silently treated as trainable.
Each selected identity is `source:original_index`, with a raw-record SHA256.

| Physical shard | Samples | Cumulative lineage | Formal role |
|---|---:|---:|---|
| `main_a_1k` | 1,000 | 1,000 | Health-check checkpoint |
| `main_b_2k` | 2,000 | 3,000 | Main experiment |
| `extra_1k` | 1,000 | 4,000 | Optional time-permitting extension |
| `reserve_4k` | 4,000 | 8,000 | Reserve only |

The manifest records source-file SHA256s, each shard's exact source counts and
output SHA256, sample membership/source/original index/raw SHA256, and a
disjointness assertion. All four shard JSONs are physical and disjoint;
`main_3k`, cumulative 4k, and cumulative 8k are lineage names, not separately
sampled files. Main 3k is the primary result; 4k is the most likely optional
extension, while 8k requires unusually ample training time.

| Source | main_a_1k | main_b_2k | extra_1k | reserve_4k | Pool 8k |
|---|---:|---:|---:|---:|---:|
| fvqa | 121 | 241 | 121 | 482 | 965 |
| livevqa | 364 | 728 | 364 | 1457 | 2913 |
| palace | 81 | 163 | 81 | 325 | 650 |
| webqa | 104 | 208 | 104 | 416 | 832 |
| wiki_art | 139 | 278 | 139 | 557 | 1113 |
| wiki_en | 96 | 192 | 96 | 382 | 766 |
| wiki_zh | 95 | 190 | 95 | 381 | 761 |

Metadata-only preparation never downloads images. For actual training, opt in
to extraction of selected members from each pinned official image ZIP.
`media_status.json` records image readiness separately: materializing images
does **not** change the fixed selection manifest checksum. Run preflight after
image extraction. Do not copy only shard JSONs without their media/manifest.

## Read-only preflight and current blockers

`scripts/preflight_sft_main.py` writes four reports under
`reports/sft_preflight/`:

- `leakage.json`: normalized question matches and decoded-image-content SHA256
  matches against the unchanged fixed Eval-300, with training ID, source,
  Eval ID, benchmark, and image hash. Overlaps are reported, never removed.
- `tool_contract.json`: every selected sample's declared tool schemas and
  actual `<tool_call>` arguments checked against current runtime declarations.
  No argument rename or schema widening occurs.
- `sequence.json`: actual pinned processor/template lengths for all four
  shards and the full 8k (min/mean/median/p50/p90/p95/p99/max, count and ratio
  over 32k), zero-supervised-token count, assistant-span cut count, and
  tool-call truncation risk. The current collator right-truncates at 32k and
  supervises assistant bodies only; a truncated assistant span may be damaged
  even when some supervised tokens remain. The audit reports this and does
  not edit trajectories.
- `summary.json`: bound to the pool manifest and frozen Eval SHA256; blocks
  formal training if images/sequence are incomplete, tool schemas drift, or
  supervised tokens/assistant spans are damaged. A leakage overlap requires
  an explicit human acknowledgement before training.

The current local metadata-only run found **32,000 declaration-schema drifts**
(four per selected sample), although its parsed tool-call arguments showed no
call-schema drift. Examples: dataset `layout_parsing.file_path`, dataset
`text_search.query/lang`, and different required arguments for
`super_resolution`. This is a genuine train-time prompt-contract mismatch
because the existing collator passes each sample's declared tools into the
template. It is **not** auto-remapped. One normalized question overlap was
also found (`wiki_art:4712` ↔ SimpleVQA `1492`). Without official images,
image-overlap and processor sequence audits remain incomplete. The user must
decide how to resolve/review these before formal 1k training; do not bypass
the failed preflight merely to start a run.

## Scheduler, checkpoints, and continuation

The trainer is a separate, explicit DDP loop; old Phase 0 smoke training is
unchanged. It saves an equivalent full **trainer state** rather than pretending
that loading an adapter alone is a resume. With 2 GPUs, global batch 8 and 2
epochs per shard, planned optimizer steps are computed from sample count:

| Boundary | This shard | Cumulative global step | Scheduler phase/horizon |
|---|---:|---:|---|
| 1k | 250 | 250 | Phase 1 / 1,000 |
| 3k | 500 | 750 | Phase 1 / 1,000 |
| 4k | 250 | 1,000 | Phase 1 / 1,000 |
| 8k | 1,000 | 2,000 | Phase 2 / 1,000 |

Phase 1 is cosine, warmup ratio 0.1 (100 steps), candidate peak LR `2e-4`.
The 1k→3k and 3k→4k transitions load the previous adapter, optimizer,
scheduler, global/phase steps, per-rank Python/NumPy/CPU/CUDA RNG, and
deterministic dataloader offset; the scheduler is never re-warmed. At 3k one
may stop and evaluate that checkpoint independently. Phase 2 starts **only**
from a complete checkpoint-4k; it keeps the model and optimizer state but
creates a new 1,000-step cosine schedule. Its peak LR is deliberately `null`
in config and must be selected explicitly after reviewing Phase 1 (for
example, perhaps `1e-4`, not assumed here).

Each checkpoint directory holds `adapter/` (PEFT safetensors and processor
copy), `optimizer.pt`, `scheduler.pt`, `rng.pt` (per-rank states),
`trainer_state.json`, and `metadata.json` with file checksums, model/revision,
pool/shard checksums, lineage, epoch/dataloader offset, global/phase steps,
current LR, warmup/horizon/peak LR, and parent checkpoint. Periodic
checkpoints are written only after optimizer steps. Final 1k/3k/4k/8k
checkpoints have distinct immutable paths; resume validates exact predecessor,
batch configuration, base identity, pool checksum, and schedule phase.
The training report additionally records finite losses, trainable parameter
audit, LoRA parameter delta, per-GPU load/train peak allocated/reserved VRAM,
overall peak VRAM, step time, samples/sec, environment versions, and
TensorBoard scalar logs. Tokens/sec is **not** reported because the multimodal
processor's variable image-token expansion and padding do not yet support a
reliable, comparable counter.

## Local/static commands (no model download, GPU, or API)

From the project root, using the project Python environment:

```bash
python scripts/prepare_sft_main.py                  # pinned JSON already local
python -m pytest
```

The first command writes `data/sft_main/{main_a_1k,main_b_2k,extra_1k,reserve_4k}.json`
and `manifest.json`; `media_status.json` is a separate readiness report. The
selection manifest is version-controlled while generated trajectories/media
are ignored; the former verifies pure planning/identity/audit logic.
Local preflight can be attempted, but it correctly exits nonzero until images
are present and the schema drift is resolved; it does not download a model.

## AutoDL acceptance order (do not skip the gates)

Run from the same project root, with dependencies from `pyproject.toml` and
the pinned raw JSON/Eval-300 parquet present. `torchrun` requires two visible
BF16 GPUs. The preparation command below explicitly permits official ZIP
downloads; omit `--download-images` if the seven archives are already local.

```bash
# 1. Materialize the independent 4B official 100-sample smoke set.
python scripts/prepare_sft_4b_smoke.py

# 2A. Benchmark micro=1/accum=4; intentionally pause at step 10, inspect,
# then verify full optimizer/scheduler/RNG resume to step 20.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --config configs/sft_4b_smoke.yaml \
  --stage smoke --run-tag micro1_accum4 --micro-batch 1 --grad-accum 4 \
  --stop-after-steps 10
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --config configs/sft_4b_smoke.yaml \
  --stage smoke --run-tag micro1_accum4 --micro-batch 1 --grad-accum 4 \
  --resume-from outputs/sft_4b_smoke/micro1_accum4/checkpoint-step-10

# 2B. Independently benchmark micro=2/accum=2 (same global batch).
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --config configs/sft_4b_smoke.yaml \
  --stage smoke --run-tag micro2_accum2 --micro-batch 2 --grad-accum 2

# 3. Compare without auto-selecting a winner: finite loss, OOM, per-GPU peak
# allocated/reserved VRAM, step time, samples/sec, and checkpoint reload.
python scripts/compare_sft_4b_smoke.py \
  --a reports/sft_4b_smoke/micro1_accum4/smoke_step20.json \
  --b reports/sft_4b_smoke/micro2_accum2/smoke_step20.json

# 4. Fresh-process Base+smoke-adapter generation on an official SFT smoke
# sample, not on the held-out Eval-300.
CUDA_VISIBLE_DEVICES=0 python scripts/reload_sft_adapter.py \
  --config configs/eval_base_300.yaml \
  --adapter outputs/sft_4b_smoke/micro1_accum4/checkpoint-20/adapter \
  --report reports/sft_4b_smoke/adapter_reload.json

# Optional: real current eight-tool Agent registry and external search API;
# this is a protocol smoke, not accuracy or Eval-300. Repeat up to 3 times.
CUDA_VISIBLE_DEVICES=0 python scripts/run_4b_agent_smoke.py \
  --config configs/eval_base_300.yaml \
  --adapter outputs/sft_4b_smoke/micro1_accum4/checkpoint-20/adapter \
  --phase3-search-tools --phase3-tool text_search \
  --report reports/sft_4b_smoke/agent_text_search.json

# 5. Materialize the fixed 8k training pool's official images and audit it.
python scripts/prepare_sft_main.py --extract-images --download-images
python scripts/preflight_sft_main.py
```

The next commands are **conditional**, not permission to ignore a failed
preflight. First inspect `reports/sft_preflight/*.json`, settle the declaration
drift without changing Eval runtime contracts or silently rewriting calls,
review the leakage record, and verify any 32k cut/zero-target findings.
Select A or B based on the two smoke reports, keep that choice fixed for the
whole lineage, and inspect finite losses, LR, LoRA delta, frozen vision,
checkpoint completeness, throughput, VRAM and malformed-data/long-sequence
errors at checkpoint-1k before continuing.

```bash
# Example only after preflight PASS and human approval; shows A's fixed batch.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --stage main_a_1k --micro-batch 1 --grad-accum 4 \
  --acknowledge-leakage

# After manual checkpoint-1k health check, same Phase 1 scheduler/state.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --stage main_b_2k --micro-batch 1 --grad-accum 4 \
  --resume-from outputs/sft_main/checkpoint-1k --acknowledge-leakage

# Optional, retaining checkpoint-3k independently.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --stage extra_1k --micro-batch 1 --grad-accum 4 \
  --resume-from outputs/sft_main/checkpoint-3k --acknowledge-leakage

# Reserve only, after a separately chosen Phase 2 peak LR and 4k health check:
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --stage reserve_4k --micro-batch 1 --grad-accum 4 \
  --resume-from outputs/sft_main/checkpoint-4k \
  --phase-2-peak-lr "$PHASE2_PEAK_LR" --acknowledge-leakage
```

For a future SFT Eval-300, use a **new** run ID and only add `--adapter` to
the unchanged Base evaluation command. The run manifest records the adapter
path/fingerprint, adapter-config fingerprint, cumulative stage, checkpoint
lineage, and pinned Base model/revision, so Base and SFT results cannot
resume into one another. Example (do **not** run during SFT-0):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id sft-3k-eval300-v1 --config configs/eval_base_300.yaml \
  --adapter outputs/sft_main/checkpoint-3k/adapter --eval300 --max-samples 200
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id sft-3k-eval300-v1 --config configs/eval_base_300.yaml \
  --adapter outputs/sft_main/checkpoint-3k/adapter --eval300
```

Judge remains the existing separate stage and must use the same scoring
protocol. Neither a 4B smoke success nor a static test success is a formal
1k/3k SFT or Eval-300 result.
