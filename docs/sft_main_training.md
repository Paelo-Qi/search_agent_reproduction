# SFT-0: fixed data, resumable 4B LoRA, and adapter evaluation

The 4B smoke A/B runs and the original formal SFT-3k training have completed.
That original checkpoint-3k is **legacy pre-image-ID-grounding** and must not
be used as the corrected SFT Eval-300 adapter. The paper's full-parameter, large-scale SFT is intentionally replaced by a small
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
  checkpointing on. Formal training is fixed at micro-batch **1** × accumulation
  **4** × 2 GPUs = global batch 8. Do not change this batch configuration
  during the formal lineage. Micro-batch 2 × accumulation 2 was tested but is
  not the chosen formal setting.
- Base Eval inference retains `image_max_pixels=1048576`, 16 Agent turns,
  `max_new_tokens=512`, temperature 0, deterministic decoding, and the same
  prompt/tools/backends/Judge. Only the optional adapter changes.

Formal 4B SFT uses `flash_attention_2` and requires a working `flash-attn`
installation in the GPU training environment. The verified AutoDL stack is
`flash-attn==2.8.3` with `torch==2.7.1+cu126`, Transformers 5.17.0 and PEFT
0.21.0. Install/verify this compiled extension against that environment before
loading the model. It is documented here rather than forced into the regular
`pyproject.toml` dependencies because wheel/build compatibility depends on
the CUDA/PyTorch platform. This changes **only SFT training**; the frozen
`configs/eval_base_300.yaml` inference attention backend remains SDPA.

The prior gradient-checkpointing failure (outer model training, inner language
model eval) has been fixed: training now sets `model.train()` and checks the
language model plus all 36 decoder layers. On the 16,723-token smoke sample,
forward peak allocation fell from about 77 GiB to 35 GiB. Smoke A
(micro 1/accum 4) completed step 10 → resume → step 20 with `passed=true`,
about 43.98 GiB peak allocated and 14.17 s/optimizer step. Smoke B
(micro 2/accum 2) also passed but used about 78.68 GiB and was slower.

## Deterministic 8k pool

`scripts/prepare_sft_main.py` reads the seven pinned source JSON files and
uses largest-remainder allocation (alphabetical tie-break) against the **raw**
36,592-source population: fvqa 965, livevqa 2913, palace 650, webqa 832,
wiki_art 1113, wiki_en 766, wiki_zh 761. Within each source, only samples
accepted by the existing Phase 0 collator format are eligible; seed-qualified
SHA256 ranks select once from that eligible list and then partition without
replacement. Frozen Eval-300 question metadata is checked before selection;
excluded candidates are skipped and the next valid, unused candidate in the
**same source rank order** fills the fixed quota. The same mechanism accepts
frozen image-overlap exclusions after media is available. This is stable
across local and AutoDL Python versions.
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
output SHA256, sample membership/source/original index/raw SHA256, exclusion
IDs/reasons, replacement IDs/sources, ranking version, and a disjointness
assertion. Manifest schema is now **version 2**; old version-1 pools cannot be
used for formal training. All four shard JSONs are physical and disjoint;
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
Images must be local **before training**; the trainer never downloads them.

The raw dataset's tool declarations are broader than the fixed Agent runtime
contract. We do not widen runtime tools or alter Base Eval-300. Preparation
retains each raw declaration verbatim in `_source_tools`, then puts the current
runtime `TOOL_DECLARATIONS` chat-template schemas in `tools`, the field the SFT
collator actually sends to Qwen. Expert calls, observations, and final answers
are unchanged. The manifest records the transform version, raw-declaration
fingerprint, effective runtime-contract fingerprint, and canonicalization flag.
Changing the runtime contract invalidates this pool until explicitly rebuilt.
The independent 4B smoke data file stays raw; its records receive the same
declaration transformation only in training memory.

## SFT image-ID grounding correction (input format v2)

The original SFT message builder attached a real PIL image at `<image>` but did
not tell the assistant that the first input image is registered as `img_1`.
For example, `fvqa:2921` starts with `<image>Which novel earned the woman in
the image the Man Booker International Prize?`; its first assistant target
calls `image_search` with `{"url":"img_1"}`. Before this correction, the
message prefix had the image and question plus the legacy source system prompt:
it mentioned `img_1` in generic examples but never registered this input image,
and it even allowed a direct URL for `image_search.url`. The effective SFT
system now replaces only that pinned `*Params*` line with a registered-ID-only
version, preserving every other source instruction. Unknown direct-URL wording
fails closed instead of silently retaining contradictory instructions. The
system prefix then ends with the runtime-style text
`Registered input images:\n- img_1: width=<actual width>, height=<actual height>`.
It also appends the existing runtime rule that `image_search.url` must be a
registered `img_n`, never an HTTP URL or file path, so the legacy generic
instruction does not contradict the effective training input.
The image remains the same PIL payload; assistant text, tool declarations and
the structured assistant-only loss mask are unchanged. No image is duplicated.

Only images in the first human turn are registered up front, in marker order.
Images appearing later in tool observations are **not** promoted to initial
images: their own `New image ID: img_n` text must register them before later
assistant calls. The read-only image-grounding audit checks this monotonic
runtime order, all image-tool `img_n` references, and reports exact sample IDs
on gaps. The local legacy 3k JSON has six trajectories with `img_3` announced
without a prior `img_2`: `fvqa:2711`, `fvqa:3371`, `fvqa:4016`,
`fvqa:3026`, `fvqa:1145`, `fvqa:219`. At least `fvqa:3371` also calls
`image_search(img_2)` before registration. Two further source trajectories,
`webqa:1884` and `webqa:3301`, supervise HTTP URLs in `image_search.url`.
Corrected SFT v2 therefore consists of **image-ID grounding, the exact
conflicting source-system line canonicalization, and the frozen data-quality
exclusions in `configs/sft_data_exclusions.json`**. Before selection, the
CPU-only `scripts/audit_sft_source_image_contract.py` scans all 36,592 pinned
source records (and verifies their pinned hashes). It finds 22 non-`img_n`
`image_search.url` targets, 79 derived-ID gap/mismatch events, and 27
ungrounded image-reference events. These overlap across records: the frozen
list has 104 unique sample IDs, including the original eight (22 with reason
`image_search_non_img_n_target`, 82 with reason `derived_image_id_gap`). No
expert assistant/tool-call target is rewritten. Deterministic same-source
replacements remove excluded selected records from the formal pool. Preflight
also fails if any non-`img_n` image_search target or ungrounded image ID
remains in the regenerated pool.

The manifest, preflight summary and checkpoint metadata now carry
`runtime-image-id-grounding-v2`; old manifests and old checkpoints cannot pass
the new gates. The mask version remains `structured-message-prefix-v1`.
Regenerating with the same pinned inputs/seed **and frozen exclusions** is
deterministic. Relative to the legacy pool, 20 selected IDs are replaced in
their own sources; the other 84 excluded IDs were outside the legacy 8k.
The 8k size, source quotas, shard sizes and selection algorithm/seed stay
fixed. The manifest hash changes due to the exclusions and message-format
version. The old checkpoint-3k must be
retired for corrected SFT comparison and training must restart from pinned
Base after all gates pass. Do not resume the old adapter/optimizer state.

On AutoDL, run the full pinned-source audit **before** backing up the old pool,
so its source hashes can be checked against the legacy manifest. The
`--verify-frozen` gate requires the committed exclusions to match the complete
source audit exactly. Then regenerate and run the full multimodal preflight:

```bash
python scripts/audit_sft_source_image_contract.py \
  --expected-manifest data/sft_main/manifest.json \
  --report reports/sft_source_image_contract.json --verify-frozen
mv data/sft_main data/sft_main_before_quality_exclusions
python scripts/prepare_sft_main.py \
  --exclusions configs/sft_data_exclusions.json --extract-images --download-images
python scripts/audit_sft_image_grounding.py \
  --previous-manifest data/sft_main_before_quality_exclusions/manifest.json \
  --output reports/sft_preflight/image_grounding_main_3k.json
python scripts/preflight_sft_main.py
python -m pytest
```

Omit `--download-images` only when all seven official ZIP archives are already
local. The source audit needs no processor, image files, model weights, GPU,
or API. The later image-grounding audit uses the locally cached pinned
processor, no model weights, GPU or API. Its report contains JSON-safe messages, tools, actual
`apply_chat_template` prompt, pre-first-tool context, a 3k grounding summary,
and `selection_change_matches_frozen_exclusions=true` (the legacy selection
identity is intentionally different). It exits nonzero on any ungrounded or
out-of-order ID. The full preflight checks all 8k shard counts (1000/2000/
1000/4000), manifest/shard hashes and membership, leakage, tool contract,
grounding, and actual processor token/mask/truncation. Confirm
`image_search_img_n == grounded_image_search_img_n` in the grounding report,
`selection_change_matches_frozen_exclusions=true`,
`image_search_non_img_n=0`, and `summary.json.passed=true` before any corrected
training. The two historical HTTP-url targets are excluded, not rewritten.

A local metadata-only 8k rebuild with all 104 source-level exclusions
preserved 1000/2000/1000/4000 shard sizes, all source quotas, and the
surviving expert conversations. Its static scan found
`image_search_non_img_n=0`, `ungrounded_runtime_image_refs=0`, and
`derived_id_gaps=0`; all 9,080 runtime image references were grounded.
This does **not** constitute a passing multimodal processor preflight.
Corrected formal training remains blocked until the pool is regenerated on
AutoDL and the full preflight passes.

## Read-only preflight and current blockers

`scripts/preflight_sft_main.py` writes six reports under
`reports/sft_preflight/`:

- `reserved_literals.json`: read-only counts of literal `<|im_start|>` and
  `<|im_end|>` in structured message bodies, by source role, plus bounded
  sample/shard/turn examples. Literals are not a reason to delete samples.
- `leakage.json`: normalized question matches and decoded-image-content SHA256
  matches against the unchanged fixed Eval-300, with training ID, source,
  Eval ID, benchmark, and image hash. The final formal pool requires **zero
  known question and image overlaps**. If image overlap appears after media
  extraction, rebuild with `--leakage-report` and rerun full preflight.
- `tool_contract.json`: raw declaration drift is reported only; effective
  training declaration drift and actual expert-call drift are separate hard
  failures. No expert argument rename or schema widening occurs.
- `sequence.json`: actual pinned processor/template lengths for all four
  shards and the full 8k (min/mean/median/p50/p90/p95/p99/max, count and ratio
  over 32k), plus four distinct outcomes based on actual multimodal processor
  tokens: zero supervised targets, partial assistant span cut, partial
  `<tool_call>...</tool_call>` cut, and complete later assistant span dropped.
  The first three are hard failures. A complete later turn dropped is
  report-only when earlier intact supervised content remains; it does not
  automatically fail. Problem and report-only sample lists are separate.
- `image_grounding.json`: initial and derived image-ID registrations and
  image-tool references in actual constructed messages; ungrounded IDs or
  nonmonotonic derived registrations fail the formal gate with sample IDs.
- `summary.json`: bound to the pool manifest and frozen Eval SHA256; blocks
  formal training unless media/sequence audits are complete, effective tool
  and call drift are zero, question/image overlap is zero, and all hard
  truncation counts are zero. It includes `structured-message-prefix-v1`,
  which the trainer requires so an older mask audit cannot authorize training.
  There is no leakage-acknowledgement bypass.

The label mask and sequence preflight derive assistant bodies from structured
message roles. For each assistant, the real multimodal processor renders a
message prefix and an empty-body prefix; verified token-prefix positions give
the assistant body and its true template end. Literal Qwen boundary spellings
inside user/tool/assistant content are never used to infer a role. Tool-call
truncation is checked by decoding only these role-derived assistant spans,
not by searching a flattened sequence for a separately encoded marker.

The previous pool exposed 32,000 raw declaration-schema drifts (four per
sample), but no parsed expert call drift. These raw differences remain
auditable; effective declarations are canonicalized. The known question
overlap `wiki_art:4712` ↔ SimpleVQA `1492` is excluded by general normalized
question matching and replaced by the next same-source ranked candidate,
`wiki_art:4506`. The four shard sizes and all per-shard source counts are
unchanged; the rebuilt local metadata preflight reports question overlap 0,
effective declaration drift 0, actual call drift 0, and raw drift 32,000.
Until official images are materialized and full preflight passes, image
overlap and processor sequence audits remain **unverified**.

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
Rank 0 also prints a flushed stage-start summary, loss/LR/step time/elapsed/ETA
at each configured optimizer-step logging interval (default every step),
checkpoint paths after save, and stage completion. Other ranks stay quiet.

## Local/static commands (no model download, GPU, or API)

From the project root, using the project Python environment:

```bash
python -m pytest
```

The backed-up preparation command above writes
`data/sft_main/{main_a_1k,main_b_2k,extra_1k,reserve_4k}.json`
and `manifest.json`; `media_status.json` is a separate readiness report. The
selection manifest is version-controlled while generated trajectories/media
are ignored; the former verifies pure planning/identity/audit logic.
Local preflight can be attempted, but it correctly exits nonzero until images
are present; raw declaration drift alone does not block it. It does not
download a model while images are missing.

## AutoDL acceptance order (do not skip the gates)

Run from the same project root, with dependencies from `pyproject.toml` and
the pinned raw JSON/Eval-300 parquet present. `torchrun` requires two visible
BF16 GPUs. The preparation command below explicitly permits official ZIP
downloads; omit `--download-images` if the seven archives are already local.

```bash
# 1. Materialize the fixed 8k pool's official images, then run full preflight.
python scripts/prepare_sft_main.py --exclusions configs/sft_data_exclusions.json \
  --extract-images --download-images
python scripts/preflight_sft_main.py

# If (and only if) complete leakage.json reports image overlap, deterministically
# replace those IDs in their own sources, materialize replacement images,
# and rerun full preflight. Do not continue until summary.json passed=true.
python scripts/prepare_sft_main.py --exclusions configs/sft_data_exclusions.json \
  --leakage-report reports/sft_preflight/leakage.json \
  --extract-images --download-images
python scripts/preflight_sft_main.py

# Verify the formerly ambiguous sample using the actual Qwen processor and
# collator, without loading 4B model weights or starting training.
python scripts/diagnose_sft_mask.py --sample-id livevqa:5212

# Check summary.json: passed=true, leakage_complete=true,
# question_overlap_count=image_overlap_count=0,
# effective_declaration_drift_count=actual_call_drift_count=0,
# zero_supervised_count=partial_assistant_span_cut_count=
# partial_tool_call_cut_count=0. The diagnostic must report
# mask_matches_structured_roles=true and tool_call_span_count as expected.

# 2. Materialize the independent 4B official 100-sample smoke set.
python scripts/prepare_sft_4b_smoke.py

# 3. Smoke A/B already passed on the verified FA2 AutoDL stack. Their reports
# document why formal training fixes micro=1/accum=4; do not rerun B as a
# candidate formal batch size. The historical smoke config is left unchanged.
python scripts/compare_sft_4b_smoke.py \
  --a reports/sft_4b_smoke/micro1_accum4/smoke_step20.json \
  --b reports/sft_4b_smoke/micro2_accum2/smoke_step20.json

# 4. Scan main_a_1k with the real untruncated multimodal processor, then run
# the selected longest sample through the unchanged training collator/model.
# This is single-GPU, no DDP/optimizer/scheduler/checkpoint, not formal training.
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_sft_vram.py \
  --config configs/sft_main.yaml --data data/sft_main/main_a_1k.json \
  --select-longest
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_sft_vram.py \
  --config configs/sft_main.yaml --data data/sft_main/main_a_1k.json \
  --select-longest --with-backward

# 5. Final worst-case two-GPU DDP/AdamW stress check. Each rank deliberately
# receives the SAME longest sample; this is NOT the formal sampler or an
# effective-global-batch-8 optimizer step. No scheduler/checkpoint is created.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/diagnose_sft_ddp_stress.py \
  --config configs/sft_main.yaml \
  --data data/sft_main/main_a_1k.json \
  --select-longest

# 6. Fresh-process Base+smoke-adapter generation on an official SFT smoke
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

```

The longest-sample scan measures **untruncated** processor IDs; for the
current `main_a_1k` preflight the maximum is approximately 30,977 tokens,
but the diagnostic discovers the index and length again rather than hard-coding
either. The selected record is then passed to `OpenSearchVLCollator` with the
formal 32,000 limit. The two commands each rescan the shard. The second runs
one BF16 forward and direct backward for a single micro-batch; it does **not**
measure a full DDP/optimizer step. Check finite loss, requested/effective FA2,
36/36 decoder training/checkpointing flags, and both peak allocated **and**
reserved VRAM. On an 80 GiB card, a backward OOM or peak reserved VRAM close
to capacity is a stop signal; retain substantial headroom (roughly 8–10 GiB
or more) for DDP, optimizer state, and run-to-run variation. Passing this
single-card stress test is necessary evidence, not a guarantee of formal
two-GPU completion. The actual single-card longest sample (`livevqa:7073`,
index 177, 30,977 tokens) passed forward/backward but reached 75.545 GiB
allocated and 76.494 GiB reserved on a 79.250 GiB GPU. That leaves too little
evidence about DDP wrapping and AdamW's first-step state allocation; the
two-rank command above is the final pre-run check. It scans with the real
processor on each rank, uses the same sample on both ranks as a worst-case
VRAM stressor, and performs only **one** micro-batch forward, backward,
`optimizer.step()`, and `zero_grad()`. It does not accumulate four micro-batches,
write a checkpoint, or launch `main_a_1k`. Inspect every rank's stage lines
and rank 0's `[SUMMARY]` (`passed=true`, finite losses, both optimizer steps,
per-rank/overall peak allocated and reserved). An `[OOM]` line identifies its
rank and stage (`ddp_wrap`, `forward`, `backward`, or `optimizer_step`) before
torchrun fails. Even a pass at near-capacity peak reserved VRAM requires
manual headroom review; this diagnostic is not formal training acceptance.
Do not launch `main_a_1k` until preflight and both stress tests are reviewed.

The next commands are **conditional**, not permission to ignore a failed
preflight. First inspect `reports/sft_preflight/*.json`, verify effective
declaration/call drift and both overlap counts are zero, and review any 32k
partial-cut/zero-target findings and report-only complete drops.
Use the fixed A batch configuration throughout the
whole lineage, and inspect finite losses, LR, LoRA delta, frozen vision,
checkpoint completeness, throughput, VRAM and malformed-data/long-sequence
errors at checkpoint-1k before continuing.

```bash
# Example only after preflight PASS and human approval; shows A's fixed batch.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --stage main_a_1k --micro-batch 1 --grad-accum 4

# After manual checkpoint-1k health check, same Phase 1 scheduler/state.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --stage main_b_2k --micro-batch 1 --grad-accum 4 \
  --resume-from outputs/sft_main/checkpoint-1k

# Optional, retaining checkpoint-3k independently.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --stage extra_1k --micro-batch 1 --grad-accum 4 \
  --resume-from outputs/sft_main/checkpoint-3k

# Reserve only, after a separately chosen Phase 2 peak LR and 4k health check:
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_main.py --stage reserve_4k --micro-batch 1 --grad-accum 4 \
  --resume-from outputs/sft_main/checkpoint-4k \
  --phase-2-peak-lr "$PHASE2_PEAK_LR"
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
