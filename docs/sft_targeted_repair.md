# Corrected-3k targeted SFT repair ablations

This is an isolated diagnostic experiment for the fake-HTTP `image_search.url`
regression. It is **not** formal SFT checkpoint-4k, and it must never be chosen
using Eval300, Dev30, or the historical 20 Eval300 debug samples. The only
checkpoint-selection set is the disjoint tool-protocol Dev50.

R1 (`argument_only`) supervises every valid `image_search` arguments fragment
in a selected trajectory, for example `"arguments": {"url": "img_2"}`.
Tool names, wrappers, reasoning, observations, and final answers are masked.
R2 (`full_tool_call`) supervises the first call of the selected category in
the designated assistant turn, including `<tool_call>` wrappers. It never simply
opens the whole assistant trajectory. Both modes reuse the formal processor,
message construction, canonicalized tool declarations and base collator;
target boundaries use complete-prompt tokenizer offsets matched to the actual
multimodal assistant-body token IDs and decoded tokens. An ambiguous or
truncated target aborts the audit
or training rather than silently widening loss.

R1 requests 180 `img_1`, 70 `img_2`, 50 `img_3` or later. R2 requests 220
image-search and 80 other image-tool records. R2 intentionally contains no
no-tool/direct-answer repair target because the original OpenSearch-VL SFT
objective is tool-first/search-first. The non-image-search portion preserves
tool-selection diversity among visual tools, not direct answering. These are soft
category goals: shortage is filled by other eligible tool categories and logged in
the manifest. No target text or image IDs are rewritten. Each mode contains
300 pinned-source records. The deterministic rank is SHA-256 over fixed seed,
mode, and sample ID. The builder rejects the 8k membership, 127 frozen bad
source IDs, Dev50 IDs, Eval300/Dev50 question overlaps, Eval300/Dev50 decoded
image overlaps, and source image-contract failures. Dev30 is a verified
subset of frozen Eval300 and is thus excluded too. Image ZIPs must exist
locally; the builder does not download them. It does not replace existing
output directories.

If an earlier R2 dataset exists, preserve it outside `data/sft_repair/` before
rebuilding only R2 with `python scripts/prepare_sft_repair.py --mode full_tool_call`.
This does not touch the R1 dataset or manifest. The old R2 manifest's no-tool
membership is rejected by the current validator; it must not be trained.

## Order of operations on AutoDL

These commands assume the corrected pool manifest SHA-256 is
`0faf210483978435e808e4ba8ce4fb2556fb27ecfe30267e948f5cc2f1c9c637`,
the Dev50 IDs/manifest have already been prepared, pinned source image ZIPs
are present under `data/raw/<source>/images.zip`, and the corrected 3k
checkpoint is complete. All commands run from the project root.

```bash
pytest
# For an existing R1 dataset, rebuild only R2 after preserving any old R2 directory.
python scripts/prepare_sft_repair.py --mode full_tool_call
python scripts/audit_sft_repair_masks.py --config configs/sft_repair_r1.yaml
python scripts/audit_sft_repair_masks.py --config configs/sft_repair_r2.yaml
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_sft_repair.py --config configs/sft_repair_r1.yaml
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train_sft_repair.py --config configs/sft_repair_r2.yaml
```

For a fresh installation with neither dataset present, run
`python scripts/prepare_sft_repair.py` instead to build both modes.

The default is 50 repair optimizer steps. `--max-steps 25` and
`--max-steps 100` are supported. Batch size is 2 GPUs × micro-batch 1 ×
gradient accumulation 4 = 8. Steps 25, 50, and (if reached) 100 are saved.
Each run starts with the **adapter weights** in
`outputs/sft_main/checkpoint-3k/adapter`; it intentionally starts a fresh
AdamW optimizer and constant 1e-5 learning-rate scheduler with zero warmup.
It does not resume the formal optimizer/scheduler horizon. The repair
checkpoint has `checkpoint_kind=targeted_repair_ablation`,
`formal_sft_stage=false`, a parent metadata SHA-256, dataset provenance,
mask version, step, and artifact SHA-256 values. It is not loadable as a
formal SFT stage by the formal checkpoint/resume contract.

Only after the CPU mask audit passes, train. Training does **not** automatically
run Dev50. Evaluate each saved adapter with a separate run ID, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_tool_protocol_dev.py --run-id r1-step25-tooldev50 --config configs/eval_base_300.yaml --adapter outputs/sft_repair/r1_argument_only/checkpoint-step25/adapter
python scripts/eval_tool_protocol.py --trajectories reports/tool_protocol_dev_runs/r1-step25-tooldev50/trajectories.jsonl --report-dir reports/sft_repair/r1-step25
CUDA_VISIBLE_DEVICES=0 python scripts/run_tool_protocol_dev.py --run-id r1-step50-tooldev50 --config configs/eval_base_300.yaml --adapter outputs/sft_repair/r1_argument_only/checkpoint-step50/adapter
python scripts/eval_tool_protocol.py --trajectories reports/tool_protocol_dev_runs/r1-step50-tooldev50/trajectories.jsonl --report-dir reports/sft_repair/r1-step50
CUDA_VISIBLE_DEVICES=0 python scripts/run_tool_protocol_dev.py --run-id r2-step25-tooldev50 --config configs/eval_base_300.yaml --adapter outputs/sft_repair/r2_full_tool_call/checkpoint-step25/adapter
python scripts/eval_tool_protocol.py --trajectories reports/tool_protocol_dev_runs/r2-step25-tooldev50/trajectories.jsonl --report-dir reports/sft_repair/r2-step25
CUDA_VISIBLE_DEVICES=0 python scripts/run_tool_protocol_dev.py --run-id r2-step50-tooldev50 --config configs/eval_base_300.yaml --adapter outputs/sft_repair/r2_full_tool_call/checkpoint-step50/adapter
python scripts/eval_tool_protocol.py --trajectories reports/tool_protocol_dev_runs/r2-step50-tooldev50/trajectories.jsonl --report-dir reports/sft_repair/r2-step50
```

Compare registered-image-ID rate, HTTP image-argument hallucination rate,
unknown-image-ID rate, provider-not-called due to bad ID, valid argument
rates for image-search/layout/crop, no-tool behavior, and duplicate tool-call
count. Never use frozen Eval300 or Dev30 for recipe selection.

R0 `standard_continue` is intentionally not implemented: it would need an
additional mode/dataset training path and is optional; R1/R2 are the required
paired ablations.
