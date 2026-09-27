# R3: derived-image-ID argument repair

R3 is an isolated ablation from the **corrected SFT checkpoint-3k adapter**.
It never resumes R1 or R2. Its 600 samples come only from the pinned corrected
SFT 8k membership (the formal pool), with Dev50, Eval300 and frozen exclusions
kept outside the repair dataset. Neither Dev50 nor evaluation trajectories are
training input.

R3 reuses R1's `argument_only` mask and its contextual-token minimum-cover /
safe-spill checks. Every eligible `image_search` argument occurrence in a
selected sample is supervised; tool names, complete tool-call wrappers,
reasoning and final answers are not. Source expert conversations are copied
unchanged. Derived image IDs are prioritized in this order: all legal `img_3+`
candidates, then legal `img_2`, then `img_1` until 600. No target-count quota
can override grounding or image-contract rejection. The manifest separates
sample category, per-sample ID coverage and argument-occurrence counts.

The builder requires the corrected pool manifest/shards, the pinned raw source
JSON files and image ZIPs, frozen Eval300 parquet/media, and the existing Dev50
IDs/manifest. It refuses an existing R3 output directory and never downloads
archives. The CPU mask audit must pass before GPU training; its R3 report is
`reports/sft_repair/r3_mask_audit.json`. It checks all 600 samples with the
pinned real Qwen processor, decoded supervised spans, source membership,
grounding, safe BPE expansion, materialized image hashes, and Eval300 overlap.
It also rechecks Dev50 ID, question and decoded-image overlap from pinned source.

From the project root on AutoDL:

```bash
sha256sum data/sft_main/manifest.json
# Must equal 0faf210483978435e808e4ba8ce4fb2556fb27ecfe30267e948f5cc2f1c9c637.
pytest -q tests/test_sft_repair.py
python scripts/prepare_sft_repair.py --mode r3
python scripts/audit_sft_repair_masks.py --config configs/sft_repair_r3.yaml
# Only after both CPU commands pass:
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train_sft_repair.py --config configs/sft_repair_r3.yaml
```

The recipe is 2 BF16 GPUs, micro-batch 1, accumulation 4 (global batch 8),
constant LR 1e-5, 100 optimizer steps, and checkpoints at 50, 75 and 100.
The 600-sample set gives 75 optimizer steps per epoch and about 1.33 epochs
for 100 steps. The output is isolated under
`outputs/sft_repair/r3_argument_only_derived/` and is not a formal SFT stage.
