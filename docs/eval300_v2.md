# Audited Eval-300 v2

The audited 300-sample JSON changes only the `question` and reference-answer
text. `scripts/build_eval300_v2.py` joins on `sample_id`, copies every other
Arrow column from frozen v1, and never opens `images[].path` or decodes images.
It refuses to overwrite either input or an existing v2 output.

```bash
python scripts/build_eval300_v2.py \
  --audit-json E:/OpenSearch_mine/eval_300/eval300_blind_audit_rebuilt_v1/samples.json
python scripts/preflight_eval300.py \
  --config configs/eval_base_300_v2.yaml
```

Frozen v2 SHA256: `b42bcf96c93437adc607c2e5427e1bf7da24912eca373e5c31d58b4acea8a961`.
The build has 300 unique IDs, 100 per benchmark, 49 changed questions and 66
changed references. The v1/v2 ID, benchmark, `image_packed`, and all other
non-text columns are identical. V1 remains unchanged and retains its original
SHA256. The parquet files are ignored by Git, so copy the **exact** v2 artifact
to the evaluation host and verify its SHA256 there; reserializing with another
PyArrow version can change the file hash even when cell values are identical.

The formal Base and SFT-3k v2 Agent runs use separate run IDs. Use the same
command with the same run ID to resume after an interruption; optional
`--max-samples 200` runs the existing balanced first batch before resuming.

```bash
# Base, audited Eval-300 v2
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id base-eval300-audited-v2 \
  --config configs/eval_base_300_v2.yaml \
  --eval300

# SFT-3k, audited Eval-300 v2: only after a v3-compatible 3k checkpoint exists.
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id sft-3k-eval300-audited-v2 \
  --config configs/eval_base_300_v2.yaml \
  --adapter outputs/sft_main_imageid_v3/checkpoint-3k/adapter \
  --eval300
```

The historical `outputs/sft_main/checkpoint-3k/adapter` has the v2
`image_search.url` protocol and is rejected by the current v3 runtime; it is
**not** a substitute for the adapter path above. No SFT training is launched
by these commands. After each complete Agent run, Judge can use the audited
references without changing its default v1 behavior:

```bash
python scripts/run_judge.py \
  --run-id base-eval300-audited-v2 \
  --dataset data/eval/combined_eval_300_v2.parquet
```

Use `--run-id sft-3k-eval300-audited-v2` for the SFT Judge. The Judge loader
checks that each Agent trajectory question equals the chosen parquet question,
so a v1 Agent run cannot silently be scored with v2 references.
