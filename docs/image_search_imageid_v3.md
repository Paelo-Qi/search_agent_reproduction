# image_search runtime protocol v3

The current model-facing `image_search` argument is `{"image_id":"img_n"}`.
`img_n` must already be registered in the current Agent episode. HTTP URLs,
filenames, filesystem paths, unknown IDs, and the former `url` argument are
invalid. Other visual tools still use their existing `image` argument.
Inference never repairs a legacy call. Invalid references are rejected before
any provider call; the trajectory retains the attempted arguments and error.

Pinned Search-VL-SFT JSON remains immutable. During formal 8k generation only,
the adapter parses each source assistant tool call and changes the direct
`image_search.arguments.url` key to `image_id` when its value is a valid
`img_n`. It does not rewrite observations, reasoning, ordinary prose, web
queries, or other tools. The original raw-record SHA256 and `_source_tools`
remain in each shard. A malformed, ambiguous, or non-`img_n` source target
fails closed. Thus the model-facing chain is:

```text
pinned raw source (image_search.url)
  -> controlled assistant-call canonicalization in SFT generation
  -> v3 shard target (image_search.image_id)
  -> unchanged multimodal processor and assistant-only loss mask
```

The separate v3 paths below preserve the historical v2 pool, checkpoints,
Dev50 manifest, and reports. `configs/sft_main_imageid_v3.yaml` changes only
artifact paths relative to `configs/sft_main.yaml`; seed, selection, data
sizes, max length, model, LoRA, batch, scheduler, and attention remain fixed.
Do **not** start training until the full preflight passes. All commands here
use local pinned files except the explicit optional official image ZIP download.

```bash
# Rebuild the fixed 8k; omit --download-images if all seven official ZIPs exist.
python scripts/prepare_sft_main.py \
  --raw-dir data/raw \
  --eval data/eval/combined_eval_300.parquet \
  --exclusions configs/sft_data_exclusions.json \
  --output-dir data/sft_main_imageid_v3 \
  --extract-images --download-images

# Rebuild Dev50 from the same pinned source, requiring exact ordered old IDs.
# This writes a NEW directory and refuses to overwrite a nonempty one.
python scripts/prepare_tool_protocol_dev.py \
  --raw-dir data/raw \
  --pool-manifest data/sft_main_imageid_v3/manifest.json \
  --eval data/eval/combined_eval_300.parquet \
  --expected-ids data/eval/tool_protocol_dev50/ids.json \
  --output-dir data/eval/tool_protocol_dev50_imageid_v3

# CPU-only static audit, including legacy/effective call counts, grounding,
# 8k selection/quotas/shards, and exact ordered Dev50 IDs/overlap metadata.
# --previous-manifest must point to the corrected-v2 8k, not a pre-exclusion pool.
python scripts/audit_image_search_v3.py \
  --pool-manifest data/sft_main_imageid_v3/manifest.json \
  --previous-manifest data/sft_main/manifest.json \
  --dev-dir data/eval/tool_protocol_dev50_imageid_v3 \
  --previous-dev-ids data/eval/tool_protocol_dev50/ids.json

# If historical Dev50 IDs/media are unavailable locally, audit only v3 SFT.
# This reports sft_passed, but deliberately keeps overall passed=false.
python scripts/audit_image_search_v3.py \
  --pool-manifest data/sft_main_imageid_v3/manifest.json --sft-only

# Full pinned Qwen processor/media/mask/leakage/tool preflight (CPU, no model).
python scripts/preflight_sft_main.py \
  --data-dir data/sft_main_imageid_v3 \
  --eval data/eval/combined_eval_300.parquet \
  --config configs/sft_main_imageid_v3.yaml \
  --report-dir reports/sft_preflight_imageid_v3
```

The read-only audit must print `passed=true`, `image_search_legacy_url=0`,
`image_search_image_id>0`, `image_search_http_target=0`,
`image_search_non_img_n=0`, and `grounding_passed=true`. The full preflight
must report `summary.json.passed=true`; it additionally verifies decoded-image
Eval300 overlap, actual processor sequence/masking, and tool declaration/call
schema. The static audit does not replace this full preflight. Dev50 category
selection still parses legacy **source** calls, not v3 shard calls. It refuses
to write if even the ordered 50 IDs differ from the historical list. Eval300
membership and scoring protocol are unchanged.

After local static/preflight gates and with a deliberate API budget, the Base
Dev50 v3 comparison can be run separately from prior protocol results:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_tool_protocol_dev.py \
  --run-id base-imageid-v3-dev50 \
  --dev-dir data/eval/tool_protocol_dev50_imageid_v3 \
  --pool-manifest data/sft_main_imageid_v3/manifest.json \
  --config configs/eval_base_300.yaml
python scripts/eval_tool_protocol.py \
  --dev-dir data/eval/tool_protocol_dev50_imageid_v3 \
  --trajectories reports/tool_protocol_dev_runs/base-imageid-v3-dev50/trajectories.jsonl \
  --report-dir reports/tool_protocol_imageid_v3/base
```

Metrics read `image_search.image_id` and retain the HTTP hallucination count
for bad values such as `{"image_id":"https://..."}` under
`image_search_http_image_id_hallucination_count`. They also count legacy
`url` arguments. No model, GPU, external API, or training is invoked by the
prepare/static/preflight commands (apart from optional official ZIP download).

The v2 SFT manifest, v2 checkpoint-1k/3k (including optimizer state), v2
Dev50 manifest, and old Dev50/Eval300 run manifests are historical artifacts,
not inputs or resume parents for v3. New formal training must start from the
pinned Base using the v3 pool/config after all gates pass. Runtime tool
fingerprints prevent old run manifests from resuming with the new schema, and
adapter identity rejects v2 metadata. Legacy targeted-repair R1/R2/R3 data,
targets, and checkpoints remain historical v2 experiments; they are not
silently converted into v3 training material. The legacy repair training
entrypoint fails closed under v3; no repair experiment is run or redesigned
by this upgrade.
