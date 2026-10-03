# v4.5.5 — BF16 SDPA Backend Diagnostic

Status: **READY FOR AUTODL BF16 SDPA BACKEND DIAGNOSTIC**.
CPU fixtures verify implementation, not real model/GPU results. This is a
forward-only forensic experiment, not Gate C, a repair, PPO, or formal RL.

## One variable, four serial variants

| Variant | Weights / adapter | Load and forward |
| --- | --- | --- |
| A0 | Original pinned base + formal checkpoint-3k dynamic PEFT | Original historical BF16 FA2 loader/forward |
| B0 | Attempt1 exact production merged checkpoint | Original historical BF16 FA2 loader/forward |
| A3 | Same base + same adapter, still dynamic PEFT | Fresh local BF16 SDPA load, BF16 CUDA autocast |
| B3 | **Exact same B0 checkpoint**, not a new merge | Fresh local BF16 SDPA reload, BF16 CUDA autocast |

Base: `Qwen/Qwen3-VL-4B-Instruct`, revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`.
The B0/B3 path comes from validated historical `group.identity.collection_attempt`;
it is never regenerated. A3 preserves default PEFT inference LoRA promotion
(often FP32 LoRA A/B with BF16 base); it never manually casts LoRA.
All non-LoRA model parameters must be BF16. The A0/A3 full LoRA dtype/target
audits must match. Resolved root/nested attention backends must match each variant.

`handoff.forward_rows` remains the authoritative forward: original prompt IDs,
response IDs, formal response/fatal mask, saved multimodal `.pt`, original
`get_rope_index` M-RoPE, response slice `[-R-1:-1]`, temperature **0.7**, and
pinned verl 0.6.1 `logprobs_from_logits(..., inplace_backward=False)`.
No retokenization or image processor is used. Real attempt1 has **1294** selected
trainable tokens; the count is validated against historical artifacts, **not
hardcoded** (CPU fixtures use four).

No TF32 setting, autocast policy or SDPA kernel selector is changed. PyTorch may
dispatch SDPA to an enabled CUDA kernel: `sdpa` is the HF backend contract, not a
claim that PyTorch's math-only implementation ran. Enabled SDPA options and TF32
flags are recorded for interpretation; no specific SDPA kernel is forced.

Audit records actual parameter/base/LoRA dtypes, saved vision dtype, model-root
input IDs/attention/position/pixel/grid dtypes, actual vision-tower input dtype,
actual forward autocast state/dtype, and actual logits dtype. BF16 CUDA autocast
and BF16 logits are checked without modifying them. Saved `pixel_values` may
be FP32; its actual dtype is preserved by the historical device-only move.

## Historical binding and protection

Before loading any model, validate the original failed Gate C report, historical
run/group/merge fingerprints, pinned base/adapter/configs, software versions,
and original alignment. Historical integration source hashes remain provenance;
they are not rebound to current diagnostic code.

Read and validate summaries **and full token JSONL** from:

- `policy_handoff_diagnostic/` (v4.5.2)
- `merge_precision_diagnostic/` (v4.5.3)
- `fp32_merged_forward_diagnostic_fa2_primary/` (v4.5.4 primary archive)
- `fp32_merged_forward_diagnostic/` (v4.5.4 secondary run)

Require consistent run/attempt/base/revision/adapter/B0/software/temperature,
token counts and complete token IDs/order. Recompute persisted historical
pair metrics from tokens. Secondary SDPA FP32 per-token values were not saved
by v4.5.4: validate its completed SDPA variant/aggregate provenance and count,
but do **not** claim an independent reconstruction of those secondary numbers.

Hash every file before and after in original Gate attempt outputs, base snapshot,
formal adapter, parent metadata, configs, all four old diagnostic directories,
and the original `gate_c_report.json`. Added/deleted/changed source files cause
failure. Refuse an existing output directory. Only new statistics are written to:

```text
reports/rl_gate_c/gate-c-v451-attempt1/bf16_sdpa_backend_diagnostic/
  summary.json
  source_checksums.json
  clip_sets.json
  token_diagnostics.jsonl
```

Successful `summary.json` is published last. Errors publish
`execution_succeeded=false`, an error/stage and available partial evidence,
then raise/nonzero. All outputs declare `diagnostic_only=true` and
`formal_rl_initialization_allowed=false`. No Gate manifest is produced.

A0 → destroy → B0 → destroy → A3 → destroy → B3 → destroy uses weak references,
GC and CUDA cache release to refuse overlapping model lifetimes. Per-variant
peak allocated/reserved bytes are recorded. No checkpoints, model weights,
FP32 merge, weight comparison, collection, vLLM, API, Judge, search, optimizer,
or update is invoked. No API key is required.

## Measurements and repeat sanity

Core pairs: `A0_vs_B0`, `A3_vs_B3`, `A0_vs_A3`, `B0_vs_B3`;
optional same-token cross pairs: `A0_vs_B3`, `A3_vs_B0`.
All use historical `pair_metrics`: count, absolute mean/max, signed mean,
importance-ratio mean/min/max, strict clip fraction (<0.8 or >1.28), six
absolute quantiles, and absolute-difference counts >0.01/0.05/0.1/0.2.

Each token retains original global/rollout/step/response-position/ID/text/special
metadata, all four logprobs, signed differences, ratios, clip flags and signs.
Pearson correlation is between `(A0-A3)` and `(B0-B3)`; zero variance yields
JSON `null`. Opposite-sign fraction uses **all aligned trainable tokens** as
denominator; a zero shift does not count as opposite. Clip sets contain global
token indices; all pairwise and three-/four-way intersection counts are saved.

Three improvements compare A3-B3 against this run's repeated A0-B0:
`(baseline - measured) / baseline` for absolute mean, maximum and clip fraction.
Zero baseline yields `null`; worsening yields a negative value, not clamping.
Original A0-B0 top50 is ranked from **v4.5.2** absolute diff, stable global-index
ties, and enriched with current A3-B3 diff/reduction. Also report current top50
for A3-B3, A0-A3, B0-B3 (and repeated A0-B0). `--top-n` controls count.

`repeat_delta_vs_history` covers v4.5.2, v4.5.3, and both v4.5.4 archives:
pair-metric deltas and per-variant repeated logprob metrics. Diagnostic-only
repeat sanity limits are mean ≤ `max(1e-4, 0.1 * historical A/B mean_abs)` and
max ≤ `max(1e-3, 0.1 * historical A/B max_abs)` for **each A0/B0**. The limits
are recorded. A material repeat deviation fails interpretation and the
invocation; these are not modified Gate thresholds or a new RL PASS criterion.

## Existing evidence and manual interpretation

The following are **user-reported AutoDL measurements**, not new local results:

- Original FSDP vs rollout: 1294 tokens, mean 0.0249463389, max 0.52434909,
  clip 0.0139103555.
- v4.5.2 A0-B0: mean 0.0248931106, max 0.440624237, clip 0.013137558;
  A0-C mean 0.0260643721, B0-C mean 0.0237095909;
  corr(A0-B0, B0-C) −0.50143304, opposite fraction 0.42194745.
- v4.5.3 A0-B1: mean 0.0229848835, max 0.443992615, clip 0.0123647604;
  only ~7.7% mean improvement. B0-B1 mean 0.0207854069.
- v4.5.4 primary FP32 + FA2 was unsupported for both variants:
  `FlashAttention only support fp16 and bf16 data type`.
- v4.5.4 secondary FP32 SDPA: 1294 tokens, mean abs
  7.999367102220585e-06, max abs 0.0002727508544921875,
  signed mean −2.1667716307839066e-07, ratio mean 0.9999997835324825,
  ratio min/max 0.9997272863386406/1.0001411537600828, clip zero and all
  four >threshold counts zero. It used FP32/no autocast/TF32 off and is
  **not** a dtype-matched baseline for this BF16 experiment.

After repeat sanity passes, interpret manually:

- If A3-B3 collapses toward 1e-5–1e-4 while A0-B0 remains ~0.025:
  “BF16 dynamic-vs-merged drift collapses under SDPA; evidence strongly
  implicates FA2/backend-specific numerical path.” This is **not** proof of
  an FA2 root cause; deeper backend/kernel interactions remain possible.
- If A3-B3 remains ~0.02: “BF16 dynamic-vs-merged mismatch persists under
  SDPA.” Investigate arithmetic ordering/dynamic vs static GEMM; do not
  automatically blame PEFT or BF16.
- At ~0.003–0.01, consider mixed backend/arithmetic contributions.
- Large A0-A3 but small B0-B3 suggests stronger dynamic-branch sensitivity;
  the inverse suggests merged-branch sensitivity.
- Large shifts in both branches require the signed correlation/opposite-sign
  and clip-overlap evidence to assess cancellation/different directions.

The script makes no automatic root-cause classification or Gate decision.

## CPU verification

```bash
PYTHONPATH=src python -m pytest tests/test_rl_bf16_sdpa_diagnostic.py tests/test_rl_merge_precision_diagnostic.py tests/test_rl_policy_handoff_diagnostic.py tests/test_rl_policy_alignment.py tests/test_rl_training_batch.py tests/test_rl_gate_c.py -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

## AutoDL (one BF16 CUDA GPU, serial/offline)

Run from repository root with the **same historical** torch/transformers/PEFT/verl
versions as attempt1. Keep all four prior report directories including the
primary archive. If output already exists, preserve/archive it outside the
protected/new target paths before intentionally making another invocation;
there is no overwrite/cleanup flag.

```bash
set -euo pipefail
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN_BASE_SNAPSHOT=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17

CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_rl_bf16_sdpa_backend.py \
  --run-id gate-c-v451-attempt1 \
  --base-model-path "$QWEN_BASE_SNAPSHOT" \
  --top-n 50 \
  --local-files-only \
  2>&1 | tee logs_gate_c_v451_attempt1_bf16_sdpa_backend_diag.txt
```

Expected terminal: `TOKENS: 1294`, four core pairs' mean/max/clip, three
backend improvements, backend-shift correlation/opposite fraction and report
path. Return the terminal summary and `summary.json`, especially A3-B3,
A0-A3, B0-B3 metrics, correlation, opposite fraction, and original A0-B0 top20
under A3-B3. No need to paste the entire token JSONL.

To extract those 20 entries:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/rl_gate_c/gate-c-v451-attempt1/bf16_sdpa_backend_diagnostic/summary.json')
s = json.loads(p.read_text())
for r in s['original_A0_vs_B0_outliers'][:20]:
    print(json.dumps({k: r[k] for k in (
        'global_trainable_index', 'rollout_index', 'step_index',
        'response_token_position', 'token_id', 'decoded_token',
        'prior_A0_vs_B0_diff', 'A3_vs_B3_diff',
        'original_outlier_abs_diff_reduction_fraction')}, ensure_ascii=False))
PY
```
