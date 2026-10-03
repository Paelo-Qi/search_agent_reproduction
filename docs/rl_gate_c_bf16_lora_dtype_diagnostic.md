# v4.5.6 — BF16 Dynamic LoRA Dtype Diagnostic

Status: **READY FOR AUTODL BF16 LORA DTYPE DIAGNOSTIC**.
CPU fixtures do not establish real CUDA/model results. This is a read-only,
forward-only forensic diagnostic, not Gate C, PPO, rollout repair or formal RL.

## Controlled variants

| Variant | Runtime parameters | Backend / forward |
| --- | --- | --- |
| A3 | Pinned BF16 base + formal checkpoint-3k dynamic PEFT; actual default inference LoRA dtype | Same v4.5.5 BF16 SDPA loader / BF16 CUDA autocast |
| A4 | Fresh identical A3 load, then only real LoRA A/B weights cast to BF16 | Same SDPA backend / same BF16 CUDA autocast |
| B3 | Fresh reload of attempt1 **exact B0 production merged weights**, BF16 | Same v4.5.5 SDPA loader / BF16 CUDA autocast |

Base: `Qwen/Qwen3-VL-4B-Instruct`, revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`.
All loads directly reuse `bf16_sdpa_diagnostic.load_bf16_sdpa` (A4 loads as A3).
No A0/B0 re-forward, B1/B2 generation, FP32 model, merge or weight checkpoint
is needed. No production loader/config/threshold is changed.

Do **not** assume A3 LoRA is FP32: inspect real tensors and record their
distribution. Mixed source dtypes are allowed. If all A3 LoRA A/B are already
BF16, fail nonzero with `experiment_informative=false` and
`experiment_not_informative=true`; do not manufacture a dtype contrast.

## Runtime-only cast safety

`cast_dynamic_lora_to_bf16` requires a real, frozen, enabled, unmerged dynamic
PEFT model in eval mode, BF16 non-LoRA parameters, and SDPA root/nested attention.
Reuse the strict vanilla LoRA roster: **36 × 7 = 252 modules / 504 tensors**,
q/k/v/o/gate/up/down projections, exactly the default active adapter. Missing,
extra, aliased, vision/projector LoRA, nonfinite or unsupported targets fail.

Before A4 cast, compare freshly loaded A3/A4 exact non-LoRA parameter hashes,
LoRA tensor hashes, names, shapes, source dtypes and execution semantics.
Only assign `lora_A['default'].weight` and `lora_B['default'].weight` storage
to their exact BF16 casts, preserving Parameter identity, device and frozen state.
Never call `model.bfloat16()` or cast the whole model.

Audit records module/tensor counts, source/target dtype distributions, per-module
rank/alpha/scaling/dropout/eval/active-adapter state, and each tensor's name,
shape, original/target dtype, source value SHA256, expected BF16 cast SHA256 and
actual cast SHA256. Non-LoRA hashes before/after must match, including all
vision/projector parameters. LoRA semantic state must match. No tensor contents
are exported. Hashing uses bounded CPU chunks, not a full extra model clone.

BF16 rounding is the intended **runtime dtype** intervention: original FP32
values may round when converted. The cast hash must equal the exact conversion
of the original value, not an arbitrary rewrite. Original source adapter/base
files are unchanged. Before and after A4 forward all LoRA tensors must remain
BF16; parameter values/base hashes/LoRA state are rechecked after forward.

## Inputs, historical binding and repeat sanity

Reuse the existing BF16 forward auditor and `handoff.forward_rows`: original
prompt/response IDs, formal response/fatal mask, saved multimodal `.pt`, M-RoPE
`get_rope_index`, response slice `[-R-1:-1]`, **temperature 0.7**, and pinned
verl 0.6.1 logprob helper. No retokenization or image processor is called.
The real attempt has **1294 trainable tokens**; validate the recorded count,
not a hardcoded constant (CPU fixtures use four).

Record base/LoRA parameter dtypes, per-module LoRA dtypes, root/nested attention,
actual CUDA autocast state/dtype, logits dtype, saved/root pixel dtype,
vision-tower input dtype and position/attention/input-ID dtypes. Preserve saved
pixel dtype on device move. Enforce the same BF16 CUDA autocast/logits contract
as v4.5.5. Do not modify TF32 flags or SDPA kernel selection; record and compare
the existing global numerical options.

Read and checksum-protect all five prior directories:

- `policy_handoff_diagnostic/` (v4.5.2)
- `merge_precision_diagnostic/` (v4.5.3)
- `fp32_merged_forward_diagnostic_fa2_primary/` (v4.5.4 primary archive)
- `fp32_merged_forward_diagnostic/` (v4.5.4 secondary)
- `bf16_sdpa_backend_diagnostic/` (v4.5.5 direct baseline)

Reuse the previous four-history validator; require v4.5.5 run/attempt/base/
revision/adapter/exact-B0/software/temperature/count/diagnostic provenance.
Validate **all** token IDs/order/text metadata and finite A0/B0/A3/B3 logprobs,
then recompute v4.5.5 pair metrics from its JSONL. Check SDPA/BF16/actual default
LoRA dtype evidence. The historical secondary FP32 aggregate remains aggregate
provenance only; it did not persist secondary FP32 per-token logprobs.

`repeat_delta_vs_v455` contains pair-metric deltas and **individual A3 and B3**
repeat metrics (a common shift cannot hide behind an unchanged A3-B3 gap).
Reuse the v4.5.5 diagnostic-only limits: per-branch mean repeat abs ≤
`max(1e-4, 0.1 * historical A3-B3 mean_abs)` and max repeat abs ≤
`max(1e-3, 0.1 * historical A3-B3 max_abs)`. Material drift fails interpretation
and the invocation. These are **not** changed Gate thresholds.

Before/after SHA256 covers complete original Gate outputs, pinned base snapshot,
formal adapter and parent metadata, configs, all five historical diagnostics,
and original `gate_c_report.json`. Any addition/deletion/change fails closed.
Historical integration source hashes are not rebound to current code.

## Measurements

Required pairs: `A3_vs_B3`, `A4_vs_B3`, `A3_vs_A4`.
Reuse complete historical `pair_metrics`: token count, mean/max abs and signed
mean, ratio mean/min/max, strict clip fraction (<0.8 or >1.28), six absolute
quantiles, and >0.01/0.05/0.1/0.2 counts.

Mean/max/clip improvements use `(A3B3_baseline - A4B3_measured) / baseline`.
Zero baseline → `null`; worsening → negative. Pearson correlation is between
per-token **A3-A4** and **A3-B3**; zero variance → JSON `null`.
Toward/away/equal fractions compare `abs(A4-B3)` with `abs(A3-B3)` using strict
<, >, equality respectively, denominator **all aligned trainable tokens**.

Clip sets are global token indices for all three pairs. Save intersections,
`A3B3 only` / `A4B3 only` relative to each other. Original top50 uses historical
v4.5.5 A3-B3 absolute diff, stable global-index ties. Enrich with repeated
A3-B3 diff, A4-B3 diff, absolute reduction, reduction fraction and original
clip→nonclip transition; also save current pair top-N. `--top-n` defaults to 50.

## Evidence and manual interpretation

**User-reported AutoDL evidence, not newly measured locally:**

- Original FSDP-vs-rollout: 1294 tokens, mean abs 0.0249463389,
  max abs 0.5243490934, clip 0.0139103555.
- v4.5.2 A0-B0 mean abs 0.0248931106; B0-C 0.0237095909;
  A0-C 0.0260643721. v4.5.3 FP32 merge arithmetic→BF16 only reduced mean ~7.7%.
- v4.5.4 true FP32+FA2 unsupported for both variants; FP32+SDPA secondary
  dynamic-vs-merged mean abs 7.999367102220585e-06,
  max abs 0.0002727508544921875, clip 0.
- v4.5.5 BF16 SDPA A3-B3 mean abs 0.0213978928, max abs 0.495536804,
  clip 0.0100463679. Backend mean reduction 0.1404090430, max reduction
  −0.1246244816, clip reduction 0.2352941176. Backend-shift correlation
  0.0246514387, opposite-sign fraction 0.2851622875.

After repeat sanity passes, interpret manually:

- A4-B3 near 1e-5–1e-4 strongly implicates dynamic LoRA runtime dtype semantics
  as a dominant contributor, **not** a confirmed root cause. BF16 operation
  ordering/kernel interactions remain possible.
- A4-B3 falling to ~0.002–0.008 suggests a material but partial dtype contribution.
- A4-B3 remaining ~0.02 suggests mismatch persists after matching A/B runtime
  dtype; investigate dynamic `Wx + ΔWx` vs static `(W + ΔW)x` BF16 ordering.
- A large A3-A4 shift without better A4-B3 alignment changes the dynamic policy
  but does not establish movement toward the merged policy.

The program reports measurements only, no automatic root-cause confirmation,
production repair, or Gate PASS.

## Outputs and lifecycle

Refuse existing output. Write only statistics under:

```text
reports/rl_gate_c/gate-c-v451-attempt1/bf16_lora_dtype_diagnostic/
  summary.json
  source_checksums.json
  lora_cast_audit.json
  clip_sets.json
  token_diagnostics.jsonl
```

Successful summary is published last. Failures publish false execution status,
stage/error/available audits/checksums and raise nonzero. All artifacts declare
diagnostic-only and disallow formal RL initialization. No Gate manifest exists.

One GPU, serial: A3 load/forward/destroy → fresh A4 load/cast/forward/destroy →
B3 load/forward/destroy. Weakrefs/GC/CUDA cache release refuse overlapping model
lifetimes; per-variant peak allocated/reserved bytes are recorded. Runtime
fingerprints add CPU transfer/hash overhead; they do not benchmark throughput.
No checkpoint, `.safetensors`, `.pt` model, cast adapter or model export is created.
No vLLM/API/Judge/search/collection/optimizer/update is called. No API key needed.

## Verification and AutoDL

```bash
PYTHONPATH=src python -m pytest tests/test_rl_bf16_lora_dtype_diagnostic.py tests/test_rl_bf16_sdpa_diagnostic.py tests/test_rl_merge_precision_diagnostic.py tests/test_rl_policy_handoff_diagnostic.py tests/test_rl_policy_alignment.py tests/test_rl_training_batch.py tests/test_rl_gate_c.py -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

Use repository root, one BF16 CUDA GPU, and the **same historical** software
versions and all five completed historical reports. No overwrite flag exists.

```bash
set -euo pipefail
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN_BASE_SNAPSHOT=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17

CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_rl_bf16_lora_dtype.py \
  --run-id gate-c-v451-attempt1 \
  --base-model-path "$QWEN_BASE_SNAPSHOT" \
  --top-n 50 \
  --local-files-only \
  2>&1 | tee logs_gate_c_v451_attempt1_bf16_lora_dtype_diag.txt
```

Return terminal summary and `summary.json`, especially A3-B3/A4-B3/A3-A4 metrics,
LoRA dtype/cast audit, improvements, correlation, toward/away/equal fractions,
and original A3-B3 top20 under A4-B3. No need to paste the entire token JSONL.

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/rl_gate_c/gate-c-v451-attempt1/bf16_lora_dtype_diagnostic/summary.json')
s = json.loads(p.read_text())
for t in s['original_A3_vs_B3_outliers'][:20]:
    print(json.dumps({k: t[k] for k in (
        'global_trainable_index', 'rollout_index', 'step_index',
        'response_token_position', 'token_id', 'decoded_token',
        'prior_v455_A3_vs_B3_diff', 'A3_vs_B3_diff', 'A4_vs_B3_diff',
        'original_outlier_abs_reduction', 'original_outlier_reduction_fraction',
        'original_clipped_to_nonclipped')}, ensure_ascii=False))
PY
```
