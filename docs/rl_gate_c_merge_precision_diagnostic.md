# v4.5.4 — FP32 Merged Forward Diagnostic

Current status: **READY FOR AUTODL FP32 MERGED FORWARD DIAGNOSTIC**.
The new entry adds A2/B2 and writes `fp32_merged_forward_diagnostic/`.
The historical v4.5.3 entry and reports remain compatible/read-only.
See the v4.5.4 section below for the new command and interpretation limits.

## Historical v4.5.3 mode

Status: **READY FOR AUTODL MERGE PRECISION DIAGNOSTIC**.
CPU tests are not real Qwen/AutoDL evidence. This tool cannot repair or pass Gate C.
It measures only dynamic PEFT HF → static merged HF; it does not investigate
static HF → vLLM. It never collects rollouts, calls tools/APIs, recomputes rewards,
updates a policy, publishes a Gate manifest, or changes production merge/thresholds.

## Variants and identical forward

| Variant | Weights | Forward |
| --- | --- | --- |
| A | Original pinned BF16 base + formal checkpoint-3k PEFT | Existing v4.5.2 BF16 forward |
| B0 | Exact historical `merged-<collection_attempt>` bound by `group.identity.collection_attempt` | Same forward |
| B1 | CPU FP32 target base + FP32 LoRA delta, actual PEFT merge, explicit BF16 cast, save, fresh reload | Same forward |
| B2 (not implemented) | FP32 merged weights, truly FP32 forward | Would separate final weight representation from merge arithmetic |

B0 is read-only and is **never regenerated**. Reuse the attempt1 `run_manifest.json`,
`group/group.json`, original `multimodal-*.pt`, failed alignment artifact,
historical merged manifest and v4.5.2 forensic summary/token JSONL. Validate their
lineage and fingerprints before any model loading. Do not bind historical source
hashes to current Gate implementation hashes or silently replace old artifacts.

Token selection comes from the existing v4.5.2 `validate_attempt` /
`diagnostic_rows_from_group` path (which uses the formal training masks).
The prior JSONL must match every response token ID, position, rollout/step and
masked token count; its A/B metrics are recomputed and checked against its summary.
For real attempt1 this count is **1294**, not a newly selected token subset.
`forward_rows` reuses original tensor inputs, M-RoPE on the full prompt+response,
the previous-position logits slice, temperature **0.7**, and the pinned verl
`logprobs_from_logits` helper. No processor/template/token re-encoding occurs.
Token text is copied from the validated prior forensic artifact, not re-tokenized.

## B1 arithmetic, staging, dtype evidence

One variant resides on the GPU at a time: A forward → destroy → B0 forward →
destroy → CPU B1 merge/save → destroy CPU model → fresh reload → B1 forward →
destroy. Weak-reference checks reject overlapping live model instances.

B1 loads the original pinned BF16 base on CPU with the formal loader and loads
the formal adapter with PEFT. It verifies the complete **36 × 7** language target
roster, default vanilla Linear LoRA, no merged/DoRA/bias/variant representation.
Only target base linear weights/biases and LoRA A/B used by the merge are promoted
to FP32. The remainder of the model need not be promoted.

Before using actual `merge_and_unload(safe_merge=True)`, the diagnostic installs
temporary guards at each target's `get_delta_weight` call. The guard asserts CPU
FP32 base and FP32 computed delta at the **actual merge invocation**, then is
removed. Exactly one guarded merge call per target is required. The result's
target weights must remain FP32 until the explicit final BF16 cast.
This is not a handwritten replacement for PEFT's model merge. The dtype checks
address the actual addition/writeback behavior in the
[pinned PEFT Linear implementation](https://github.com/huggingface/peft/blob/v0.21.1/src/peft/tuners/lora/layer.py).
Production `merge_actor_adapter` is neither called nor modified.

The report records actual A base/LoRA dtypes, per-target A/B dtypes,
initial B1 base/LoRA dtypes, promoted target/delta dtypes, CPU arithmetic dtype,
mixed whole-model dtypes before cast, explicit cast dtype, in-memory save dtypes,
**serialized safetensors header dtypes**, fresh reload parameter dtypes, and
BF16 forward autocast. Reload alone could conceal a wrong saved dtype; the
header audit therefore independently rejects non-BF16 floating tensors.
Counts are provided globally and by each of the seven target suffixes.

Before loading the CPU merge model or creating staging files, check free disk:
twice the original safetensors weight bytes + adapter bytes + 1 GiB safety margin.
The diagnostic saves **no FP32 full checkpoint**. CPU RAM must still accommodate
BF16 base, FP32 target weights and the largest delta/safe-merge working buffers;
the script does not guarantee a particular CPU RAM minimum. Inspect AutoDL RAM
and disk before execution.

Staging is a new `.fp32-merge-*` direct child of the diagnostic directory.
After full save, asset copy, complete shard validation, serialized dtype audit,
CPU model destruction, fresh plain-model reload, BF16/attention validation and
file hashing, publish it as `tmp_fp32_merge`. Only then use it for forward.
Copied processor assets are immutable B0 files; no processor is invoked.
All model metadata declares `diagnostic_only=true`, `evidence_only=true`,
`formal_rl_initialization_allowed=false`, `not_for_rollout=true`,
`not_for_training=true`.

By default, only this invocation's validated direct-child temporary models are
deleted after weight/token analysis (also cleaned on failure). Their dtype/file
hash evidence remains in the report. `--keep-diagnostic-model` retains them for
inspection, including unpublished staging on failure; it never makes them formal.
Existing diagnostic output is refused, not overwritten. To repeat, explicitly
archive the **new** diagnostic directory; leave all original attempt artifacts
and the old `policy_handoff_diagnostic` directory untouched.

## Statistics and denominator definitions

For every target, compute on the complete weight tensor in bounded chunks:

```text
target_fp32 = float32(base_bf16) + delta_fp32
error = float32(bf16(target_fp32)) - target_fp32
```

Record delta/error absolute mean/max, mean(|error|)/mean(|delta|),
max(|error|)/max(|delta|), nonzero-delta/unchanged counts, layer/suffix and dtype.
Suffix summaries use element-weighted means (not the mean of module means).

`fraction_delta_rounded_to_zero_effectively` = count(bf16(W+delta)==W) / all
elements. This includes zero deltas and **does not mean delta itself became zero**.

`fraction_merged_weight_unchanged_despite_nonzero_delta` (alias
`unchanged_despite_delta_fraction`) =
count(delta!=0 AND bf16(W+delta)==W) / count(delta!=0).
If the denominator is zero, report JSON `null`. Suffix `unchanged_fraction`
uses this same nonzero-delta denominator.

Compare every B0/B1 target weight by streaming one module at a time from
safetensors: mean/max absolute weight difference and nonzero-difference fraction,
per layer/suffix and aggregated by suffix. Do **not** export full W/delta/error
or weight-difference tensors.

Token comparisons reuse v4.5.2 `pair_metrics` for A-B0, A-B1, B0-B1:
mean/max/signed logprob difference, mean/max importance ratio,
clip fraction outside [0.8, 1.28], p50/p90/p95/p99/p99.5/p100 absolute differences,
and counts above .01/.05/.10/.20. Improvements use current repeated A/B0 as the
baseline: `(production_metric - B1_metric) / production_metric`; zero baselines
produce JSON `null`, worsening produces negative values.

Original top-N outliers are ranked by **prior v4.5.2 A-B0** absolute difference
with global index as stable tie-breaker. Each retains prior/current A-B0,
current A-B1 and its original-outlier reduction fraction. Default N=50.
Repeat-forward delta metrics separately expose new A/prior A and new B0/prior B0
differences; do not attribute those differences automatically to merge precision.

## Read-only evidence and outputs

Before/after SHA256 maps must match for the whole historical output tree, pinned
base snapshot, formal adapter and metadata, `rl_main.yaml`,
`sft_main_imageid_v3.yaml`, old forensic directory and original `gate_c_report`.
Mutation/addition/deletion fails closed with nonzero exit and failed diagnostic
summary. Hash checks detect changes; they cannot undo an external concurrent writer.
Missing protected inputs fail at the checksum precheck before creating diagnostic
output; failures after directory creation leave a failed diagnostic summary.
No production output lock, marker, manifest or checkpoint is written.

New reports only:

```text
reports/rl_gate_c/gate-c-v451-attempt1/merge_precision_diagnostic/
  summary.json
  token_diagnostics.jsonl
  module_merge_stats.json
  weight_compare_stats.json
  source_checksums.json
```

Successful summary is published last. `execution_succeeded=true` means the
measurement completed, **not** that drift passed a Gate threshold. Failure records
the stage/error and rethrows; no automatic root-cause classification or Gate PASS.

## Offline single-GPU AutoDL command

Run in the project root, with the **same historical pinned environment** and all
attempt1 + v4.5.2 artifacts present. No API key, API calls, vLLM or torchrun.

```bash
set -euo pipefail
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN_BASE_SNAPSHOT=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17

CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_rl_merge_precision.py \
  --run-id gate-c-v451-attempt1 \
  --base-model-path "$QWEN_BASE_SNAPSHOT" \
  --top-n 50 \
  --local-files-only \
  2>&1 | tee logs_gate_c_v451_attempt1_merge_precision_diag.txt
```

Expected terminal: `TOKENS: 1294`, three pairwise mean/max/clip measurements,
`IMPROVEMENT` fractions and `REPORT` path. Numbers are not predetermined.
Please return: (1) terminal summary; (2) `summary.json`; (3)
`module_merge_stats.json`; (4) original A-B0 top20 outliers with B1 diff.
Do not paste the whole token JSONL. Extract (4) offline:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/rl_gate_c/gate-c-v451-attempt1/merge_precision_diagnostic/summary.json')
s = json.loads(p.read_text())
for r in s['original_production_outliers'][:20]:
    print(json.dumps({k: r[k] for k in (
        'global_trainable_index', 'rollout_index', 'step_index', 'response_token_position',
        'token_id', 'decoded_token', 'prior_dynamic_vs_production_diff',
        'dynamic_peft_vs_production_merge_diff', 'dynamic_peft_vs_fp32_merge_bf16_diff',
        'original_outlier_abs_diff_reduction_fraction')}, ensure_ascii=True))
PY
```

## Interpretation limits

Final BF16 representation may discard small deltas even after FP32 arithmetic.
B0/B1 can be identical; tests must not assume an improvement. Dynamic two-GEMM
LoRA and merged one-GEMM arithmetic remain different. CPU versus GPU delta
calculation is another variable, and kernel/environment repeatability can affect
logprobs. Without B2, do not claim arithmetic and final representation have been
fully separated. True B2 would require a supported genuinely FP32 attention/
forward path; BF16 autocast over FP32 weights is **not** B2. The historical v4.5.3
mode keeps the original FA2/BF16 forward and implements only B1; the new v4.5.4
entry below adds separate true FP32 variants without changing that old mode.
Even substantial A-B1 improvement is evidence for a subsequent decision, not
proof that BF16 merge is the confirmed root cause, nor permission to change Gate C.

## Local validation

```bash
PYTHONPATH=src python -m pytest tests/test_rl_merge_precision_diagnostic.py tests/test_rl_policy_handoff_diagnostic.py tests/test_rl_policy_alignment.py tests/test_rl_training_batch.py tests/test_rl_gate_c.py -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

New CPU tests use actual tiny PEFT merge and actual safetensors save/reload/header
inspection, known BF16 rounding, weighted statistics, lineage/tokens, serial
orchestration, cleanup and failure injection. They do not load real model weights,
use CUDA, run PPO or call external providers. No commit or push is performed.

## v4.5.4 scope and reported AutoDL evidence

The user's v4.5.3 evidence (not locally reproduced): A-B0 mean/max/clip
0.0248931106 / 0.440624237 / 0.013137558; A-B1
0.0229848835 / 0.443992615 / 0.0123647604. Mean improves ~7.7%, clip ~5.9%,
maximum worsens slightly. This does not confirm production BF16 arithmetic as
the main cause. The new mode separates additional variables where supported.
Production merge, Gate thresholds/alignment/rollout, reward/RLOO/fatal, training
batch and optimizer are unchanged. No collect, judge, search, vLLM or API calls.

| Variant | Definition | Primary backend/autocast |
| --- | --- | --- |
| A0 | Original pinned BF16 base + formal dynamic PEFT | FA2 / original BF16 autocast |
| B0 | Exact historical attempt1 static checkpoint; never regenerated | Same |
| B1 | Original BF16 base + CPU FP32 target merge → BF16 save/fresh reload | Same |
| A2 | Base explicitly loaded FP32 + formal PEFT, all base/LoRA floating params FP32, no merge | FA2 / autocast disabled |
| B2 | Original BF16 base + CPU FP32 target merge → all floating weights FP32 → F32 save/fresh reload | FA2 / autocast disabled |

A2 is implemented and requires no save. B2 is a separate FP32 checkpoint, never
upcast from B1. The explicit FP32 loader uses `dtype=torch.float32`, local/offline
files, the original revision/trust setting and original attention backend.
Floating buffers are checked too. B2's serialized header and reload parameters
must be F32/FP32 independently; a wrong BF16 checkpoint fails closed.

To avoid cloning an entire 4B state, B1 and B2 each run the same CPU FP32 merge.
Record `shared_fp32_merge_source=false`, `merge_repeat_count=2` and each
`pre_cast_merge_source_fingerprint`. Stream a SHA256 over the entire pre-final-cast
CPU state_dict (names/shapes/dtypes/tensor bytes); no tensor dump or full clone.
If both variants complete, fingerprints must be identical, or the diagnostic
fails. Before final casting that state contains FP32 merged targets plus untouched
original BF16 non-target weights. B1 then casts the entire model to BF16; B2
promotes the remaining untouched floating weights to FP32, never back to BF16.

The execution order is A0 → destroy → B0 → destroy → B1 → destroy → A2 → destroy
→ B2 → destroy. Each variant gets separate peak allocated/reserved bytes and
weak-reference destruction checks. The CPU merge model is destroyed before fresh
reload. Checkpoints are separate `tmp_fp32_merge_bf16` and `tmp_fp32_merge_fp32`
direct children, staged and reload-validated before publication/use; default
cleanup and `--keep-diagnostic-model` semantics apply to both.

### True FP32 forward and support evidence

`forward_rows_fp32` reuses original `load_row_inputs`, full prompt+response IDs,
response mask, pixel/image-grid tensors, Qwen `get_rope_index`, temperature 0.7,
previous-position slice and the pinned verl helper (`inplace_backward=False`).
No re-tokenization and no full-vocabulary logit export. Historical floating vision
inputs are promoted to FP32; integer input/position/grid/mask dtypes are retained.
Actual saved/input dtypes are recorded per row for A2/B2. Observational hooks on
BF16 variants record root inputs and vision tower inputs without altering the
original forward. Qwen internally aligns pixel dtype with the visual model;
the actual before/after observations are retained.

FP32 forward disables autocast even if the caller has an ambient autocast context.
Temporary module hooks reject non-FP32 floating inputs/outputs or nested enabled
autocast. TF32 matmul/cuDNN flags are temporarily disabled and restored on exit;
this prevents calling reduced-mantissa TF32 GEMM a true FP32 control.

FA2 FP32 availability is NOT inferred from a version number. Probe the **original
first multimodal row**, then forward remaining original rows if it succeeds.
The pinned HF FA2 integration has dtype-selection logic which could cast QKV
before the kernel; see the
[official 4.57.1 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/integrations/flash_attention.py).
Therefore this diagnostic additionally wraps the real installed HF `_flash_fn` /
`_flash_varlen_fn` entry points temporarily. It verifies FP32 QKV and output at the
actual kernel boundary, records dtype counts, and restores original functions on
every exit. Hooks/guards exist only in this diagnostic process, not in production
files or formal execution. FA2 success requires an audited kernel call; parameter
dtype alone is insufficient evidence.

Recognized FA2 dtype-not-supported errors and detected implicit half downcasts
produce `supported=false`, exception type/message/stage. They are **not** overall
measurement failures. Missing/mismatched lineage/tokens, nonfinite baseline or
FP32 results, source mutation, output corruption, OOM and unrelated runtime bugs
remain real nonzero failures; they are not relabeled as FA2 support evidence.

When A2/B2 are unsupported:

```json
{
  "execution_succeeded": true,
  "fp32_same_backend_supported": false,
  "primary_fp32_forward_available": false
}
```

Their support records preserve the errors. Corresponding unavailable token
values/diffs and pair metrics are `null`, including in original top-N outliers.
A0/B0/B1 and weight results are retained. Weight representation statistics remain
available even if FP32 forward fails, provided B2 save/reload completed. If one
variant succeeds and the other fails, report asymmetric support and refuse A2-B2.
This is NOT a successful arithmetic-versus-representation separation.

### Historical binding, repeated results and statistics

Require the complete v4.5.2 and v4.5.3 summary/token artifacts before GPU work.
Validate v4.5.3 successful diagnostic state, software, run/adapter/B0 lineage,
collection attempt, count, original token IDs/order/positions and recomputed
three pairwise metrics. Never trust a hardcoded 1294 alone; bind the actual mask
count (1294 for the real attempt) to the existing artifacts.

`repeat_delta_vs_v453` reports current-minus-old A0-B0/A0-B1/B0-B1 metrics. B1
per-token repeated logprob differences are also reported. No bitwise equality is
required. A conservative **diagnostic integrity** guard rejects gross B1 repeat
changes: repeated mean > max(0.1, 4×old A0-B1 mean), or repeated maximum >
max(1.0, 4×old A0-B1 maximum). These are transparent diagnostic sanity bounds,
not Gate thresholds, not adjustable PASS criteria and not production changes.
Smaller changes remain visible for human review, not silently ignored.

Preserve A0-B0/A0-B1/B0-B1 metrics. Canonical new keys are
`dynamic_vs_production_merge`, `dynamic_vs_fp32_merge_bf16`, and
`production_vs_fp32_merge_bf16`; historical v4.5.3 compatibility keys remain too.
When available add A0-B2, A2-B2, A0-A2,
B1-B2 and B0-B2 using the original `pair_metrics`. Retain the original v4.5.2
A0-B0 top50 ranking and enrich those SAME tokens with all new values/diffs,
including A2-B2 and B1-B2; never rank a different token population.

Weight analysis retains B0-B1 and adds B0.float()-B2, B1.float()-B2. The latter
is final BF16 representation error, measured fully in streamed target-weight
chunks, per layer/suffix and element-weighted suffix aggregates: absolute mean/max,
nonzero fraction and mean/max error relative to the original FP32 LoRA delta.
Only statistics are saved, never complete weight/delta/diff tensors.

### New outputs and optional secondary

All original outputs, pinned base, formal adapter/metadata/configs, v4.5.2,
v4.5.3 and original Gate report retain before/after SHA256 protection.
New output (existing directory refuses overwrite):

```text
reports/rl_gate_c/gate-c-v451-attempt1/fp32_merged_forward_diagnostic/
  summary.json                    # successful summary published last
  token_diagnostics.jsonl
  module_merge_stats.json
  weight_representation_stats.json
  source_checksums.json
```

Metadata remains diagnostic/evidence-only, not for training/rollout, with
`formal_rl_initialization_allowed=false`. No Gate manifest is written.

An optional `--allow-sdpa-fp32-secondary` is implemented, **default off**. Only
explicit opt-in loads A2/B2 under SDPA with the same FP32 forward checks.
Results are under `secondary`, marked `secondary_backend_changed=true`,
`exploratory_only=true`, `excluded_from_primary_metrics=true`; its A2-B2 metric
is separately named. It cannot serve as a pure dtype control against FA2 A0/B1.
Do NOT enable it for this first AutoDL run. If unsupported, first return the FA2
evidence and decide whether a later secondary experiment is worthwhile.

### v4.5.4 AutoDL command

Use one A800 80GB and the unchanged historical pinned environment (actual run
manifest takes precedence; torch 2.8.0 / transformers 4.57.1 / PEFT 0.21.1 /
verl 0.6.1). Inspect CPU RAM/disk too: FP32 4B parameters are roughly 16 GB before
activations/workspaces/logits; this is not a peak-VRAM guarantee. This entry does
not automatically switch to CPU on GPU OOM or downgrade dtype.
CPU merge promotes the target weights and needs the largest delta/safe-merge
working buffers. Precise CPU peak RSS is not collected. Free-space checks account
for FP32 save separately (4×source safetensors bytes + adapter bytes + 1 GiB free
at that stage); B1 and B2 coexist on disk until analysis/cleanup. No GPU models
coexist. Real FP32 GPU support and peak memory are not validated locally.

```bash
set -euo pipefail
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN_BASE_SNAPSHOT=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17

CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_rl_fp32_merged_forward.py \
  --run-id gate-c-v451-attempt1 \
  --base-model-path "$QWEN_BASE_SNAPSHOT" \
  --top-n 50 \
  --local-files-only \
  2>&1 | tee logs_gate_c_v451_attempt1_fp32_merged_forward_diag.txt
```

Expect `TOKENS: 1294`, repeated BF16 comparison metrics, `FP32 SAME BACKEND
SUPPORTED: true/false`, available FP32 comparisons or clear unsupported reasons,
repeat deltas and report path. Numeric outcomes are not predetermined.

If supported, return terminal summary, `summary.json`, A2-B2/A0-A2/B1-B2 metrics,
representation suffix summary and original top20 token changes under A2/B2.
If unsupported, return terminal summary, exception type/message and summary.
Offline top20 extraction:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('reports/rl_gate_c/gate-c-v451-attempt1/fp32_merged_forward_diagnostic')
s = json.loads((p / 'summary.json').read_text())
for r in s['original_production_outliers'][:20]:
    print(json.dumps({k: r[k] for k in (
        'global_trainable_index', 'token_id', 'decoded_token',
        'prior_dynamic_vs_production_diff', 'dynamic_peft_vs_fp32_merge_bf16_diff',
        'dynamic_fp32_vs_merged_fp32_diff', 'fp32_merge_bf16_vs_fp32_merge_fp32_diff')}, ensure_ascii=True))
PY
```

### Human interpretation only

- A0-B2 far below A0-B1/B0: evidence implicating BF16 representation/forward,
  but both parameter and computation dtype changed; do not claim weight rounding alone.
- A2-B2 near zero, A0/B1 still different: stronger evidence of a BF16 numerical
  path rather than a mathematical dynamic/merged mismatch.
- Even A2-B2 remains large: investigate ordering/module/GEMM/backend numerical paths.
- Small mixed improvements: mixed-cause evidence, not a single confirmed root cause.
- Unsupported/asymmetric primary: no FP32 dtype-only conclusion. SDPA secondary
  additionally changes backend, so keep it separate.

No case automatically declares a root cause, modifies production merge, repairs
Gate C, changes thresholds or grants Gate PASS. Current local readiness only:
**READY FOR AUTODL FP32 MERGED FORWARD DIAGNOSTIC**.
