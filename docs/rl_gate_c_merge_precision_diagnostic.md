# v4.5.3 — Merge Precision Diagnostic

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
forward path; BF16 autocast over FP32 weights is **not** B2. This round intentionally
keeps the original FA2/BF16 forward and implements only mandatory B1.
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
