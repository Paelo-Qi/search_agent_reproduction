# Gate C v4.5.2 — Policy handoff forensic diagnostic

Status: **READY FOR AUTODL POLICY HANDOFF DIAGNOSTIC**.
Local CPU fixtures cannot confirm numerical root cause or Gate C PASS.
No Gate C configuration, threshold, reward, RLOO, fatal mask or update path changes.

## Why the real attempt1 failure matters

The user-reported AutoDL `gate-c-v451-attempt1` stopped at
`pre_update_policy_alignment`, with optimizer_step_count=0. Both FSDP ranks
reported identical statistics over 1294 trainable response tokens:

| Metric | Observed value |
| --- | ---: |
| temperature / rollout_temperature | 0.7 / 0.7 |
| mean absolute logprob difference | 0.024946338922444147 |
| maximum absolute difference | 0.5243490934371948 |
| mean signed difference | -0.0032114278333751217 |
| mean importance ratio | 0.9985741278485162 |
| minimum / maximum ratio | 0.5919405369778351 / 1.3909131382256403 |
| initial clipping fraction | 0.013910355486862442 |

Approximately 18/1294 tokens exceed [0.8, 1.28]. A mean ratio near 1 does
not establish a trustworthy handoff: opposing signed drift can cancel while
individual sampled actions already fall outside PPO clipping bounds.
This was a useful fail-closed finding, NOT a reason to relax the gate.
The unchanged requirements remain clipping_fraction=0 and max_abs_diff<0.1.

## A/B/C decomposition

- **A: dynamic PEFT HF** — pinned BF16 Qwen base plus the ORIGINAL formal
  checkpoint-3k adapter, eval/no-grad, no merge.
- **B: exact static-merged HF** — the already-published merged directory
  actually consumed during attempt1 collection, BF16 plain HF, no active PEFT.
- **C: recorded vLLM rollout** — sampled response IDs and processed logprobs
  already persisted in the committed group. No vLLM process is launched.

Compare A-C, A-B and B-C token by token. The group supplies collection_attempt;
ONLY `outputs/rl_gate_c/<run>/merged-<collection_attempt>` is accepted.
Missing exact weights fail with `exact rollout merged checkpoint unavailable`.
There is no regeneration or substitute Gate B model.

The BF16 merge hypothesis is a possibility, not a conclusion: dynamic PEFT
computes W_bf16 x + LoRA(x), while static merge computes with
round_bf16(W_bf16 + delta_W). Some small updates may be rounded during merge.
A-B measures the actual existing handoff without changing base/LoRA/merge dtype.
There is NO FP32 merge experiment here.

## Historical provenance and read-only safety

The diagnostic does not call Gate C context preparation, run binding or
finalization. Historical integration source hashes need not match v4.5.2 code.
Instead, it verifies the old identity self-hash and cross-artifact bindings:

- gate_c_report: stage pre_update_policy_alignment, passed=false, integer
  optimizer_step_count=0. A verified update or PASS/ambiguous manifest is refused.
  update_started may exist, but must refer to the same identity.
- committed group payload and all saved tensor file hashes/path containment;
  original alignment group/policy/context/temperature/token-count evidence.
- exact merge manifest completeness, absence of PEFT files, model-file hashes
  and canonical merged fingerprint matching the group/source adapter.
- fixed `outputs/sft_main_imageid_v3/checkpoint-3k/adapter`, its formal metadata,
  adapter/config/checkpoint fingerprints and lineage matching the old source actor.
- old formal config hashes and pinned base model/revision/offline file hashes.
  Forward library versions must match the historical environment, including verl0.6.1.

The entire attempt output tree, base snapshot, adapter, metadata, formal config
files and original gate_c_report are SHA256-snapshotted before validation and
after computation. Changed content, additions or deletions fail the diagnostic.
No locks, markers, models, checkpoints or new files are written into outputs/.
The original gate_c_report is never overwritten. Diagnostic outputs are separate;
an existing diagnostic directory is protected from automatic overwrite.

## Exact token/vision forward

Reuse formal `training_rows` with irrelevant zero placeholder advantages solely
to extract its existing masks; no reward/estimator computation or training occurs.
Fatal-generation responses remain included. Entire post-fatal rows are omitted.
Only response_mask==1 contributes to reports. Prompt, observation and padding
do not contribute; no response text is tokenized.

For each original generation row, load the checksum-bound .pt with
weights_only=True, require saved input_ids[0] to equal actual prompt_ids,
and reuse pixel_values/image_grid_thw. No PIL, chat rendering or processor call.
Build prompt_ids+response_ids with all-one attention mask, without padding.
Use the real Qwen get_rope_index on that full sequence and image grid.

The pinned [verl actor non-rmpad path](https://github.com/volcengine/verl/blob/v0.6.1/verl/workers/actor/dp_actor.py)
provides the forward semantics: eval/no_grad, CUDA BF16 autocast, use_cache=False,
temperature-scaled logits, and response slice `[:, -response_length-1:-1, :]`.
Use the actual [verl logprobs_from_logits](https://github.com/volcengine/verl/blob/v0.6.1/verl/utils/torch_functional.py)
with original response IDs and inplace_backward=False. This preserves the
chosen-token calculation; no separate production softmax implementation exists.
temperature=.7, top_p=1 and top_k=-1 make C's processed distribution untruncated,
so full-softmax HF logprobs are comparable without filtering.

Exactly one visible BF16 CUDA GPU is required, no distributed launch. Run A on
each row, retain only sampled-token logprobs on CPU, delete A, collect garbage,
empty CUDA cache and verify weakref release BEFORE loading B. Repeat for B.
Full-vocabulary logits are transient per row and never serialized or retained
across rows/models. Nonfinite forward logits/sampled logprobs fail; finite
numerical drift is a successful MEASUREMENT, not script failure or Gate PASS.

## Reports and interpretation

Output folder:
`reports/rl_gate_c/gate-c-v451-attempt1/policy_handoff_diagnostic/`

- summary.json: execution success/failure (NOT Gate PASS), lineage, original
  FSDP alignment, all three measured pairs, original-vs-dynamic metric deltas,
  per-row metrics, top outliers, model destruction and peak CUDA memory.
- token_diagnostics.jsonl: ALL trainable tokens (1294 for the reported attempt,
  validated from actual mask counts rather than hardcoded). Contains row/step/
  position/global index, ID, tokenizer token representation and decoded string,
  special-token flag, A/B/C values and all three signed differences/ratios.
  Control characters remain JSON escaped; empty decoded tokens are retained.
- row_diagnostics.json: comparisons grouped by rollout_index/step_index.
- source_checksums.json: before/after checksums and unchanged-source evidence.

Every diagnostic artifact says formal_rl_initialization_allowed=false.
No gate_manifest is created. Summary is published last; incomplete publication
cannot claim successful execution. Failure is nonzero and writes only a separate
diagnostic failure summary where possible. Originals are not repaired/deleted.

Each pair includes token count; mean/max absolute and mean signed difference;
ratio mean/min/max; fraction outside [0.8,1.28]; p50/p90/p95/p99/p99.5/p100
absolute-difference quantiles; counts above .01/.05/.10/.20. Top-N defaults to50,
sorted by absolute difference with deterministic index tie-breaks. The terminal
prints only aggregate comparisons, old FSDP values and report location.

Interpret evidence manually, including individual tokens and generation rows:

| Pattern | Evidence to investigate, NOT automatic root cause |
| --- | --- |
| A-B resembles A-C; B-C small | Dynamic PEFT -> BF16 static merge drift |
| A-B small; B-C resembles A-C | Static HF -> vLLM backend/numerical drift |
| A-B and B-C both substantial | Mixed contributions, possibly cancellation |
| A-C differs substantially from original FSDP-C | Plain dynamic HF may not be an adequate FSDP2 proxy; plan a separate FSDP diagnostic |

No heuristic assigns a root cause and no report modifies any Gate check.
No new rollout, judge, reward cache request or external API is needed.

## AutoDL command

Run from repository root using the SAME verified pinned RL Python environment.
No source-root, judge/search/layout configuration or provider API keys are required.

```bash
set -euo pipefail
export PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export QWEN_BASE_SNAPSHOT=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17

CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_rl_policy_handoff.py \
  --run-id gate-c-v451-attempt1 \
  --base-model-path "$QWEN_BASE_SNAPSHOT" \
  --top-n 50 --local-files-only \
  2>&1 | tee logs_gate_c_v451_attempt1_policy_handoff_diag.txt
```

If your verified snapshot is elsewhere, change only that locator; hashes must
still match attempt1. Missing original AutoDL artifacts cannot be reconstructed
from the local Windows checkout. Preserve the original attempt regardless of
success/failure. For another diagnostic invocation, first explicitly archive
the separate diagnostic directory (never alter the attempt output directory).

Return summary.json, row_diagnostics.json and terminal log, plus representative
outliers from all three pairs. Preserve token_diagnostics.jsonl (all tokens) and
source_checksums.json for deeper offline inspection. Do not rerun collection or
begin any optimizer update based solely on these diagnostic measurements.

## CPU tests

```bash
PYTHONPATH=src python -m pytest tests/test_rl_policy_handoff_diagnostic.py \
  tests/test_rl_policy_alignment.py tests/test_rl_training_batch.py \
  tests/test_rl_gate_c.py -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

Fixtures test pair decomposition/quantiles/outliers; exact masks; real-ID use;
M-RoPE handoff and previous-logit slice; historical artifact bindings and
fail-closed validation; lazy imports; serial destruction; source mutation;
drift-as-measurement; and protected originals/output publication. They do not
prove real GPU forward, HF/vLLM equivalence or BF16 merge root cause.
