# v4.5.7 — Actor-side old-logprob recompute feasibility (NOT a Gate)

This is a read-only historical-attempt diagnostic. It does not repair Gate C,
resume a Gate, collect trajectories, launch vLLM, calculate rewards/RLOO, run a
policy loss/backward/update, export a model, or authorize RL initialization.
Formal configs, clipping bounds and `max_abs < 0.1` are unchanged.

## Inputs and independent computations

Read `gate-c-v451-attempt1`, its original group, original failed alignment, pinned
base snapshot and **formal** `outputs/sft_main_imageid_v3/checkpoint-3k/adapter`.
The existing artifact validators bind adapter/base/group/run fingerprints,
response/fatal masks, software versions and all six forensic histories. This
must be the zero-optimizer-step attempt, not a subsequently updated actor.

Reuse `construct_actor` from `verl_actor_gate`: real 2-GPU FSDP2, dynamic PEFT,
BF16 mixed precision, FA2, frozen vision/projector, checkpointing. Reuse formal
`configure_one_update` **only to configure** the actor: runtime flag remains
`use_rollout_log_probs=True`. The unchanged constructor creates the formal
optimizer object, but it is never used; its instance `step` and actor instance
`update_policy` are guarded against accidental execution.

Reuse `training_batch.build_dataproto` and saved multimodal tensors, without
re-tokenization or image processing. `load_processor` reads local pad-token ID
only. The pinned model's M-RoPE helper constructs the exact formal positions.
Saved training-row advantages are retained when available; otherwise the existing
forward-only helper supplies irrelevant zero placeholders, with no reward or
advantage recomputation. Both ranks use the complete original group (as Gate C
does), so 1294 masked tokens are **not doubled to 2588**.

- **R**: saved rollout/vLLM processed logprobs, carried as the same FP32
  `old_log_probs` used by formal alignment. No rollout logprobs are regenerated.
- **O**: first `actor.compute_log_prob` on its own deep-cloned DataProto.
- **C**: second independent `actor.compute_log_prob` on a different deep clone.

Pinned verl returns `(log_probs, entropy)` here. Each computation must really
enter the actor module; a cache-only/copy-only result is rejected. The outer
`torch.no_grad()` is the same as formal alignment; verl owns eval mode and BF16
autocast. Forward hooks observe eval/no-grad, disabled dropout and real CUDA
BF16 autocast. RNG state must not change. No temperature arithmetic is added:
metadata is exactly 0.7 and the installed forward implementation is audited.

Typed recursive SHA256 covers all DataProto tensor fields, metadata and tensors
inside the multimodal object array. O/C before/after and the untouched template
must match. Full parameter fingerprints hash **every actual local shard** with
name, dtype, local/global shape and placements, in bounded CPU chunks. No full
4B state gather/export is used. pre_O/post_O/pre_C/post_C must match on each rank;
no gradients may be created. These checks fail closed on mutation or randomness.

## Metrics and interpretation

The three comparison labels use **candidate minus anchor**, consistent with the
formal initial PPO ratio (explicit directions are recorded): O−R, C−O, C−R.
All existing `pair_metrics` quantiles/counts/ratios are preserved. Additionally
report the O-to-C initial ratio **exp(C−O)**: mean/min/max/clip fraction.
Per-token records also expose literal R−O/O−C/R−C and the corresponding forward
ratios, historical token identities/text and deterministic top50 lists.

Numerical feasibility is diagnostic-only: finite aligned masked values, count
equal, zero clipping, and O/C maximum difference ≤ min(1e−3, 1% of original
R/C maximum). This allows floating-point-level differences, not arbitrary .01
drift, and is **not** the formal Gate .1 threshold. A failing O/C check exits
nonzero before interpreting the rollout gap, retaining partial O/C evidence.

Project newly measured C/R through the unchanged `compare_policy_logprobs`
helper, then compare each rank to its corresponding original alignment. The
repeat guard covers count and all seven scalar statistics; diagnostic tolerances
are max(1e−4, 10% of the original scalar), or max(one masked token fraction,
10% of original clip fraction) for clipping. A significant non-repeat exits
nonzero and refuses interpretation. Original C/R is expected to remain failing
the formal Gate; this diagnostic never calls `require_policy_alignment`.

**Historical per-token FSDP C was not persisted by the original Gate C.** Thus
historical R/C top50 and historical individual differences are unavailable/null,
not reconstructed or substituted with v4.5.2 HF proxy values. All three **new**
top50 lists are provided, with original aggregate/per-rank repeat evidence.

## Installed verl semantic audit (AutoDL evidence required)

Windows does not have installed verl 0.6.1; CPU mocks are not semantic evidence.
Rank zero inspects the actual installed actor class, config class and resolved
loss registry function. It discovers FSDP worker/trainer definitions in the
installed package, selects FSDP via actual utility imports (not a filename guess),
and follows the trainer's actual correction-helper import. AST checks trace
flag branches, denominator assignment, loss arguments and `exp(current−old)`
(including vanilla's stability clamp). Actual runtime values are recorded.
Source file paths, SHA256, symbol locations and relevant line/snippet hashes are
saved in `verl_source_audit.json`. No Ray worker/trainer/loss is executed.
Unsupported or incomplete patterns produce `undetermined`, not a guessed fix.

The official v0.6.1 source suggests an important distinction, which must still
be confirmed against AutoDL's installed files:

- flag **true** reads the caller-supplied `model_inputs["old_log_probs"]`.
- flag **false**, one mini-batch and one epoch, uses the **same update forward**
  `log_prob.detach()`; otherwise it reads that same supplied old field.
- trainer non-bypass recomputation is a separate path: worker compute →
  `DataProto(old_log_probs=log_probs)` → `batch.union(old_log_prob)`.
- trainer bypass maps rollout logprobs into the old field in its correction
  helper; this is controlled by `rollout_correction.bypass_mode`, not this flag.

References: [actor](https://github.com/volcengine/verl/blob/v0.6.1/verl/workers/actor/dp_actor.py),
[trainer](https://github.com/volcengine/verl/blob/v0.6.1/verl/trainer/ppo/ray_trainer.py),
[worker](https://github.com/volcengine/verl/blob/v0.6.1/verl/workers/fsdp_workers.py),
[loss](https://github.com/volcengine/verl/blob/v0.6.1/verl/trainer/ppo/core_algos.py),
[bypass helper](https://github.com/volcengine/verl/blob/v0.6.1/verl/trainer/ppo/rollout_corr_helper.py).

`ppo_semantic_acceptability` concerns the **assumed false-flag candidate**. A
proven conditional same-forward branch is `not_supported_by_verl_implementation`,
even if a separate trainer actor-recompute path exists and O/C≈0. Unknown source
patterns are `undetermined`. Numerical feasibility alone does not establish PPO
theoretical correctness or authorize any production change. Trajectory-generating
static-merged vLLM representation, dynamic training representation and chosen
denominator are different concepts. This tool changes **none** of them.

## Execution and safety

From the AutoDL repository, using the same installed versions as attempt1:

```bash
set -euo pipefail
export PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Set to your EXISTING pinned snapshot; no downloads.
export QWEN_BASE_SNAPSHOT=/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/diagnose_rl_actor_old_logprob_recompute.py \
  --run-id gate-c-v451-attempt1 \
  --base-model-path "$QWEN_BASE_SNAPSHOT" \
  --top-n 50 --local-files-only \
  2>&1 | tee logs_gate_c_v451_attempt1_actor_old_logprob_recompute_diag.txt
```

No API keys needed. Exactly two visible BF16 CUDA GPUs are required. Rank zero
alone reserves/writes a **new**, overwrite-refused directory:
`reports/rl_gate_c/gate-c-v451-attempt1/actor_old_logprob_recompute_diagnostic/`.
Do not delete an earlier report to force a repeat. All source attempt outputs,
base, formal adapter/metadata/configs, six old diagnostics and original Gate report
are protected with before/after checksums, excluding the new directory. Relevant
installed verl source hashes are also rechecked. Existing files are never repaired.

False summary is published first; bounded NCCL stage coordination propagates
peer errors. Ordinary exceptions persist false summary/integrity audit and exit
nonzero; a process death leaves the initialized false report/stage evidence.
There is no failure-cleanup barrier. No Gate manifest/update marker is created.
Success here means only diagnostic execution succeeded; formal initialization
remains false and optimizer_step_count=0. All token fields are diagnostic output,
not a training dataset or checkpoint.

Return terminal summary and these `summary.json` fields: comparisons, actor-side
initial ratio, numeric feasibility, installed semantic audit/acceptability,
historical repeat, optimizer count. If semantics is undetermined, also provide
`verl_source_audit.json`. Full token JSONL is not needed.

## CPU verification

```bash
PYTHONPATH=src python -m pytest tests/test_rl_actor_old_logprob_diagnostic.py \
  tests/test_rl_bf16_lora_dtype_diagnostic.py tests/test_rl_bf16_sdpa_diagnostic.py \
  tests/test_rl_policy_alignment.py tests/test_rl_training_batch.py \
  tests/test_rl_verl_policy_update.py tests/test_rl_gate_c.py \
  -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

The previously absent `test_rl_verl_policy_update.py` now tests only the unchanged
formal configuration and `(log_probs, entropy)` read-only alignment contract.
Local status: **READY FOR AUTODL ACTOR OLD-LOGPROB RECOMPUTE FEASIBILITY DIAGNOSTIC**.
No Gate C PASS, repair claim or smoke20 authorization follows from CPU tests.
