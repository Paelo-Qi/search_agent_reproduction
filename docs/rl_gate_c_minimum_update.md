# Gate C — Minimum RL Integration

Status: **READY FOR AUTODL EXECUTION — NOT YET GPU PASSED**.
CPU tests are not rollout, live Judge, FSDP2 policy-update or GPU reload evidence.

## Scope and immutable inputs

This isolated gate consumes ONE real quality-clean schema-v3 smoke record,
collects exactly TWO trajectories, and calls the real verl actor policy update
exactly ONCE. No smoke20/main400 trainer, epoch loop, scheduler progression or
production throughput infrastructure is implemented. The record's ORIGINAL
question is used, not the Gate B diagnostic crop question. Direct answers with
zero tools are legal. All original input images are hash-verified and retained.

Gate A2.2 proved FSDP2/backward/save/reload using a temporary supervised loss.
Gate B proved static merge, fresh HF reload, real rLLM generation ownership and
actual derived PIL round-trip. Gate C connects real rollout tokens and rewards
to actual RL policy loss, gradients, one optimizer update and save/reload.

The ONLY actor initializer is:

```text
outputs/sft_main_imageid_v3/checkpoint-3k/adapter
```

Formal SFT complete-stage metadata, adapter/config/file fingerprints, stage
`main_b_2k`, lineage `main_a_1k -> main_b_2k`, base/revision, LoRA and current
runtime protocol are validated with the existing lineage functions. The Gate B
manifest must be PASS with every literal check, matching source-SFT fingerprint
and pinned base/protocol. It is EVIDENCE ONLY; its updated/merged weights are
never loaded as Gate C's initializer. There is no CLI adapter override.

Base: Qwen/Qwen3-VL-4B-Instruct, revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`; BF16, FA2, frozen vision/projector,
existing rank16/alpha32/dropout0.05 SFT LoRA and seven original target modules.
Use the same locally verified pinned base snapshot used for A2.2/B; no downloads.
Snapshot model/config/weight file hashes are bound to the run identity; changing
offline files between collect/update/finalize fails closed. This hash pass reads
the base weights on CPU and may take time; it does not load a GPU model.
Formal RL/SFT/data/Eval membership/configuration is not changed.

`configs/rl_gate_c.yaml` contains ONLY gate settings: n=2, actor world size=2,
one update, lr=1e-6, AdamW weight decay=0, low/high clip=.2/.28, entropy=0;
vLLM TP1, context8192, per-response cap512, temperature=.7/top_p1/top_k-1,
GPU memory utilization=.6, max turns16. Formal main remains n=4. The short cap
is gate-only; there is no 70k expansion or silent truncation.

## Pinned APIs checked before implementation

The compatibility layer is a DATA/INTERACTION layer, not a local PPO loss:

- [verl 0.6.1 DataParallelPPOActor](https://github.com/volcengine/verl/blob/v0.6.1/verl/workers/actor/dp_actor.py):
  `update_policy(DataProto)` requires responses, response_mask, input_ids,
  attention_mask, position_ids, old_log_probs and advantages; temperature is
  metadata, and `multi_modal_inputs` lives in `non_tensor_batch`.
- [verl RLOO and loss](https://github.com/volcengine/verl/blob/v0.6.1/verl/trainer/ppo/core_algos.py):
  actual `compute_rloo_outcome_advantage` and `get_policy_loss_fn('vanilla')` /
  `compute_policy_loss_vanilla`; no outcome standard-deviation normalization.
- [DataProto](https://github.com/volcengine/verl/blob/v0.6.1/verl/protocol.py) and
  [MM extraction](https://github.com/volcengine/verl/blob/v0.6.1/verl/utils/model.py):
  vision dict tensors are concatenated by the actor; `DataProto.to` does NOT
  move nested non-tensor dict tensors, so the converter moves them explicitly.
- [rLLM pinned MultiTurnWorkflow](https://github.com/rllm-org/rllm/blob/c5c02a49780e26ae9cb6f1fb56731d1e594d59f0/rllm/workflows/multi_turn_workflow.py):
  upstream `run` remains inherited. LocalEngine passes actual ModelOutput to the
  matching ProjectAgent Step through a checked per-generation handoff because
  upstream calls `update_from_model(output.text)`, not `update_from_model(output)`.
- [vLLM 0.11 SamplingParams](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/sampling_params.py) and
  [Sampler](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/sampler.py):
  logprobs=1 returns the chosen token even when it is not the top token.
  Crucially, default raw_logprobs are BEFORE temperature. Gate C explicitly
  configures `LLM(logprobs_mode='processed_logprobs')` so old logprobs describe
  temperature=.7 sampling. Top-p1/top-k-1 do not truncate the distribution.
- [Qwen3-VL 4.57.1 M-RoPE](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py):
  actual `get_rope_index(input_ids, image_grid_thw, attention_mask)` supplies
  multimodal positions; no generic text-only arange replacement.

Software contract: torch2.8.x, transformers4.57.1, PEFT0.21.1, verl0.6.1,
vLLM0.11.x, rLLM0.2.1. Installed real rLLM component source hashes, repository
integration source hashes and software versions are recorded. Changing code or
config changes the run identity and cannot silently reuse an old run-id.

## Three-stage architecture

```text
collect: original checkpoint-3k -> safe static merge -> fresh HF reload
         -> destroy HF -> vLLM TP1 -> actual rLLM workflow (n=2)
         -> two independent live judges per trajectory -> shutdown vLLM
         -> validate entire group -> atomic directory publication
update:  torchrun 2 -> original checkpoint-3k -> real verl FSDP2 actor
         -> official RLOO -> fatal clamp -> actual token DataProto
         -> both-rank pre-update policy alignment -> update_policy once
         -> native checkpoint + PEFT export
         -> destroy actor -> fresh FSDP2 actor + native reload -> finite forward
finalize: verify immutable run/group/mask/checkpoint/rank evidence
          -> final report PASS -> gate_manifest PASS LAST
```

No vLLM and actor coexist. No Ray cluster, custom generation loop or
`AgentRuntime.run` bypass. Existing phase3_registry/cache/tool retry semantics
remain unchanged; actual remote tools and local Pillow tools are used.

Each rank receives the SAME complete one-group per-generation training batch.
This gate-only redundant data-parallel layout keeps FSDP collective counts equal
for variable-length trajectories without invented all-masked examples. FSDP
averages the duplicated gradients: it is not four distinct rollouts and does
not turn n=2 into n=4. Actual microbatch1, one full mini-batch, ppo_epochs1;
there is internal accumulation over the group's generation rows and exactly
one optimizer step. `seq-mean-token-mean` means each generation row receives
equal weight. This transparent step-view conversion is NOT a production
multi-prompt sampler or a claim of trajectory-length-normalized formal training.
That design/performance work remains outside this gate.

**Post-Gate-C smoke design review item:** before formal smoke20, review whether
multi-turn trajectory weighting should be trajectory-normalized or
token-normalized. Gate C's unchanged per-generation-row representation proves
only the real multimodal PPO update chain; this task does not implement either
weighting redesign.

## Actual token and vision evidence

Every real rLLM Step stores prompt_ids, response_ids, logprobs, ModelOutput,
token count, parser kind, finish reason and processed-logprobs provenance.
IDs come from `RequestOutput.prompt_token_ids` and `CompletionOutput.token_ids`;
old logprobs are the actual chosen-token values, length-matched and finite.
No response-text re-encoding or zero logprob placeholders are used.
After both live judges succeed, bind the actual total to the real rLLM
Trajectory.reward and Step.mc_return, with terminal-only Step.reward and
reward_computed=true. The workflow's deferred environment transition 0.0 is
not used as the outcome reward or training advantage. Provider failures never
reach this binding/publication step.

Shared `prepare_qwen_vl_processor_inputs` renders unchanged messages then
processes ordered actual PIL objects. Processor prompt IDs must match actual
vLLM prompt IDs EXACTLY, or the gate fails. Processor input tensors, including
pixel_values/image_grid_thw for initial and any derived images, are stored as
checksummed per-step tensor-only `.pt` files, loaded with weights_only=True.
The actor uses these real vision tensors and actual Qwen M-RoPE positions.

One generation = one policy row. Prompt/context contains system/user/previous
assistant/tool observations. ONLY the generated response suffix receives
policy loss. Prompt, observation, padding and post-fatal responses do not.
Generated tool-call response tokens ARE policy tokens. Rows with completely
post-fatal masks are omitted from the actor mini-batch (to avoid an empty-loss
denominator), but their zero masks remain in `training_masks.json`.

Left padding aligns context lengths; right padding aligns response lengths.
Only padded positions have dummy pad IDs/zero old logprob storage; their
attention/response masks are zero. References are never SFT labels or policy
targets. `model_task` copies exactly sample_id/question/images; reference_answer
is available ONLY to the separate reward orchestration.

## Group identity, atomicity and recovery

`trajectory_group_id` is a canonical hash of prompt/source ID, pre-update
policy fingerprint, rollout config fingerprint, a NEW collection-attempt UUID
and the frozen Gate run context. Both members must match all fields and have
indices0/1. Prepared smoke/main/shards remain untouched and have no group ID.
New attempt or changed policy/config/context means a different group ID.

An OS advisory run lock permits only one writer and releases on process death.
Unpublished `.group-<attempt>` contains forensics only, NEVER a usable half
group. Each member must contain real aligned Step tokens, bound vision files,
complete reward components and the exact reward formula. All two members and
file hashes are validated; group.json is fsynced atomically, then the whole
staging directory is renamed to `group/` as ONE publication point.

Provider/Judge interruption keeps failure history and success-only tool/reward
caches, returns nonzero, and commits no incomplete group. Repeat EXACTLY the
same collect command/run-id after the provider recovers. A new attempt recollects
both rollouts, never mixes halves. A previously fully committed group is verified
and reused; collect does not run either member again. Successes from other runs
and SFT/A/B artifacts are not mutated.

The update creates an `update_started.json` marker. If update/save/reload dies
before `update_verified.json`, execution/optimizer state is ambiguous: it fails
closed and requires a NEW Gate C run-id from original checkpoint-3k. No guessed
checkpoint continuation/double optimizer step. If update_verified exists,
repeating update skips the step and proceeds to finalize, which rechecks hashes.
Collection provider interruptions do NOT create the update_started marker.
The marker means entry into the actor update PHASE, not proof that AdamW stepped.
An alignment failure reports optimizer_step_count=0 but retains the same
new-run-id recovery rule; it is not automatically resumed at the optimizer.

Stage reports are persisted at initialization, rollout collecting, rewarding,
group rewarded/publication, training-batch-ready, actor updating/saving/reloading,
and finalization; distributed logs record per-rank memory/stage failures.
Partial failures are append-only under reports/.../failures/.

## Reward, fatal and RLOO contracts

Reward remains EXACTLY `r_fmt * (0.8*r_acc + 0.2*r_query)`.

- Format: reuse frozen deterministic `format_reward`, current parser/schema and
  image-id-v3 rules. Existing range is fractional [0,1], not a new binary rule.
  Fatal is not an unconditional format-zero switch. The legacy scorer's
  cascade-start prefix is preserved; it is separate from the LIVE stop cutoff.
- Accuracy: existing DeepSeek correctness prompt/parser, same client transport,
  bounded RetryPolicy, thinking disabled and provider classification. No exact
  string match. For no final answer, send REAL null model_answer with explicit
  abnormal/fatal/no-final metadata and a versioned conservative suffix; the
  provider must still return valid verdict JSON. Never invent an answer.
- Query: SEPARATE request using the existing five-criterion query-utility prompt
  and strict score/reason parser, with the same actual DeepSeek transport/retry.
  Full relevant tool trace/final answer, not just the last turn, is assessed.
- Success-only reward caches are separated into accuracy/query namespaces.
  Keys bind kind, prompt/version/messages, judge provider/model/config,
  question/reference and complete relevant trace/termination/images/final
  answer. Cached evidence retains real-provider request and retry provenance.
  No provider error is cached or converted to a zero reward.

Authentication/configuration/quota/persistent429/5xx/network/exhausted malformed
judge responses are run-level interruptions AFTER the provider's bounded retry.
Existing reason enums are reused (quota_exhausted, auth_failed,
provider_misconfigured, provider_unavailable, network_unavailable,
judge_unavailable, malformed_provider_response, manual_interrupt, etc).
Unknown/unattributed failures are hard FAIL, not silent model penalties.

Live fatal K=3 counts shared classifier's MODEL-caused tool errors. A successful
tool resets; neutral no_results breaks a consecutive error cascade as in the
existing classifier. Provider failure aborts without incrementing the counter.
At the THIRD error, store fatal tool-turn and generation-step indices, stop
before another tool execution/generation. Retain the ENTIRE actual generated
response containing that third error; mask all LATER generation responses.
For multi-call text generated before tool execution, the whole response is
already generated at detection; no fourth tool is executed. This is the explicit
generation-boundary cutoff, not guessed character-to-token slicing. Max turns
or response-length termination is fatal at the last captured generation.

Official verl `compute_rloo_outcome_advantage` uses BOTH real rewards, including
fatal members, with the common group index. For n=2 raw advantages are
`[r1-r2, r2-r1]`, without standard-deviation normalization. Only AFTER these
group statistics, reuse `clamp_fatal_advantages`: fatal A_final=max(A_raw,0),
other advantages unchanged. Broadcast each trajectory's final advantage to its
per-generation response tokens, then apply the response/fatal mask.

If rewards are equal, or the actual update has zero/nonfinite LoRA gradients,
the gate FAILS. No dummy advantages, altered rewards or hidden extra attempts.
Use a new explicit run-id/sample-index for another real gate attempt; record it
as such, never cherry-pick silently within a completed group.

## Pre-update policy handoff alignment

v4.5.1 audits the handoff from checkpoint-3k PEFT -> static merge -> vLLM
processed old_log_probs to checkpoint-3k PEFT -> FSDP2 actor current logprobs.
Incorrect merge/reload, image binding, M-RoPE or sampling semantics must fail
BEFORE a policy update, not merely produce finite PPO loss.

After building the real multimodal DataProto and taking the pre-update LoRA
snapshot, BOTH ranks call the pinned verl 0.6.1 actor's
`compute_log_prob(data, calculate_entropy=False)`. The exact same DataProto
object then goes to `update_policy`; no response-text tokenization or rebuilt
batch is allowed. The audit is no-grad, changes no optimizer state/parameters,
and never overwrites the actual vLLM sampled old_log_probs.

Pinned `compute_log_prob` internally calls actor_module.eval(): LoRA dropout
0.05 is disabled only by inference mode, not by a changed dropout setting.
Pinned `update_policy` restores train mode. DataProto temperature must equal
both 0.7 and the frozen rollout temperature. With top_p=1 and top_k=-1,
processed vLLM logprobs and actor temperature-scaled logits describe an
untruncated distribution. Future top_p<1 or finite top_k requires a new
old-logprob semantics audit; it is NOT handled by this gate.

Comparison selects ONLY response_mask==1. Padding and post-fatal mask-zero
tokens are ignored even if their stored values are nonfinite. Shape mismatch,
nonbinary/empty/inconsistent masks, masked NaN/Inf logprobs or ratios fail closed.
Report mean/max absolute and mean signed difference, and
`ratio=exp(current-old)` mean/min/max without clamping. Require zero fraction
outside [1-.2, 1+.28] = [0.8, 1.28] AND strictly max_abs_logprob_diff < 0.1.
The Gate-only bound permits small BF16/kernel/merge drift, not bitwise equality;
0.1 itself fails. No GPU evidence justifies loosening that bound.

`outputs/rl_gate_c/<run>/pre_update_policy_alignment.json` records source-policy
fingerprint, context/group identity, bounds, all metrics and both ranks' separate
checks/stats. Means are averaged across duplicate group views, not extra
rollouts. Rank0 cannot override a failed rank1. `update_verified.json` binds its
SHA256 and per-rank evidence; CPU finalize rechecks hash, numeric checks,
rank evidence, token counts and group/source-policy lineage. Final Gate report
includes the complete pre_update_policy_alignment summary.

Numerical failure persists a failed alignment artifact and false final report
at stage `pre_update_policy_alignment`, with optimizer_step_count=0 and the
specific failed checks. A compute/publication exception also stops there before
the optimizer. No PASS manifest is published. Use a NEW run-id after correction.
Post-update reload requires a finite forward, NOT same-policy ratios near 1.

## Actual update and save/reload proof

Reuse A2.2 construct_actor/activate training/FSDP2 wrapping/LoRA-only AdamW and
native checkpoint helpers. Configure official actor use_rollout_log_probs=True
(otherwise its single-mini-batch on-policy shortcut would ignore supplied old
logprobs). Vanilla clipped RL loss, low=.2/high=.28, no entropy/KL bonus.
No CE, reference labels, homegrown surrogate loss or scheduler step.

The real AdamW.step is temporarily audited: finite nonzero LoRA gradients,
all frozen gradients None, exactly the expected LoRA optimizer parameters,
and exactly ONE call. Snapshot all local trainable shards before/after; require
finite changed tensors on every rank. Require a changed post-export fingerprint.
Both ranks report checks/metrics/gradient audit, optimizer count and peak memory.
Every UPDATE_CHECKS flag starts false and is set from its own completed evidence;
there is no blanket true assignment. Preserve ALL metrics actually returned by
verl. Pinned vanilla loss returns actor/pg_clipfrac, actor/ppo_kl and
actor/pg_clipfrac_lower in addition to actor/pg_loss and actor/grad_norm from
update_policy. No synthetic ratio/PPO metric keys are manufactured.

Reuse native FSDPCheckpointManager (model/optimizer/extra) with scheduler=None
and official full-state gather for PEFT export. Destroy original actor and
optimizer with weakref proof. Build a genuinely fresh base+exported LoRA FSDP2
actor; match updated parameter shards; load native model/optimizer checkpoint
without deleting it; match again and perform finite multimodal logprob forward.
No reference-based CE is used even for reload validation.

All Gate C manifests/metadata say formal_rl_initialization_allowed=false;
the adapter lacks formal SFT complete-stage metadata and cannot pass formal
RL initialization validation. Future smoke20 MUST start checkpoint-3k again.

## AutoDL commands (two A800 total; stages are serial)

Run from the repository root in the already verified pinned RL environment.
Set the SAME real source-image root and offline base snapshot used in A2.2/B.
Supply API keys through environment variables, not command-line strings or JSON.

```bash
set -euo pipefail
export PYTHONPATH=src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Set these locators to YOUR previously verified local inputs.
export RL_SOURCE_ROOT=/absolute/path/to/Search-VL-RL-8K
export QWEN_BASE_SNAPSHOT=/absolute/path/to/pinned/Qwen3-VL-4B-Instruct/snapshot

# MUST execute, not skip, in the installed real rLLM environment (CPU fake model).
python -c 'import rllm.workflows.multi_turn_workflow'
python -m pytest tests/test_rl_gate_c.py::test_real_installed_rllm_tokens_survive_crop_and_step_if_available \
  -o addopts= -q -rs -p no:cacheprovider

# Use the identical COMMON args in all three stages; do not hand-edit artifacts.
COMMON=(--run-id gate-c-v451-attempt1 --config configs/rl_main.yaml \
  --gate-config configs/rl_gate_c.yaml --data data/rl/smoke20.json --sample-index 0 \
  --source-root "$RL_SOURCE_ROOT" --base-model-path "$QWEN_BASE_SNAPSHOT" \
  --gate-b-manifest outputs/rl_gate_b/a22-tp1-attempt2/gate_manifest.json \
  --judge-config configs/judge.example.yaml \
  --search-config configs/search_backends.example.yaml \
  --layout-config configs/layout_parsing.example.yaml)

# 1. Real one-GPU collection + TWO independent live judges per rollout.
CUDA_VISIBLE_DEVICES=0 python scripts/validate_rl_minimum_update.py collect "${COMMON[@]}"

# 2. After vLLM has shut down: exactly one real two-GPU FSDP2 policy update.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/validate_rl_minimum_update.py update "${COMMON[@]}"

# 3. No GPU/model loading in finalize; immutable artifacts/checks must all match.
python scripts/validate_rl_minimum_update.py finalize "${COMMON[@]}"
```

Correctness/query Judge share the existing DeepSeek config; configure the real
provider/search/layout environment keys as for earlier gates. Collection uses
the existing formal tool-cache directory by default; --tool-cache-dir can point
to an existing shared tool cache without changing any cache key/namespace.

Successful artifacts:

```text
outputs/rl_gate_c/<run>/run_manifest.json
outputs/rl_gate_c/<run>/group/group.json + real multimodal tensor files
outputs/rl_gate_c/<run>/training_masks.json
outputs/rl_gate_c/<run>/pre_update_policy_alignment.json
outputs/rl_gate_c/<run>/updated_actor/distributed/{model,optim,extra_state}_*.pt
outputs/rl_gate_c/<run>/updated_actor/adapter/*
outputs/rl_gate_c/<run>/updated_actor/gate_only_metadata.json
outputs/rl_gate_c/<run>/update_verified.json
reports/rl_gate_c/<run>/gate_c_actor_stages.json
reports/rl_gate_c/<run>/gate_c_report.json
outputs/rl_gate_c/<run>/gate_manifest.json
```

All COLLECT_CHECKS and UPDATE_CHECKS plus artifact verification and Gate-only
restriction must be literal true. Finalizer binds group/context, n=2, both rank
identities, all group/vision/mask/checkpoint hashes, pre/post policy, real loss,
one-step counts, gradients, frozen audit and fresh reload proof.
Both pre-update rank alignments and the alignment artifact SHA256 must pass.
Final PASS report is written first; authoritative gate_manifest is the LAST
durable PASS artifact. Publication failure revokes any unsafe PASS marker before
writing false failure reports. Collection or update alone can NEVER publish PASS.

Unverified real-GPU risks: actual HF/vLLM prompt-ID expansion alignment,
temperature-processed logprobs, dropout/BF16 static-merge numeric differences,
derived-image context size, zero reward variance, live Judge/provider quotas,
FSDP2 multimodal forward/loss memory and native checkpoint/reload API behavior.
These must be resolved with REAL Gate evidence, not relaxed checks.

## Local tests and next boundary

```bash
PYTHONPATH=src python -m pytest tests/test_rl_group.py tests/test_rl_reward.py \
  tests/test_rl_fatal.py tests/test_rl_rloo.py tests/test_rl_training_batch.py \
  tests/test_rl_gate_c.py tests/test_rl_reward_judges.py tests/test_rl_workflow_adapter.py \
  tests/test_rl_policy_alignment.py \
  tests/test_rl_actor_gate.py tests/test_rl_gate_b.py -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

After real Gate C PASS, a separate task may implement smoke20's complete-group
training, transactional update/resume, aggregation/metrics and production batch
layout, still initializing the original checkpoint-3k. Gate C adapters must not
be continued. Do NOT start that phase here; no main400 loop is implemented.
