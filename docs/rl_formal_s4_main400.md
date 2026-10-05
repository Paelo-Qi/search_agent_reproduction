# Formal RL S4 Main400

Implementation/CPU validation is not Main400 GPU/API PASS. The user has verified
S2 and S3 on AutoDL; S3 completed 20 prompts, n=2, five updates, using about
86 GB on disk. No other AutoDL artifact fields, memory peaks or timings are
inferred here. Local frozen Main400 inputs and runtime artifacts are not
provisioned; CPU fixtures never authorize runtime PASS.

## Frozen experiment and identity

S4 uses `configs/rl_main.yaml`, the existing quality-selected Main400 and four
ordered 100-row shards. Full pinned-source/quality/overlap/image preflight is
mandatory before bootstrap. Iteration zero ONLY uses the original
`outputs/sft_main_imageid_v3/checkpoint-3k/adapter`; Smoke/Gate/S2/staging
initialization is forbidden.

400 prompts / n=4 / K=4 / W=4 / 100 windows; one AdamW step per window,
lr=1e-6, weight decay=0, PPO epochs=1, per-generation-row mean weighting,
clip=.2/.28, entropy=0, vanilla loss, FSDP2/BF16/FA2. Original saved LoRA
dropout stays .05; live RL dropout stays zero. Reward, reference isolation,
fatal semantics, live tools and judges are unchanged. The shared S2 production
update method performs official verl RLOO, O/C, LoRA change/match, native
model/AdamW/RNG save and destructive fresh multimodal reload.

`formal-s4-main400-v5` binds ordered source identities, frozen data/manifest and
config hashes, SFT lineage, pinned offline base, software/source hashes,
algorithm/rollout/reward/tool contracts, seed, retention contract/milestones and
seed scheme. Every member's seed is `initial_seed + 4*global_position + index`.
It also explicitly binds `search_behavior_version=3` and
`provider_reliability.serpapi_google_lens` (retry-v3, five attempts per stage,
backoffs `[5, 10, 20, 30]` seconds), rather than relying only on source hashes.
GPU locators, collection parallelism, pause and disk controls are operational.
The validated stack remains torch 2.8.x, transformers 4.57.1, PEFT 0.21.1,
verl 0.6.1, vLLM 0.11.0, rLLM 0.2.1, with installed FA2/source identities bound.

### Main-only context budget (no truncation)

The rollout identity additionally binds `remaining-context-no-truncation-v1`.
Main's independent `formal_main.MAIN_ROLLOUT` uses `max_model_len=16384`;
`max_new_tokens=512` and all other rollout/training parameters stay unchanged.
After FULL HF multimodal processing (`truncation=False`), each call gets
`min(current_sampling.max_tokens, 16384 - processor_token_length)` tokens.
Only a deep copy of that call's SamplingParams changes; all other fields,
member seeds, exact prompt IDs and saved vision tensors remain unchanged.
Gate B/C and S3 remain at 8192, opt out by default and retain their strict
fixed-budget check. CLI locators cannot override the frozen Main ceiling.

If real generation ends with `finish_reason=length`, the unchanged pinned
rLLM MultiTurnWorkflow produces `max_response_length_exceeded`. If a later
prompt has no context space, LocalEngine emits the same upstream termination
without generating or adding a phantom processor capture. Existing actual
Steps/prefix, fatal cutoff, reward and advantage rules remain authoritative.
An initial zero-step overflow instead fails closed with prompt ID, processor
length, model limit and remaining-token diagnostics; no member is fabricated.
Generation receipts and captured Step info include requested/effective token
budgets, remaining context, policy and context-limited flag. Exhaustion
diagnostics are also persisted in episode/trajectory metadata.

The user reports that `formal-main400-s4-attempt2` first-window runtime path
completed, but 8k training-quality acceptance FAILED: 10/16 (62.5%) members
were context-fatal (`max_response_length_exceeded`), versus six `env_done`.
Observed processor prompt lengths were roughly 9k--12.3k (8968--12263).
Nine of the ten fatal members had final advantage zero after the unchanged
fatal clamp. These are supplied runtime/audit conclusions; AutoDL artifacts
are not local, and no additional stage details, memory or timings are inferred.

The user reports that attempt3 (v3) successfully initialized Main 16k vLLM
(`max_model_len=max_seq_len=16384`) and ran four collectors in parallel.
Three groups committed; one collector failed at publication lock acquisition:
the shared `groups/.publication.lock` used non-blocking `LOCK_NB` and rejected
benign contention between different immutable group destinations. Coordinator
correctly reaped peers and forbade update. This was NOT a 16k context failure;
16k context remains unchanged. These are supplied forensic facts, not locally
inspected AutoDL artifacts.

Retain attempt1 (v1), attempt2 (v2/8k) and attempt3 (v3/16k) as forensic evidence.
Their sealed identities are incompatible with v4: do not resume, migrate or edit
them, including attempt3's three committed groups. Start a NEW
`formal-main400-s4-attempt4` from the ORIGINAL SFT checkpoint-3k; never inherit
an earlier attempt's policy. The context-budget policy version stays v1.
This patch is CPU/static validated only, NOT attempt4 4xA800 GPU/API PASS.

## Exclusive process lifecycle and resume

Metadata recovery -> ONE shared single-GPU static merge/fresh HF validation ->
merge process/descendants dead -> up to four isolated single-GPU workers ->
one complete n4 group per prompt, atomic publication -> ALL workers/descendants
dead -> exact K4 same-policy groups -> durable merge-retirement authorization
and exact deletion -> fresh four-rank update -> immutable verified successor
and verified attempt -> retention -> next window. No resident GPU actor crosses
a phase boundary. Linux subreaper supervision tracks PID birth identities,
reaps detached descendants and forbids the next phase if cleanup fails.

Only formal group publication uses `run_lock(..., blocking=True)` to serialize
the short inventory/seal/atomic-publication critical section. Default
`run_lock(..., blocking=False)` remains fail-fast for coordinator, formal
authority and Gate C locks. Linux uses blocking `flock`; Windows publication
uses synchronous native `LockFileEx` (no sleep polling or CRT retry timeout).
OS advisory locks release on context exit/file close/process death; stale lock
files are never deleted. Immutable group identity, inventory, seal, atomic
publication and resume contracts are unchanged.

On any worker/provider failure, peer workers are killed/reaped and update is
forbidden. Whole committed groups survive. Partial `.collect-UUID` forensic
files are never spliced or reused. Missing prompts get fresh UUIDs with the
same deterministic seeds. Failure reports/logs are per prompt/invocation.
Resume repeats the same run-id and semantic inputs. Uncertain updates reload
the verified parent through the existing S2 live capability; post-publication
crashes roll forward without repeating an optimizer step.

## Retention and scalable verification

`formal-s4-retention-v1`: latest checkpoint FULL; policy25/50/75/100 FULL;
every adapter, immutable checkpoint manifest, rank evidence, attempt and
authorization retained. Only exact manifest-declared historical native,
optimizer and RNG files may be deleted after a verified immediate successor
and verified attempt. The immutable compaction receipt binds run, original
inventory/roles and both checkpoint identities. No broad directory deletion.
Current/milestone/SFT deletion is forbidden. Crash cleanup accepts only missing
files specifically authorized by a matching receipt.

Merge retirement likewise binds the current merge inventory/manifest, policy,
four group IDs/payload hashes and retention version before deletion/fsync.
Incomplete-window reuse FULL verifies the existing shared merge in the merge
worker before collectors start. Retired bytes are not needed for recovery.

Routine recovery reads sealed historical metadata/attempts/receipts and names,
FULL verifies the current resume checkpoint and unconsumed K groups, and
checks small bound rank metadata. It does not hash historical large checkpoint
or consumed multimodal tensor files. Default S1/S2/S3 FULL readers stay FULL.
Publication/consumption/new verified events are still FULL boundaries.
Instrumented SHA tests cover retained historical FULL milestones too.

Final heavyweight reconstruction verifies all 400 group artifact trees,
all full milestones and every retained compacted file/authorization/attempt,
exact 100 steps, 1600 members, no duplicate consumption, unresolved attempt,
partial current collection, active merge or unauthorized missing artifact.
All checkpoints are `main_checkpoint`, `eligible_for_main_init=false`.
Final report is durable FIRST; output `manifest.json passed=true` is LAST.
Publication failure revokes the PASS marker. CPU fixtures cannot publish it.

The roughly 100 GB retained / below 200 GB normal peak target is an estimate,
not a runtime guarantee. Guards default to a 250 GiB run cap and 30 GiB free
floor, with 18 GiB merge and 32 GiB update staging headroom. Disk reports show
current/free bytes (outputs plus this run's reports/logs), full/compacted checkpoints, active merges, groups and
observed peak across resumes (phase-boundary observations, not continuous
filesystem telemetry). Provision extra headroom if actual group sizes demand it.
Disk-accounting full/compacted checkpoint lists are sorted by zero-padded
policy name, independently of filesystem enumeration order; retention and
compaction authorization rules are unchanged.

## Historical v4 AutoDL acceptance commands (NOT commands for this v5 checkout)

The commands below document the original attempt4 v4 bootstrap. Do NOT execute
them with this v5 checkout or reuse attempt4 with modified sources. They are
not a continuation recipe and not a recommendation to restart from SFT3k.

Do not run these locally. Provision the exact frozen Main400/quality/source,
original SFT, offline base and private provider credentials first. Use the
same overlap manifests used to prepare the frozen RL dataset; see
`docs/rl_data_resume_contract.md`. Set the following locators to their actual
AutoDL paths (do not regenerate or alter membership to satisfy this command).

```bash
cd /path/to/OpenSearch-VL-Reproduction
export RL_SOURCE_ROOT=/absolute/path/to/pinned/Search-VL-RL-8K
export RL_SOURCE_PARQUET="$RL_SOURCE_ROOT/rl_data.parquet"
export RL_BASE_SNAPSHOT=/absolute/path/to/pinned/Qwen3-VL-4B-Instruct

MAIN_ARGS=(
  --run-id formal-main400-s4-attempt4
  --config configs/rl_main.yaml --data data/rl/main400.json
  --source-root "$RL_SOURCE_ROOT" --source-parquet "$RL_SOURCE_PARQUET"
  --base-model-path "$RL_BASE_SNAPSHOT"
  --eval-overlap-manifest data/rl_overlap/eval_v422.json
  --sft-overlap-manifest data/rl_overlap/sft_v422.json
  --sft-adapter outputs/sft_main_imageid_v3/checkpoint-3k/adapter
  --judge-config configs/judge.example.yaml
  --search-config configs/search_backends.example.yaml
  --layout-config configs/layout_parsing.example.yaml
  --collection-gpus 0,1,2,3 --collection-parallelism 4
  --update-gpus 0,1,2,3
)
PYTHONPATH=src python scripts/run_rl_formal_main.py "${MAIN_ARGS[@]}" --stop-after-window 1
# Only after actual four-GPU collection/update/reload/retention/disk inspection:
PYTHONPATH=src python scripts/run_rl_formal_main.py "${MAIN_ARGS[@]}"
```

Replace example provider config locators with the actual private files, keeping
their bytes stable across resume. Secrets stay in existing environment
variables, not command arguments. Start the coordinator with plain Python,
not torchrun; it launches the exclusive four-rank workers itself. Can reduce
collection parallelism to 1..4 for provider troubleshooting without changing
member seeds/policy. Update remains four GPUs.

The first command must return `paused_at_verified_boundary`, `passed=false`,
policy/global step=1, four complete groups/16 trajectories, no active merge,
FULL policy1 and no final PASS manifest. Inspect all rank receipts, logs and
process teardown. Then the second command resumes the same immutable run for
the remaining 99 windows. Only final heavyweight reconstruction/publication
may claim Main400 runtime PASS.

## CPU acceptance (no model/API initialization)

```bash
PYTHONPATH=src python -m pytest tests/test_rl_group.py \
  tests/test_rl_formal_s4_main.py \
  -o addopts= -q -p no:cacheprovider
git diff --check
```

This publication-lock patch runs targeted tests ONLY. Full pytest is deferred until
the user reviews the diff; it is not part of this development invocation.

## SerpAPI reliability v3 and Main continuation blocker

Upload and Google Lens each have an independent bounded five-attempt budget:
wait 5, 10, 20, then 30 seconds, deterministically, without jitter. Timeout,
network failure, HTTP 429/5xx and normalized SerpAPI JSON `provider_error` retry;
401/403, missing credentials, invalid input and malformed responses fail fast.
Serper/Jina and the global `RetryPolicy()` stay at three attempts with their
existing backoff. Exhausted retries remain errors: no fallback, synthetic
success or weakened Formal `ProviderInterruption`/peer teardown/update gating.

Failure metadata includes provider, `failure_stage` (`upload`/`lens`), normalized
`error_type`, `attempt_count`, `upload_attempt_count` and `lens_attempt_count`.
`attempt_count` retains its existing max-of-stages meaning; the two new counts
disambiguate stages. Pre-request validation has zero actual stage attempts.
Safe stage/count diagnostics are also included in the normalized error text,
so the unchanged Formal interruption detail/report can identify the failing phase.
Successful results carry both stage counts; cache hits do not replay those
runtime counters. Request credentials and provider error payloads are not
included in diagnostic metadata or error messages.

**Goal A is implemented; Goal B is BLOCKED and is NOT implemented.**
Do not upgrade the running `formal-main400-s4-attempt4` in place. No parent
policy number is known locally, and no latest-parent resolution is claimed.
AutoDL artifacts are absent: this finding is based on code, not a parent
checkpoint/native/AdamW/RNG/retention inspection.

The existing architecture has no legal inherited-prefix authority:

- `checkpoint.initialize_formal_run` and `_load_anchor` require this run's
  exact original-SFT iteration-zero policy; `run_state.reconstruct_consumed_ledger`
  requires policy zero and a single matching run hash throughout all checkpoints
  and groups. A nonzero copied/resealed anchor is rejected.
- `formal_policy_update.load_formal_actor` FULL-verifies the native checkpoint
  and requires its run identity and PolicyIdentity to match the current run.
  A parent checkpoint cannot be loaded as a child capability without a new,
  explicit, verified continuation authority (not an identity bypass).
- Main recovery indexes historical policies by absolute iteration in a complete
  same-run chain; retention indexes successors as `checkpoints[step]` and binds
  compaction to that run. A suffix beginning at an arbitrary N is not supported.
- Final reconstruction requires 100 local same-run checkpoints, 400 local groups
  and 1600 members. It has no receipt schema/validator to prove a parent prefix
  plus child suffix, or to retain and verify cross-namespace historical evidence.

Consequently, a CLI-only handoff or checkpoint copy cannot satisfy the requested
contract. This patch stops that part as explicitly permitted by the task.
No `--continue-from-run` flag, bootstrap copy, mutable parent resolution, optimizer
reset, adapter-only fallback or arbitrary Main initialization is introduced.
There is intentionally no runnable continuation command yet. A future narrowly
versioned continuation schema must address all four authority layers, frozen
latest-parent resolution, full child-owned artifact materialization/disk guards,
inherited ledger evidence and combined final verification **before** permitting
collection/update. Parent current-window partial groups must then be excluded
from handoff; model/AdamW/RNG/absolute steps/global-position seeds must continue
from the frozen verified boundary. None of those features is claimed by this patch.

Same-run source/semantic identity checks and `eligible_for_main_init=false` remain
intact. Do not treat this v5 provider patch as authorization to restart Main from
SFT3k or resume attempt4 with v5. Keep the active historical run and its namespace
unchanged while the continuation architecture is resolved.

Targeted CPU acceptance for this reliability patch (no full pytest):

```bash
PYTHONPATH=src python -m pytest tests/test_phase3_search.py \
  tests/test_phase4_reliability.py tests/test_rl_formal_s4_main.py \
  tests/test_rl_formal_main_continuation.py tests/test_v3_provenance_guards.py \
  -o addopts= -q -p no:cacheprovider
git diff --check
```

The continuation test file proves existing rejection barriers and parent metadata
immutability only; it does not prove a working handoff or authorize GPU execution.
