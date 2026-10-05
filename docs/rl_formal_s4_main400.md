# Formal RL S4 Main400

Implementation/CPU validation is not Main400 GPU/API PASS. The user has verified
S2 and S3 on AutoDL; S3 completed 20 prompts, n=2, five updates, using about
86 GB on disk. No other AutoDL artifact fields, memory peaks or timings are
inferred here. Local frozen Main400 inputs and runtime artifacts are not
provisioned; CPU fixtures never authorize runtime PASS.

## Frozen experiment and identity

S4 uses `configs/rl_main.yaml`, the existing quality-selected Main400 and four
ordered 100-row shards. Full pinned-source/quality/overlap/image preflight is
mandatory before bootstrap. Fresh-run iteration zero ONLY uses the original
`outputs/sft_main_imageid_v3/checkpoint-3k/adapter`; Smoke/Gate/S2/staging
initialization is forbidden. The separate, explicit v6 continuation authority
below may inherit a verified nonzero v4/v5 Main boundary; it is not arbitrary init.

400 prompts / n=4 / K=4 / W=4 / 100 windows; one AdamW step per window,
lr=1e-6, weight decay=0, PPO epochs=1, per-generation-row mean weighting,
clip=.2/.28, entropy=0, vanilla loss, FSDP2/BF16/FA2. Original saved LoRA
dropout stays .05; live RL dropout stays zero. Reward, reference isolation,
fatal semantics, live tools and judges are unchanged. The shared S2 production
update method performs official verl RLOO, O/C, LoRA change/match, native
model/AdamW/RNG save and destructive fresh multimodal reload.

`formal-s4-main400-v6` binds ordered source identities, frozen data/manifest and
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

## Historical v4 AutoDL acceptance commands (NOT commands for this v6 checkout)

The commands below document the original attempt4 v4 bootstrap. Do NOT execute
them with this v6 checkout or reuse attempt4 with modified sources. They are
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

This continuation patch runs targeted tests ONLY; full pytest is not part of
this development invocation.

## Frozen SerpAPI reliability v3 (Goal A)

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

## Verified cross-version continuation (Goal B, v6)

`formal-main-continuation-v1` supports original, non-continued Main v4/v5 parents
only. Gate/Smoke/S2/CPU/unverified parents, nested continuation, in-place upgrades
and arbitrary adapter initialization fail closed. `eligible_for_main_init=false`
is unchanged. Never resume attempt4 with v6 or edit/reseal its historical identity.

First stop and reconcile the parent under its original code. Handoff acquires
the parent's existing coordinator/formal/publication locks read-only and rejects
active locks, unpublished update staging, ambiguous attempts or corrupt latest
state; it does not silently fall back to an older policy. It dynamically verifies
the original SFT anchor, complete checkpoint/attempt/retention chain and actual
consumed groups, then FULL-verifies the latest checkpoint's SHA inventory and W4
rank evidence. The actual parent policy N is unknown locally and is never hardcoded.

The sealed child run binding and `continuation/receipt.json` freeze N/global step,
parent run/checkpoint/policy identities, original artifact role/file hashes,
exact first 4N ordered prompts/group identities and prefix manifest hashes.
Only enumerated provider/reliability and continuation control-plane source files
may differ. Dataset/ordered sources, base/SFT, execution, PPO/AdamW/reward/judge,
generation/context/tool/image protocol, seed, retention and software are compared
exactly. Goal A's five-attempt/backoff strategy is not changed by this handoff.

The child disk guard covers bootstrap copy plus receipt/anchor metadata. All
adapter, native model, AdamW, RNG and bound metadata files are SHA-verified and
copied (not symlinked) to child staging. Existing durable publication fsyncs all
bytes and directories, writes the receipt last in staging, then atomically
publishes `continuation/bootstrap/policy-NNNNNN` with its ORIGINAL parent
checkpoint manifest intact. The child anchor expresses a distinct inherited
logical policy and explicit cross-run edge; it does not launder parent identity.
Failed staging is forensic, never authority. A post-receipt/pre-anchor crash
finishes from the frozen child-owned receipt without re-resolving parent latest.

Materialization alone NEVER authorizes collection. A fresh four-rank bootstrap
uses the original native loader with a narrow verified bootstrap capability.
It loads actual model/LoRA, AdamW and RNG and verifies all four rank snapshots,
global step N and execution/dropout contracts before publishing the separate
`continuation_reload/receipt.json` activation. All workers must exit under the
unchanged supervisor before the next phase. Missing/mismatched state rejects;
there is no optimizer/RNG/step reset or SFT fallback. Only this inherited initial
handoff uses the capability; child policy N+1 onward uses ordinary same-run
checkpoint loading and the unchanged S2 update implementation.

The ledger starts with the verified prefix, not a fabricated `N*4` counter.
Parent current-window partial groups are NOT copied or reused. Window N+1
collects all K4 prompts afresh, then updates to absolute step/policy N+1.
Seeds remain `initial_seed + 4*original_global_prompt_position + rollout_index`.
`--stop-after-window 25` is absolute: N<25 runs N+1..25; N>=25 adds no update
(a completed policy100 may undergo final verification). Parent namespace,
reports, caches and retention are never written; explicit parent cache locators
are rejected. Keep parent history as forensic/final-audit evidence.

Routine child recovery validates its frozen metadata authority and current own
checkpoint/bootstrap, not parent historical large bytes or a newly resolved N.
Later parent progress cannot drift the frozen boundary. Child retention uses
absolute steps and only child-local successors; it never compacts parent or its
own inherited bootstrap. Final heavyweight audit independently verifies the
frozen parent prefix and child suffix as EXACT ordered 400 prompts, 100 windows/
steps, 400 groups/1600 members and final policy100, with retained milestones
25/50/75/100 and no partial collection, active merge or unresolved update.
Parent checkpoint/group evidence must remain accessible and valid at final
audit, including its legally authorized compaction and full milestone artifacts.
The final report/manifest records parent identity, frozen N, inherited/local
counts, receipt hash and old/new reliability semantics; it cannot claim a fresh
child-only history. Final PASS publication ordering remains unchanged.

### Future AutoDL handoff command (NOT executed locally)

Use the same frozen locators/config bytes as the parent; remove the historical
`--run-id formal-main400-s4-attempt4` entry from the `MAIN_ARGS` array above.
Choose a NEW child run ID, after stopping the parent under its original code:

```bash
export PARENT_RUN=formal-main400-s4-attempt4
export RUN_ID=formal-main400-s4-continuation-v6-attempt1  # choose an unused NEW ID
PYTHONPATH=src python scripts/run_rl_formal_main.py \
  "${MAIN_ARGS[@]}" --run-id "$RUN_ID" \
  --continue-from-run "$PARENT_RUN" --stop-after-window 25
# Same child resume: repeat the command, keeping --continue-from-run unchanged.
# After inspecting the real verified boundary, omit ONLY --stop-after-window:
PYTHONPATH=src python scripts/run_rl_formal_main.py \
  "${MAIN_ARGS[@]}" --run-id "$RUN_ID" --continue-from-run "$PARENT_RUN"
```

No manual N is needed or accepted. A changed CLI parent/semantic input fails
closed; never edit receipts to make resume pass. Provision extra disk for the
child-owned FULL bootstrap in addition to normal update/merge headroom. Parent
history remains a final-audit dependency, not a routine recovery native loader
dependency. This first schema does not support continuation of a continued child.

Targeted CPU acceptance only (includes retention/run-state regressions):

```bash
PYTHONPATH=src python -m pytest \
  tests/test_rl_formal_main_continuation.py tests/test_rl_formal_s4_main.py \
  tests/test_rl_formal_contracts.py tests/test_rl_formal_update.py tests/test_rl_group.py \
  -o addopts= -q -p no:cacheprovider
git diff --check
```

CPU tests cover multiple dynamic N values, fail-closed latest/authority/native
reload, immutable parent bytes, crash recovery, absolute seeds/stops, milestone
crossing and real fixture prefix+suffix policy100 accounting. They do not prove
AutoDL native compatibility, actual disk/VRAM/timing or runtime PASS. AutoDL
artifacts are absent; those details can only be audited from code and the user's
provided run conclusions. No GPU/model/API/training was executed for this patch.
