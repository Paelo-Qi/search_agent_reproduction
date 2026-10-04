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

`formal-s4-main400-v2` binds ordered source identities, frozen data/manifest and
config hashes, SFT lineage, pinned offline base, software/source hashes,
algorithm/rollout/reward/tool contracts, seed, retention contract/milestones and
seed scheme. Every member's seed is `initial_seed + 4*global_position + index`.
GPU locators, collection parallelism, pause and disk controls are operational.
The validated stack remains torch 2.8.x, transformers 4.57.1, PEFT 0.21.1,
verl 0.6.1, vLLM 0.11.0, rLLM 0.2.1, with installed FA2/source identities bound.

### Main-only context budget (no truncation)

The rollout identity additionally binds `remaining-context-no-truncation-v1`.
Frozen numeric settings stay `max_model_len=8192`, `max_new_tokens=512`.
After FULL HF multimodal processing (`truncation=False`), each call gets
`min(current_sampling.max_tokens, 8192 - processor_token_length)` tokens.
Only a deep copy of that call's SamplingParams changes; all other fields,
member seeds, exact prompt IDs and saved vision tensors remain unchanged.
Gate B/C and S3 opt out by default and retain their strict fixed-budget check.

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

The first-window `formal-main400-s4-attempt1` failed under old v1 behavior;
retain it as forensic evidence. Its sealed identity is incompatible with v2:
do not resume, migrate or edit it. Start a NEW `formal-main400-s4-attempt2`.
This patch is CPU/static validated only, NOT Main400 4xA800 GPU/API PASS.

## Exclusive process lifecycle and resume

Metadata recovery -> ONE shared single-GPU static merge/fresh HF validation ->
merge process/descendants dead -> up to four isolated single-GPU workers ->
one complete n4 group per prompt, atomic publication -> ALL workers/descendants
dead -> exact K4 same-policy groups -> durable merge-retirement authorization
and exact deletion -> fresh four-rank update -> immutable verified successor
and verified attempt -> retention -> next window. No resident GPU actor crosses
a phase boundary. Linux subreaper supervision tracks PID birth identities,
reaps detached descendants and forbids the next phase if cleanup fails.

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

## Next AutoDL acceptance: REAL run, pause after window 1

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
  --run-id formal-main400-s4-attempt2
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
PYTHONPATH=src python -m pytest tests/test_rl_context_budget.py \
  tests/test_rl_gate_b.py tests/test_rl_gate_c.py tests/test_rl_rollout_sync.py \
  tests/test_rl_workflow_adapter.py tests/test_rl_formal_s4_main.py \
  tests/test_rl_formal_s3_smoke.py tests/test_rl_formal_contracts.py \
  tests/test_rl_formal_update.py tests/test_rl_formal_s2_validation.py \
  tests/test_rl_run_state.py tests/test_rl_group.py tests/test_rl_rloo.py \
  tests/test_rl_training_batch.py tests/test_rl_reward_judges.py \
  -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```
