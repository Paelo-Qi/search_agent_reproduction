# Formal RL S3 — Smoke20 coordinator

**FORMAL RL S3 SMOKE20 COORDINATOR IMPLEMENTATION COMPLETE — NOT YET SMOKE20 GPU/API PASS**

S2 two-window GPU validation was already PASS on the user's AutoDL 2×A800.
That is a verified user-provided fact, not evidence that S3 Smoke20 has run.
AutoDL S2 artifacts are absent locally: any exact GPU memory, timing, stage
details or concrete runtime artifact example requires those artifacts. This
implementation was validated locally using CPU/static orchestration fixtures
only. No model/GPU, real rLLM/vLLM, API, torchrun, Main400 or S4 was executed.

## Frozen scope

`configs/rl_smoke.yaml` / `data/rl/smoke20.json`: exactly 20 ordered prompts,
n=2 (NOT `rl_main.yaml` n=4), K=4, five complete windows, one PPO/AdamW step
per window, epochs=1, `per_generation_row_mean_v1`, W=2, FSDP2/BF16/FA2.
No scheduler. Existing S2 O/C thresholds, reduction, residency, native save,
fresh reload and runtime LoRA dropout p=0/source metadata .05 are unchanged.
Rollout uses the existing validated static bridge settings: temperature .7,
top_p=1, top_k=-1, 512 new tokens, max_model_len=8192, max_turns=16, TP=1,
processed sampled logprobs, GPU utilization .6. These settings are bound to the
S1 run semantics; they do not modify frozen RL data/config files.

Iteration0 is **only** the original
`outputs/sft_main_imageid_v3/checkpoint-3k/adapter`. Iteration N>0 loads the
previous immutable verified Formal checkpoint's actual adapter for static
merge, and its native model/AdamW/RNG for update. Gate artifacts, diagnostic
S2 runs, staging checkpoints, foreign runs, stale merges and mutated bytes
are rejected. Windows0–3 publish `smoke_continuation`; step5 publishes
`smoke_final`. Every checkpoint has `eligible_for_main_init=false`.
Future Main400 MUST restart from original SFT checkpoint-3k, not Smoke final.

## Lifecycle and authority

The coordinator is CPU-only; heavyweight imports are lazy. A first isolated
two-rank bootstrap calls `load_formal_actor` and publishes the S1 initial
anchor using the **actual** empty AdamW and seeded native RNG identities.
It never invents optimizer/RNG hashes. Run identity binds frozen membership,
source/data/SFT/base/revision/offline snapshot content, execution/rollout/
optimizer/PPO/reward/tool/image semantics, installed integration sources and
software versions. Machine directories remain locators; credentials are
inherited in worker environment only and are not command/identity/report data.

Each window:

1. Recover S1 immutable anchor/groups/checkpoint chain/complete attempt logs;
   ignore disposable `latest.json`, `state.json`, stdout and progress counters.
2. Launch a single-GPU collection worker. Validate the current Formal policy,
   reuse the existing PEFT→plain-HF merge implementation and fresh HF forward,
   then initialize real vLLM/rLLM with that exact static policy. Stale merge
   bindings fail closed; all four groups share the same pre-update policy.
3. Collect each prompt's two whole members with live Phase3 tools, bounded
   existing provider retry/cache, accuracy/query judges and actual token/
   processor/logprob/fatal artifacts. Model-task construction is a whitelist;
   reference answers are available only to reward/judges. Publish each complete
   Formal group by atomic directory publication, never a half-group.
4. Close vLLM and exit the collection process. The Linux coordinator acts as a
   child subreaper, waits, kills/reaps owned descendants and verifies `/proc`
   before proceeding. PID birth identities prevent signaling recycled PIDs.
   This also covers torch elastic's detached rank sessions, not merely the
   launcher's process group. Failure forbids launching the next GPU phase.
5. Launch a fresh `torch.distributed.run --nproc_per_node=2` update worker.
   Reread all four committed groups; use official verl RLOO, formal generation
   rows, deterministic equal rank plans, CPU-backed multimodal DataProto,
   strict independent O/C and the durable `step_may_have_run` marker. Execute
   exactly one official PPO/AdamW update, native save and destructive fresh
   reload/finite multimodal verification. Rank evidence is hashed **inside**
   the immutable checkpoint. Publish checkpoint, then append verified attempt.
6. Destroy actor/update process; recover immutable successor before collecting
   the next ordered four prompts. No collection/update overlap or resident
   actor reuse. Zero-signal windows stop without fabricating an update/reward.

## Safe same-command resume

Rerun the exact command below with the **same run-id and unchanged inputs**.
No `--retry-failed` or mutable state repair is needed. Do not modify immutable
receipts, delete history, substitute data, or change semantic configs/software
in an existing run. Conflicts/tampering/duplicate consumption fail closed.

- Collection/provider interruption: retain committed current-policy groups;
  recollect only missing prompts. Incomplete private `.collect-<UUID>` attempts
  remain forensic; a new UUID recollects all n=2 members. Never splice members,
  pad missing rollouts, turn infrastructure errors into reward=0, or start an
  optimizer with a partial window. Bounded provider retries are unchanged.
- Unpublished update (including `step_may_have_run`): after processes exit,
  S1 rollback/retry plans a new update UUID. The worker actually reconstructs
  verified parent model/native AdamW/RNG (iteration0 restores original SFT and
  initial seed). S2 live reload capability is required before retry. Committed
  groups are reused; uncertain actor/staging is never reused.
- Immutable successor published but index/report/verified-event write failed:
  roll forward from that directory, append the missing verified event only
  if its complete attempt chain exactly matches the checkpoint, and skip the
  already-consumed window. NEVER repeat its optimizer step.
- Immutable contradictions: stop, retain forensic evidence, require inspection.

Outputs are isolated under `outputs/rl_formal_smoke/<run-id>/` and
`reports/rl_formal_smoke/<run-id>/`. Progress reports retain current policy,
group/member/reward/cache/fatal/tool error totals, provider interruption,
subprocess exit/elapsed information; immutable receipts remain the authority.
Worker logs are redacted and informational only.

Finalization rereads all artifacts: 20 unique prompts/groups, 40 complete
members, five ordered K=4 windows, steps/iterations1..5, complete attempt
chains, no unresolved update/double consumption, final policy from checkpoint5,
kind `smoke_final`, no Main initialization eligibility. Runtime rank O/C,
one-step, changed full LoRA, native/optimizer/RNG fresh reload evidence is
rechecked. CPU fixtures cannot publish PASS.

**Only runtime `outputs/rl_formal_smoke/<run-id>/manifest.json` with
`passed=true` after complete reconstruction is final PASS authority.**
It is published LAST, after durable final `reports/.../report.json`. Start and
failure revoke a stale PASS marker first; publication failures (even after
atomic replace) revoke it again. Report-only PASS is not a successful run.

## AutoDL command — do not run before provisioning frozen inputs/providers

Use the S2-verified environment (torch2.8, transformers4.57.1, PEFT0.21.1,
verl0.6.1, vLLM0.11.0, rLLM0.2.1, FA2) and the exact pinned offline base
snapshot. Provision original SFT, prepared frozen Smoke20 + manifest + quality
audit, source images and real provider configs/API credentials separately.
No download or model initialization is performed by the CPU coordinator.

```bash
cd /path/to/OpenSearch-VL-Reproduction
PYTHONPATH=src python scripts/run_rl_formal_smoke.py \
  --run-id formal-smoke20-s3-attempt1 \
  --config configs/rl_smoke.yaml \
  --data data/rl/smoke20.json \
  --source-root /absolute/path/to/pinned/Search-VL-RL-8K \
  --base-model-path /absolute/path/to/pinned/Qwen3-VL-4B-Instruct \
  --sft-adapter outputs/sft_main_imageid_v3/checkpoint-3k/adapter \
  --judge-config configs/judge.example.yaml \
  --search-config configs/search_backends.example.yaml \
  --layout-config configs/layout_parsing.example.yaml \
  --rollout-gpu 0 --update-gpus 0,1
```

Use explicit private config locators instead of the example filenames if your
provider settings differ. Keep those same config bytes/locators for resume.
Secrets belong in the providers' existing environment variables, not CLI args.
Optional `--tool-cache-dir` / `--reward-cache-dir` reuse caches without changing
their existing key semantics. Defaults are isolated inside this Formal output,
not legacy `outputs/rl_smoke` or Gate outputs. Explicit cache locators may not
overlap data/config, SFT, Eval, Gate, S2 diagnostic or immutable Formal receipts.

## Local CPU acceptance

```bash
PYTHONPATH=src python -m pytest tests/test_rl_formal_s3_smoke.py \
  tests/test_rl_formal_contracts.py tests/test_rl_formal_update.py \
  tests/test_rl_formal_s2_validation.py tests/test_rl_run_state.py \
  tests/test_rl_group.py tests/test_rl_rloo.py tests/test_rl_training_batch.py \
  tests/test_rl_reward_judges.py -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

No synthetic fixture result is a real rollout/PPO/Smoke20 PASS. Real five-window
collection, all live provider rewards, sequential process teardown, native
continuation and final publication still require AutoDL GPU/API acceptance.
