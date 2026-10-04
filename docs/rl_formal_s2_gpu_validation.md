# Formal RL S2 GPU validation entrypoint

This is a fixed **two-window diagnostic validation**, not S3, a Smoke20 trainer,
or evidence that formal RL is ready. Local CPU tests never issue GPU PASS.
Frozen configs, datasets, Gate code and S1/S2 schemas are unchanged.

## AutoDL invocation

Use the already verified pinned environment (Transformers 4.57.1, verl 0.6.1,
PEFT 0.21.1, BF16/FA2, two visible CUDA GPUs), real source images and the pinned
local Qwen snapshot. No network/provider/model sampling is performed.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/validate_rl_formal_s2.py \
  --run-id formal-s2-gpu-v1-attempt1 \
  --config configs/rl_main.yaml \
  --data data/rl/smoke20.json \
  --source-root "$RL_SOURCE_ROOT" \
  --base-model-path "$QWEN_BASE_SNAPSHOT" \
  --sft-adapter outputs/sft_main_imageid_v3/checkpoint-3k/adapter
```

The offline **file-map** SHA256 (not an individual weight-file hash) must equal
`5c655eb7bd80fb959428f2194acb217424d0e658dfd54ff1eb27f30bcedc236b`.
The original complete checkpoint-3k lineage and source dropout .05 are checked;
live RL LoRA dropout is p=0. Constructor/load/update are existing production
FSDP2, BF16/FA2, 36-layer gradient-checkpointed, LoRA-only primitives.

## Fixed diagnostic inputs and identity

First four frozen Smoke20 source prompts (optional `--prompt-start`) supply real
questions/images, NOT reference answers in model inputs. Real pinned processor
renders and encodes PIL images, saves its full CPU inputs including all multimodal
metadata. Short `Yes.`/`No.` completions, and an additional `Proceed.` generation
in window1, are encoded by the actual tokenizer plus EOS. Saved R=-1 is a finite
artificial diagnostic value, not a sampled model probability.

Each K=2/n=2 group has deterministic rewards .9/.1; **official verl RLOO** gives
nonzero +/- .8 advantages. Window0 has 4 logical/4 physical rows, 2 per rank;
window1 has 5 logical rows uniformly replicated twice into 10 physical rows,
5 per rank. Its extra row is a new real processor encoding of the original
question/image, prior diagnostic completion and a diagnostic follow-up user turn.
This is not formal collection, fatal/tool behavior, or the Smoke20 K=4/n=4 identity.

The unchanged historical group schema requires `token_origin=vllm.RequestOutput`.
Here that field is **wire compatibility only**: every step explicitly records
`actual_token_origin=deterministic_processor_encoded_completion`,
`diagnostic_fixture=true`, `rollout_executed=false`. Reports/run semantics also
identify this recipe. No vLLM/rLLM/API/Judge/reward provider is started or called.

## Two iterations and evidence

Original SFT actor -> formal committed groups/window/rank-local DataProto -> two
independent O/C recomputations -> existing strict alignment guard -> one official
verl vanilla PPO/RLOO AdamW step (lr=1e-6, weight decay=0, micro1, one epoch,
clip .2/.28, entropy0) -> native staging save -> destroy old actor/GC/empty_cache
-> fresh native model/optimizer/RNG load -> finite real multimodal forward -> S1
immutable verified checkpoint1. No scheduler or alternative loss is implemented.

The fresh verification actor is destroyed. Iteration1 constructs **again** from
checkpoint1 with real native continuation, no reseed: nonempty AdamW moment step1,
model/optimizer/native-RNG hashes must match the saved rank state. Then the second
window follows the same update/save/fresh-forward/commit path to checkpoint2.

Both rank reports include source/run/software/git/GPU identity, row plan and
uniform multiplicity, official RLOO, loss/gradient/frozen/p0 audit, O/C and
informational R/O, optimizer steps [1,2], saved/reloaded native RNG/model/optimizer
hashes, immutable checkpoint identities and fresh forward receipts. O/C uses
the unchanged bound <.1, ratio [.8,1.28], zero initial clipping, finite mask-bound
token counts. R/O magnitude is not a policy-alignment gate (structural finite
evidence is required for these deliberately finite short diagnostic outputs).

Cross-rank consistency hashes **full reconstructed LoRA DTensors** in deterministic
name order, never unequal local shard fingerprints. We also require weights changed.
Residency observes the original nested CPU carrier before/after and actual active
microbatch CUDA pixel/grid devices/shapes (including object-array containers).
CUDA allocated/reserved/peak allocated/peak reserved bytes are sampled after
actor load, before O, after O/C, after update, after save and after fresh reload.
No hardcoded VRAM threshold is used.

## Artifacts and fail-closed semantics

- `outputs/rl_formal_s2_validation/<run-id>/`: S1 anchor/groups/attempts, verified
  `checkpoints/policy-000001` and `policy-000002`, final `manifest.json`.
- `reports/rl_formal_s2_validation/<run-id>/`: false-first rank progress and final
  `report.json` aggregating both ranks.

All runs are one-shot: any existing output/report path requires a **new run-id**,
including after failure. No production resume/retry is added. Failures after a
possible step preserve attempts/native artifacts for forensics, never auto-retry.
Ordinary stage errors are collected across ranks; fatal process/NCCL failure can
leave only false last-stage evidence and relies on torchrun/timeout teardown.
No failure cleanup barrier is used. Source/config/data/Gate paths are protected.

Both checkpoints use the existing `smoke_continuation` schema slot only;
`eligible_for_main_init=false` stays unchanged. Diagnostic identity/artifacts must
never be used as formal Smoke20/main initialization. The wrapper refuses reuse
even though S1's generic same-run checkpoint resume eligibility is unchanged.

Only after both ranks/all checks complete and process-group teardown succeeds:
atomically write final passed report, then publish output manifest **last**.
Publication error revokes the manifest, preserves false failure evidence and exits
nonzero. A report alone is not PASS authority; require the final manifest.

Local status: **FORMAL RL S2 GPU VALIDATION ENTRYPOINT COMPLETE**.
GPU execution/VRAM/timing are not verified locally. Actual AutoDL artifacts are
required before claiming an S2 GPU validation PASS.
