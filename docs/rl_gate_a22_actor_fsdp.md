# Gate A2.2 — real distributed actor infrastructure

Status: **READY FOR AUTODL EXECUTION — NOT YET PASSED**. Windows CPU tests do
not establish that Qwen3-VL, CUDA, FlashAttention or verl actually run together.
This gate neither starts formal RL nor changes its frozen contracts/data.

## Scope and version contract

The same `scripts/validate_rl_actor_fsdp.py` runs under `torchrun` with 2, 4 or
another reasonable number of ranks. Each rank processes one distinct sample
from the first `world_size` records of the quality-clean `smoke20.json`. The full
small JSON is read for checksum validation, but only these records have images
opened/tokenized. `--max-samples`, if supplied, must be >= launcher world size;
it never creates additional steps. World size is read from the process group,
not from SFT's historical two-GPU training setting or logical RLOO group size.
`rollout_n` is not used by this gate.

Verified API assumptions: verl 0.6.1, Torch 2.8.x, Transformers 4.57.1, PEFT
0.21.1; a mismatch fails closed. Use the existing **RL environment**, not the
SFT/Windows environment's Transformers 5.17.0. No dependency files are changed
and the script never installs anything. FlashAttention-2 must already work.

The selected path is **FSDP2**, which verl 0.6.1 explicitly supports; this is
not a claim that FSDP2 universally supersedes FSDP1 for every model. The gate
mirrors the supported FSDP2 branch of the
[official ActorRolloutRefWorker](https://github.com/volcengine/verl/blob/v0.6.1/verl/workers/fsdp_workers.py)
without constructing Ray workers or importing a rollout backend into the gate.
Actual reused components:

- [`apply_fsdp2`, `fsdp2_load_full_state_dict`, `get_shard_placement_fn`, `get_fsdp_full_state_dict`](https://github.com/volcengine/verl/blob/v0.6.1/verl/utils/fsdp_utils.py):
  the actual verl wrapping/loading/gathering functions, default model
  `_no_split_modules` policy, full-world one-dimensional mesh, reshard after
  forward, BF16 forward parameters with FP32 gradient reduction, no CPU offload.
- [`DataParallelPPOActor`](https://github.com/volcengine/verl/blob/v0.6.1/verl/workers/actor/dp_actor.py):
  real actor object owns the sharded model and optimizer. Remove-padding,
  sequence parallelism, compiled entropy and fused policy kernels are disabled
  for this minimal gate, not frozen as future RL settings.
- [`FSDPOptimizerConfig` / `build_optimizer`](https://github.com/volcengine/verl/blob/v0.6.1/verl/workers/config/optimizer.py):
  Torch AdamW, only trainable LoRA parameters, gate LR 1e-6, weight decay zero.
- [`FSDPCheckpointManager`](https://github.com/volcengine/verl/blob/v0.6.1/verl/utils/checkpoint/fsdp_checkpoint_manager.py):
  native model/optimizer/RNG shard save/load and FSDP metadata, scheduler=None.

The thin compatibility portion is standalone lifecycle/model construction and
the temporary supervised forward/backward/step. We do **not** call
`DataParallelPPOActor.update_policy`, launch `ActorRolloutRefWorker`, exercise
PPO/RLOO loss, or validate the full verl trainer. Merely passing this gate does
not establish rollout, reward, rLLM, vLLM synchronization or RL correctness.

## Input, objective and freezing

The source adapter must be the verified, completed `checkpoint-3k` with lineage
`main_a_1k → main_b_2k`. Existing `build_rl_lineage` validates metadata,
checksums, pinned base revision, current tool protocol and LoRA settings.
The smoke manifest must be schema v3, checksummed, and have the frozen quality
`status == ok` prefix revalidated against the full local quality audit bundle.
This does not replace the already completed full RL data preflight.

For each rank, the original question and `reference_answer` become an in-memory
`human → gpt` sample, with original `image_relpaths` resolved against
`--source-root`. Source records/files are never rewritten. Existing
`load_processor`, `OpenSearchVLCollator`, `build_messages`, image loading,
chat template and assistant-only label mask are reused. Max length remains
32000; image max pixels remains 262144. Input IDs, attention mask, labels,
pixel values and image grid must exist; only shape/dtype summaries are printed.

**Temporary supervised objective != RL objective.** The model's standard
assistant-token loss is used solely to prove that a real multimodal backward
can update an actor. There is one forward, one backward, one AdamW step and
zero-grad; no accumulation, scheduler, clipping, reward or rollout. No decrease
in validation loss is required (fresh validation uses eval mode).

The base is loaded in BF16; PEFT loads the existing trainable adapter (no new
LoRA initialization). PEFT's trainable adapter master dtype is reported, not
silently cast to BF16 to bypass wrap issues. The FSDP2 mixed-precision forward
uses BF16. The existing vision freeze and parameter audit reject full-finetune
or vision/projector trainables; optimizer ownership must exactly match LoRA.
`activate_sft_training_mode` checks the real Qwen language stack and all 36
decoder layers' train mode/checkpointing; FSDP2 wrapping of all 36 is checked.

## Save, destroy, reload and PASS

Output is separate from all SFT/RL inputs and refuses overwrite/resume. Native
verl artifacts are under `distributed/` (model, optim, extra_state per rank,
processor/config and `fsdp_config.json`). All ranks participate in verl's
full-state gather; rank0 then exports only PEFT weights/config under `adapter/`.
The adapter file fingerprint is the same shared algorithm used by formal SFT
adapter provenance; a gate adapter does not pretend to be a formal SFT stage.
Native checkpoint files include frozen model shards, so allow disk space for
the full model plus optimizer, not merely a small adapter.

Before/after comparisons use every local LoRA shard on every rank. All present
LoRA gradients must be finite and at least one must be nonzero; all updated
LoRA values must be finite and at least one must change. The optimizer has no
base/vision parameters, so it cannot update those frozen weights.

The original actor/model/optimizer references are deleted, GC/empty-cache runs,
and weak references must confirm destruction before a barrier. A fresh base
and **saved adapter** are loaded and FSDP2 wrapped. All LoRA local shard values
must exactly match their saved pre-destruction values, including changed ones,
and the saved file fingerprint must match. Then the native verl checkpoint is
reloaded into this fresh actor and optimizer; LoRA equality and nonempty restored
optimizer state are checked again. A real fresh multimodal forward must have
finite loss. No custom long-term RL checkpoint-resume protocol is introduced.

All required checks must be literal true on **every** rank. Only after collective
success does rank0 atomically publish `passed=true` to `gate_manifest.json` and
the report. Reports include logical model/revision, input/output fingerprints,
lineage, software, world size, sample IDs, losses, parameter counts, memory peaks
per rank, stages and checkpoint file hashes. Machine snapshot/source locators
are informational only, outside the deterministic identity hash. World size
is inside the identity; ws2 and ws4 are independent hardware records.

## AutoDL commands (do not run locally)

Run from the repository root in the existing RL environment. Set locators to
the actual local snapshot and **root of source image_relpaths**. Do not use an
image archive filename or assume an absolute AutoDL directory in code.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=src
export BASE_SNAPSHOT=/your/local/pinned/snapshot/ebb281ec70b05090aa6165b016eac8ec08e71b17
export RL_SOURCE_ROOT=/your/local/Search-VL-RL-8K/source-root

# Minimum supported hardware validation: 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  scripts/validate_rl_actor_fsdp.py \
  --config configs/rl_main.yaml --gate-config configs/rl_gate_a22.yaml \
  --data data/rl/smoke20.json --source-root "$RL_SOURCE_ROOT" \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --base-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --base-model-path "$BASE_SNAPSHOT" \
  --sft-adapter outputs/sft_main_imageid_v3/checkpoint-3k/adapter \
  --output-dir outputs/rl_gate_a22/ws2 --report-dir reports/rl_gate_a22/ws2 \
  --max-samples 2 --seed 20260506 --local-files-only

# Recommended additional validation: 4 GPUs, EXACT same implementation/config
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  scripts/validate_rl_actor_fsdp.py \
  --config configs/rl_main.yaml --gate-config configs/rl_gate_a22.yaml \
  --data data/rl/smoke20.json --source-root "$RL_SOURCE_ROOT" \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --base-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --base-model-path "$BASE_SNAPSHOT" \
  --sft-adapter outputs/sft_main_imageid_v3/checkpoint-3k/adapter \
  --output-dir outputs/rl_gate_a22/ws4 --report-dir reports/rl_gate_a22/ws4 \
  --max-samples 4 --seed 20260506 --local-files-only
```

Omit `--base-model-path` to use the existing offline HF cache with pinned logical
name/revision. Offline is enforced even when `--local-files-only` is omitted.
Missing cache/dependencies fail; the gate cannot download a replacement.
Only use fresh output/report paths; after a failed invocation preserve the
report/checkpoint for diagnosis and run a new path (e.g. `ws2-attempt2`).

Expected success: per-rank memory/model/batch logs, nonzero gradients and changed
LoRA tensors, native checkpoint/adapter export, weakref-confirmed destruction,
fresh reload/forward, and one rank0 `Gate A2.2 PASS` with a passed report.
ws2 PASS does not imply ws4 PASS or decide final training topology/throughput.

## Failure localization and next gate

`[GATE FAIL]` records rank/stage/error; memory logs show allocated/reserved and
both peaks without requiring NVML. Stages distinguish load/wrap, collate,
forward, backward, optimizer, save, destroy and reload. Ordinary exceptions
are collectively aggregated into a false report and nonzero exit. Hard process
death or an interrupted NCCL collective may prevent a final error aggregate;
the initial/stage report remains **passed=false**, and torchrun/NCCL timeout
(180 seconds) surfaces the failure. Cleanup destroys the process group without
a failure-path barrier. Preserve torchrun logs alongside the report in that case.

After a real PASS, Gate B should separately validate actor-to-rollout weight
synchronization (the known static PEFT merge workaround), real multimodal
generation plus runtime tool-protocol integration and interruption handling,
without yet claiming an RL update loop. Do not infer rollout readiness from
this gate, and do not start main400/RL smoke until those contracts are validated.

CPU checks:

```bash
PYTHONPATH=src python -m pytest tests/test_rl_actor_gate.py -o addopts= -q -p no:cacheprovider
PYTHONPATH=src python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```
