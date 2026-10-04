# Formal RL S2: production update primitives

Status: **FORMAL RL S2 UPDATE PRIMITIVES COMPLETE** (CPU contracts only).
S2.1: **FORMAL RL S2.1 OFFLINE LOCATOR HARDENING COMPLETE** (CPU contracts only).
Not Smoke20 ready, not GPU verified, not formal training PASS. No coordinator,
collector, run CLI, rollout/API/Judge execution, scheduler or final PASS manifest
is introduced. Frozen configs, data, reward/RLOO/fatal semantics, membership and
LoRA execution versions are unchanged. Gate A/B/C retain their orchestration.
The supplied AutoDL Gate C PASS is accepted as a runtime fact; without those
artifacts we do not claim its memory, timing, stage details or JSON examples.

The subsequent isolated [S2 GPU validation entrypoint](rl_formal_s2_gpu_validation.md)
wraps these unchanged APIs for two diagnostic K=2/n=2 windows on two GPU ranks.
It is not S3 or a Smoke20 trainer; CPU completion does not establish GPU PASS.

## Dataplane APIs, not a training entry point

In `rl/formal_policy_update.py`:

- `load_formal_actor`: actual source or native load, returns an owned
  `LoadedFormalActor` plus an in-process `FormalReloadReceipt`.
- `update_formal_window`: validates one K-group window and its actual batch,
  computes O/C, durably marks uncertainty, calls the existing audited official
  PPO update once, and returns checkpoint-staging-phase evidence.
- `save_formal_staging`: official native save plus PEFT export at the actual
  global optimizer step, with complete role maps and per-rank runtime snapshots.
- `fresh_reload_staging`: destroys the old actor/optimizer, fresh constructs,
  loads all native state, verifies and executes a finite multimodal forward,
  then returns fresh actor and S1.1-compatible reload evidence.

The caller must supply an initialized distributed process group/device mesh,
processor, authoritative S1 run identity, and committed groups. Production
defaults reuse `construct_rl_actor`, `checkpoint_manager`, `save_checkpoint`,
`configure_one_update` and `audited_policy_update`; there is no alternate PPO
loss implementation. Backend injection is explicitly CPU-fixture-only, cannot
issue runtime checkpoint evidence, and is not a fallback for missing verl.
The production path requires installed verl **0.6.1**.

The intended one-window sequence is:

```text
verified policy N + actual model/optimizer/RNG load receipt
  -> K committed parent-policy groups + official grouped RLOO
  -> formal_training_rows -> deterministic_rank_plan
  -> build_rank_local_dataproto (CPU)
  -> O (one actor batch call) -> C (second independent call)
  -> all-rank strict O/C checks -> durable step_may_have_run
  -> one audited official PPO update
  -> save_formal_staging -> destroy old -> fresh native reload + MM forward
  -> S1 checkpoint manifest inputs (NO final commit or run PASS here)
```

Immutable commit and policy/ledger advancement remain S1 APIs, invoked by a
future S3 coordinator. An uncommitted staging description or reload evidence
does not authorize the next policy iteration. The verification actor is not a
ready-to-update capability for an unpublished successor. Dispose of it before
calling `load_formal_actor` for the newly published verified checkpoint; do not
retain overlapping actor/module/optimizer references.

## Actual rows and masks

`formal_training_rows` binds the window/run/policy/groups and rechecks sealed
RLOO outcomes against trajectory rewards and the existing fatal clamp. No
baseline is recomputed across prompts or generation rows. Every usable actual
generation has a unique logical hash of `(group_id, prompt_id, rollout_index,
step_index, member_id)`; its trajectory's final advantage is broadcast to all
its generated tokens. R remains `rollout_log_probs`, never `old_log_probs`.

Context, observations and left/right padding are not supervised. Response masks
cover actual assistant/tool-call generation only. A fatal formal member must
provide an explicit integer `fatal_step` in its immutable member metadata; a
missing/invalid cutoff fails closed instead of guessing from text. The fatal
generation/prefix remains trainable. Entirely post-fatal generations retain
all-zero forensic masks but are omitted from PPO as in Gate, not introduced as
dummy masked rows. Physical rank replication never adds RLOO members.

The batch builder rereads actual committed group receipts, verifies saved
processor file hashes, and requires exact prompt ID agreement. It preserves
processor attention and uses the real model's CPU `get_rope_index` for full
prompt+actual-response M-RoPE. Saved positions, if provided, must agree. Qwen
versions requiring `mm_token_type_ids` must receive the saved processor field;
it is extended with text modality for actual response IDs and padded together
with the sequence. Missing modality metadata fails closed, never retokenizes
or fabricates a multimodal prompt. Extra processor vision fields are retained.

`FormalBatchReceipt` binds the actual DataProto instance, inputs, masks,
advantages, R, row IDs, window and rank plan. Mutation cannot enter O/update.
All DataProto tensors and nested multimodal tensors remain CPU-backed outside
active microbatches. `materialize_actor_microbatches` intercepts only official
`_forward_micro_batch`, rejects batch sizes other than one, recursively moves
the active microbatch's nested vision tensors to its input device, and restores
the method in `finally`. Original CPU carriers are not mutated. Device copies
belong only to the active forward/backward graph, never a window-wide CUDA
multimodal array. This does not rely on `DataProto.to()` moving nested objects.

## Rank plan and objective scaling

The existing S1 deterministic plan uses `LCM(L, W)` physical rows and uniform
multiplicity, with equal rank-local counts. The batch receipt records logical
and physical counts, replication factor, all assignments and every physical
index's logical/rank mapping. No drop-last, unequal weights or dummy rows.
Sequence padding uses common window maxima; response token counts may differ.

Pinned official sources inspected before implementation:
[actor update/compute](https://github.com/volcengine/verl/blob/v0.6.1/verl/workers/actor/dp_actor.py),
[loss aggregation](https://github.com/volcengine/verl/blob/v0.6.1/verl/trainer/ppo/core_algos.py),
[native checkpoint manager](https://github.com/volcengine/verl/blob/v0.6.1/verl/utils/checkpoint/fsdp_checkpoint_manager.py).

For fixed microbatch=1, `ppo_mini_batch_size=local_row_count`, epochs=1,
shuffle=false, dynamic batching=false and `seq-mean-token-mean`, official verl
accumulates each row loss with `1/local_row_count`. Under W-way FSDP gradient
averaging, the result is the global physical row mean. Uniform multiplicity m
gives `sum(m * row_loss)/(m * L) = sum(row_loss)/L`, preserving
`per_generation_row_mean_v1`. No custom compensating scale is added. CPU tests
compare scalar objectives AND gradients for multiple L/W/replication cases;
actual distributed FSDP2 gradient equivalence is still an AutoDL requirement.

## Continuation and reload authority

### S2.1 canonical identity vs offline runtime locator

Both `load_formal_actor` and `fresh_reload_staging` require explicit
`canonical_config` and `runtime_config`. Canonical identity is always
`Qwen/Qwen3-VL-4B-Instruct` at revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`; it is used for execution-contract,
original SFT/adapter lineage and run/policy validation. Actual actor construction
receives the runtime config, whose `model.name_or_path` is a local snapshot.
Every other config field must equal the canonical config, including revision,
LoRA, FA2, image limits, freeze flags and other training settings. No config or
source adapter is rewritten by these primitives.

`offline_snapshot_files` is the CPU-only validator extracted from Gate C, whose
existing checks and relative file-map identity remain unchanged. Formal callers
use `strict=True`: directory and `config.json` must exist, model type must be
`qwen3_vl`, text layers must be 36, a present `_commit_hash` must equal the pinned
revision, and `model*.safetensors` must exist and be nonempty. All root-level
JSON and safetensors files are hashed in deterministic filename order. An
absent `_commit_hash` is not guessed from the directory name: canonical revision
and the already-frozen content digest remain mandatory.

Before creating the S1 run/policy anchor or collecting any groups, the caller
binds the validated file-map digest in the existing extensible S1 semantics:

```python
from copy import deepcopy
from opensearch_vl_repro.eval_subset import canonical_json_sha256
from opensearch_vl_repro.rl.offline_snapshot import offline_snapshot_files

# canonical_config is the unchanged formal SFT configuration.
files = offline_snapshot_files(local_snapshot, revision=canonical_config["model"]["revision"], strict=True)
semantics["base_model"] = {
    "name": canonical_config["model"]["name_or_path"],
    "revision": canonical_config["model"]["revision"],
    "offline_snapshot_sha256": canonical_json_sha256(files),
}
# Build the S1 run with these semantics; absolute paths may live in locators only.
runtime_config = deepcopy(canonical_config)
runtime_config["model"]["name_or_path"] = str(local_snapshot)
# load_formal_actor(run, canonical_config=canonical_config,
#                   runtime_config=runtime_config, ...)
# fresh_reload_staging(..., canonical_config=canonical_config,
#                      runtime_config=runtime_config, ...)
```

No S1 schema/version change is needed: all semantic fields already participate
in `training_behavior_fingerprint` and `run_identity_sha256`; the base content
digest also reaches initial native and policy identity. Identical snapshot
contents in two directories therefore preserve logical identities. Different
contents, missing binding, noncanonical base/revision, or non-locator runtime
drift fail closed before construction. Previously unbound S1 run identities
remain schema-valid for historical/control-plane use, but S2.1 refuses to load
them; it never silently reseals/upgrades an existing run anchor. Bind the content
in a new authoritative run before collecting groups.

The in-process reload receipt includes snapshot files in its live, local source
inventory, so later file mutation also invalidates update authority. Its local
inventory digest is **not** a run, training-behavior or policy fingerprint.
Fresh verification revalidates the content binding before destroying the old
actor and again before issuing evidence; it also checks the original canonical
config hash. N>0 reload validates the same canonical/base-content binding and
uses the same validated runtime locator. Verified checkpoint run comparison uses
the existing S1 semantic identity check, not machine-specific locator equality.
Only original complete formal checkpoint-3k SFT is valid at iteration zero;
relocating a base snapshot does not authorize a Gate/diagnostic adapter.

Iteration zero loads the original complete checkpoint-3k SFT adapter using the
existing full RL/SFT lineage validator: base/revision, exact source adapter and
metadata hashes, stage completion, LoRA and lineage must match run semantics.
The directory is `checkpoint-3k`; the metadata stage is its SFT stage (normally
`main_b_2k`), not an invented RL iteration. The caller supplies the frozen
initial seed. After actual construction, fresh AdamW and native-manager RNG
snapshots are gathered; their complete rank-map hashes become S1 initial policy
optimizer/RNG identities. A supplied initial policy must match those identities.
Initialize the S1 anchor using this actual policy, not placeholder fingerprints.

For N>0, an adapter alone is rejected. `read_verified_checkpoint` must validate
the immutable S1 manifest, complete file inventory/role hashes, scope and run;
`checkpoint_policy` must equal the requested policy. Official manager restores
model, AdamW and native RNG/extra state. Actual local parameter/optimizer/RNG
fingerprints must equal the hash-bound saved per-rank runtime metadata, and
native LoRA weights must match the independently loaded PEFT export. Restored
optimizer LR/weight decay must match frozen run semantics.

The official extra file holds RNG (and null scheduler here); its global-step
argument is bookkeeping, NOT a persisted step field. S2 therefore explicitly
saves sealed `runtime_state_rank_<rank>.json` in metadata, checks its global
step against S1 policy and the actual AdamW moment step counters, and never
claims an adapter or an empty AdamW is a native continuation.

Only a successful actual load issues the in-process reload capability with
model/adapter/native/optimizer/RNG/global-step/execution/dropout checks. It is
bound to the actor object, actual state and unchanged source files. JSON booleans
or manually clearing `recovery_reload_required` do not authorize S2 update.
The recovery FSM can accept a verified live receipt and preserves the flag as
history. Real update still independently requires the capability every time.

## O/C, transaction and runtime policy

O is computed once over the rank-local batch, internally microbatched by verl,
detached to CPU, cloned and sealed as the denominator carrier. C is a separate
second pre-update batch computation with identical actual inputs and policy.
Parameters, AdamW state, RNG, R and carriers must remain unchanged; observed
forwards must be deterministic eval/no-grad, p0, and CUDA BF16 when applicable.
Formal APIs permit populated AdamW and arbitrary W, unlike Gate wrappers.

All ranks must satisfy existing strict checks: `max_abs_diff < 0.1`, ratio in
`[0.8, 1.28]`, no initial clipping and finite masked logprobs. Rank token counts
need not match. Aggregate means are token-count weighted; these diagnostics do
not change the row-mean objective. Receipt identity and installed O are checked
again at the update boundary. R/O is a lightweight rank/window-bound summary,
`informational_only=true`, `gate_blocking=false`; no large per-token forensic.
An overflowing informational ratio is null/flagged, not an alignment authority.

Only after all-rank checks, rank0 durably writes prepared -> started ->
step_may_have_run through the existing S1 append-only chain. Peers synchronize
before `actor.update_policy` (including its zero_grad). The shared audited update
guards actual train/grad-enabled forwards, all 252 default LoRA targets p0,
frozen base/vision/projector and LoRA-only optimizer. Exactly one optimizer step
and actual AdamW step `before+1` are required; no scheduler is constructed.
Any ambiguous failure leaves no consumption/policy advancement and requires
S1 rollback from the verified parent, never reuse of uncertain memory.

S2.1 additionally requires explicit `actor.config.policy_loss.loss_mode ==
"vanilla"`, using pinned verl 0.6.1's `policy_loss.get` field. Missing/implicit
mode, `gpg`, `rollout_correction`, `clip_cov` and any other mode fail before O/C
or optimizer execution. The guard runs again at the final update boundary; it
does not change PPO loss implementation, frozen clipping or reduction semantics.

Source and exported `adapter_config.json` keep dropout **0.05**; only live RL
Dropout modules use **0.0**. Both existing execution-version constants stay
unchanged. Gate empty-optimizer/source-SFT/two-identical-rank contracts remain.

## Staging and verification

Rank0 alone reserves an exclusive hidden staging directory before native save
collectives. All native model/optim/extra rank files are required, plus adapter
config/weights and sealed per-rank runtime snapshots. Official PEFT export
keeps source dropout metadata. Layout is partitioned into S1.1 adapter/native/
optimizer/RNG/optional-metadata role maps; hashes cover ALL files and ALL ranks,
not just rank0. Native checkpoint global step is the actual window step (Gate's
default remains 1).

Fresh verification checks old actor AND optimizer weakrefs are gone before
constructing. It restores native state into a fresh exported-adapter actor,
reapplies p0, verifies model/PEFT/AdamW/native RNG/global step against the
saved evidence and executes one finite actual multimodal microbatch per rank.
Only then is complete `reload_evidence` produced. Scope remains `cpu_fixture`
for injected CPU tests; it cannot authorize a runtime checkpoint. The output
is staging evidence only. S2 never publishes a final Smoke/formal PASS.

## CPU checks and remaining GPU validation

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_rl_formal_contracts.py tests/test_rl_run_state.py tests/test_rl_training_batch.py tests/test_rl_verl_policy_update.py tests/test_rl_policy_alignment.py tests/test_rl_actor_semantics.py tests/test_rl_gate_c.py tests/test_rl_formal_update.py -o addopts= -q -p no:cacheprovider
python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

Local tests include torch CPU AdamW continuation across two windows, full native
CPU save/fresh reload/immutable S1 commit, masks, rank objective/gradient
equivalence, capability and transaction failures, and the installed real Qwen
M-RoPE method on metadata-only self (no Qwen model construction). Installed
PEFT tiny-model regressions are included in existing actor-semantics tests.
Local verl is absent; production imports/execution are not claimed tested.
S2.1 tests additionally cover canonical/local separation, content-bound directory
relocation, changed/missing/invalid snapshots, original canonical SFT lineage,
native continuation/fresh reload, non-locator drift and explicit vanilla loss.

Pending AutoDL: real FSDP2 distributed gradients, real multi-rank native
optimizer/RNG reload, actual Qwen multimodal residency/VRAM, GPU O/C numerics.
S3 still owns the 20-prompt coordinator, collector/rollout lifecycle, static
handoff/restarts, provider retry/resume, CLI and five-window Smoke20 execution.
