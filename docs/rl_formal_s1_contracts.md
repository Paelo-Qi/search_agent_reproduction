# Formal RL S1: CPU control-plane contracts

Status: **FORMAL RL S1 CPU CONTRACTS COMPLETE**. This is not Smoke20 ready,
not a formal training entry point, and not GPU verification. No collector,
actor, optimizer update, model loading, provider/Judge request, or real weight
checkpoint is implemented here. Gate A/B/C retain their existing paths and
schemas. The user's AutoDL Gate C PASS is a supplied runtime fact; absent
AutoDL artifacts, S1 makes no claim about its memory, timing or stage evidence.

## Frozen future design and generic interfaces

Future Smoke20 uses 20 prompts, n=2, K=4 groups/window, five windows, one
optimizer step/window, PPO epochs=1, `per_generation_row_mean_v1`. Main400
must restart the original SFT checkpoint-3k, never the Smoke final adapter.
No config or membership is changed. The APIs do not hardcode prompt count,
rollout n, K or world size: n>=2, any nonempty window and any positive W are
accepted. Main n=4 and four physical 100-prompt shards remain future inputs;
physical shard boundaries are not RLOO group boundaries.

`checkpoint.py` owns independent formal version constants (all initially 1):
`RL_TRAINING_BEHAVIOR_VERSION`, `RL_RUN_SCHEMA_VERSION`,
`RL_CHECKPOINT_SCHEMA_VERSION`, `RL_GROUP_SCHEMA_VERSION`,
`RL_WINDOW_SCHEMA_VERSION`. Attempt/state schemas also start at 1.

`build_training_run_identity` accepts explicit semantic metadata and ordered
unique prompt IDs. It binds dataset checksum/split, pinned base/revision,
source SFT adapter/metadata/lineage, execution contract, rollout behavior/config,
n, optimizer/PPO/world size, reward version/semantics, tool/image versions,
integration source hashes and weighting. All semantic keys, including extra
keys, are hashed. `training_behavior_fingerprint` excludes run ID;
`run_identity_sha256` additionally binds run ID. Paths belong in `locators`,
outside both identities. `require_same_training_run` rejects changed semantics.
These are schemas, not a claim that supplied metadata was tied to a live actor;
actual integration/actor identity binding remains S2.

`initial_policy` anchors iteration/step zero to the source SFT identity and
explicit optimizer/RNG identities. `checkpoint_policy` derives the successor
and cumulative consumed IDs from a checkpoint manifest. FSM advancement and
ledger consumption additionally require its **published immutable directory**;
a constructed manifest, attempt phase or `latest.json` is insufficient.

## Groups, windows and rewards

The original Gate group API remains n=2. `formal_group_identity`,
`validate_formal_group`, `publish_formal_group` and `read_formal_group` add a
separate schema: prompt/source/run, policy iteration/effective fingerprint,
parent checkpoint, rollout config, collection UUID/index and expected n.
Members must be complete, uniquely identified, indices exactly 0..n-1, with
finite composed rewards and unchanged actual token/logprob validation.
Trajectory/multimodal artifact hashes are committed as a whole group.

`build_training_window` binds ordered group IDs/hashes, parent policy/checkpoint,
window ID, iteration, n, weighting, next optimizer step and behavior fingerprint.
Duplicate prompt/group/member, mixed policy/iteration/rollout, uncommitted and
already consumed groups fail closed.

`assemble_window_rloo` lazily calls **official verl** with one outcome row per
generation and distinct prompt-group IDs. There is no homemade fallback when
verl is missing. All members, including fatal members, enter their own group
baseline; only then is the existing fatal `max(advantage, 0)` clamp applied.
Output keeps raw/final advantages and group/member identities. All final values
zero means `zero_signal`, not resampling or fabricated signal. A zero-signal
window cannot authorize a new optimizer checkpoint. Explicit injected estimators
are flagged as test evidence and cannot authorize runtime checkpoints.

## Transactions and recovery

Formal metadata paths are isolated from any `rl_gate*` directory:

```text
identity/run.json                       immutable run/SFT anchor
groups/<group-id>/group.json             immutable whole-group receipt
attempts/<uuid>/<sequence>-<phase>.json  append-only event chain
checkpoints/policy-<iteration>/checkpoint.json
latest.json, state.json                  disposable/rebuildable indexes
```

Attempt phases: prepared -> started -> step_may_have_run -> checkpoint_staging
-> verified (or failed). Future S2 must durably record `step_may_have_run`
**before** any optimizer operation. Full event histories are checksum chained;
missing, overwritten or mixed histories fail closed. Failed attempts consume
nothing. Retry plans require restoring parent adapter/native policy,
optimizer and RNG, discarding uncertain memory, and using a fresh attempt UUID.
Committed parent-policy groups are reused unchanged; old attempt events remain.
`rollback_interrupted_state` is a separate rollback operation, not a legal
updating -> collecting transition. It marks reload required: S1 does not execute
or claim a real reload, and cannot start another update without S2 verification.

Checkpoint metadata binds source lineage, parent, iteration/step, all four
artifact roles (adapter/native/optimizer/RNG), execution/behavior, consumed
group IDs/hashes, window/attempt/reward receipt and matching reload evidence.
Eligibility distinguishes Gate, Smoke continuation, Smoke final and Main.
Formal checkpoints allow same-run resume; `eligible_for_main_init=false` keeps
new Main runs anchored to original SFT rather than Smoke/other RL artifacts.

Under the run advisory lock: write staging artifacts -> fsync files -> write
manifest last -> fsync nested/staging directories -> same-parent atomic rename
-> fsync parent. No committed directory is overwritten. The directory rename
is the publication point. A pre-rename failure leaves hidden staging, never a
verified checkpoint. A post-rename failure (including failed index/parent-fsync
publication) requires scanning immutable manifests and **rolling forward** if
the complete successor exists; do not repeat its optimizer step. Conflicting
successors, missing/changed artifacts and duplicate consumption fail closed.
`recover_formal_run` reconstructs ledger/latest/state from the immutable chain,
ignoring hidden staging and untrusted index contents.

POSIX production uses real file and directory fsync. Windows only supports
explicit `cpu_fixture=True` tests; production transactions reject unsupported
directory durability before publication. Fixtures are marked `cpu_fixture`,
cannot be mixed into a runtime chain, and are **not actual training proof**.

## Rank planning and tests

`deterministic_rank_plan` assigns unique logical row IDs once when divisible by
W; otherwise it repeats the **entire** list to LCM(B,W). All rows have the same
multiplicity and all ranks equal row counts; no drop-last, dummy or partial
duplication. It reports logical/physical counts, replication, multiplicity and
rank assignments. Real FSDP gradient equivalence is explicitly deferred to S2.

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_rl_group.py tests/test_rl_rloo.py tests/test_rl_run_state.py tests/test_rl_checkpoint.py tests/test_rl_training_batch.py tests/test_rl_gate_c.py tests/test_rl_formal_contracts.py -o addopts= -q -p no:cacheprovider
python -m pytest -o addopts= -q -p no:cacheprovider
git diff --check
```

All filesystem simulations use pytest `tmp_path`, with explicitly injected CPU
RLOO fixtures. No formal Smoke outputs or Gate artifacts are created/modified.
