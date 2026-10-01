# RL data and future run contract

This phase prepares data and schemas only. It does not run rollouts, APIs,
optimizers, checkpoints, tool caches, or a trainer. The 50-row local fixture is
**not** the formal RL dataset. Formal smoke20/main400 must be regenerated on
AutoDL from all 7,992 rows of pinned `OpenSearch-VL/Search-VL-RL-8K` revision
`8ef567289043eef004b13da83b0e7bb7f5ae2daa`.

## Data selection and provenance

The CLI takes a source parquet and source root; it never downloads data.
Source row index defines `rl_000000`-style IDs. `prompt_id` and
`trajectory_group_id` equal that stable source ID. Samples retain source
question, reference answer, dataset name and *relative* image paths. Question
hashes reuse local question normalization; image hashes reuse the existing
decoded-pixel SHA256 implementation. The reference answer is preserved for
both correctness and query-utility judges.

One SHA256-ranked stream (seed, dataset ID/revision, source ID) drives selection.
Eval question/image overlaps are recorded, skipped and backfilled from later
candidates. SFT question/image overlaps are recorded but **not excluded**.
Fail closed if there are too few eligible rows. `smoke = main[:smoke_count]`;
`main` is split in order into equal physical shards. The first main shard is
the pilot; there is no separate pilot100 selection. Every dataset/shard has a
deterministic manifest with source SHA, ordered membership and content hashes;
timestamps are deliberately excluded from hashed content. `overlap_audit.json`
records excluded Eval IDs and selected SFT overlaps.

Overlap manifests are independent hash lists generated from frozen Eval-300
parquet and SFT shards/images by `scripts/build_rl_overlap_manifest.py`.
For formal runs, the SFT image audit must be complete. The development-only
`--allow-missing-images` / `--allow-incomplete-sft-audit` flags leave the
incompleteness explicit and cannot pass formal preflight. Generated artifacts
are never overwritten by the preparation command.

Formal commands, with paths set to the actual AutoDL checkout and source
location (the exact AutoDL source root is not known to this local workspace):

```bash
export RL_SOURCE_ROOT=/path/to/Search-VL-RL-8K
export SFT_DIR=data/sft_main_imageid_v3
python scripts/build_rl_overlap_manifest.py --kind eval \
  --eval-parquet data/eval/combined_eval_300_v2.parquet \
  --output data/rl_overlap/eval_v2.json
python scripts/build_rl_overlap_manifest.py --kind sft \
  --sft-dir "$SFT_DIR" --output data/rl_overlap/sft_v3.json
python scripts/prepare_rl_data.py \
  --source-parquet "$RL_SOURCE_ROOT/rl_data.parquet" \
  --source-root "$RL_SOURCE_ROOT" \
  --dataset-id OpenSearch-VL/Search-VL-RL-8K \
  --dataset-revision 8ef567289043eef004b13da83b0e7bb7f5ae2daa \
  --output-dir data/rl --seed 20260506 --expected-source-rows 7992 \
  --smoke-count 20 --main-count 400 --main-shard-size 100 \
  --eval-overlap-manifest data/rl_overlap/eval_v2.json \
  --sft-overlap-manifest data/rl_overlap/sft_v3.json
python scripts/preflight_rl.py --config configs/rl_main.yaml \
  --eval-overlap-manifest data/rl_overlap/eval_v2.json \
  --sft-overlap-manifest data/rl_overlap/sft_v3.json
```

The final preflight also validates the existing SFT adapter provenance. It
must not be replaced with `--data-only` for a formal run. The source parquet
must be readable by the active PyArrow environment.

## Cache, group resume, interrupt and progress (contract only)

`tool_cache_key` SHA256 includes tool name, canonical JSON arguments,
provider, tool behavior version and runtime protocol version; changing any
version isolates old results. It does not yet cache or coalesce requests.
Future runtime must share one in-flight provider request among simultaneous
identical tool calls (for example, four rollouts requesting one text search),
then persist one result for later resume. Cache keys are not just query text.

`RLRunState` supports running/interrupted/completed/failed and records shard,
group, optimizer, actor/config fingerprints, cache version, checkpoint and
interrupt details. Indexes are zero-based. An n-trajectory prompt group is the
smallest resumable unit: only one complete attempt with all indices `0..n-1`
may enter RLOO statistics or an optimizer update. If trajectory 2 interrupts,
old trajectories 0/1 are discarded for training; resume re-runs the **entire**
group. `group_ready_for_update` encodes this constraint. It does not save a
trainer checkpoint or implement resume I/O.

Quota/auth/persistent-rate-limit/provider/network/judge/exhausted-malformed
response failures are **recoverable run interrupts**, not low model rewards.
Future trainer must stop new group scheduling; mark the active group incomplete;
exclude it from reward/advantage/update; flush caches; atomically save run
state and checkpoint; print `RLProgressSnapshot`; exit nonzero. After provider
recovery, resume from the next uncommitted group using the same run ID and
config fingerprints. Manual interruption uses the same contract. Model-caused
invalid tool arguments and K=3 fatal cascades remain trajectory-level events;
ordinary no-result search does not become a provider interrupt.

`RLProgressSnapshot` defines shard/group/trajectory counts, optimizer step,
mean reward, fatal/tool/cache counters and rates, elapsed time and ETA. No
terminal progress loop is implemented. Config paths reserve `checkpoints/`,
`run_state/`, `rollout_cache/`, `reward_cache/`, and `tool_cache/` under
`outputs/rl_main/` (and corresponding smoke paths); no directories are created
until a future trainer uses them.
