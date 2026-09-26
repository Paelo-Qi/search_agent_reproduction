# Phase 6A: Base Eval-300 formal-run preparation

This document records the Phase 6A workflow under the earlier Agent prompt.
Phase 6B.1 sets `AGENT_BEHAVIOR_VERSION=4`; validate the current protocol on
`base-dev30-v4` first, then use a new run ID for any later full Eval-300 run.
Do not reuse existing `base-eval300-v1` artifacts with the new behavior.

Phase 6A scales the already accepted Base Agent protocol to the frozen
`data/eval/combined_eval_300.parquet`; it does not change Agent generation,
tools, cache/retry behavior, or Judge semantics. The one formal run ID is
`base-eval300-v1`.

## Identity and balanced invocations

`--eval300` validates the frozen SHA256
`f9d0ca74f98d3f73cd6ad1f60b7f2294c5e5d59d4956b60ad95084dc8ee36cf4`,
requires exactly 100 globally unique samples from each benchmark, and creates
one deterministic 300-sample ordering. The first 200 positions contain 67
SimpleVQA, 67 MMSearch, and 66 VDR-Bench samples. The remaining positions
contain 33, 33, and 34 respectively. IDs are sorted within each benchmark and
interleaved, so the plan is stable across restarts.

The full 300-sample ordered-ID checksum is part of `run_manifest.json`.
`--max-samples` is only an invocation cap and is deliberately absent from run
identity. Therefore the first invocation creates all 300 status entries, runs
200, and leaves 100 pending. A later invocation with the same run ID and no cap
skips success and failed records and executes only pending records.
`--retry-failed` remains an explicit, independent operation for ordinary
per-sample failures. After a provider exhausts its bounded retries with
`quota_error`, or returns `authentication_error`/`configuration_error`, Agent
batch execution stops after durably saving the current failed trajectory.
That sample is `pending` (not permanently `failed`), later samples remain
`pending`, and completed successes remain untouched. Status/summary record
the latest interruption; retries preserve the prior error trajectory in
`attempt_history`. The command exits nonzero even with `--max-samples`.
After fixing credentials or quota, repeat **the same command and run ID**;
no `--retry-failed` is needed for the interrupted sample. Ordinary timeout,
network, provider, and invalid-response failures keep their per-sample
behavior and still need `--retry-failed` if marked failed. Existing `failed`
records with a saved systemic tool-turn error are also restored to `pending`
on resume; already successful records are never demoted.
The cap always denotes the same prefix of the full universe; rerunning the
first-batch command after an interruption fills only unfinished items within
that prefix and never spills into the remaining 100.

The default `.eval-runtime/cache` remains shared with compatible Dev-30 tool
requests. Run status, trajectories, summaries, and Judge artifacts remain
isolated under `reports/eval_runs/base-eval300-v1/`.

## AutoDL commands

The preflight reads parquet metadata/configuration and environment-variable
presence only. It does not load Qwen, create a runtime/provider, or make an HTTP
request. Jina authentication is reported but remains optional, matching the
existing backend.

```bash
# A. Read-only preflight. Add --require-judge-env if Judge will follow now.
python scripts/preflight_eval300.py \
  --config configs/eval_base_300.yaml \
  --search-config configs/search_backends.example.yaml \
  --layout-config configs/layout_parsing.example.yaml \
  --judge-config configs/judge.example.yaml \
  --require-agent-env

# B. First balanced 200 Agent attempts in the one formal 300-sample run.
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id base-eval300-v1 \
  --config configs/eval_base_300.yaml \
  --search-config configs/search_backends.example.yaml \
  --layout-config configs/layout_parsing.example.yaml \
  --cache-dir .eval-runtime/cache \
  --eval300 \
  --max-samples 200

# C. Inspect completion, per-benchmark tool calls, provider-backend executions,
# cache hits/misses, recovery guards, and provider errors.
python -m json.tool reports/eval_runs/base-eval300-v1/summary.json

# E. Resume only the remaining pending samples in the same formal run.
# If B stopped for a systemic error, first repeat B verbatim to finish its
# fixed prefix; do not add --retry-failed for that interruption.
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id base-eval300-v1 \
  --config configs/eval_base_300.yaml \
  --search-config configs/search_backends.example.yaml \
  --layout-config configs/layout_parsing.example.yaml \
  --cache-dir .eval-runtime/cache \
  --eval300

# Optional and separate: retry only failed Agent samples.
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id base-eval300-v1 \
  --config configs/eval_base_300.yaml \
  --search-config configs/search_backends.example.yaml \
  --layout-config configs/layout_parsing.example.yaml \
  --cache-dir .eval-runtime/cache \
  --eval300 \
  --retry-failed

# F. Final Judge, only after Agent pending/running are both zero.
python scripts/run_judge.py \
  --run-id base-eval300-v1 \
  --config configs/judge.example.yaml \
  --dataset data/eval/combined_eval_300.parquet

python -m json.tool reports/eval_runs/base-eval300-v1/judge/judge_summary.json
```

There is intentionally no first-200 Judge command. The existing parent guard
requires the complete 300-sample Agent universe to have zero pending/running
items before it creates Judge artifacts. This preserves one stable Judge
manifest and prevents a partial result set from becoming the formal run.

Agent `summary.json` provides overall and per-benchmark totals, processed,
success/failed/pending, completion rate, all eight tool-call counts, cache
hits/misses, external backend executions (`real_tool_executions`), duplicate
call/unknown-image recovery counts, and provider-error counts. A cache miss
means the external tool backend executed; it is not claimed to equal exactly
one HTTP request because some backends perform multiple HTTP operations.
Provider HTTP 429 only stops the batch after the existing bounded provider
retry is exhausted; a transient 429 followed by success or a cache hit does
not. Jina's ordinary page-read failures retain snippet fallback, while
quota/auth/config errors do not use that fallback.

Judge `judge_summary.json` retains the existing successful-Judge accuracy and
macro definition, and also reports `end_to_end_accuracy = correct / total`
overall and per benchmark. Agent completion remains available in the adjacent
Agent summary, so upstream failure and Judge failure are not conflated.
