# Phase 5A/5B: Dev-30 and correctness judge

## Dev-30

Dev-30 is an engineering and judge-validation set, not an independent final
test set. `scripts/prepare_dev30.py` takes 10 IDs from each frozen Eval-300
benchmark. It stable-sorts the 100 IDs in each group, samples without
replacement with seed `20260506`, then stable-sorts each selected group. Output
order is SimpleVQA, MMSearch, VDR-Bench. It stores only IDs and provenance under
`data/eval/dev30/`; images and rows remain in `combined_eval_300.parquet`.

The manifest binds the source revision and combined parquet SHA256, per-group
canonical ID checksums, the combined canonical checksum, seed, strategy, and
script/schema versions. `created_at` is excluded from semantic identity. Agent
batch execution accepts `--selection-manifest`; ID lookup is independent of row
order. The Agent reader deliberately requests only `id`, `benchmark`,
`question`, and `image_packed`, so a reference answer cannot enter model input.
During a later Base-300 run these 30 items are rolled out and judged again; old
Dev-30 outputs are not reused.

## Judge boundary and verdicts

The default live configuration uses DeepSeek model `deepseek-flash` at
`https://api.deepseek.com`.

The DeepSeek correctness judge receives only `question`, `reference_answer`,
and `model_answer`, plus sample/benchmark audit IDs. It never receives tool
observations, full trajectories, hidden reasoning, or cache data. Its system
prompt treats every input field as untrusted data and asks only for semantic
correctness. Equivalent wording, formatting, capitalization, and harmless
extra detail are accepted; a core factual error is incorrect.

The response contract is a JSON object:

```json
{"verdict": "correct", "reason": "brief reason"}
```

The strict parser strips one optional Markdown fence, requires a JSON object,
and accepts only `correct` or `incorrect`. Parse failure has
`status=error,error_type=invalid_response`, is not retried, and is never counted
as incorrect. The reason is audit text and does not affect the verdict.
`JUDGE_PROMPT_VERSION=1` is part of run identity.

## Retry, fail-fast, and resume

The adapter reuses Phase 4 `RetryPolicy`. Network errors, timeouts, HTTP 429,
and HTTP 5xx are retried for at most three attempts with bounded backoff. HTTP
401/403, configuration/local-input errors, and valid-HTTP structured parse
failures are not retried. An exhausted 429 is `quota_error`.

Authentication, quota, and configuration errors are systemic: the current
sample becomes failed and later samples remain pending. Status is persisted as
`pending/running/success/failed`; a stale `running` becomes
`failed/interrupted`. Default resume skips success and failed records, while
`--retry-failed` explicitly retries failed ones. A judge manifest binds the
parent Agent fingerprint, public judge config, model, prompt version, and exact
judge inputs. Incompatible reuse is rejected. The API key value is never part
of identity, output, or logs and all artifacts pass through secret redaction.

Judge may start only after the parent Agent batch has no `pending` or `running`
samples. Agent `failed` samples are allowed: they are complete rollout attempts
and become `upstream_agent_failure` without a provider call. Before any Judge
artifact is created, the guard also requires the completed IDs and outcomes in
`status.json` to match `trajectories.jsonl` exactly.

Agent failures or missing final answers bypass the provider and become
`upstream_agent_failure`, not `incorrect`. Accuracy is
`correct / (correct + incorrect)` among successful judge responses only.
Summaries contain overall and per-benchmark counts and the mean of available
per-benchmark accuracies as `macro_accuracy`.

## Commands

```bash
python scripts/prepare_dev30.py

# Future real Agent run; does not expose answers to the Agent.
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id base-dev30 \
  --selection-manifest data/eval/dev30/dev30_manifest.json

# Fully offline fake-provider acceptance.
python scripts/run_phase5_judge_smoke.py

# Explicit opt-in live call after setting DEEPSEEK_API_KEY.
python scripts/run_deepseek_judge_smoke.py \
  --question "What is the capital of France?" \
  --reference-answer "Paris" \
  --model-answer "Paris"

# Optional second acceptance call for the negative verdict.
python scripts/run_deepseek_judge_smoke.py \
  --question "What is the capital of France?" \
  --reference-answer "Paris" \
  --model-answer "London" \
  --expected-verdict incorrect

# Judge an existing Agent run.
python scripts/run_judge.py --run-id base-dev30
```

Judge outputs are separate from Agent trajectories under
`reports/eval_runs/<run_id>/judge/`: `judge_manifest.json`,
`judge_status.json`, `judge_results.jsonl`, and `judge_summary.json`.
