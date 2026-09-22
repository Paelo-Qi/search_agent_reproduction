# Phase 4: reliability, cache, resume, and trajectory logging

Phase 4 adds execution infrastructure only. It does not add tools, change the
eight model-visible schemas, judge answers, compute benchmark scores, or
implement SFT/RL.

## Retry policy

Serper, Jina Reader, SerpApi upload, and Google Lens requests use at most three
attempts with bounded 1 s / 2 s backoff. Timeout, connection failure, HTTP 429,
and HTTP 5xx are retryable. Authentication failures, configuration errors,
invalid arguments, invalid local input, and invalid responses are not. Retry
details remain metadata; the model sees only the final observation.

Jina Reader page failures are classified. `timeout`, `network_error`,
`invalid_response`, and `provider_error` may fall back to the corresponding
Serper snippet while other pages continue. Even if every page has one of those
transient/page-level failures, snippet-only success is allowed when snippets
exist. `authentication_error`, `configuration_error`, and `quota_error` are
systemic: `text_search` immediately returns a tool error instead of silently
changing capability for the rest of a batch.

PaddleOCR is intentionally different. Image submission occurs exactly once.
After a `job_id` is received, only transient GET failures while polling that
same job are retried. Every retry uses the same URL/job ID and the original
`max_poll_seconds` deadline; neither a new job nor a new deadline is created.
The whole OCR job is never automatically retried.
After the job reports done, the signed JSONL result download independently uses
the same bounded transient policy. Download timeout, network failure, 429, and
5xx are retried; 401/403 and invalid responses are not. Download retry neither
submits another job nor restarts the polling deadline.

## Tool-result cache

Caching is opt-in through `cache_dir` and applies only to `text_search`,
`web_search`, `image_search`, and successful `layout_parsing`. Local `crop`,
`sharpen`, `super_resolution`, and `perspective_correct` calls are never
cached. The default shared location used by the batch command is:

```text
.eval-runtime/cache/v1/<tool>/<first-two-hash-chars>/<sha256>.json
```

The key is SHA256 of deterministic, sorted compact JSON containing tool name,
normalized effective input, cache schema version, and a behavior namespace.
The behavior namespace fingerprints result-affecting configuration such as
result/length limits, provider selection, formatter version, and layout model.

- `text_search`: stripped query, normalized language, effective `top_k`.
- `web_search`: stripped query and normalized language.
- `image_search`: SHA256 of canonical RGB dimensions/pixels, not `img_n` or a
  file path.
- `layout_parsing`: the same canonical image hash plus parsing flags.

Image hashes therefore match for identical decoded pixels stored under
different paths or trajectory-local IDs, while two different `img_1` images
do not collide. Previous observations, messages, sample/run IDs, trajectory
history, timestamps, latency, and credentials never enter a key. History is
excluded because a tool function receives only its own arguments; the model
turn between tools is responsible for transforming earlier evidence into the
next call.

Lookup is a direct hash path. A valid hit returns the persisted successful
observation and result metadata without calling or rewriting the provider.
A miss calls the real backend and atomically writes only a success. Timeout,
network, quota, authentication, provider, and argument errors are not cached.
Malformed, incomplete, or version-incompatible entries become misses and add
a `cache_warning`; they do not abort a batch and are overwritten only after a
new successful real call.

Persisted metadata contains stable result facts such as provider, counts,
dimensions, and truncation. Per-execution metadata is never persisted in the
entry: `cache_hit`, `cache_key`, `cache_version`, `attempt_count`,
`latency_seconds`, and `cache_warning` are attached at runtime. A hit has
`attempt_count=0`; a clean real call has 1; a success on the third attempt has
3. For tools involving several requests, `attempt_count` is the maximum retry
attempt index across their request stages—not HTTP request count or API usage.
PaddleOCR also records `poll_attempt_count` and `download_attempt_count`.
Observation text is identical on hit and miss.

The cache schema remains version 1. The search behavior version is 2 because
Jina systemic failures can no longer produce cacheable snippet-only successes;
this prevents reuse of older text-search entries created under that unsafe
failure policy. The observation formatter and cache JSON layout are unchanged.

One cache namespace can be shared by Base, SFT, and SFT+RL runs. Identical
calls then receive identical evidence; distinct model queries naturally use
different entries. Phase 4 provides this facility but runs none of those
formal experiments.

## Batch state and resume

`scripts/run_agent_batch.py` is a sequential runner with no judge or scoring.
Each run directory contains:

```text
reports/eval_runs/<run_id>/
  run_manifest.json
  status.json
  trajectories.jsonl
  summary.json
```

## Run identity

The manifest is created before any status or trajectory write. Its deterministic
`run_config_fingerprint` covers model name/revision and checkpoint identity,
inference-config content, actual dataset SHA256 and the frozen eval-manifest
identity, `start`/`limit`, maximum Agent turns, search/layout configuration,
cache and behavior versions, and a fingerprint derived directly from all eight
tool declarations. API values, creation time, report path, hostname, PID,
latency, and GPU identity are excluded.

Remote models use their repository ID plus pinned revision. Local checkpoints
record their resolved path and a lightweight artifact fingerprint: small
configs/weights are fully hashed, while large weights use size plus first/last
1 MiB hashing, avoiding a full multi-GB startup read. The dataset path is
recorded for diagnosis, while checksum and frozen manifest identity protect
semantics if bytes at that path are replaced.

Reusing a `run_id` is permitted only when the persisted and current identities
match. A mismatch reports only differing field names and aborts before touching
manifest, status, summary, or trajectories. `--retry-failed` cannot bypass this
check. An older run directory containing artifacts but no manifest is also
refused rather than silently adopted. This prevents Base/SFT/RL, checkpoint,
dataset, selection, or tool-contract trajectories from being mixed.

The run manifest does not affect `.eval-runtime/cache/`: different compatible
or model-comparison run IDs still share evidence for identical tool calls.

Statuses are `pending`, `running`, `success`, and `failed`. Before a sample is
executed, `running` is written atomically. Its completed record and final status
are persisted immediately afterward. On startup, stale `running` records
become `failed` with `error_type=interrupted`. By default both `success` and
`failed` are skipped; only `pending` runs. `--retry-failed` makes the failed
set eligible once in that invocation—nothing is requeued in a loop.

`trajectories.jsonl` is atomically upserted by sample ID, so explicit retry
replaces the prior record instead of creating conflicting lines. Each record
contains sample ID, benchmark, question, status, final answer, error, elapsed
time, tool-call count, and the existing complete Agent trajectory. Turns keep
raw assistant output, parsed tool name/arguments, observation, status,
metadata, latency, errors, and derived IDs. Image summaries contain only
`img_n`, decoded-pixel SHA256, dimensions, kind, parent/lineage, and metadata;
PIL objects, raw bytes, and base64 are never serialized.

Cache entries, status, summary, and trajectory rewrites use a temporary file,
flush/fsync, and `os.replace`. Known API credential values are recursively
redacted from cache and batch artifacts.

`summary.json` reports sample-state counts, `cache_hits`, `cache_misses`, and
`real_tool_executions`. The last field equals cache misses among cache-enabled
tool calls whose backend executed. A cache miss is not an HTTP/provider request
count: one `text_search` can issue Serper plus several Jina requests and retry;
one `image_search` can upload and then query Lens. Phase 4 deliberately does
not claim HTTP/API usage accounting.

## Commands

Completely offline reliability smoke:

```bash
python scripts/run_phase4_reliability_smoke.py
```

The smoke uses fake providers/models and synthetic samples to demonstrate a
miss then hit, transient retry, failure, normal resume, explicit failed retry,
and trajectory persistence.

An explicitly bounded real Agent batch can be run later without scoring:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id engineering-check --start 0 --limit 4
```

Optional provider-only real cache check (not part of pytest):

```bash
python scripts/run_search_backends_smoke.py --tool text_search \
  --cache-dir .eval-runtime/cache --report reports/text_search_cache_first.json
python scripts/run_search_backends_smoke.py --tool text_search \
  --cache-dir .eval-runtime/cache --report reports/text_search_cache_second.json
```

With an initially empty cache, the first report should contain
`cache_hit=false`, `attempt_count>=1`; the second should contain
`cache_hit=true`, `attempt_count=0` and perform no provider request.
