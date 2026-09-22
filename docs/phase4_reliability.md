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

PaddleOCR is intentionally different. Image submission occurs exactly once.
After a `job_id` is received, only transient GET failures while polling that
same job are retried. Every retry uses the same URL/job ID and the original
`max_poll_seconds` deadline; neither a new job nor a new deadline is created.
The whole OCR job is never automatically retried.

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
3. Observation text is identical on hit and miss.

One cache namespace can be shared by Base, SFT, and SFT+RL runs. Identical
calls then receive identical evidence; distinct model queries naturally use
different entries. Phase 4 provides this facility but runs none of those
formal experiments.

## Batch state and resume

`scripts/run_agent_batch.py` is a sequential runner with no judge or scoring.
Each run directory contains:

```text
reports/eval_runs/<run_id>/
  status.json
  trajectories.jsonl
  summary.json
```

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
