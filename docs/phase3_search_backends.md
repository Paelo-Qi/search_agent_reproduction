# Phase 3: real search backends

`create_phase3_tool_registry()` preserves the audited eight model-visible tool
names and argument schemas. The Phase 2 registry remains unchanged and retains
mock search tools. Phase 3 uses the existing real local visual tools, the
PaddleOCR AI Studio layout adapter, and three independent search adapters:

| Tool | Data flow |
| --- | --- |
| `web_search` | Serper Google Search → normalized title, URL, snippet |
| `text_search` | Serper top-k URLs → Jina Reader `r.jina.ai` per URL → bounded passages |
| `image_search` | `url: "img_n"` → local PIL/path → SerpApi Image API multipart upload → temporary `image_id` → Google Lens visual matches |

There is **no LLM summarization** and no call to `s.jina.ai`. Search results
are normalized into plain-text observations; raw provider JSON is not passed to
the model. `image_search` is informational: it does not register remote search
images as derived images. `perspective_correct` remains an identity fallback.

Configuration is in [search_backends.example.yaml](../configs/search_backends.example.yaml).
Set `SERPER_API_KEY` and `SERPAPI_API_KEY` privately in the environment. A
`JINA_API_KEY` is supported for Reader authentication; without it Reader uses
the unauthenticated endpoint, subject to its provider limits. See
[`.env.example`](../.env.example). No secret belongs in YAML, observations,
metadata, reports, or exception messages.

`text_search.top_k` accepts finite integers 1–10, including `5.0`, and rejects
fractional/zero/oversized values. The default is 5. Reader failures are counted
per URL. If some pages fail, successful passages and snippets remain; if all
Reader calls fail but Serper snippets exist, the tool returns a clearly
recorded snippet-only fallback. Metadata includes `reader_success_count`,
`reader_failure_count`, `reader_fallback_used`, and `truncated_result_count`.
Passages are cut deterministically to 6,000 characters per page and the full
observation to 20,000 characters by default. No model is used for compression.

SerpApi's Image API accepts in-memory PNG/JPEG multipart uploads up to 500 KB.
The adapter uses a deterministic JPEG fallback for oversized PNGs and fails
clearly if still too large. Lens is queried with `engine=google_lens`,
`type=visual_matches`, and the returned short-lived `image_id`; no public URL
for `img_n` is required. An empty `visual_matches` list is a valid protocol
success with zero semantic matches, not an HTTP failure. No images are fetched
from result thumbnail URLs.

All three tools map missing credentials, 401/403, 429, timeout, connection
failure, 5xx, malformed responses, invalid arguments, and absent results into
explicit `ToolResult` status/error categories. Reader failure is partial where
possible. This stage deliberately adds no retry framework, cache, resume,
judge, benchmark evaluation, SFT, or RL.

Run local tests without any API call:

```bash
pytest
```

Real API smoke commands are opt-in and are **not** part of pytest:

```bash
python scripts/run_search_backends_smoke.py --tool web_search
python scripts/run_search_backends_smoke.py --tool text_search
python scripts/run_search_backends_smoke.py --tool image_search
```

Reports are independent under `reports/<tool>_smoke.json`. Web smoke requires
at least one title/URL. Text smoke requires at least one successful Reader
passage. Image smoke requires successful upload and Lens protocol; zero matches
are recorded separately and may still pass. These live calls must not be
reported as passed until run with real credentials.

Provider references: [Serper](https://serper.dev/),
[Jina Reader](https://jina.ai/reader/), [SerpApi Image API](https://serpapi.com/image-api),
[SerpApi Google Lens API](https://serpapi.com/google-lens-api).
