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
The adapter tries PNG, then JPEG quality 85/70/55. If the image is still too
large, it resizes with Lanczos at deterministic 0.8 scale intervals and
re-encodes, with a 12-resize bound and a 32-pixel minimum dimension. It fails
clearly only if those bounded attempts cannot fit. Metadata records original
and uploaded dimensions, byte size, and whether resizing occurred; no image
bytes enter the report. Lens is queried with `engine=google_lens`,
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
are recorded separately and may still pass. The three provider-only real API
smokes were reported as passed in the preceding Phase 3 acceptance. That does
**not** prove Qwen-Agent integration.

## Phase 3 final integration smoke

This separate opt-in path loads the real Qwen3-VL-4B model, registers one
synthetic image as `img_1`, and connects the Agent runtime to the eight-tool
Phase 3 registry. It uses the existing `SERPER_API_KEY`, `JINA_API_KEY`, and
`SERPAPI_API_KEY` environment variables; the layout token is needed only if
that unrelated tool is called. Run each command on the AutoDL GPU host with
the relevant credentials set:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_4b_agent_smoke.py \
  --phase3-search-tools --phase3-tool text_search \
  --report reports/4b_phase3_text_search_smoke.json

CUDA_VISIBLE_DEVICES=0 python scripts/run_4b_agent_smoke.py \
  --phase3-search-tools --phase3-tool image_search \
  --report reports/4b_phase3_image_search_smoke.json
```

The first prompt asks Qwen to emit `text_search` with a fixed query; Serper
returns URLs, Jina Reader returns at least one passage, and that observation
must appear in the next Qwen `generate()` input before a nonempty final answer.
The second prompt asks for `image_search({"url":"img_1"})`; the registered
image is uploaded to SerpApi, searched by Google Lens, and its observation
must likewise enter the next Qwen turn. Zero Lens matches remain valid if
the upload and Lens request succeed.

Each report records the requested/called tool, status, real provider metadata,
trajectory status, observation re-entry, final-answer presence, elapsed time,
and peak VRAM. `passed` requires the full tool-to-next-model chain, not merely
a final answer. Reports redact known API credential values even if echoed in a
trajectory. These integration smokes are **not yet passed** in this checkout:
offline tests verify the wiring, but Qwen plus live APIs must be run on the
GPU host. Phase 3 is complete only after both reports pass there. Existing
mock and Phase 2 modes remain available and are not prerequisites.

Provider references: [Serper](https://serper.dev/),
[Jina Reader](https://jina.ai/reader/), [SerpApi Image API](https://serpapi.com/image-api),
[SerpApi Google Lens API](https://serpapi.com/google-lens-api).
