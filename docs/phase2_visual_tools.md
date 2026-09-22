# Phase 2: derived-image visual tools

The opt-in `create_phase2_tool_registry()` keeps the eight audited tool names
and schemas unchanged. `crop`, `sharpen`, `super_resolution`, and
`perspective_correct` now return actual PIL images through `ToolResult`.
`AgentRuntime` alone registers each image as the next trajectory-local `img_n`,
records its parent/tool/dimensions/metadata, and puts the PIL image in a
`role=tool` multimodal content part alongside a short text observation. The
next `QwenAgentModel.generate` passes that message to
`processor.apply_chat_template`; an image ID alone is not the visual input.
CPU tests verify this processor handoff with a fake model. Actual Qwen3-VL-4B
vision-token processing and generation still require AutoDL/CUDA validation.
The existing 4B Agent smoke now has an opt-in real visual-tool mode; its
default mock mode is unchanged:

```bash
python scripts/run_4b_agent_smoke.py \
  --config configs/eval_4b.yaml --index 0 \
  --local-visual-tools --report reports/4b_visual_agent_smoke.json
```

This uses a deliberately synthetic crop instruction on one frozen sample and
requires a successful derived-image crop plus final answer. It is not an
evaluation result. It must not be reported as passed until actually run on a
BF16 CUDA host. An optional `--layout-config
configs/layout_parsing.example.yaml` enables the provider adapter in this
same opt-in registry.

The local backends are deliberately small:

- `crop`: real Pillow crop; rectangle is clipped to bounds, empty intersections fail.
- `sharpen`: real Pillow `ImageEnhance.Sharpness`, amount 0–4.
- `super_resolution`: Pillow Lanczos resize, scale 1–4 and 16 MP output cap.
  This is a lightweight upscale baseline, **not learned SR**; it cannot recover
  lost detail. The backend can later be replaced without changing the tool schema.
- `perspective_correct`: identity fallback. It copies pixels into a new derived
  image and reports `changed=false`, `backend_mode=identity_fallback`; it does
  **not** perform perspective detection or correction.

The three search tools remain mock-only **in the Phase 2 registry**; Phase 3
has a separate real-search registry. `layout_parsing` is optional and
returns `configuration_error` when no provider is configured. Thus the local
smoke makes no API calls or downloads:

```bash
python scripts/run_local_visual_smoke.py
```

It writes `reports/phase2_local_visual_smoke.json` with each tool's status,
lineage, dimensions, derived IDs, and final registry state. This is a CPU
protocol/backend check, not a real Qwen or benchmark result.

## Optional PaddleOCR AI Studio layout provider

`LayoutParsingBackend` is a provider-neutral interface returning `LayoutDocument`
blocks. `PaddleOCRAiStudioBackend` uses the official hosted **asynchronous job
API**, not the former synchronous Qianfan endpoint. The example configuration
specifies `https://paddleocr.aistudio-app.com/api/v2/ocr/jobs`, model
`PaddleOCR-VL-1.6`, per-request timeout, poll interval, and a monotonic maximum
poll duration. Defaults are 120 seconds per HTTP request, 5 seconds between
polls, and a 600-second maximum polling window. A quick job returns immediately
on `done`; it does not wait the full window. The old Qianfan-specific backend
was removed.

The credential is read at request time from `PADDLEOCR_ACCESS_TOKEN`; see
[the example config](../configs/layout_parsing.example.yaml) and
[`.env.example`](../.env.example). Load the actual token into your shell, never
commit it. `.env` files are ignored by Git. No paid request occurs unless a
registry configured with `layout_config=...` actually invokes `layout_parsing`.

The adapter converts the registered PIL image to PNG bytes in memory, submits
multipart `file` plus `model` and JSON-string `optionalPayload`, polls
`pending` / `running` until `done` or `failed`, then downloads only the returned
JSONL URL. It never downloads `markdown.images` or `outputImages`. The
model-visible optional arguments map to `useChartRecognition` and
`useDocOrientationClassify`; absent arguments are omitted. Because there is no
model-visible unwarping argument, `useDocUnwarping` is always `false`.

Structured `prunedResult.parsing_res_list` blocks take precedence in reading
order; pages without readable structured blocks use `markdown.text` as a
fallback. The result is stable plain text, for example:

```text
<observation>
Content:

[Title]
Report

[Paragraph]
Results are summarized here.

[Table]
| A | B |
</observation>
```

Bounding boxes, job ID, page count, and block count are metadata only; raw JSON,
result URLs, and the token are not sent to the model. Failure categories include
`configuration_error`, `authentication_error`, `quota_error`, `timeout`,
`network_error`, `provider_error`, `invalid_response`, and `invalid_image`.
All API unit tests use a fake HTTP session and make no paid request.

The project's prior real-API acceptance, as reported by the user, established:

| Check | Status |
| --- | --- |
| AI Studio authentication | PASS |
| Multipart image submission | PASS |
| Async job polling | PASS |
| JSONL download | PASS |
| Structured block parsing | PASS |
| SearchVL-style observation | PASS |
| Credential redaction | PASS |

The reported synthetic document result was `status=success`, `error_type=None`,
`provider=paddleocr_aistudio`, `page_count=1`, `block_count=5`. A natural/anime
image such as `anime_0` may contain no readable OCR content; that is not by
itself a transport/backend failure. For a reproducible layout check, the smoke
now draws the same local document on every run and does not read eval parquet
by default:

```bash
# Set PADDLEOCR_ACCESS_TOKEN privately in the environment first.
python scripts/run_layout_parsing_smoke.py \
  --report reports/layout_parsing_smoke.json
```

For an explicit evaluation-image diagnostic, use `--use-eval-sample --index 0`.
Without a token the smoke reports `configuration_error` and does **not** pass.
Success requires a readable `Content:` observation; it is not a benchmark
result. This code change itself did not make a new live request. API shape
follows [the official PaddleOCR AI Studio API documentation](https://ai.baidu.com/ai-doc/AISTUDIO/fml7mozw5).
