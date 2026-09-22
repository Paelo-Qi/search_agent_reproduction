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

The three search tools remain mock-only. `layout_parsing` is optional and
returns `configuration_error` when no provider is configured. Thus the local
smoke makes no API calls or downloads:

```bash
python scripts/run_local_visual_smoke.py
```

It writes `reports/phase2_local_visual_smoke.json` with each tool's status,
lineage, dimensions, derived IDs, and final registry state. This is a CPU
protocol/backend check, not a real Qwen or benchmark result.

## Optional Baidu layout provider

`LayoutParsingBackend` is a provider-neutral interface returning `LayoutDocument`
blocks. `BaiduLayoutParsingBackend` implements an optional Qianfan PaddleOCR-VL
HTTP transport. Endpoint, model, key environment-variable name, and timeout
come from [the example config](../configs/layout_parsing.example.yaml); the key
itself is read at request time from `BAIDU_QIANFAN_API_KEY`. Copy
[`.env.example`](../.env.example) only as a template and load the actual secret
into your shell. `.env` files are ignored by Git. Nothing calls the paid API
unless `create_phase2_tool_registry(layout_config=...)` is used and a model
invokes `layout_parsing`.

The adapter sends a base64 PNG and preserves optional
`use_chart_recognition` / `use_doc_orientation_classify` flags. Provider JSON
is normalized to stable plain text in provider reading order, for example:

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

Bounding boxes and request ID are kept in trajectory metadata, not dumped to
the model. Explicit failure categories include configuration, authentication,
timeout, network, quota, provider, and invalid-response errors; the tool
returns a concise failure observation and does not crash the agent. Unit tests
use an injected fake HTTP transport, not a paid request. A real API response
and credential handling remain to be validated with a real key.

API shape is based on [Baidu Qianfan PaddleOCR-VL API documentation](https://cloud.baidu.com/doc/qianfan-api/s/zmho8omz3).
