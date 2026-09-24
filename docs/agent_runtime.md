# Qwen3-VL-4B inference and Agent smoke

The original Agent smoke remains infrastructure-only and uses mock tools.
Phase 2 adds separate real local visual backends and an optional layout API
adapter; it does not add search backends, benchmark scoring, SFT, or RL.

See [Phase 2 visual tools](phase2_visual_tools.md) for the new opt-in registry,
CPU smoke, and provider configuration.

The model is pinned to `Qwen/Qwen3-VL-4B-Instruct` revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`. The eight declarations mirror the
local SearchVL-SFT audit, including the observed `image_search.url` argument.

## CPU-only protocol smoke

```bash
pytest
python scripts/run_agent_mock_smoke.py
```

The scripted model executes `image_search`, receives a plain-text observation,
executes `text_search`, receives another observation, and then returns a final
answer. No API or model weight is used.

## Phase 5C stabilization contract

Each episode exposes the registered input IDs (`img_1`, `img_2`, ...) in the
model-visible system context. Image tools accept only those runtime IDs, never
dataset filenames, filesystem paths, or HTTP URLs. Invalid references return
the currently available IDs so the model can recover. Derived-image IDs remain
visible through their producing tool observations.

An episode-wide signature of tool name plus canonical JSON arguments prevents
an exact call from executing twice, including `A-B-A` loops. A duplicate still
consumes and records an Agent turn, but produces a synthetic observation before
the registry/cache/provider path, so it cannot call or pollute shared backends.
Different tools or arguments remain eligible. The system guidance also asks the
model to stop once evidence is sufficient; `max_agent_turns` remains 8 for the
formal evaluation configuration.

This policy is identified by `agent_behavior_version` in every run manifest.
Older manifests with a different version cannot resume under the new runtime.
Batch execution prints flushed START/DONE lines after the corresponding safe
persistence points, using the number eligible in that invocation as `[i/N]`.

## Phase 6B tool-use guidance

The earlier reproduction prompt was intentionally minimal. Phase 6B expands
the model-facing instructions toward the original OpenSearch-VL Visual
Investigation Agent policy: assess the question and image quality, verify
external facts, choose tools for specific information gaps, and chain useful
steps such as `crop -> layout_parsing` or `image_search -> text_search`.
The reproduction's registered `img_n` contract and actual search output formats
remain authoritative. It does not adopt the original prompt's path/URL image
arguments, summarization component, or mandatory output tags.

`AGENT_BEHAVIOR_VERSION=3` separates these trajectories from earlier Agent
runs. Base, SFT, and RL comparisons made after this change must use the same
prompt version. Previous `base-dev30-v2` and `base-eval300-v1` artifacts must
not be resumed under version 3; use a new run ID. For Dev-30 validation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id base-dev30-v3 \
  --config configs/eval_4b.yaml \
  --search-config configs/search_backends.example.yaml \
  --layout-config configs/layout_parsing.example.yaml \
  --cache-dir .eval-runtime/cache \
  --selection-manifest data/eval/dev30/dev30_manifest.json

python scripts/run_judge.py \
  --run-id base-dev30-v3 \
  --config configs/judge.example.yaml \
  --dataset data/eval/combined_eval_300.parquet
```

## Phase 6B.1 image and question contract

`AGENT_BEHAVIOR_VERSION=4` makes the actual width and height of each registered
input image visible in the system message. Tool observations now give the new
`img_n` ID and the actual derived image dimensions, including after crop
clipping. The crop guidance asks for pixel coordinates within the listed image
bounds and a meaningful nonempty region; backend clipping and validation are
unchanged.

Before model generation, the Agent removes only a leading dataset wrapper of
the form `image_id: <filename.jpg/png/...> Question: <question>` (allowing
whitespace and case variations). The frozen question remains unmodified in the
batch trajectory record's `question` field. Other question text is preserved.
This prevents dataset filenames from competing with registered `img_n` IDs in
the model's input.

Use a new Dev-30 run ID for the version 4 protocol; existing v3 artifacts
cannot resume under this version. The model, selection manifest, tool configs,
cache directory, eight-turn limit, and Judge configuration remain the same:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id base-dev30-v4 \
  --config configs/eval_4b.yaml \
  --search-config configs/search_backends.example.yaml \
  --layout-config configs/layout_parsing.example.yaml \
  --cache-dir .eval-runtime/cache \
  --selection-manifest data/eval/dev30/dev30_manifest.json

python scripts/run_judge.py \
  --run-id base-dev30-v4 \
  --config configs/judge.example.yaml \
  --dataset data/eval/combined_eval_300.parquet
```

## AutoDL CUDA validation

Install the pinned requirements and a matching CUDA build of PyTorch first.
Then run:

```bash
# Model/processor load, eval mode, BF16 CUDA environment, and load peak VRAM.
python scripts/run_4b_smoke.py \
  --config configs/eval_4b.yaml \
  --load-only \
  --report reports/4b_load_smoke.json

# One frozen sample: processor + multimodal generation.
python scripts/run_4b_smoke.py \
  --config configs/eval_4b.yaml \
  --index 0 \
  --report reports/4b_smoke.json

# One sample from each frozen benchmark (combined file is sorted by benchmark).
python scripts/run_4b_smoke.py \
  --config configs/eval_4b.yaml \
  --indices 0 100 200 \
  --report reports/4b_three_benchmarks_smoke.json

# Real 4B model + parser + all mock-backend Agent machinery.
# This prompt explicitly requests a tool call and is not a benchmark result.
python scripts/run_4b_agent_smoke.py \
  --config configs/eval_4b.yaml \
  --index 0 \
  --synthetic-tool-prompt \
  --report reports/4b_agent_smoke.json
```

The last command fails rather than claiming success if the synthetic prompt
does not result in at least one parsed and successfully executed tool call.
Running it without `--synthetic-tool-prompt` leaves the frozen question intact;
a direct final answer is valid, and the report separately records whether a
tool-call chain was observed.

## Smoke diagnostics and CUDA initialization

Both CUDA smoke wrappers explicitly resolve `cuda:0`, call `set_device`, and
initialize CUDA before the first peak-memory reset. This ordering avoids the
PyTorch lazy-initialization failure where the first memory-statistics call can
raise `RuntimeError: Invalid device argument` even though CUDA is available.
All later reset/read operations reuse the same resolved `torch.device`.

On failure, `error` is an object containing `type`, `message`, `stage`, and
`traceback`; representative stages include `config_load`, `cuda_init`,
`cuda_memory_reset`, `model_load`, `sample_load`, `preprocessing`,
`generation`, `agent_runtime`, and `report_write`. Successful reports retain
`"passed": true` and `"error": null`.
