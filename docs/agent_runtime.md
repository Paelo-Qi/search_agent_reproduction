# Qwen3-VL-4B inference and Agent smoke

This stage provides infrastructure only. It does not implement real tool
backends, external APIs, benchmark scoring, SFT, or RL.

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
