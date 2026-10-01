# RL-0: framework-independent foundation

This is a local, CPU-testable scaffold. It is **not** an RL trainer and does not
perform online rollouts. It leaves Base Eval, SFT, Eval-300, Agent prompts and
the v3 `image_search.image_id` contract unchanged.

The paper/original implementation motivate the multiplicative reward
`r_fmt * (0.8*r_acc + 0.2*r_query)`, three consecutive model-caused tool
errors for fatal detection, fatal-prefix preservation, and a one-sided fatal
advantage clamp. The local Agent's parser, declarations, `AgentTrajectory`,
and image-ID protocol are authoritative. Infrastructure failures abort reward
computation; they are never scored as model mistakes. A search with no results
is neutral for fatal detection and breaks the error cascade.

`rl/reward.py` reuses the existing DeepSeek correctness judge interface. Query
utility has an independent prompt/parser and an injected request callback;
the callback is not connected to a live provider in RL-0. Its rubric covers
relevance, progression, complementarity, evidence usefulness, redundancy and
noise. Malformed/provider responses fail closed. Future integration must
provide bounded retries, scheduling and API credentials.

`rl/checkpoint.py` calls the existing `adapter_identity` implementation to
verify SFT adapter checksums, model/revision and protocol. It then checks
LoRA structure and complete-stage lineage against the local formal SFT
configuration. The default RL plan expects the existing v3 `checkpoint-3k`
adapter; if that checkpoint is absent locally, preflight correctly fails.
The RL run manifest is a *schema/function only*; no RL checkpoint is saved.

The three configs specify logical rollout group sizes (2 for smoke, 4 for
pilot/main), not physical GPU batch sizes. Their `runtime_protocol: current`
is resolved from the local `tool_contracts` constant. No FSDP/vLLM/Ray or
memory topology has been selected.

Run read-only checks:

```bash
PYTHONPATH=src pytest tests/test_rl_*.py
python scripts/prepare_rl_data.py  # schema-only dry run; no download/write
python scripts/preflight_rl.py --config configs/rl_smoke.yaml
```

With an absent adapter, the last command exits nonzero by design. A later
local source JSON may be inspected with `prepare_rl_data.py --input ...
--dataset-id ... --dataset-revision ... --limit ...`; it only prints a
deterministic plan. Source records need `source_sample_id`, `question`, and
local `image_paths`. Image hashes use decoded pixels; question hashes use
existing SFT normalization. `overlap_audit()` accepts externally computed
frozen Eval-300/SFT question/image hash sets. No RL-8K download or final
300–400 prompt membership is made in this phase.

Deferred until AutoDL Gate A: selecting/validating an actual rLLM/verl/vLLM
version; converting local trajectories to framework episodes; correct
token-level fatal response masks; a trainer implementing RLOO and the
advantage clamp; live DeepSeek query-judge transport and retries; GPU rollout,
memory and distributed tuning; RL checkpoint saving and Eval-300 comparison.
