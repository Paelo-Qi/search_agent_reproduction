# Gate B — actor → rollout multimodal round-trip

Status: **READY FOR AUTODL EXECUTION — NOT YET GPU PASSED**.
CPU tests are not Gate B PASS. A2.2 ws2/ws4 PASS is a prerequisite input already
verified by the user; this change does not rerun A2.2 or start formal RL.

## Scope and ownership

The mandatory chain is:

```text
verified A2.2 FSDP2 exported adapter
→ fresh pinned BF16 base + PEFT safe static merge
→ full model/processor saved and fresh plain-HF forward checked
→ both HF model lifetimes destroyed
→ fresh local vLLM 0.11 engine
→ actual rLLM 0.2.1 MultiTurnWorkflow.run
→ model-generated crop(img_1), real project Pillow backend
→ derived img_2 (parent img_1), image-first tool observation
→ actual derived PIL passed to second vLLM generation
→ final answer before the small Gate turn limit
```

This tests infrastructure, not benchmark accuracy, reward, RLOO, an RL update,
group caching or throughput. It uses a deterministic **diagnostic question**
in memory, asking for one central 50% crop with coordinates computed from the
real first source image. It never alters the frozen smoke question or records.
The model must generate the tool call itself; no injected/scripted model output
is allowed in the GPU entry point. An immediate first-turn answer fails this
particular round-trip probe. Ordinary zero-tool/direct-answer episodes remain
valid in `RLWorkflowAdapter`.

Crop is mandatory because it exercises the complete image path locally without
quota/network variability. The eight original declarations remain unchanged.
Four real local visual backends are available; remote backends explicitly fail
with `gate_tool_disabled` and `provider_called=false`. No provider credentials,
Judge or network API is needed. This Gate does not test live search.

### Actual rLLM components and compatibility layer

The API was inspected in the original project's pinned source at commit
`c5c02a49780e26ae9cb6f1fb56731d1e594d59f0` (rLLM 0.2.1):

- [MultiTurnWorkflow](https://github.com/shawn0728/OpenSearch-VL/blob/c5c02a49780e26ae9cb6f1fb56731d1e594d59f0/RL/rllm/rllm/workflows/multi_turn_workflow.py):
  real `run`, `reset`, generation scheduling and turn limit.
- [Workflow](https://github.com/shawn0728/OpenSearch-VL/blob/c5c02a49780e26ae9cb6f1fb56731d1e594d59f0/RL/rllm/rllm/workflows/workflow.py):
  real `run_with_termination_handling`, `collect_trajectories` and lifecycle.
- Real `TimingTrackingMixin`, `BaseAgent`, `BaseEnv`, `RolloutEngine`,
  `ModelOutput`, `Action`, `Step`, `Trajectory`, `Episode`,
  `TerminationEvent` and `TerminationReason` are used by that workflow.

`build_rllm_workflow` constructs thin `BaseAgent`/`BaseEnv` implementations and
a synchronous-local-vLLM `RolloutEngine` bridge. Only the postprocess hook is
overridden to **avoid** upstream reward/correctness computation. Gym's required
numeric step reward and rLLM dataclass defaults are unused placeholders, marked
`reward_computed=false`; no reward result is calculated or used. The Gate does
not invoke the original reward-integrated DeepResearch workflow.
`workflow.run.__func__ is MultiTurnWorkflow.run` is checked at runtime.
Installed component source hashes and the API-reference commit are recorded;
there is no locally copied substitute generation loop and no `AgentRuntime.run`
call. A real-rLLM CPU lifecycle test is included and skips only when rLLM is
not available in the local environment.

Gate B does not require rLLM Step `prompt_ids`/`response_ids`/`logprobs` to be
saved; these token-level training fields belong to Gate C and are not
implemented in this round.

`RLWorkflowAdapter` owns **one step's** initialization/messages, parsing,
reference validation, duplicate prevention, tool execution, observation commit,
image registration and trajectory export. It has no `run` generation loop.
`AgentInteraction` contains the former Eval runtime's unchanged helpers and
guidance; both Eval and RL inherit them. Eval keeps its original generation
loop and failure handling. The moved helpers and constants were compared to
the prior Git version and are AST-identical; relevant Agent tests cover the
behavior. `image_search(image_id)` remains `runtime-image-id-grounding-v3`:
valid registered IDs only, no legacy `url`, filenames or HTTP references.
Invalid IDs never invoke the backend. No tool schema/prompt/provider/cache key
or frozen Eval/SFT/RL contract is changed.

## Actor identity and static checkpoint

Prefer `--actor-adapter outputs/rl_gate_a22/ws2-attempt2/adapter` together with
that run's `--actor-gate-manifest`. Validation requires literal `passed=true`,
FSDP2, pinned base/revision, world size and identity hash, all A2.2 checks, saved
adapter checksums/output fingerprint, input fingerprint and original verified
SFT lineage (`main_a_1k` → `main_b_2k`). LoRA semantics must match the source SFT.
Missing/wrong metadata or changed weights fail closed.

Explicit fallback: pass the original verified
`outputs/sft_main_imageid_v3/checkpoint-3k/adapter` and omit
`--actor-gate-manifest`. It must match the verified source fingerprint; an
A2.2 artifact without formal SFT metadata cannot masquerade as that fallback.
Fallback validates SFT → rollout, **not** A2.2 → rollout.

All Gate outputs record `formal_rl_initialization_allowed=false`. A2.2's
one-step temporary supervised update is an infrastructure artifact, not a
trained RL policy or long-term formal initialization. Neither input adapters
nor `configs/rl_main.yaml` are modified: formal RL still starts from the
original image-ID-v3 checkpoint-3k.

`rollout_sync.merge_actor_adapter` loads a fresh base, attaches the verified
existing PEFT adapter, uses `merge_and_unload(safe_merge=True)`, checks that the
result is not a PeftModel and contains no active PEFT flag/`lora_` parameters,
then saves full BF16 safetensors + model/processor/tokenizer config. It checks
weight keys and shard indices, destroys the merge-time model, and fresh-loads
the plain full checkpoint for a finite multimodal HF forward. That second HF
model is also destroyed before vLLM is constructed. A tiny CPU regression
exercises the **real** PEFT/safetensors merge/reload path without a GPU model.

Fresh HF validation and the vLLM bridge share `prepare_qwen_vl_processor_inputs`:
collect actual PIL image blocks in message/content order, render the unchanged
messages/tools with `apply_chat_template(tokenize=False, add_generation_prompt=True)`,
then call `processor(text=[prompt], images=[images], return_tensors="pt", truncation=False)`.
HF then moves the inputs to CUDA and checks finite logits as before. This avoids
the Transformers 4.57.1 multimodal `tokenize=True` traversal that treats string
system/tool content as a list of blocks. No message schema is rewritten; zero
images or non-PIL image blocks fail closed, without loading paths/URLs/base64.

Files are built in a hidden same-parent staging directory and atomically
renamed only after validation. Failed staging remains unpublished for
inspection. Existing outputs are rejected, not overwritten/resumed. Source
base/adapter are read-only; the actor fingerprint is rechecked after merge.
Free disk is checked against twice the actual base weight-file size plus
adapter size, not an arbitrary fixed GB assumption. Output/report must be
separate fresh paths outside protected data/model inputs.

`merged_model/merge_manifest.json` binds the pinned logical base/revision,
actor fingerprint/A2.2 identity, original SFT fingerprint/lineage, merge method,
runtime image protocol, software versions, config/processor/tokenizer hashes
and each merged weight hash. The resulting merged checkpoint fingerprint is
deterministic over those fields. Informational timestamps and machine absolute
locators do not enter the identity. The raw A2.2 manifest hash is recorded as
provenance separately because that manifest may contain machine locators.

Dynamic multimodal LoRA is deliberately not retried: the reported vLLM path
already rejected the Qwen-VL dynamic module. There is no `enable_lora` option
and no `LoRARequest`; vLLM loads only this standalone full checkpoint.

## AutoDL commands

Use the existing compatible RL environment: Python ≥3.10, torch 2.8.x,
transformers 4.57.1, PEFT 0.21.1, vLLM 0.11.x and the inspected rLLM 0.2.1.
FlashAttention's installed version is recorded; this Gate does not reinstall
it. No system nvcc change/build, dependency installation, Ray cluster or
torchrun is part of the Gate. All HF access is local-only; a missing cached
base/processor fails rather than downloading a replacement. Set the following
two locators to the existing local pinned snapshot and source image root:

```bash
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BASE_SNAPSHOT=/your/local/pinned/snapshot/ebb281ec70b05090aa6165b016eac8ec08e71b17
export RL_SOURCE_ROOT=/your/local/Search-VL-RL-8K/source-root

# CPU adapter/lifecycle regression; real rLLM case runs when installed.
python -m pytest tests/test_rl_rollout_sync.py tests/test_rl_workflow_adapter.py \
  tests/test_rl_gate_b.py -o addopts= -q -p no:cacheprovider

# Mandatory: one GPU, A2.2 exported actor as input.
CUDA_VISIBLE_DEVICES=0 python scripts/validate_rl_rollout_roundtrip.py \
  --config configs/rl_main.yaml --gate-config configs/rl_gate_b.yaml \
  --data data/rl/smoke20.json --source-root "$RL_SOURCE_ROOT" --sample-index 0 \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --base-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --base-model-path "$BASE_SNAPSHOT" \
  --actor-adapter outputs/rl_gate_a22/ws2-attempt2/adapter \
  --actor-gate-manifest outputs/rl_gate_a22/ws2-attempt2/gate_manifest.json \
  --output-dir outputs/rl_gate_b/a22-tp1-attempt2 \
  --report-dir reports/rl_gate_b/a22-tp1-attempt2 --local-files-only

# Optional additional TP=2 probe; NOT required and NOT rollout_n=2.
CUDA_VISIBLE_DEVICES=0,1 python scripts/validate_rl_rollout_roundtrip.py \
  --config configs/rl_main.yaml --gate-config configs/rl_gate_b.yaml \
  --data data/rl/smoke20.json --source-root "$RL_SOURCE_ROOT" --sample-index 0 \
  --base-model Qwen/Qwen3-VL-4B-Instruct \
  --base-revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --base-model-path "$BASE_SNAPSHOT" \
  --actor-adapter outputs/rl_gate_a22/ws2-attempt2/adapter \
  --actor-gate-manifest outputs/rl_gate_a22/ws2-attempt2/gate_manifest.json \
  --tensor-parallel-size 2 \
  --output-dir outputs/rl_gate_b/a22-tp2-attempt1 \
  --report-dir reports/rl_gate_b/a22-tp2-attempt1 --local-files-only
```

The two locators above are placeholders, not presumed AutoDL paths. If the
verified A2.2 run has a different name, change **both** actor paths together.
For a rerun after failure, retain the report/staging files and use a fresh
output/report attempt; never delete/overwrite A2.2 inputs. Omitting
`--base-model-path` selects only an already cached pinned Hub snapshot.
In particular, retain `a22-tp1-attempt1` as the failed fresh-HF processor audit;
the corrected single-GPU command above uses new `a22-tp1-attempt2` paths.

Default Gate parameters: BF16, TP=1, max_model_len=8192, max_new_tokens=256,
temperature=0, max_turns=4 and memory utilization=0.6. Actual expanded
multimodal processor length plus response allowance must fit; no truncation
or simplified fake processor is allowed. TP is unrelated to rollout_n. TP=2
still performs merge/HF validation on GPU0, then starts fresh vLLM workers.

## Evidence, PASS and failure diagnosis

Successful stdout ends with `Gate B PASS`. Exit status is zero **only** when
every named `checks` entry is the literal boolean `true`, including:

- Valid actor/source provenance, verified first real source image, complete
  static checkpoint and fresh finite plain HF reload, no active PEFT, both HF
  models destroyed, then fresh vLLM loaded.
- Real rLLM workflow, first image accepted, exactly one successful model-generated
  crop with arguments identical to the diagnostic `crop_probe_arguments` (img_1,
  dynamic central-crop x/y/width/height), no extra successful crop, and derived
  img_2 registered with parent img_1 and producing_tool=crop.
- Image hash/dimensions and an actual-PIL receipt at the second vLLM call
  matching img_2, second generation successful, final answer, successful
  trajectory, no tool errors or external provider, vLLM shutdown and artifacts.

`reports/rl_gate_b/<run>/gate_b_report.json` starts atomically with `passed=false`
and records stages. `trajectory.json` records the diagnostic question,
assistant outputs, parsed calls, observations, image parent/hash/dimension
chain, final answer, termination, real rLLM episode, actor/merged fingerprints
and generation receipts. These contain image metadata, **not** image bytes,
base64 or PIL objects; secret values are redacted. The second-generation
receipt is taken directly beside `LLM.generate`, from the same PIL objects
passed to `multi_modal_data.image`; an img_2 textual mention alone cannot pass.
The source record still has its full multimodal membership, while the Gate
deterministically selects only image index 0 (explicitly recorded).

`outputs/rl_gate_b/<run>/gate_manifest.json` is published only for final PASS.
After all checks and successful vLLM shutdown, the final `passed=true` report
is written atomically first; the PASS manifest is atomically published last.
On a caught failure, any manifest from this invocation is revoked before
writing the `passed=false` failure report, and the invocation exits nonzero.
The merge manifest's `merge_complete=true` is **not** a Gate PASS declaration.
HF memory peaks are reported by torch; vLLM worker/KV memory is separate and
this is not a comprehensive worker-memory profiler.

Failures exit nonzero and keep `passed=false` with exception/stage/partial
trajectory and receipts where available. Inspect:

| Stage / origin | Check |
| --- | --- |
| software / actor_provenance / source_probe | Versions, A2.2 hashes/lineage, frozen quality smoke manifest and local image hashes |
| merge_hf_load / peft_static_merge / merged_checkpoint_save | Offline base, adapter compatibility, GPU/disk capacity; hidden staging is not a published model |
| merged_checkpoint_validation / fresh_hf_reload / fresh_hf_forward | Full shards/config/processor, plain model and finite multimodal logits |
| fresh_vllm_init | Actual vLLM Qwen3-VL support, TP/visible devices, worker memory; no dynamic LoRA fallback |
| rllm_owned_roundtrip / model_generation_error | Framework/import/API error versus vLLM generation failure; inspect rllm_error and generation receipts |
| malformed_tool_call / unknown_tool / unknown_image_id | Generated syntax, declaration or registered-ID error; no provider invocation for invalid ID |
| tool_backend_error / max_turns / rllm_workflow_failure | Real crop bounds/backend observation, repeated calls, missing final answer or lifecycle failure |
| vllm_shutdown | Incomplete cleanup is not PASS |

Generation errors remain `model_error`; tool errors retain their distinct turn
metadata and observations. A later answer cannot hide a tool failure from the
Gate checks. No error is converted into a successful rollout/reward=0.

## Next boundary: Gate C (not implemented)

Only after a real Gate B PASS: minimum RL integration must validate whole
prompt rollout groups, format reward, accuracy/query-utility Judge reward,
`r_fmt * (0.8*r_acc + 0.2*r_query)`, RLOO advantages, K=3 model-caused fatal
errors and post-fatal token masks, provider recoverable run interruption,
whole-group atomic commit/resume and one minimal actor RL update. Gate B
implements none of those and does not alter the frozen smoke20/main400,
quality files, rollout_n, reward/fatal contract, SFT adapters or Eval datasets.
