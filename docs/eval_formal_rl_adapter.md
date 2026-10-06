# Eval300 with a Formal RL Main PEFT adapter

This change is for the **evaluation machine only**. Do NOT pull this source
change onto the machine continuing the existing Formal Main child: Main's
`software_binding()` hashes all Python files in src AND scripts. The new
inference code and preflight script change that inventory even though no RL
training/control-plane source or frozen semantics were modified. They cannot
be used as an in-place training upgrade; do not rewrite/reseal its identity.

## Portable eval bundle

The supported bundle is the original, byte-for-byte manifest plus the entire
original adapter directory:

```text
outputs/rl_eval/policy25/
  checkpoint.json
  adapter/
    adapter_config.json
    adapter_model.safetensors
    ...all other files declared in the adapter role (e.g. README/index)...
```

Continue passing just `--adapter outputs/rl_eval/policy25/adapter`. Adapter-only
copies without parent metadata fail closed. Never create a fake SFT
`metadata.json`, copy SFT metadata onto RL, edit checkpoint.json, or change the
exported adapter. The bundle parent need not be named `policy-000025`; identity
comes from the manifest, not the directory name.

## Validation and provenance

`inference.adapter.adapter_identity()` keeps the existing SFT branch and exact
identity structure. If SFT metadata exists, failure is NOT retried as RL.
Only when SFT metadata is absent and `checkpoint.json` exists does RL validation
run. It uses `rl.checkpoint.validate_checkpoint_manifest()` and the existing
training-run identity validation, requiring runtime `main_checkpoint` eligibility.
The bound base name/revision and current tool/image protocols must match.
Policy iteration/global step must be positive integer counters and agree.

The adapter role must explicitly include config and safetensors under adapter/.
Formal role validation checks role maps against the global sealed inventory;
`verify_artifacts()` then verifies **only the adapter directory**, with exact
file membership and SHA256 checks, including every declared ancillary file.
Missing/extra/tampered files are rejected; declared README/index files are
allowed. Config base_model_name_or_path must match the evaluation base.
Native, optimizer, RNG, rank evidence and trajectory bytes need NOT be copied
to the evaluation machine. Their manifest metadata is still validated, not
deleted or rewritten. This is not full distributed reload, chain recovery,
or a new GPU training verification.

The existing Eval manifest stores the returned adapter identity unchanged,
including `training_origin=formal_rl_main`, `policy_iteration`,
`global_optimizer_step`, `run_identity_sha256`, base/revision, source SFT
lineage and runtime tool/image protocols. `checkpoint_identity` is the existing
`checkpoint_manifest_sha256`, not a newly invented checkpoint ID. The raw
checkpoint manifest SHA, PEFT file/config fingerprints and full adapter-role
fingerprint are also bound. PEFT loading itself is unchanged and inference-only.
Existing SFT identities gain no fields, preserving their resume fingerprints.

## Repackage on the training machine (without pulling this Eval change)

Run only against the already committed, immutable policy25 directory:

```bash
tar -C outputs/rl_formal_main/formal-main400-s4-continuation-v7-attempt1/checkpoints/policy-000025 \
  -czf /tmp/rl-policy25-eval-bundle.tar.gz checkpoint.json adapter
```

Transfer that archive to the evaluation machine. Extract into a NEW/empty
destination (do not mix files from different copies/checkpoints):

```bash
mkdir -p outputs/rl_eval/policy25
tar -xzf /path/to/rl-policy25-eval-bundle.tar.gz -C outputs/rl_eval/policy25
```

If policy25 was copied previously with only adapter/, replace the incomplete
eval copy using a clean bundle destination. Do not overwrite training artifacts.

## Offline preflight and Eval300

```bash
python scripts/preflight_eval_adapter.py \
  --adapter outputs/rl_eval/policy25/adapter \
  --config configs/eval_base_300.yaml
```

Preflight reads config/metadata and hashes adapter files only. It does not read
the Eval dataset, load a model, initialize CUDA or call an API. Errors propagate
with nonzero exit; success prints origin, policy/step, checkpoint identity and
adapter fingerprint, then `PASS — adapter identity only`.

After real bundle preflight succeeds on the evaluation machine:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_agent_batch.py \
  --run-id rl-policy25-eval300 \
  --config configs/eval_base_300.yaml \
  --adapter outputs/rl_eval/policy25/adapter \
  --eval300
```

For audited v2, use `configs/eval_base_300_v2.yaml` and a distinct run ID. Do not
mix v1/v2 or adapter checkpoints within an existing run manifest. If an old run
ID already has an incompatible manifest, choose a new ID; never edit it to
force resume. Dataset, generation, tools, prompts and judging are unchanged.

All packaging/Eval commands above are instructions, not locally executed
results. No actual AutoDL policy25 artifacts were available for this change.
Synthetic runtime-shaped CPU fixtures test the validators only; they are not
evidence of a real RL update or policy25/Eval300 PASS.

Targeted CPU acceptance only (no full pytest):

```bash
PYTHONPATH=src python -m pytest \
  tests/test_inference_rl_adapter.py tests/test_sft_main.py \
  -k 'adapter or manifest or rl' -o addopts= -q -p no:cacheprovider
git diff --check
```
