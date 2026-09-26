# Corrected SFT tool-protocol diagnostics

This is a protocol-regression workflow, not a new QA benchmark or a change to
SFT/Eval-300. Do not tune on Eval-300 or Dev30; the earlier 20 Eval-300 examples
are historical bug evidence only. All scripts fail closed unless the corrected
8k pool manifest has SHA256
`0faf210483978435e808e4ba8ce4fb2556fb27ecfe30267e948f5cc2f1c9c637`.

Run the CPU audits where the corrected shards, their extracted images, and the
pinned Qwen processor are available. No model weights or GPU are loaded:

```bash
python scripts/audit_sft_runtime_prompt_alignment.py
python scripts/audit_sft_supervised_url_patterns.py
python scripts/audit_sft_tool_distribution.py
# Optional, much slower: include extra_1k and reserve_4k in the label-only scan.
python scripts/audit_sft_supervised_url_patterns.py --include-full-8k
```

Reports appear under `reports/sft_diagnostics/`. Prompt alignment samples 15
trajectories per 1k/2k shard and exports real SFT `build_messages`, processor
render, collator labels, and AgentRuntime initial messages rendered with the
same processor. `semantic_drift` denotes a behaviorally meaningful structural
or instruction difference, `exact_match` denotes equality of the compared
contract, and `cosmetic_only_drift` denotes a known render-stage difference.
It is a diagnostic, not an automatic prompt rewrite. In particular, the
training trajectory contains literal `<tool_call>` text while the Agent
runtime stores structured `tool_calls` after execution; the report displays
both. The URL audit only counts decoded `labels != -100` assistant tokens as
supervised; system, user, and tool observations have separate masked groups.

Local CPU-only metadata rebuild reproduced the corrected pool manifest SHA256
exactly, then produced this expert-target distribution (not model behavior):

| Metric | main_a_1k | main_b_2k |
|---|---:|---:|
| Tool-containing samples | 99.2% | 98.5% |
| Mean tool calls/trajectory | 2.641 | 2.614 |
| image_search calls | 897 | 1,788 |
| layout_parsing calls | 74 | 147 |
| crop calls | 128 | 264 |
| image_search share of these three visual tools | 81.62% | 81.31% |
| image_search followed by text_search | 96.96% | 97.68% |

No large 1k-to-2k expert-target tool distribution change is apparent. This
does not explain the observed model regression by itself. The local machine
cannot produce the actual prompt/label URL reports because it lacks both the
materialized SFT images and pinned processor dependencies.

Static code/source inspection already identifies a plausible **semantic
train/serve prompt difference**: the pinned SFT source system describes a
"Tool-First Mindset" and says text_search **MUST** be used for external
facts, while `AGENT_SYSTEM_GUIDANCE` explicitly allows no tool for a clear,
directly answerable image and forbids repeating identical calls. The SFT
message builder appends image registration before its runtime-rule suffix;
the Agent runtime appends registration after its guidance. SFT expert calls
are literal `<tool_call>` text, whereas the runtime stores executed calls as
structured `tool_calls` in subsequent history. These are audit hypotheses,
not proof of causal attribution; the actual processor report above is still
required before changing any prompt or training recipe.

Build the fixed, Eval-300-isolated dev set from pinned source JSON and the
**local official source image ZIPs**. No ZIP is downloaded and no image is
copied into Git:

```bash
python scripts/prepare_tool_protocol_dev.py
```

The script checks the corrected pool checksum, pinned source hashes and
revision, frozen Eval-300 checksum, normalized question overlap, decoded-pixel
image overlap, SFT 8k membership, all 127 frozen exclusions, and the raw
runtime image-ID contract **before writing** `ids.json` and
`tool_protocol_dev50_manifest.json` under `data/eval/tool_protocol_dev50/`.
Its deterministic seed is `20260927`. Stratified target categories are
image_search(img_1), registered later-image search, layout_parsing, crop,
other image tools, no-tool, non-image_search visual calls, and multi-tool.
Categories can overlap. Scarce categories are reported instead of fabricated.
Each item includes its selection category, all protocol tags, and expert
reference tool sequence; no
`expected_tool_label` is asserted unless later manually curated and
re-checksummed, because the expert's first tool is not necessarily unique
ground truth.

The local source-only pass found 28,423 candidates after corrected 8k,
frozen-exclusion, format, question-overlap, and image-contract filtering.
Among them, 244 support registered later-image search, 311 have no tool call,
1,961 include layout_parsing, and 2,841 include crop. These are **not yet
image-overlap-certified**; no final dev50 IDs are written until decoded-pixel
hash checks against all 300 Eval images pass.

Once the dev set is verified, run the production Agent runtime on it with
separate run IDs. These commands may call the configured external tool APIs,
so set their credentials and budgets first. Never point this runner at
Eval-300 data; it reads only the dev manifest and pinned source ZIPs:

```bash
python scripts/run_tool_protocol_dev.py --run-id base-tooldev50
python scripts/run_tool_protocol_dev.py --run-id sft1k-tooldev50 \
  --adapter outputs/sft_main/checkpoint-1k/adapter
python scripts/run_tool_protocol_dev.py --run-id sft3k-tooldev50 \
  --adapter outputs/sft_main/checkpoint-3k/adapter
python scripts/eval_tool_protocol.py \
  --trajectories reports/tool_protocol_dev_runs/base-tooldev50/trajectories.jsonl
```

Repeat the metrics command for each SFT run (choose a separate `--report-dir`
to retain all three outputs). The BatchRunner's run manifest and status/resume
semantics are reused; re-running the same command and run ID resumes pending
samples. Metrics concern registered IDs, invalid HTTP/path arguments,
unknown IDs, provider-not-called, duplicate calls, tool selection when a
reliable label exists, and call counts. They do **not** score final answers.

Local development machines without source image ZIPs, materialized SFT images,
or the pinned processor cannot certify prompt/mask alignment, image overlap,
or dev50 construction. A successful unit test or source-only distribution
report must not be misreported as those gates passing.
