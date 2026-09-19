# Upstream provenance

## OpenSearch-VL code reference

- Repository: https://github.com/shawn0728/OpenSearch-VL
- Reference commit: `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`
- Commit date: 2026-07-30
- License: Apache-2.0

Files consulted to define the Phase 0 interface:

- `README.md`
- `SFT/README.md`
- `SFT/data/dataset_info.json`
- `SFT/examples/agentic_full/qwen3_vl_full_sft_8b_ray.yaml`
- `SFT/src/llamafactory/data/converter.py`

No OpenSearch-VL or LLaMA-Factory source file is copied into this repository. The
local implementation uses Transformers and PEFT as dependencies. It reproduces
the documented ShareGPT column mapping and the paper's policy-token-only loss
rule, adapted for a small BF16 LoRA smoke test.

## Dataset reference

- Repository: `OpenSearch-VL/Search-VL-SFT-36K`
- Hugging Face revision: `2c1c460af4fa15bd63210cbf426a96664b959944`
- Format: seven ShareGPT JSON files plus one `images.zip` per source
- Sources: FVQA, LiveVQA, Palace, WebQA, WikiArt, Wiki-en, Wiki-zh

The smoke-data script uses seeded stratified reservoir sampling and extracts
only the images referenced by the selected records. Because images are released
as ZIP archives, each selected source archive must still be downloaded in full.

## Base-model reference

- Repository: `Qwen/Qwen3-VL-2B-Instruct`
- Hugging Face revision: `89644892e4d85e24eaac8bacfd4f463576704203`
