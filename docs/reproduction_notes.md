# Phase 0 reproduction notes

Status: **READY FOR GPU EXECUTION - NOT YET PASSED**

This file is completed automatically from `reports/*.json` after running
`bash scripts/run_phase0.sh` on the target 2 x A800 80GB machine. Until those
artifacts exist, training-dependent acceptance items remain intentionally
unchecked.

## Fixed references

- OpenSearch-VL repository: https://github.com/shawn0728/OpenSearch-VL
- Reference commit: `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`
- Dataset: `OpenSearch-VL/Search-VL-SFT-36K`
- Dataset revision: `2c1c460af4fa15bd63210cbf426a96664b959944`
- Base model: `Qwen/Qwen3-VL-2B-Instruct`
- Base model revision: `89644892e4d85e24eaac8bacfd4f463576704203`
- Training method: BF16 LoRA; vision tower and multimodal projector frozen
- LoRA targets: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## Planned smoke configuration

| Item | Value |
| --- | --- |
| Samples | 100, stratified across 7 sources |
| GPUs | 2 |
| Per-device batch | 1 |
| Gradient accumulation | 4 |
| Effective global batch | 8 |
| Optimizer steps | 20 |
| Max sequence length | 32,000 |
| Attention | SDPA |
| Checkpoint | `outputs/phase0/qwen3_vl_2b_lora/adapter` |

## Runtime evidence

Run `python scripts/render_reproduction_notes.py` after Phase 0. It fills in:

- Python, PyTorch, CUDA, Transformers and PEFT versions
- exact GPU names and count
- trainable parameter count and percentage
- dataset source counts and label-mask statistics
- first/final loss and finite-loss check
- peak allocated/reserved GPU memory
- proof that a LoRA parameter changed
- adapter reload result and generated text

## Acceptance checklist

- [x] independent reproduction repository scaffolded
- [x] upstream code and dataset revisions recorded
- [x] environment versions pinned/documented
- [x] deterministic 100-sample preparation implemented
- [x] full preprocessing inspection implemented
- [x] prompt/observation masking logic unit-tested locally
- [ ] Qwen3-VL-2B loaded on CUDA
- [ ] multimodal forward passed
- [ ] exactly 100 official SFT samples prepared on target host
- [ ] LoRA trainable/frozen parameter audit passed
- [ ] at least 10 optimizer steps completed (configured: 20)
- [ ] all losses finite
- [ ] a LoRA parameter changed numerically
- [ ] adapter checkpoint saved
- [ ] adapter reloaded in a fresh process and forward/generation passed

## Known constraints

The official dataset stores images in seven large ZIP archives. A 100-sample
run only extracts selected images, but downloading the relevant complete ZIPs is
unavoidable with the published layout. GPU-dependent results cannot be produced
on the current Windows host (Python 3.7, no PyTorch/CUDA); they must be generated
on the requested AutoDL A800 machine.
