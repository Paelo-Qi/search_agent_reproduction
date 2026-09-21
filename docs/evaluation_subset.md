# Frozen Phase 1 evaluation subset

The Base, SFT, and SFT+RL stages must all read the same ID manifests and must
not draw new samples.

## Provenance and sampling

- Dataset: `Osilly/Vision-DeepResearch-Eval`
- Revision: `deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9`
- Seed: `20260506`
- Strategy: fixed random sampling without stratification, after sorting stable IDs
- Selected: SimpleVQA 100/300, MMSearch 100/171, VDR-Bench 100/500

The actual source schemas differ slightly. SimpleVQA stores `id` as `int64`;
MMSearch and VDR-Bench store it as `large_string`. VDR-Bench alone contains
`question_original`. Output IDs are normalized to strings. The VDR-specific
field remains in `vdr_bench_100.parquet`; the combined file contains only the
seven genuine common fields:

```text
id, benchmark, question, answer, images, image_caption, image_packed
```

## Frozen output checksums (SHA-256)

```text
simplevqa_100.parquet
9a73313156061329c0d90c1ca6465805d1c9cc18ffc5a0d893f630b33e4f12a8

mmsearch_100.parquet
919d4d3f5299d9e11d5e0a2069f4ca1cc8184f15326b07ca6b37d39ff23ac9e8

vdr_bench_100.parquet
8190d466708589ad0970291013fa33759d35bfc716e020cadf917748d5d1f522

combined_eval_300.parquet
f9d0ca74f98d3f73cd6ad1f60b7f2294c5e5d59d4956b60ad95084dc8ee36cf4
```

All 300 selected samples have non-empty IDs, questions, and answers. All 300
packed images were decoded fully with PIL; image decode failures were zero.
Running the preparation command twice reproduced identical ID and parquet
checksums.

These subsets support controlled comparisons in this reduced reproduction.
They do not constitute complete official benchmark results. MMSearch will be
reported with final-answer accuracy only, not its official end2end, requery,
rerank, or summarization composite score.
