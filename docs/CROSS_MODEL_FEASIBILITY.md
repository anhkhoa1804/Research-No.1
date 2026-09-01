# Cross-model WPRD — a concrete feasibility assessment

The gate on every general claim in this programme is running WPRD on published
SGG checkpoints. This records what was actually checked, so the blocker is a
documented engineering fact rather than an assumption.

---

## What was checked

| repo | PredCls checkpoints? | framework | required torch / CUDA |
|---|---|---|---|
| **Scene-Graph-Benchmark.pytorch** (Tang) | **yes** — Motifs/VCTree/VTransE, causal TDE variants | maskrcnn-benchmark | torch ~1.4–1.9, CUDA 10/11 |
| **DRM** (CVPR 2024) | **yes** — `DRM_VG_Stage1/2_PredCls`, Google Drive | maskrcnn-benchmark | **torch 1.9.1, CUDA 11.1** (stated) |
| **SGG-Benchmark** (Maelic) | **no** — SGDet only (REACT++/YOLO) | modern, torch 2.2.1 | torch 2.2.1, py 3.11 |
| PENET, ST-SGG | code released; checkpoints not confirmed | maskrcnn-benchmark | torch 1.x |

**Every repository that ships VG150 PredCls weights is built on
maskrcnn-benchmark and pinned to torch 1.x.** The one modern, torch-2.x
codebase ships **no** PredCls checkpoints.

## The hardware blocker, measured

```
torch 2.9.1+cu129   CUDA 12.9
device NVIDIA L4    capability sm_89
arch list ['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']
```

The L4 is **sm_89** (Ada). torch 1.9.1 predates Ada and its binaries target at
most **sm_86**. It may or may not run via PTX JIT from an embedded
`compute_86` — that is genuinely uncertain and would have to be tested, not
assumed. Installing a torch-1.9.1 + CUDA-11.1 environment and compiling
maskrcnn-benchmark's custom CUDA extensions against a driver-580 / CUDA-13 host
is a substantial and failure-prone piece of work.

**Assessment: the blocker is environment compatibility, not the metric, and not
the science.**

## Why this does not block the metric

`tools/sgg_evaluation_table.py --extra name=path.pt` takes only per-pair
predicate logits:

```
{"model_term": Tensor(132556, 50)}                  # GT-row aligned
{"per_image_logits": [Tensor(n_pairs_i, 50), ...]}  # cache order per image
```

Nothing about WPRD needs the model, the framework, or this machine. Any party
who can run one of those checkpoints anywhere can produce that file.

## Ranked paths, cheapest first

1. **Ask for prediction files rather than checkpoints.** Tang's evaluation
   already writes `checkpoints/MODEL/inference/.../eval_results.pytorch`. One
   such file for VG150 PredCls answers the question with **zero** GPU and zero
   porting. This is by far the best option and should be pursued first.
2. **CPU inference in an isolated torch-1.x venv.** PredCls with GT boxes on
   10,401 images is slow but bounded, and sidesteps sm_89 entirely because no
   CUDA kernel is needed. Custom CUDA ops still have to compile, or be stubbed —
   many are only needed for detection, not for PredCls with GT boxes.
3. **Port a single predicate head.** Motifs' PredCls head is an LSTM over object
   labels plus ROI features; re-implementing it against released weights is
   tractable but is real work and risks silent divergence.
4. **Retrain a small SGG model here.** Last resort. Answers a weaker question —
   *our* model, not *published* models — and re-introduces exactly the
   single-checkpoint limitation the study exists to remove.

## Consequence for claims, restated

Until one of these lands, **every result in this programme is a measurement of
one checkpoint.** That is stated in `docs/DIAGNOSIS.md`, the literature audit,
and the Paper-1 table, and it is the reason no novelty claim has been made.

The p49 rank-inversion result — `Spearman(mR@50, WPRD) = −0.650` — is the one
that most needs this, because its interest is entirely in whether it generalises
beyond a family of scoring functions built from a single cache.
