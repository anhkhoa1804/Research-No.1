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

---

# GPU contention log — 2026-09-01 20:2x UTC

Checked before launching the pre-registered test-split pass, as the resource
policy requires:

```
pid 26415  /home/leanhkhoa150204/.venvs/vlm-screen/bin/python
           research/experiments/screen_vlm.py --device cuda --role page region
           15,242 MiB of 23,034 MiB   97% utilisation
```

Different venv (`vlm-screen`), different script, different session scratchpad —
**this is the other project.** It is a substantial process by any reading: 66% of
VRAM and 97% utilisation.

**Action: did not launch.** `runs/p54_test_relfeat_cache` is pre-registered
(`docs/TEST_SPLIT_REPLICATION_PREREGISTRATION.md`) and stays unlaunched until the
L4 is free. ~7.8 GB of VRAM was technically available and `p36` only needed
~4.75 GB, so the job would have *fit* — but at 97% utilisation both jobs would
have roughly halved in throughput, which is competing, not sharing.

All work continued on CPU.

---

# Update — 2026-09-02. Assets secured, blocker narrowed to one file.

Re-checked the ranked paths. Three findings change the picture; the gate is
still not cleared and **no cross-model number exists.**

## 1. Path 1 (ask for prediction files) is DEAD as written

I read Tang's README directly. **Scene-Graph-Benchmark.pytorch distributes no
pre-computed evaluation results** — only weights (Causal MOTIFS-SUM for
SGDet/SGCls/PredCls, via OneDrive) plus a reference results table. The
`eval_results.pytorch` file this document hoped for is produced *locally* by a
user's own evaluation run and is not published. No mirror of such a file was
found for any repo. Path 1 was the cheapest option and it is not available.

## 2. A much better codebase exists than the ones previously surveyed

`bknyaz/sgg` (BMVC 2020 / ICCV 2021, built on Neural Motifs) is a materially
better target than anything in the original table:

| property | Tang / DRM | **bknyaz/sgg** |
|---|---|---|
| framework | maskrcnn-benchmark | **self-contained, `torchvision.models.detection`** |
| torch pin | 1.4–1.9 | **>= 1.2**, no upper pin |
| custom CUDA ops | many | **none** — one pure-Cython ext (`draw_rectangles`) |
| PredCls checkpoints | yes | **yes** (IMP+, IMP++, IMP++GAN, +GraphN) |
| PredCls inputs | GT boxes | **GT boxes** |

**The sm_89 blocker this document was built around does not apply to it.** There
is no CUDA kernel to compile against Ada.

## 3. The checkpoint is downloaded and verified complete

`IMP+` (the Neural-Motifs/IMP baseline arm of the ICCV 2021 paper) downloaded
successfully, 2.54 GB. Inspected with `torch.load(weights_only=True)` — a safe
load that cannot execute code from an untrusted pickle:

```
top keys      state_dict, optimizer, epoch, global_batch_iter
entries       86
parameters    387,507,278
detector.*    40 entries  (VGG16 backbone, RPN, roi_heads.box_head/box_predictor)
union_boxes.* 14 entries
rel_fc.*      2 entries   <- the predicate head, the thing WPRD needs
```

**It is a complete self-contained model.** The separate Zellers detector
download (which failed on a Google Drive quota error) is **not needed** — the
detector weights are inside this checkpoint.

## 4. The one remaining blocker: `VG-SGG.h5`

The model's dataloader needs the standard VG150 annotation bundle
(`VG-SGG.h5`, `VG-SGG-dicts.json`, `image_data.json`). Status:

- Published only inside **`VG.tar`, 26,021,427,200 bytes (26 GB)** on Yandex
  Disk, which bundles the VG images. We already have the images locally
  (`~/VG150_dataset_extract/images`, 15 GB) and only 49 GB of disk is free, so
  downloading the whole archive is neither necessary nor safe.
- The server **does** honour HTTP Range (`Accept-Ranges: bytes`, verified with a
  206 response), so selective extraction is possible in principle. I scanned for
  512-byte tar headers at seven offsets across the final 4 GB and found **none**,
  meaning that region is the payload of a small number of very large members;
  the header offsets were not located within the time budget.
- **Not on Hugging Face.** `VG-SGG` returns zero hits in both the model and
  dataset APIs; the VG150 datasets that are on HF (`maelic/VG150-coco-format`,
  `JosephZ/vg150_*`) are in other formats and contain no `.h5`.

## 5. A structural point that makes the eventual test cheaper than assumed

**WPRD is split-agnostic.** It needs only, per candidate pair: the model's
predicate logits, the GT predicate, and the subject/object labels. It does *not*
need the model evaluated on *our* images or joined to *our* cache. A cross-model
run can therefore be computed entirely inside the other model's own evaluation,
on its own split, with its own prior control verified to read 0.5000 there.

This removes the alignment work the original plan assumed and means the only
real cost is *running the model at all*.

## 6. Why no number is reported

A ported model whose recall has not been checked against its published number is
not evidence. If `rel_fc`'s outputs were mis-assembled, WPRD would still return
a plausible-looking value near 0.55 and we would have no way to know it was
wrong. **Reproducing the published R@50/mR@50 first is a precondition for
trusting the WPRD**, and that check itself requires `VG-SGG.h5`.

So the honest status is unchanged in substance and much improved in position:

> **Cross-model WPRD: NOT YET RUN.** One published PredCls checkpoint is now on
> disk and verified structurally complete, on a codebase with no CUDA blocker.
> The remaining step is obtaining `VG-SGG.h5` — a byte-range extraction from a
> 26 GB archive, or any mirror of that one file.

Every claim in this programme remains a measurement of one checkpoint.


---

# Correction — 2026-09-02, verified from disk

**Section 3 above ("The checkpoint is downloaded and verified complete") is
STALE. The IMP+ checkpoint is NOT on disk.**

Verified: `downloads/` is empty; a filesystem-wide search for `*.h5`, `*.tar`,
`*IMP*.pth` and for any file >500 MB outside the repo returns only
`~/VG150_dataset.zip`. The 2.54 GB download described above was evidently made
into a session scratchpad that has since been cleared. **It must be re-downloaded
before any cross-model work; do not plan as if it were available.**

Also re-checked this session, with network confirmed working
(`curl -I https://huggingface.co` -> 200):

- **`VG-SGG.h5` is still not on Hugging Face.** Searches over the dataset and
  model APIs for `VG-SGG`, `VG_SGG`, `vg150`, `scene-graph-benchmark` and
  `visual genome scene graph` return no `.h5` artifact.
- **A new and better path was found.** `maelic/VG150-coco-format` is the
  **standard VG150 split** (top-150 objects / 50 predicates, Xu et al.
  selection) in COCO JSON, published as parquet, produced for SGG-Benchmark and
  used in the REACT paper. It contains the standard split membership, boxes and
  vocabulary — i.e. **everything `VG-SGG.h5` carries** — in a format that is
  freely downloadable and needs no byte-range extraction from a 26 GB archive.

**Revised cheapest path to the cross-model gate:**

1. Download `maelic/VG150-coco-format` (parquet, no images needed for PredCls
   with GT boxes).
2. Write a converter COCO-JSON -> `VG-SGG.h5` schema (`labels`, `boxes_1024`,
   `boxes_512`, `img_to_first_box`/`last_box`, `relationships`, `predicates`,
   `img_to_first_rel`/`last_rel`, `split`). This is a documented, fixed schema.
3. Re-download the `bknyaz/sgg` IMP+ checkpoint (no CUDA blocker; sm_89 is fine).
4. **Run ordinary PredCls evaluation and reproduce the published R@50/mR@50.**
   This is the gate. A converter bug shows up here, not in WPRD.
5. Only after step 4 passes, compute WPRD.

Estimated: ~1 day of engineering, ~1-2 GPU-hours for step 4. Step 2 is the risk;
step 4 is the check that makes the risk survivable.
