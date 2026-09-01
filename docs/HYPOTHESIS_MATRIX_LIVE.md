# Live hypothesis matrix — H1–H10 accounting scheme

Updated **2026-09-01 after `p46`, `p47`, `p48`**. `p36`/`p37` executing; rows
depending on them marked **PENDING**. Numbering follows the directive's scheme;
earlier revisions used a different numbering and are retained as history.

Classes: **VF** verified fact · **MR** measured · **INF** inference · **HYP** hypothesis.

**Reference scale for every WPRD claim** (0.5 = no image-conditioned relational
information):

```
prior (control)            0.5000   exactly, CI [0.5000, 0.5000], every stratum
random null                0.5046
PURE text head (EVALUATED) 0.5542   <- 47% of the way to the box ceiling
PURE classifier (discarded)0.5728   <- 63%
geometry linear            0.5961
geometry MLP  (box ceiling)0.6153   <- lower bound on the achievable ceiling
pixels ceiling             PENDING (p37)
```

---

| # | Hypothesis | Supporting evidence | Contradicting evidence | Status | Next test |
|---|---|---|---|---|---|
| **H1** | **Prior dominance** | **MR** prior alone 66.59 R@50; model adds +0.575 (`p24`). **MR** prior is **85.8%** of the model term's variance (`p42`). **MR** deployed system is **97.45%** pair-constant-predictable vs reality's 69.23% (`p44`) | — | **ESTABLISHED** (not novel — Zellers; Plesse quantified 50–75%) | settled |
| **H2** | **Pair identity carries the composition gain** | **MR** `full − pair_matched_null` = +0.031 ± 0.188 (`p29`), replicating `p26`; third replication in `p32` (−0.129) | **MR** `p42`: WPRD is invariant to the between-group part, so this is a statement about the *composed metric*, not about the term | **ESTABLISHED for the composed gain** | settled |
| **H3** | **Calibration / decision artifact** | **MR** tau moves mR 22.3→26.4 with no image access. **MR** geometry has best WPRD and **worst** mR@50 (`p47`) | **MR** a learned per-class rule cannot even match tau (`p29`) | **ESTABLISHED for the mR headline** | settled |
| **H4** | **Image-conditioned information exists** | **MR** WPRD 0.5542 / 0.5728, CI excludes 0.5 by ~9 half-widths (`p33`). **MR** the non-layout residual excludes chance in both heads (`p42`) | **MR** converts to ≈0 at the additive operating point (`p29`) | **ESTABLISHED BUT WEAK** — 47% of the box ceiling | `p37` **PENDING** |
| **H5** | **Readout bottleneck** | **MR** the two heads differ by 0.0186 with disjoint CIs; discarded head wins 13/13 strata (`p35`) | **MR** `p48`: plain CE already near-optimal at *extracting* within-pair signal from fixed features | **PENDING** — `p37` is the test | `p37` |
| **H6** | **Representation bottleneck** | **MR** both heads cap at ≈0.57 vs a box ceiling of 0.6153 (`p46`). **MR** `p45` shows the composition route is bounded by signal quality. **MR** `p48` refutes the objective route on the box channel | **MR** the readout gap of 0.0186 is real, so some loss is downstream | **PENDING — NOW THE LEADING CANDIDATE**, by elimination | **`p37`** |
| **H7** | **Geometry shortcut** | **MR** geometry-linear 0.5961 and geometry-MLP **0.6153** beat both heads (`p39`, `p46`); margin *grows* toward the tail, +0.123 at tail–tail (`p40`) | **MR** `p41`: on the field's composed metric the model **beats** geometry at tau ≤ 0.05, because discrimination ≠ calibration | **ESTABLISHED for discrimination, REVERSED for the composed metric** | run geometry on other checkpoints |
| **H8** | **Annotation artifact** | **MR** VG multi-labels one instance pair; those rows tie at 0.5 | **MR** excluding same-instance rows **raises** WPRD 0.5542→0.5592 (`p35`) | **PRESENT BUT CONSERVATIVE** — cuts against our finding | quantify per bucket |
| **H9** | **Candidate-generation bottleneck** | — | **MR** GT in prior top-5 for 89.5%; scorer EXHAUSTED 9/9 (`p28`) | **FALSIFIED** | settled |
| **H10** | **Benchmark metric mismatch** | **MR** `p47`: Spearman(R@50, WPRD) = **+1.000**, Spearman(mR@50, WPRD) = **−0.400**, Spearman(Pareto, WPRD) = −0.400. The best-discriminating arm has the worst mR@50 | **n = 4 arms.** ρ=−0.400 is not distinguishable from zero at that size | **SUGGESTIVE, NOT ESTABLISHED** | **more models in the table — the gate on Paper 1** |

## Retired hypotheses

| | |
|---|---|
| **Text-lookup** (the term is non-visual) | **FALSIFIED** — `text_logits = normalize(rel_feat)·normalize(pred_emb)ᵀ`, a cosine against an image feature. Source-reading error, retracted. |
| **Prior absorption** (the model over-absorbed the prior) | **FALSIFIED** — model term alone 68.97% pair-constant-predictable vs reality's 69.23% (`p44`) |
| **Prior overwrite** (composition discards *usable* evidence) | **DESCRIPTIVELY TRUE, PRESCRIPTIVELY WEAK** — restoring it gains +1.07 vs a registered +2.0, at ~1:1 cost (`p45`) |
| **Objective bottleneck** | **REFUTED on the box channel** — contrastive 0.6020 < CE 0.6163 (`p48`); re-run on `rel_feat` pending |

## Current scientific model

The prior decides most rows. The checkpoint's within-pair variation has roughly
the right **magnitude** but weakly correct **direction**, reaching **47%** of a
ceiling that 19 numbers from two rectangles already attain. Composition discards
~91% of it — and that is close to rational, because the variation is mostly
wrong. On the **33.4%** of rows where the prior is wrong, the model fixes 7.9%,
geometry 6.8%, both 10.5%; **~89% is recovered by nothing.**

**By elimination, the binding constraint is the representation** (H6): the
composition route is bounded (`p45`), the objective route is refuted on the one
channel where it could be tested (`p48`), and the readout route is a real but
small 0.0186. `p37` tests H5 vs H6 directly.

**A dataset-level explanation has appeared and is not yet excluded:** only
**19%** of VG150 train groups contain ≥2 distinct predicates, so the supervision
that could teach within-pair discrimination is scarce by construction (`p48`).

## What would overturn this

- `p37`: a probe on `rel_feat` ≫ 0.6153 ⇒ H6 dead, H5 up, cheap successor justified.
- WPRD ≈ 0.5 on published checkpoints ⇒ we measured a PURE quirk; H10 and the
  benchmark claim collapse.
- A within-pair objective beating CE **on `rel_feat`** ⇒ `p48` was channel-specific.
