# `runs/p57` — LAYOUT-PARTIAL: `rel_feat` keeps size and overlap, and throws away *where*

Run: exit 0, 159 s. Pre-registered in
`docs/GEOMETRY_DECODABILITY_PREREGISTRATION.md` (commit `db6dec3`), before the
tool existed. No threshold moved.

This is the run that gives **H6 positive content**. After `p55` and `p56` closed
the readout, objective and supervision routes, H6 survived only by elimination —
which cannot direct an architecture. It now names a deficit.

## Validity gates — all PASS

| gate | requirement | observed |
|---|---|---|
| X1 | null controls near zero, **two-sided** | N1 −0.0058, N2 −0.0072 |
| X2 | `rel_feat` (132556, 768), finite | PASS |
| X3 | folds identical to `p37` | `[26483, 26856, 27190, 26586, 25441]` |

## Result — out-of-fold R² of `rel_feat` predicting each box feature

| feature | R² | | feature | R² |
|---|---|---|---|---|
| **log_area_ratio** | **0.889** | | dy_img | 0.410 |
| **obj_containment** | **0.799** | | obj_cy | 0.332 |
| **obj_w** | 0.785 | | subj_cy | 0.327 |
| **subj_h** | 0.784 | | obj_logaspect | 0.232 |
| **obj_h** | 0.749 | | subj_logaspect | 0.227 |
| **subj_w** | 0.733 | | dy_rel | 0.223 |
| **subj_containment** | 0.704 | | subj_cx | 0.210 |
| **iou** | 0.679 | | obj_cx | 0.178 |
| feat19 | 0.606 | | dx_img | 0.119 |
| | | | **dx_rel** | **0.052** |

```
R2_geom (mean over the 19 features) = 0.4756   ->  LAYOUT-PARTIAL
D2  rel_feat -> the geometry probe's 50 logits   R2 = 0.6503
D3  geometry -> rel_feat (orientation)           R2 = 0.3123
```

## The finding

The 19 features split cleanly into two groups, and the split is not arbitrary.

**Retained (R² 0.68–0.89) — everything about SIZE and OVERLAP:**
area ratio, both widths, both heights, IoU, both containments.

**Discarded (R² 0.05–0.41) — everything about RELATIVE POSITION:**
all four centre coordinates, both image-scale offsets, both pair-relative
offsets, both aspect ratios.

**The single worst-decoded feature in the whole set is `dx_rel` at R² = 0.052 —
the pair-relative horizontal displacement.** It is, to a linear readout,
essentially *not in the representation at all*.

There is also a consistent vertical/horizontal asymmetry: `dy_img` 0.410 vs
`dx_img` 0.119, and `dy_rel` 0.223 vs `dx_rel` 0.052. The encoder retains
roughly 3–4x more about vertical displacement than horizontal. A CLIP-derived
feature having some vertical structure (sky above, ground below) while
collapsing left–right is a plausible reading, and is offered as a reading, not a
measurement.

## Why this matters

Relative displacement is precisely what separates the directional predicates —
`above`/`under`, `in front of`/`behind`, `left of`/`right of` — and it is the
part of the layout the encoder has dropped. Meanwhile size and containment,
which it keeps, mostly support the *partitive and support* predicates
(`has`, `of`, `part of`, `on`) that the frequency prior already predicts well.

This is consistent with `p40` from the opposite direction, where geometry won
the spatially decidable contrasts (`above` vs `next to`, `behind` vs `under`)
and the model won the functional ones (`holding` vs `wearing`, `of` vs
`part of`). `p57` supplies the mechanism for that split: **the model cannot win
spatial contrasts because the spatial quantity that decides them is not linearly
present in its features.**

## The verdict is LAYOUT-PARTIAL, and that is the honest reading

`R2_geom = 0.4756` sits in the middle band. The registered label is
**LAYOUT-PARTIAL**, not LAYOUT-ABSENT, and the aggregate should not be quoted as
"the layout is missing". The *aggregate* is unremarkable; the **structure** is
the result, and the structure is specific: position is gone, size is kept.

## Limitations, as registered

R² measures **linear** decodability. Low R² does not prove the information is
absent in an information-theoretic sense — only that it is not linearly
available. That is nonetheless the relevant sense here, because the geometry arm
that beats `rel_feat` is itself a **linear** probe on those same 19 numbers: the
comparison is like-for-like. A nonlinear decoder might recover more, and that is
untested.

Single checkpoint, validation split, PredCls with GT pairs.

## Execution note — attempt 1 was bugged and its verdict is withdrawn

`runs/p57_FAILED_constant_column` printed **LAYOUT-ABSENT**. That was an
artifact: `_standardise` appends a constant bias column, whose zero variance
made its R² evaluate to −1.4e13 and dragged the mean to −6.8e11. Gate X1 also
**passed spuriously**, because it tested `n < 0.02` one-sided and a large
negative garbage value satisfies that.

Both were my errors, both are fixed (constant targets screened, X1 made
two-sided), and the archived directory carries a `STOPPED.md`. **Every
per-feature number above is byte-identical between the two attempts** — the bug
lived only in the aggregate and the gate, not in the measurements.
