# Benchmark failure modes — a running list, NOT a benchmark design

Status: **LIST ONLY. Nothing here is implemented, and nothing here is claimed to
be novel.** Per the research directive §11, no benchmark is built yet. This file
exists so the failure modes we actually measured are not lost, and so a future
benchmark is derived from the diagnosis rather than invented alongside it.

## Hard preconditions before any of this becomes a project

1. **Literature search is mandatory first.** Winoground, SpatialSense, VSR,
   ARO/CREPE, SugarCrepe, Cola, EQBEN, and the VG150 debiasing line all overlap
   parts of this space. Novelty must be established against them, not assumed.
   Several items below are *probably already covered* and are marked as such.
2. **A benchmark is only justified if a real model gap survives.** As of this
   writing the diagnosis is incomplete (see `PAIR_PRIOR_DISTILLATION_RESULT.md`).
3. The distinguishing idea, if there is one, is **not** "another spatial
   relation dataset". It is: *does the prediction change when the RELATION
   changes while the lexical/object shortcut is held fixed?*

## Failure modes, each tied to evidence we actually have

| # | failure mode | our evidence | likely already covered? |
|---|---|---|---|
| F1 | **Prior dominance** — a frequency prior alone reaches most of the score | prior-only R@50 66.59 / mR 22.30 on the full split, model adds +0.575 R (`p28`) | yes — the SGG debiasing literature |
| F2 | **Subject–object shortcut** — pair identity alone reproduces the model's contribution | `p26`: pair-matched null +2.026 vs full +1.911, difference −0.114 ± 0.265 | partly — ARO/CREPE test word order, not pair-conditioned priors |
| F3 | **Calibration masquerading as capability** — tau moves mR without any visual input | `p25`: no-vision arms average −1.2 Pareto; tuning tau beats learning a per-class rule | **under-covered.** Most benchmarks report a single operating point |
| F4 | **Top-1 vs global ranking** — a model can improve argmax while worsening rank | model worsens mean GT rank 1.83→2.78 yet yields +762 net beneficial flips (`p28`) | **under-covered** |
| F5 | **Role binding / direction** — does `<a, on, b>` survive swapping a and b | NOT measured here. Role-swap regularizer exists in the checkpoint but at weight 0 | yes — SpatialSense, VSR |
| F6 | **Same pair / different relation** — fixed (s,o), the relation must decide | NOT measured. This is the one F2 most directly motivates | **plausibly under-covered** — the natural test of F2 |
| F7 | **Same relation / different pair** — generalization of a predicate across pairs | NOT measured | partly |
| F8 | **Counterfactual relational change** — minimal image edit that changes only the relation | NOT measured; would need generated or paired imagery | EQBEN is closest |
| F9 | **Long-tail predicate robustness under a fixed prior** | head/body/tail reported throughout; tail moves mostly via calibration | yes |
| F10 | **Pair-sparsity honesty** — 33% of full-split rows have a (s,o) pair seen once | `p30` audit: 33.2% of rows are not estimable out-of-fold | **under-covered, and it bites evaluators too** |

## The one that looks most defensible

**F6, controlled by F2 and F3.** The measured fact is that a pair-conditioned
null reproduces the checkpoint's contribution to within noise. The benchmark
that *would* have caught this is one where (subject, object) is held fixed by
construction across items with different ground-truth relations, so that pair
identity carries **zero** information and the score is forced through the image.

Under such a construction a pure pair prior scores at chance by design — which
is exactly what VG150 cannot do, because there P(p | s,o) is most of the answer.

Two things must be settled before this is worth building:
- whether F6 is genuinely uncovered (literature search),
- and whether any current model actually fails it, or whether it merely fails
  our checkpoint — one model's failure is not a benchmark.

## Reporting requirements a future benchmark should inherit from this work

Independent of what it tests, it should require what our own protocol had to
learn the hard way:

- report an **operating-point separation**, never a single tau;
- report **out-of-fold** numbers with the fold rule fixed in advance;
- resample the fold partition — one partition is one draw (`p22` +2.894 vs
  `p25` +1.911 ± 1.056 was the same experiment);
- ship an **identity-destroying null and an identity-preserving null**, since
  the gap between them is where the actual claim lives;
- state the **denominator** and never compare across two of them.
