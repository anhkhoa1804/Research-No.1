# Prototype protocol: prior-controlled relational grounding on existing VG150

**No new data for the components marked BUILT.** Two components (C6 strong,
C9) were checked and found **not** answerable from VG150; that is recorded
rather than assumed. The protocol is the contribution; a dataset is only justified if
the protocol first shows that existing data cannot answer a component.

The question the protocol asks:

> **Did the model use the visual evidence that distinguishes the relation, after
> the object-pair prior has been neutralised?**

Status per component: **BUILT** = implemented and run; **SPEC** = defined, not
yet built; **NEEDS DATA** = not answerable from VG150.

---

## C1 — WPRD, the core metric · **BUILT** (`p33`, `p35`, `p42`)

Within an (s,o) group the prior is constant to 9.4e-05, so the double
difference cancels it exactly, along with per-class calibration, tau and any
temperature. WPRD is the AUC of `score[·,a] − score[·,b]` separating GT-`a` rows
from GT-`b` rows **inside one group**.

- **Measures:** relational discrimination with P(p|s,o) made *exactly*
  non-informative.
- **Validation:** the prior control reads **0.5000, CI [0.5000, 0.5000]**, in
  every stratum. This is the property that distinguishes the protocol from
  curation- or re-split-based approaches, which only reduce the prior's
  informativeness.
- **Coverage on VG150 val:** 75,366 rows (56.9%) are decidable.
- **Invariance:** `WPRD(term) == WPRD(term − group mean)` exactly, so it reads
  only the within-group half of any score.

## C2 — the baseline panel · **BUILT** (`p33`, `p39`, `p41`)

Every model reports against four references on identical cells:

| baseline | must score | credit |
|---|---|---|
| pair prior P(p\|s,o) | **exactly 0.5** | — |
| random tensor | ≈0.5 | — |
| **language/prior-only** | — | **SpatialSense (ICCV 2019)** |
| **2D/geometry-only** (19 box features, train-fitted) | 0.5961 here | **SpatialSense (ICCV 2019)** |

A model that does not clear the geometry baseline has not demonstrated
relational grounding beyond layout. **PURE does not clear it** (text 0.5542,
classifier 0.5728).

## C3 — the discrimination/calibration split · **BUILT** (`p41`, `p42`)

The protocol reports WPRD **and** the composed R@50/mR@50, because `p41` showed
they can disagree: geometry out-discriminates the checkpoint yet the checkpoint
wins the composed metric at tau ≤ 0.05. `p42` explains why — WPRD reads the
within-group half (14.2% of the text head's variance), the composed metric is
driven by the between-group half (85.8%).

- **Measures:** whether a model's metric performance comes from grounding or
  from being calibrated against the prior.
- **This is arguably the protocol's most transportable idea** and it is not
  specific to SGG.

## C4 — same (s,o), different predicate · **BUILT** (it is C1's population)

The decidable subset *is* this test. VG150 already contains it at scale
(75,366 rows). No collection needed.

## C5 — long-tail relational discrimination · **BUILT** (`p35`, `p40`)

WPRD stratified by predicate bucket. Finding: for the evaluated head, body–tail
[0.4822, 0.5570] and tail–tail [0.4174, 0.6246] **contain chance**; geometry's
advantage *grows* toward the tail (+0.033 head–head → +0.123 tail–tail).

- **Known weakness:** tail–tail rests on **61 cells**. This is the component
  most in need of more data, and **Haystack** already supplies rare-predicate
  annotations with explicit negatives — reuse before collecting.

## C6 — role swap / directional reversal · **DOWNGRADED after counting**

I specced this as SPEC and then counted it, as the spec required. The count
kills the strong version and weakens the weak one.

| quantity | VG150 validation |
|---|---|
| groups whose role-swapped pair (B,A) also occurs | 11,626 (25.5%), 75,996 rows |
| ...with **both** directions decidable (≥2 GT) | 1,960 groups, **40,188 rows** |
| role-swapped pairs with **different** GT sets | 11,442 of 11,626 |
| role-swapped pairs co-occurring **in the same image** | **0** |

Two conclusions:

1. **The strong version is NEEDS DATA, not SPEC.** Same image, same two object
   instances, roles exchanged: **zero instances exist.** VG annotates
   (A, rel, B) once; it does not annotate the reverse direction for the same
   instance pair. No amount of analysis recovers this from VG150.
2. **The weak version does not inherit the protocol's key property.** WPRD's
   prior cancellation works because it compares rows *within one group*. A
   role-swap comparison is *between two different groups*, (A,B) and (B,A),
   which have **different priors** — so the prior does **not** cancel, and the
   test would be measuring prior asymmetry confounded with role binding. It
   would need a different construction and would not be prior-free.

**Corrected status: role binding is not testable in a prior-free way on VG150.**
It is a genuine gap in the protocol and an honest motivation for future data —
alongside C9, and for the same underlying reason (VG contains no controlled
relational interventions).

## C7 — prior-adversarial cases · **SPEC**

Restrict to rows where the GT is **not** the pair's majority predicate — the
cases where following P(p|s,o) is actively wrong. Report accuracy and WPRD
there.

- **Measures:** whether the model can override the prior when the image
  contradicts it.
- **Feasible now:** the pair-constant ceiling on decidable rows is 69.23%, so
  ~30.8% of decidable rows are prior-adversarial by construction.
- **Caution:** VG's incomplete annotation means "not the majority predicate" is
  partly an annotation artifact. Must be reported with C8.

## C8 — annotation-artifact control · **BUILT, partial** (`p35`)

VG labels one instance pair with several predicates; those rows tie at 0.5 in
WPRD. Excluding them **raises** every estimate (text 0.5542 → 0.5592), so the
artifact is *conservative* — it dilutes rather than manufactures the finding.
Any component above must report the excluded-same-instance variant alongside.

## C9 — relational flip consistency · **NEEDS DATA**  *(with C6-strong, the only two)*

Same objects, same image up to a minimal change, relation genuinely different.
VG gives *same-(s,o)-different-image*, which is not a controlled intervention.
**This is the only component that actually requires new data**, and it should
not be collected until C1–C7 are shown to be insufficient.

---

## What a submission would report

```
WPRD            macro / weighted, with the prior control shown at 0.5000
                stratified by predicate bucket and by pair support
baselines       prior (0.5), random, language-only, geometry-only
C3 split        WPRD vs composed R@50/mR@50 at matched operating points
C7              prior-adversarial subset accuracy
C8              same-instance-excluded variant of every number
```

## Honest positioning

- The **confound** is known (Zellers; Plesse et al. 2020 quantify the 50–75%
  per-pair majority).
- The **baselines** are SpatialSense's.
- The **goal** is shared with SpatialSense, VG-OOD and the debiasing line.
- What is new is the **conditioning**: making P(p|s,o) *exactly* non-informative
  by construction on existing annotations, verified by a control that reads
  0.5000, rather than approximately less informative by curation or re-splitting.

## Gate before any of this is called a benchmark

**WPRD has been run on one checkpoint.** Until it is run on published models
(Motifs, VCTree, a TDE variant), this is a measurement of PURE, not a benchmark.
That test is the precondition for the benchmark claim, not a follow-up to it.
