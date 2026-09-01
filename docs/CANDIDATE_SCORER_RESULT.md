# Result — the learned candidate-restricted scorer

Pre-registration: `docs/CANDIDATE_SCORER_PREREGISTRATION.md` (commit `ea12228`)
Runs: `runs/p18_candidate_scorer`, `runs/p19_candidate_scorer_balanced`,
`runs/p21_scorer_pareto_frontier`, `runs/p22_scorer_nested` (authoritative)
Superseded: `runs/p20` (×100 Pareto scaling bug)
Compute: CPU only. No GPU. No visual encoder was trained.

Status: **EXPLORATORY / SCREENING** on the 3,000-image analysis set.
Confirmation is pre-registered in `docs/FULL_VALIDATION_PREREGISTRATION.md`.
Nothing here may be quoted as a headline until that run reports.

---

## 1. The pre-registered criterion: EXHAUSTED, 4/4 cells

ΔR of the `full` arm against the achieved additive arm, out-of-fold, 5-fold
cross-fitted by image:

| cell | full − achieved | full − prior_only | null − prior_only | model_only ΔR |
|---|---|---|---|---|
| tau=0.00, k=3 | +0.263 | +0.788 | +0.026 | −0.134 |
| tau=0.00, k=5 | +0.208 | +0.778 | −0.003 | −0.131 |
| tau=0.05, k=3 | +0.717 | +0.681 | +0.032 | −0.113 |
| tau=0.05, k=5 | +0.728 | +0.794 | +0.029 | −0.089 |

Against a pre-registered SUCCESS bar of +2.0 and INCONCLUSIVE of +1.5:
**EXHAUSTED in every cell.** On the R axis, learned candidate-restricted scoring
on frozen features does not justify a reranker. That verdict stands.

Two MEASURED facts inside that table matter more than the verdict:

- **`model_only` is negative in all four cells** and fails the R floor at
  tau=0.05. Fitted optimally within the candidate set, the model score alone
  cannot beat the prior's argmax. This is direct support for **H6
  (representation bottleneck)**.
- **`full − prior_only` is +0.68 to +0.79, stable across every cell.** This is
  the model's genuine complementary contribution: cross-fitted, out-of-fold, and
  null-controlled. It is the cleanest measurement of C' the project has, because
  C' itself was neither cross-fitted nor null-controlled.

## 2. A pre-registered null condition that I mis-specified

The pre-registration said: *if `shuffled_model` also beats `A`, the pipeline is
manufacturing gain and no arm may be reported.* It fired at tau=0.05, k=3.

It fired for the wrong reason, and the condition was wrong, not the pipeline.

**VERIFIED FACT**: a correct null here carries no model information, so the fit
falls back on the prior features and *reproduces `prior_only`*. That is exactly
what it does: `|null − prior_only| ≤ 0.032` in all four cells. The pipeline is
clean.

What the as-written condition actually detects is *a learned decision rule
beating C'* — a real finding about calibration, not a pathology. The gate is now
`null − prior_only > 1.0`. Both verdicts are computed and stored in the
artifact, so the mis-specification stays visible rather than being dropped.

This is recorded as a protocol correction, not as a result.

## 3. The class-balance dial, and why the R axis was the wrong axis

Plain cross-entropy maximises accuracy, which on VG150's label distribution
means maximising head recall. Row weight `count(GT class)^-beta` parameterises
it. beta=0 is plain CE; beta=1 is fully balanced. Counts come from the training
rows of each fold only.

At the endpoints (tau=0, k=5): beta=0 gives R@50 67.70 / mR@50 19.92; beta=1
gives R@50 28.25 / mR@50 **42.21**. A +20-point mR swing and a −38.6-point R
collapse from one scalar.

So the learned scorer, like tau, is a one-parameter family trading the same two
axes. Judging it on ΔR alone — as the pre-registration did — measures one end of
a dial and calls it the method. **The correct comparison is between the two
frontiers**, because only one of them has seen an image.

## 4. The frontier comparison — the falsifiable form of the question

`runs/p21`, tau=0, k=5, Pareto gap = mR minus the prior-only tau frontier's mR
at the same R@50, in mR points (`cprime_analysis.pareto_gap`):

| arm | beta | R@50 | mR@50 | Pareto gap | floor 66.5 |
|---|---|---|---|---|---|
| achieved additive C' | — | 67.474 | 22.837 | +0.861 | ok |
| prior_only | 0.20 | 66.134 | 24.963 | −1.054 | **FAIL** |
| shuffled null | 0.20 | 66.192 | 24.847 | −0.963 | **FAIL** |
| **full** | 0.20 | **66.657** | **25.165** | **+2.862** | **ok** |

The null tracks `prior_only` within 0.1 at every beta.

## 5. The operating point removed: nested selection

beta=0.20 was read off the table in §4, so as it stands it is a
validation-optimised operating point. `runs/p22` removes it: per outer fold, beta
is chosen on a disjoint *inner* fold by maximising mR@50 subject to R@50 ≥ the
floor, then refitted on all non-outer folds. No outer-fold label reaches either
the fit or the choice.

**tau=0, k=5, no chosen constant anywhere in these numbers:**

| arm | R@50 | mR@50 | ΔR | ΔmR | Pareto gap | floor | betas chosen |
|---|---|---|---|---|---|---|---|
| prior only (tau=0) | 66.802 | 21.976 | — | — | 0 (defines it) | ok | — |
| achieved additive C' | 67.474 | 22.837 | +0.673 | +0.861 | +0.861 | ok | — |
| prior_only | 65.651 | 26.354 | −1.151 | +4.378 | **+0.059** | **FAIL** | [.2,.3,.15,.3,.2] |
| shuffled null | 65.495 | 26.608 | −1.306 | +4.632 | **+0.224** | **FAIL** | [.25,.3,.15,.3,.2] |
| **full** | **66.673** | **25.161** | −0.129 | +3.185 | **+2.894** | **ok** | [.3,.15,.2,.05,.2] |

`full − prior_only`: ΔR +1.022. `full − null`: ΔR +1.177.

> **⚠ This table is ONE fold partition.** `runs/p25` resamples it and the
> absolute gap does not survive at this magnitude. Read §5b, not this table.

## 5b. Resampled over 5 independent fold partitions (`runs/p25`) — authoritative

Same nested procedure, five deterministic re-partitions of the images
(`fold_of_image` salts 0–4). This is the number of record; §5 is salt 0 alone.

| salt | full R@50 | full mR@50 | full Pareto | floor | prior_only Pareto | null Pareto |
|---|---|---|---|---|---|---|
| 0 | 66.657 | 24.182 | +1.879 | ok | −0.026 | +0.037 |
| 1 | 66.857 | 23.777 | +1.802 | ok | −0.814 | −0.608 |
| 2 | 66.809 | 24.917 | +2.941 | ok | −1.419 | −1.361 |
| 3 | 66.216 | 25.921 | +0.244 | **FAIL** | −1.831 | −1.833 |
| 4 | 66.741 | 24.803 | +2.690 | ok | −1.937 | −1.922 |
| **mean ± sd** | **66.656 ± 0.257** | | **+1.911 ± 1.056** | **4/5** | **−1.205 ± 0.793** | **−1.137 ± 0.838** |

**Two findings of opposite sign, and both are the result.**

**WEAKENED — the absolute magnitude.** The +2.894 reported in §5 was a
favourable draw. The resampled mean is **+1.911 ± 1.056**, ranging +0.244 to
+2.941, and on salt 3 the arm **fails the R@50 floor** (66.216). It holds the
floor on **4 of 5** partitions, so it is not yet a reliably usable operating
point. Any future quotation of "+2.894" is a single-partition number and must
be labelled as one.

**STRENGTHENED — the separation from the nulls.** Per-partition
`full − prior_only` = [+1.904, +2.615, +4.360, +2.075, +4.627], minimum
**+1.904**; `full − shuffled_model` minimum **+1.842**. Every partition, without
exception, separates the model-bearing arm from both nulls by more than 1.8
Pareto points. The separation is far more stable than the absolute level,
which is expected: the absolute gap also carries the noise of where nested
selection lands, while the difference cancels it.

**CORRECTION to §5's wording.** On salt 0 the calibration-only arms sat
essentially *on* the tau frontier (+0.059, +0.224). Across five partitions they
sit **slightly below it** (mean −1.205 and −1.137) and clear the R@50 floor on
**0 of 5**. So the correct statement is stronger than the one made from salt 0:
a learned per-class decision rule with no visual input does not merely fail to
*beat* tau — on average it does not *match* it. Tuning tau is better than
learning a calibration.

**What to carry forward.** The defensible quantity is the **separation**
(≥ +1.84 Pareto points over both nulls on every partition), not the absolute
gap. `full` has *lower* mR than `prior_only` and a *better* Pareto gap; that is
not a contradiction, because the gap measures mR at matched R and `prior_only`
bought its mR with recall it could not afford.

## 6. What this does and does not establish

**INFERENCE, supported:** the model's complementary information is real and
small, and the additive alpha/tau formulation is a poor converter of it. A
candidate-restricted, class-reweighted decision rule converts roughly three
times as much of it into Pareto movement, at the same R@50 floor.

**NOT established:** that this survives the full validation split; that it
survives on the test split. **And the magnitude is now measured NOT to be
stable** (§5b): it ranges +0.244 to +2.941 across fold partitions and the arm
fails the R@50 floor on one of five. N here is 38,053 rows over 3,000 images,
and the frontier is estimated, not analytic.

**Explicitly not claimed:** that global ranking improved. It did not — see
`docs/ORACLE_CEILING_RESULT.md` §2 and the mechanism report. Nothing here
contradicts the finding that the model worsens mean GT rank.

## 7. Bugs found and fixed in this stage

1. **Candidate-0 overwrite** (`889483e`). Forcing `cand[:,0] = argmax` duplicated
   a column on the 522 tied rows and silently shrank the candidate set to k−1;
   top-3 coverage read 85.2% against the true 85.5%. Fixed by the shared
   canonical ordering.
2. **Pareto gap scaled ×100** (`889483e`... frontier column). `pareto_gap`
   already returns mR points. The achieved arm read +86.079 instead of +0.861 —
   and +0.861 is exactly its ΔmR, the tau=0 lower-bound branch of the frontier,
   which is the identity that exposed it. `runs/p20` is superseded by `p21`.
3. **Mis-specified null condition** (§2).

## 8. Decision

Do **not** train a reranker. Do **not** scale architecture.

Extend the cache to the full validation split with one frozen forward pass and
run the three analyses unchanged, against the criterion pre-registered in
`docs/FULL_VALIDATION_PREREGISTRATION.md` before the pass was launched.
