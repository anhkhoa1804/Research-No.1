# Pre-registration — can a LEARNED candidate-restricted scorer beat the additive arm?

Status: PRE-REGISTERED. Written and committed before the tool produces any number.
Branch: research/architecture-breakthrough
Cache: runs/p10_model_recalibration/pair_logits.pt (read-only, frozen)
Compute: CPU only. No GPU. No training of any visual encoder.

## Why this experiment exists

The oracle ceiling (runs/p14, runs/p15) returned EXHAUSTED on its realizable arm
in 27/27 cells: `model_rerank`, the argmax of the raw model score inside the
prior's top-k, is worse than the prior's own argmax everywhere except a 10%-of-rows
budget, where it reaches +0.42 dR against the additive arm's +0.67.

That arm is the alpha -> infinity limit. It is forced to use the raw score and it
DISCARDS the prior entirely inside the candidate set. So it bounds the raw score
as a ranker; it does NOT bound the learnable capacity of the cached features. A
learned scorer that sees both terms strictly generalizes the prior baseline --
the prior's own top-1 is always inside its candidate set, so it can always
reproduce the prior -- and can only lose through estimation error.

Until that is measured, "EXHAUSTED" is an inference, not a result. This is the
cheapest experiment that converts it into one, and it is the gate on whether any
GPU time is justified.

## Question

Can a scorer restricted to the prior's top-k candidates, using ONLY the frozen
cached features, beat the achieved additive arm at the R@50 floor, out-of-fold?

## Arms

All four arms use the same rows, the same denominator, the same vocabulary
ordering, the same folds, and the same fitting pipeline. Only the feature block
differs. Every reported number is OUT-OF-FOLD.

| arm | features | what a gain would mean |
|---|---|---|
| `prior_only` | prior logit, candidate rank, prior margin, prior entropy, class log-marginal, class identity | a learned DECISION RULE / per-class calibration. No visual information whatsoever. This is the calibration-only null. |
| `model_only` | model term only | the visual/model score as a learned ranker |
| `full` | prior_only + model_only | the quantity of interest |
| `shuffled_model` | prior_only + model term permuted across rows | null. Refit through the identical pipeline; its mean is NOT assumed to be zero. |

`class identity` in `prior_only` is deliberate: a free per-class additive bias is
exactly a learned generalization of tau. If `prior_only` reproduces the gain,
then the gain is calibration and not model information (directive section 9).

## Protocol

- Rows: the 38,053 GT rows over 3,000 images already in the cache.
- Candidates: the prior's top-k, k in {3, 5}. The prior's top-1 is always a
  candidate by construction, so every arm can reproduce the prior baseline.
- Model: multinomial logistic over the k candidates (listwise softmax +
  cross-entropy against GT). Linear first. An MLP is fitted ONLY if the linear
  arm shows signal, and is reported separately.
- Cross-fitting: 5 folds split by IMAGE id, never by row. Rows from one image
  never appear in both the fit and the evaluation of the same fold. Predictions
  are taken out-of-fold only.
- Operating points: tau in {0.0, 0.05}, reported separately and never mixed
  (directive section 13).
- Seed: 0, fixed and recorded. Fold assignment is a deterministic hash of the
  image id.

## Primary metric

dR@50 in points against the prior-only argmax at the SAME tau, out-of-fold.

## Floor

R@50 >= 66.5, binding on every arm. An arm below the floor is rejected
regardless of its mR.

## Secondary metrics (reported, not criteria)

mR@50, head/body/tail mR, per-predicate contribution, candidate coverage,
Pareto gap against the prior-only tau frontier, fraction of rows whose argmax
changes.

mR is NOT a criterion here for the reason the mechanism report established: the
mR axis at these operating points is carried by two predicates and its bootstrap
CI spans zero.

## Decision rule — committed before the run

Let `A` = achieved additive dR (+0.673 at tau=0, +0.612 at tau=0.05) and let
`F` = out-of-fold dR of the `full` arm, subject to the R@50 floor.

- **SUCCESS**, H4 (candidate-ranking bottleneck) supported, GPU justified:
  `F > A + 2.0` and `full` clears the floor and `full - shuffled_model > 1.0`.
- **INCONCLUSIVE**: `A + 1.5 < F <= A + 2.0` at the floor.
- **EXHAUSTED**, H6 (representation bottleneck) supported, no reranker:
  `F <= A + 1.5`.

Independently of the above, and reported either way:

- If `full - prior_only <= 0.25` points, then the model features add nothing
  beyond a learned decision rule. This is recorded explicitly as a POSITIVE
  finding for branch B (decision formulation), not as a failure.
- If `prior_only > A`, then a rule with NO visual input reproduces or exceeds
  the entire C' gain, and the C' headline must be re-stated in the reports.

## What would falsify the framing itself

If `shuffled_model` also beats `A`, the pipeline is manufacturing gain from the
fitting procedure rather than from any feature, and no arm may be reported.

## Compute budget and stop rule

Budget: CPU only, under 30 minutes total. If the linear arms are all EXHAUSTED
and `full - prior_only <= 0.25`, STOP. Do not fit the MLP, do not touch the GPU,
and move to branch B.
