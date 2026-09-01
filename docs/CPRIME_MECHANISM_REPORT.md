# C′ mechanism — the model is a bounded tie-breaker, and its mR gain is two predicates

Follow-up to `docs/MODEL_RECALIBRATION_C_RESULT.md`. **No GPU, no training, no
new pass.** Everything below is CPU over the existing C′ cache
`runs/p10_model_recalibration/pair_logits.pt`, read-only.

| | |
|---|---|
| Analysis | `runs/p12_cprime_mechanism/` — exit 0, 23.3 s, CPU |
| Code | `tools/cprime_mechanism.py`, `tests/test_cprime_mechanism.py` (14 tests) |
| Cache | unchanged, SHA256 `9f8d7931…f4d4` |
| Scope | raw50, α = 3.75, 3,000 images, 38,053 GT rows, 50 classes |
| C′ artifacts | **not modified** — no doc, no result, no `runs/p11_*` file was touched |

---

## 0. DEFECT FOUND IN THE C′ ANALYSIS — reported, not fixed

**D-1 — `cprime_analysis.Bench.ensemble_term` does not reproduce the evaluator's
model term.** It slices to the 50 foreground columns and *then* standardises.
`evals.py::_normalize_eval_logits` (line 975, called at 1431–1432 and 1496–1497)
standardises over the **full 51 columns, background included**, and the
background is suppressed afterwards. The orders are not interchangeable:

```
max |ensemble_term(0) − stored model_logits|  = 1.607e-01     (mean 1.496e-02)
max |fixed_ensemble(0) − stored model_logits| = 3.576e-06
```

The identity is the proof of which order is right: the cache's stored
`model_logits` **are** the evaluator's model term at `ensemble_alpha = 0`, and
normalise-51-then-slice reproduces them across all 3,000 images while
slice-then-normalise-50 does not.

*Why S10 passed.* Cache check S10 computes the correct, full-width comparison —
it is right, and the cache is genuinely self-consistent. It simply cannot see a
defect that lives on the *analysis* side of the column slice, and it samples only
the first 200 images. The cache is not in question; the analysis tool was.

**Blast radius.** `arm_E_ensemble_alpha`, `criterion5_heldout` and
`criterion3_predicate_decomposition` are built on `ensemble_term` and are
affected. `arm_A`, `arm_B_model`, `arm_D`, the nulls, the complementarity block
and the Q4 strata all use the **stored** term and are untouched.

**Impact on published conclusions: none reverses.** Corrected values shift by
≤ 0.15 R points, ≤ 0.15 mR points, ≤ 0.13 Pareto points.

| τ = 0.05, ea = 0 | R@50 | mR@50 | Pareto |
|---|---:|---:|---:|
| as published (buggy `ensemble_term`) | 67.07 | 24.27 | +2.29 |
| corrected (`fixed_ensemble`, = stored term) | **67.09** | **24.29** | **+2.31** |

The two-axis dominance claim holds either way, marginally *more* strongly. The
`ea = 0 beats ea = 1` finding (CORRECTION 2) also survives: at τ = 0.05,
corrected, ea = 0 gives +2.31 and ea = 1.0 gives +1.55.

**A second, cosmetic defect (D-2).** The RESULTS and NULL tables in
`MODEL_RECALIBRATION_C_RESULT.md` are `arm_E` values throughout, but the null
table's column is captioned `real (arm B)`. `arm_B` differs in the third
significant figure. Separately, `arm_Bp_classifier` and `arm_B_text` feed **raw,
unnormalised** branches at α = 3.75, which is exactly the scale confound
`ensemble_term`'s own docstring warns against — but no published claim rests on
them (the only quoted branch numbers are `arm_C` at α = 0, where per-row
standardisation cannot move an argmax, so those numbers are valid).

**Nothing here was corrected in place.** Fixing `ensemble_term` and recomputing
arm E and criteria 3 and 5 is a decision, not a cleanup.

---

## 1. The question, restated correctly

The question as posed mixes two operating points. The rank statistics
(1.83 → 2.78, 4,675 worsened / 2,013 improved, +256 net flips) are the C′
`complementarity` block at **τ = 0**. The **+2.29** is a Pareto gap at
**τ = 0.05**. They are different points on the τ curve and must not be read as
one measurement. Both are analysed below.

## 2. The four claims, separated

| | Claim | Verdict | Evidence (τ = 0, α = 3.75) |
|---|---|---|---|
| 1 | information **exists** in the model | **YES, weakly** | model alone R@50 38.86, R@5 62.07, MRR 0.502, mean GT rank 9.25 over 50 classes. But the majority-class baseline (`on`) is **36.53** — model-alone top-1 beats "always say *on*" by **+2.33 points**. |
| 2 | information **changes scores** | **YES** | argmax changes on 2,930 / 38,053 rows = 7.70 % |
| 3 | information improves **global ranking** | **NO — falsified** | mean rank 1.826 → 2.782; MRR 0.7998 → 0.7815; R@2 84.68 → 81.70; R@5 95.72 → 91.69; R@10 98.81 → 95.07. **Every** ranking metric degrades. |
| 4 | information improves the **final top-1** | **YES, robustly** | +256 net flips; ΔR@50 **+0.673** points, bootstrap CI **[+0.403, +0.963]**, positive in **100.0 %** of 2,000 image-resamples |

Claim 3 "improves" at τ = 0.05 (mean rank 4.516 → 3.193) only because τ *itself*
wrecked the ranking first: τ = 0.05 alone takes mean rank 1.826 → 4.516. The
model repairs damage τ caused; it does not improve on the untouched prior.

## 3. Why the three facts are not in tension

They describe **disjoint row populations**.

**The model has a bounded budget.** It is per-row standardised (unit sd) and
enters at α = 3.75 against a log-prior whose top1−top2 margin has median 5.08.
It can only move an argmax where the prior is nearly tied — and it provably
does not move anything else:

| prior top1−top2 margin decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rows whose argmax changes | **45.97 %** | 18.16 % | 10.36 % | 2.73 % | 0.83 % | **0.00 %** | 0.00 % | 0.00 % | 0.00 % | 0.00 % |

Mean prior margin: **0.989** on changed rows, **5.998** on unchanged rows.

**Inside the budget it is informative.** Rescues concentrate exactly where a
tie-breaker can act — 679 of 1,042 rescues (65 %) come from rows where GT sat at
prior rank 2:

| prior GT rank | 2 | 3 | 4 | 5 | 6+ |
|---|---:|---:|---:|---:|---:|
| rescue rate | **10.64 %** | 3.60 % | 0.94 % | 0.42 % | 0.18 % |

On the rows it rescues the model ranks GT top-1 **68.14 %** of the time, against
**38.86 %** overall. On the rows it destroys it puts GT 2.60 below its own
maximum, against 0.31 on rescued rows. It flips where it is unusually right.

**Outside the budget it is noise on an ordering that was already good.** The
prior's sub-top-1 ranking is strong (R@2 84.68 %), and the model scatters it:

| rows whose prior GT rank was | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| mean rank change | +0.05 | **+2.40** | **+2.92** | **+3.64** | **+3.80** |

The rank-21+ bucket grows **60 → 1,022 rows**. That is the entire mean-rank
degradation: catastrophic damage to a few hundred rows the top-1 decision never
consults.

## 4. Falsification battery — it is information, not noise

At τ = 0, all controls rescored through the identical pipeline (5 seeds):

| arm | Δ R pts | Δ mR pts | net flips | Pareto |
|---|---:|---:|---:|---:|
| **real model term** | **+0.673** | **+0.861** | **+256** | **+0.861** |
| N1 pair-shuffled within image | −1.274 | −0.664 | −485 | −5.054 |
| N2 shuffled across the split | −2.490 | −1.442 | −948 | −6.531 |
| Gaussian, matched per-row scale | −1.733 | −0.618 | −660 | −5.272 |

Every null is strongly negative. The top-1 gain is not a noise or
regularisation artefact.

**The scale sweep is peaked, not monotone** — the signature of a tie-breaker,
not of a ranker. A term carrying genuine global ranking information would keep
improving as you weight it up:

| scale c | 0.125 | 0.25 | 0.5 | **1.0** | 2.0 | 4.0 | 8.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| net flips | +169 | +207 | +244 | **+256** | −120 | −1,518 | −3,880 |
| Pareto | +0.17 | +0.39 | +0.45 | **+0.86** | −2.01 | −8.36 | −17.03 |

**Restricting the model to the prior's top-3 is strictly better than letting it
act globally.** Its influence outside the prior's top few candidates is net
harmful:

| restriction | Δ R pts | net flips | Pareto |
|---|---:|---:|---:|
| prior top-2 only | +0.660 | +251 | +1.072 |
| **prior top-3 only** | **+0.786** | **+299** | **+1.086** |
| prior top-5 only | +0.786 | +299 | +1.029 |
| unrestricted (all 50) | +0.673 | +256 | +0.861 |

**It is ordering, not magnitude.** A rank transform that destroys every
magnitude in the model term keeps most of the effect (ΔmR +0.755 of +0.861 at
τ = 0; at τ = 0.05 it *beats* the real term, Pareto +2.72 vs +2.31). And it is
not merely the model's own top-1 guess: an argmax-only vote recovers just
**51 %** of ΔR and **13 %** of ΔmR.

## 5. The mR gain is two predicates, and it does not survive them

This is the finding that most constrains the C′ interpretation.

| | ΔmR points | bootstrap CI (2,000 image-resamples) | draws positive |
|---|---:|---|---:|
| τ = 0 | +0.866 | [+0.165, +1.527] | 99.3 % |
| τ = 0.05 | +0.082 | **[−0.572, +0.723]** | **60.3 %** |

At τ = 0.05 — the operating point C′ headlines — **the model's mR contribution
is indistinguishable from zero.** And at both points the delta is carried by two
low-count predicates:

| τ = 0 | ΔmR |
|---|---:|
| all 50 classes | **+0.861** |
| dropping `riding` (n = 140, recall 18.57 → 60.00) | **+0.033** |
| dropping `riding` and `walking on` (n = 56, 0.00 → 26.79) | **−0.524** |

The same two predicates, in the independent held-out half (τ = 0.1, scale 0.5):
ΔmR +0.632 → +0.152 without `walking on` → −0.261 without both. The
concentration is a stable property, not a split artefact.

By contrast the **R@50 gain is broad-based**: +207 of the +256 net flips are
head-class rows, spread across `on`, `has`, `holding`, `wearing`, and it is
positive in 100 % of resamples.

## 6. What the +2.29 Pareto gap actually measures

At **fixed** τ = 0.05 the model adds **+0.612 R and +0.068 mR**. The gap is
+2.31 because the prior frontier is steep there: buying +0.61 R by lowering τ
would cost the prior ≈ 2.3 mR. The gap is an **R-axis** measurement expressed in
mR units — not evidence that the model generates mR.

Decomposing C′'s "beats on both axes" (67.07/24.27 vs 66.80/21.98) honestly:

| step | R@50 | mR@50 | who did it |
|---|---:|---:|---|
| prior, τ = 0 | 66.80 | 21.98 | — |
| prior, τ = 0.05 | 66.47 | 24.22 | **τ** — a zero-information scalar |
| + model | 67.09 | 24.29 | **the model:** +0.61 R, +0.07 mR (CI spans 0) |

The mR came from τ. The model's job was to pay for it in recall.

## 7. Held-out confirmation (image-disjoint 50/50, seed 20260901)

Selection on half A over (scale × τ) chose **scale = 0.5, τ = 0.1**; read once on
half B:

| half B arm | R@50 | mR@50 |
|---|---:|---:|
| prior only, τ = 0 | 66.68 | 22.29 |
| prior only, τ = 0.1 | 66.26 | 26.23 |
| **model (scale 0.5) + prior, τ = 0.1** | **66.53** | **26.86** |

Held-out Pareto gap **+2.72**, null **−0.51**, margin **+3.23**. The effect
generalises. Note selection chose **half** the historical model weight — the
protocol's α = 3.75 against a unit-variance model term over-weights the model.

---

## ESTABLISHED FACTS

1. `Bench.ensemble_term` in the C′ tool mis-orders standardisation against the
   column slice (D-1). Corrected values move ≤ 0.15 points; no C′ conclusion
   reverses. `arm_E`, `criterion5` and `criterion3` are affected and remain
   unrecomputed. Nothing was edited.
2. The model changes the top-1 decision on 7.70 % of rows, and **only** where
   the prior's top1−top2 margin is small — 0.00 % of rows above the 5th margin
   decile change at all.
3. The top-1 gain is real and null-exceeding: +256 net flips, ΔR +0.673
   [+0.403, +0.963], 100 % of image-resamples positive; all three nulls
   between −485 and −948 net flips.
4. Global ranking is **degraded**, not improved: mean rank 1.826 → 2.782,
   MRR 0.7998 → 0.7815, R@5 95.72 → 91.69, rank-21+ rows 60 → 1,022.
5. Rank damage and top-1 gain occur on disjoint populations — the damage is to
   prior-rank-2..5 rows (+2.4 to +3.8 places), the gain is on prior-rank-2 rows
   inside the margin budget (10.64 % rescue rate).
6. The mR gain is carried by `riding` and `walking on`. Removing them turns
   ΔmR negative in both halves. At τ = 0.05 ΔmR = +0.082, CI [−0.572, +0.723].
7. The scale response is peaked at c ≈ 1 and negative by c = 2; held-out
   selection prefers c = 0.5. Restricting the model to the prior's top-3 is
   strictly better than global action (+0.786 vs +0.673 R).
8. A rank transform retains most of the effect; an argmax-only vote retains
   ~half the R and almost none of the mR.

## MECHANISM

The model term is a **bounded, ordinal tie-breaker on the prior's top few
candidates**, not a ranker and not a calibrator. Per-row standardisation gives
it unit scale; α = 3.75 against a log-prior with median top-2 margin 5.08 gives
it a hard budget it cannot exceed. Inside that budget — the ~30 % of rows where
the prior is nearly tied — it carries genuine image-conditioned information: it
ranks GT top-1 on 68 % of the rows it rescues versus 38.9 % overall, and every
shuffled or noise-matched control is strongly negative. Outside the budget it
adds unit-variance noise to a sub-top-1 ordering the prior already gets right,
which is what destroys mean rank while leaving the argmax untouched. Its net
effect on the task is to **restore recall that τ spends**, making a higher τ
affordable; the mR then comes from τ, not from the model.

## ALTERNATIVE EXPLANATIONS

| explanation | status |
|---|---|
| **Calibration** — the model repairs miscalibrated prior confidence | **Falsified as the primary account.** A rank transform destroying all magnitude keeps 88 % of the effect at τ = 0 and *beats* the real term at τ = 0.05. The information is ordinal. |
| **Sparse top-1 correction** — only the model's own argmax matters | **Falsified as a complete account.** Argmax-only voting recovers 51 % of ΔR, 13 % of ΔmR. The model's full ordering within the prior's top-k is used. |
| **Genuine global ranking information** | **Falsified.** Every ranking metric degrades; model-alone MRR 0.502, mean rank 9.25, top-1 only +2.33 over a constant `on` predictor. |
| **Prior/model interaction** | **Supported, and it is specifically the τ interaction.** τ promotes rare predicates globally; the model re-suppresses them where the image disagrees. They are partially inverse operations. |
| **Noise / implicit regularisation** | **Falsified.** N1 −485, N2 −948, scale-matched Gaussian −660 net flips. |
| **A two-predicate artefact** | **Supported for the mR axis only.** `riding` + `walking on` carry all of ΔmR in both halves. ΔR is broad-based and survives. |

## LIMITATIONS

- One checkpoint, PredCls, GT pairs, 3,000 validation images, one prior file,
  α = 3.75 throughout. Nothing here is a claim about SGDet or about training.
- The bootstrap resamples images **from this split**; it estimates sampling
  error, not dataset shift. It cannot tell whether `riding` generalises.
- mR@50 over 50 classes with per-class n as low as 41 is structurally
  high-variance. Every mR conclusion here is weaker than the R conclusions **by
  construction**, which is precisely why the C′ criterion needed the R@50 floor
  its own limitations section already flagged.
- Held-out selection used one seed and a scale × τ grid only.
- The stratum-level ΔmR in the low-entropy band (+2.583 on 2 net flips) is a
  small-denominator artefact and should not be read as an effect.
- `riding` and `walking on` may simply be verbs CLIP's pretraining covers well;
  that is a statement about the encoder, not about relational reasoning.
- D-1 is reported, not repaired. Arm E and criteria 3 and 5 remain as published.

## SINGLE HIGHEST-VALUE NEXT EXPERIMENT

**Measure the ceiling of the tie-breaker channel itself — CPU-only, on this same
cache, before any architecture is considered.**

Section 4 shows the model's usable influence is confined to the prior's top-3
and that restricting it there is strictly better than global action. The
unanswered question is not "can we build a better reranker" but **"how much is
in that channel at all"**: an oracle that always picks GT whenever GT lies in
the prior's top-k *and* the row is inside the margin budget bounds every
possible tie-breaker at this α. If that ceiling is near the +0.79 R already
achieved, the channel is exhausted and no reranking architecture is worth GPU
time. If it is far above, the gap is the actual research target and its size
sets the budget.

Pre-register before running, with the R@50 floor the C′ criterion lacked:

- **Arms:** oracle@top-k for k ∈ {2, 3, 5}, crossed with margin budgets
  {∞, 5.08 (median), 0.93 (decile 1)}; the real model term as the achieved point.
- **Success:** the oracle exceeds the achieved ΔR by > 2.0 R points **at
  R@50 ≥ 66.5** → the channel has real headroom, and a candidate-restricted
  reranker becomes the justified next GPU experiment.
- **Failure:** oracle ΔR < +1.5 points over achieved → the tie-breaker channel
  is near-exhausted; the bottleneck is candidate *generation* or the α/scale
  calibration, not the ranker, and no new head should be built.
- **Cost:** minutes of CPU, zero GPU, no new cache.

Do **not** fix D-1 as part of that run. Recomputing arm E and criteria 3 and 5
is a separate, explicitly-scoped decision on an already-published result.
