# Experiment C′ result — the model does carry complementary information, and it is small

Pre-registered in `docs/MODEL_RECALIBRATION_C_PREREGISTRATION.md`, committed at
`2813a0c` **before** the GPU pass. Criterion fixed before any result was read.

| | |
|---|---|
| GPU pass | `runs/p10_model_recalibration/` — exit 0, **3,673.7 s**, 3,000 images, 38,053 pairs |
| Cache | `pair_logits.pt`, 42 MB, **CACHE VALID (12/12)** — `cache_validation.json` |
| CPU analysis | `runs/p11_cprime_analysis/` — exit 0, 80.5 s, no GPU |
| Prior | `datasets_vg150_clean/frequency_prior_train.json` (leak-free, train-derived) |
| Checkpoint | `pure_best_adapt_light_mR50.pt`, SHA256 `8845c3af…` — matches the manifest |

---

## FACTS

- **Exactly one GPU pass was run.** Everything else is CPU over its cache.
- The prior-only τ frontier computed here reproduces
  `runs/p7_prior_temperature_sweep/` — an independent tool, reading the prior
  file rather than the cache — to **0.00 points** at every shared τ.
- **All five registered criteria are met.**
- The effect is **real, reproducible, null-exceeding, and small.**

## METHOD

One frozen protocol, one prior file, one predicate set per comparison. The
instrument is the **Pareto gap**: an arm's mR@50 minus the prior-only τ
frontier's mR@50 at the **same R@50**, referenced to a null that goes through
the identical pipeline. A raw ΔmR criterion was rejected in advance because
τ = 0.05 clears any such bar with *negative* information content.

## SOURCE TRACE

```
MODEL     score_m = a·norm(cls_logits)/T_cls + (1−a)·norm(text_logits)/T_text
          norm(x) = (x − mean(x)) / max(std(x), 1e−4)          per row, over P
          a       = eval_sgg_predicate_ensemble_alpha          historical: 0.0

PRIOR     score_p(p|s,o) = log P(p|s,o)
          fallback: pairs[s‖o] → ½(subjects[s]+objects[o]) → subjects[s]
                    → objects[o] → global

TAU       score_p_tau(p|s,o) = log P(p|s,o) − τ·log P(p)

COMBINED  score = score_m + α·score_p_tau        α = freq_bias_alpha = 3.75
          score[:, background] = −1e4
          prediction = argmax over the 50 foreground columns
```

**Four suppression points**, in order of severity: **S-1** `a = 0.0` discards
the trained classifier entirely; **S-2** per-row standardisation destroys
calibrated magnitude; **S-3** α = 3.75 against a unit-variance model term;
**S-4** `_apply_eval_logit_adjustment`, provably inert (`runs/p8_tau_path_bug/`).

## CACHE

`pair_logit_dump_v2` (`runs/p10_model_recalibration/cache_schema.md`). Stores
both ensemble branches separately, prior rows **raw** (pre-τ, pre-α), and GT
**raw** (never alias-normalised) plus the alias map, so one cache yields both
predicate schemes. Validated independently of the analysis before it was read.

**Two defects were caught before the GPU pass, not after:**

1. The dump was being handed **alias-normalised** GT while its docstring claimed
   raw. `near→next to` and `wears→wearing` would have been collapsed inside the
   artifact, and the 50-class arm would have been unrecoverable from 61 GPU-minutes.
2. Under `a = 0.0` the composed term **is** the text branch alone. A null on it
   would have said nothing about the model's classifier. Schema v2 stores both.

**A third defect was caught during analysis, by the p7 gate.** `log P(p)` was
initially *inferred* from the cache as the modal prior row. That row is the
**uniform** `default_log_prob` fallback (−3.932 in every column, 616 rows), not
the class marginal. Subtracting a constant cannot move an argmax, so τ was a
silent no-op: τ = 0.5 produced mR@50 23.39 where p7 measures 38.15. A τ = 0-only
check would have passed it. `log P(p)` is now **read** from the prior file, and
the reproduction gate is permanent.

## RESULTS

Prior-only frontier vs model + prior (raw50, α = 3.75, ensemble_alpha = 0.0):

| τ | prior R@50 | prior mR@50 | model+prior R@50 | model+prior mR@50 | Pareto gap |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 66.80 | 21.98 | 67.49 | 22.86 | **+0.88** |
| 0.025 | 66.54 | 22.90 | 67.29 | 23.60 | **+1.62** |
| **0.05** | **66.47** | **24.22** | **67.07** | **24.27** | **+2.29** |
| 0.1 | 66.16 | 26.00 | 66.36 | 25.70 | +0.86 |
| 0.5 | 44.64 | 38.15 | 48.59 | 38.40 | +1.79 |
| 1.0 | 11.81 | 34.82 | 16.79 | 37.15 | +1.83 |

**The clearest statement of the result needs no interpolation at all.** The
prior's best achievable R@50 is 66.80, at τ = 0. Model + prior at τ = 0.05
reaches **R@50 67.07 and mR@50 24.27** — better on **both** axes simultaneously.
No recalibration of the prior can reach R@50 > 66.80, because τ only trades R
away. That is complementary information by construction.

Model **alone** (α = 0) is far weaker than the prior: R@50 38.86 / mR@50 10.72
(text), 45.46 / 15.73 (classifier), against the prior's 66.80 / 21.98.

## NULL

Both nulls refit/rescored at every τ, 5 seeds, identical pipeline and denominator.

| τ | real (arm B) | N1 pair-shuffled | N1 SD | margin | 2 SD gate | N2 split-shuffled |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | +0.88 | −5.04 | 0.18 | **+5.92** | 0.36 | −6.53 |
| 0.05 | +2.29 | −3.80 | 0.15 | **+6.09** | 0.30 | −5.17 |
| 0.1 | +0.86 | −2.72 | 0.11 | **+3.58** | 0.22 | −4.15 |
| 0.5 | +1.79 | −0.28 | 0.23 | **+2.08** | 0.46 | −1.66 |

The null is **strongly negative**: a model whose rows are permuted across pairs
*damages* the frontier. The real model helps. Margins of +2 to +6 points against
SDs of 0.1–0.3 clear the 2 SD gate by a wide factor at every τ.

Note this is the opposite sign to experiment D's null (+0.43 for shuffled
appearance). The difference is that D's probe was *fit* on the shuffled features
and could learn to ignore them, whereas here the shuffled model term is injected
without refitting. Both are the matched null for their own arm.

## PARETO ANALYSIS — the ensemble_alpha counterfactual

Sweeping the knob the historical protocol pins at 0.0, with **both** branches
normalised as `evals.py` does:

| ensemble_α | τ=0.05 gap | τ=0.1 gap | τ=0.5 gap | τ=1.0 gap |
|---:|---:|---:|---:|---:|
| 0.00 (historical) | +2.29 | +0.86 | +1.79 | +1.83 |
| 0.25 | +2.28 | +3.55 | +2.14 | +2.61 |
| 0.50 | +2.02 | +3.75 | +3.17 | +3.66 |
| 0.75 | +1.73 | +3.51 | +3.28 | +4.51 |
| 1.00 (classifier only) | +1.41 | +3.43 | +3.64 | +4.74 |

> **CORRECTION.** Before running C′ I identified S-1 (`ensemble_alpha = 0.0`
> discarding the trained classifier) as the most severe suppression point. **At
> usable operating points that is wrong.** At τ = 0.05 the historical
> `ensemble_alpha = 0.0` is the *best* setting (+2.29), and restoring the
> classifier makes it slightly *worse* (+1.41). The classifier only overtakes
> the text branch at τ ≥ 0.1, and its advantage is largest exactly where R@50
> has collapsed. S-1 is real as a mechanism but is **not** the binding
> constraint on this checkpoint.

## TOP-5 RESCUE

At τ = 0, α = 3.75, raw50:

| | |
|---|---:|
| prior top-5 coverage | 89.69 % |
| prior top-1 accuracy | 66.80 % |
| model+prior top-1 accuracy | 67.47 % |
| **Q2** rescued wrong→right | **1,042** |
| **Q1** destroyed right→wrong | **786** |
| net flips | **+256** |
| rescue rate of prior errors | 8.25 % |
| destruction rate of prior hits | 3.09 % |
| **Q3** GT rank improved / worsened | 2,013 / **4,675** |
| mean GT rank: prior / model-only / combined | 1.826 / 9.249 / **2.782** |

**The model is a poor ranker and a useful tie-breaker.** It moves GT *down* the
ranking more often than up (4,675 vs 2,013) and raises mean GT rank from 1.83 to
2.78 — yet it still converts 1,042 prior errors into hits against 786 losses.
The gain is concentrated in *flips at the top*, not in a globally better ordering.

## UNCERTAINTY STRATA (Q4)

Pareto gap of model + prior at τ = 0.1, computed **within** each prior-entropy
tercile against that tercile's own frontier:

| stratum | n | prior mR@50 | model mR@50 | Pareto gap |
|---|---:|---:|---:|---:|
| low entropy | 12,686 | 16.15 | 20.16 | **+0.84** |
| mid entropy | 12,698 | 19.75 | 23.12 | **+3.26** |
| high entropy | 12,669 | 24.86 | 27.68 | **−0.29** |

The model helps most where the prior is **moderately** uncertain, and does not
help where the prior is most uncertain — the opposite of the intuition that
vision should rescue the hardest cases.

## HEAD / BODY / TAIL (Q5)

At τ = 0.05, raw50:

| | R@50 | mR@50 | head | body | tail |
|---|---:|---:|---:|---:|---:|
| prior | 66.47 | 24.22 | 42.62 | 16.59 | 15.98 |
| model + prior | 67.07 | 24.27 | 43.11 | 16.79 | **15.39** |

The model's contribution at this operating point is **head- and body-weighted,
and slightly negative on the tail**. Its long-tail gains appear only at high τ,
where the prior's own recalibration is already doing the work.

## CRITERION AUDIT

| # | Criterion | Verdict |
|---|---|---|
| 1 | positive Pareto gap | **MET** — positive at every τ; strict two-axis dominance at τ ≤ 0.05 |
| 2 | ≥ 2 SD over matched null | **MET** — margins +2.1 to +6.1 vs SDs 0.1–0.3 |
| 3 | reproducible across strata, no single predicate > 50 % | **MET** — largest single predicate `watching` = **6.6 %** of positive gain; gains spread over head (`wearing`), body (`watching`, `and`), tail (`walking in`, `eating`, `on back of`) |
| 4 | sign invariant to denominator | **MET** — raw50 +2.29 / eval48 +2.44 at τ = 0.05; identical pattern throughout |
| 5 | no selection leak | **MET** — selected on image-level half A, read on half B: gap **+4.22** vs null **+1.55**, SD 0.28 |

## LIMITATIONS

1. **The effect is small.** At the best usable operating point it is
   **+0.27 R@50 and +2.29 mR@50** over a zero-parameter baseline. It is real; it
   is not a solution.
2. **The largest gaps sit at degenerate operating points.** τ = 1.0 gives
   +4.74, but at R@50 ≈ 18 %. The registered criterion has no R@50 floor, so the
   held-out selection in criterion 5 chose τ = 1.0. That selection is honest but
   its operating point is not useful, and the criterion should carry an R floor
   in future registrations.
3. **These are GT pairs (PredCls) on 3,000 validation images**, one predicate per
   pair, K inert. It says nothing about detection or pair proposal.
4. **One checkpoint.** "The model" here is `pure_best_adapt_light_mR50.pt` under
   the historical protocol, not architectures in general.
5. **head/body/tail buckets** come from the evaluated split's own GT counts.
   Identical across arms, so cross-arm comparisons are sound; not comparable to
   runs at other N.
6. **The prior differs from the p5 arm.** C′ uses the leak-free train-derived
   prior for both arms; `runs/p5_model_vs_leakfree_prior/` used the historical
   prior for its model arm. C′'s comparison is internally like-for-like; the two
   experiments' model arms are not directly comparable.

## DECISION

> ### A. COMPLEMENTARY INFORMATION CONFIRMED

All five registered criteria are met, under both predicate schemes, with a
held-out selection and a matched null that is strongly negative. The decisive
observation needs no interpolation: **model + prior reaches R@50 67.07 / mR@50
24.27, beating the prior's best-R@50 point (66.80 / 21.98) on both axes at once**,
which no recalibration of the prior can do.

> **CORRECTION — my registered prediction was wrong.** The pre-registration
> recorded "C or B" as the predicted outcome, reasoning that the model pushes
> head while τ pushes tail and the two would cancel. The measurement says **A**.
> The prediction is recorded as made and is hereby marked wrong.

Decision **B** is specifically *not* supported: restoring the discarded
classifier branch does not improve matters at usable operating points, so the
decision rule is not destroying a large hidden signal.

## NEXT STEP

Per the standing constraint: because C′ is **not** null, the deliverable is
still analysis, not architecture. **No architecture is to be implemented or
trained on this result.** The open question C′ raises — and does not answer — is
why a model that ranks GT *worse* on average (mean rank 1.83 → 2.78) nonetheless
produces a net +256 flips at the top. That is a question about the model's
confidence calibration, and it is answerable on the existing cache, on CPU,
with no new GPU time.
