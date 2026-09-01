# RESEARCH_STATE — living research map

Updated: 2026-09-01. Branch: `research/architecture-breakthrough`.
This file is the single place that says what is currently believed, what is not,
and what is running. Every claim carries its evidential class.

> **Denominator warning.** Three incompatible baselines appear in this project's
> history and must never be compared. `docs/EXPERIMENT_MATRIX.md` quotes the
> **historical** prior on the **full 10,401-image** validation split
> (64.37 / 20.30). Everything in sections 2–4 below uses the **train-derived
> leak-free** prior on the **3,000-image** analysis subset (66.80 / 21.98), and
> section 4b uses that same leak-free prior on the **full 10,401-image** split
> (66.59 / 22.30). Different prior, different N. A delta against one is not a
> delta against the other.

---

## 1. Strongest validated result

**MEASURED.** A leak-free train-derived frequency prior alone reaches
R@50 66.80 / mR@50 21.98 on the 3,000-image analysis set (tau=0), and the
79.9M-parameter historical visual model adds **+0.673 R points** on top of it in
the additive alpha=3.75 formulation.

The model's contribution is real, small, and **not** a ranking improvement: it
worsens mean GT rank (1.83 → 2.78) and worsens more rows than it improves, while
still producing +256 net beneficial top-1 flips. It behaves as a bounded
tie-breaker on an argmax the prior has almost already decided.

## 2. Strongest negative results

1. **Candidate generation is not the bottleneck. H1 falsified.** GT is inside the
   prior's top-5 for 89.7% of rows, top-3 for 85.5%. (`runs/p17`)
2. **The raw model score is not a ranker.** Allowed to decide inside the prior's
   candidate set it costs up to −15.6 R points, and clears the R@50 floor only
   in the 10%-of-rows margin budget. (`runs/p17`)
3. **The additive composition already beats the best pure reranking** (+0.673 vs
   a best-cell +0.423), so composition is not what is failing. (`runs/p17`)
4. **"The model restores recall that tau spent" is falsified in the operating
   region.** ΔR *falls* from +0.673 to +0.205 across tau ∈ [0, 0.1]; it grows
   only at tau ≥ 0.2 where the prior has already lost 4.8–55.0 R points.
   (`runs/p17`)
5. **A learned calibration does not even match tau.** With beta chosen
   out-of-fold and resampled over 5 fold partitions, a learned per-class rule
   with no visual input averages a Pareto gap of **−1.205** and its shuffled
   null **−1.137** — *below* the tau frontier — and neither clears the R@50
   floor on any partition (0/5). Tuning tau beats learning a per-class rule.
   (`runs/p25`; the single-partition `runs/p22` read +0.059 / +0.224 and was
   the optimistic draw.)
6. **Appearance scoring on frozen CLIP L/14-336 adds nothing beyond
   calibration.** (`docs/APPEARANCE_TAU_INTERACTION_RESULT.md`)

## 2b. The result that reframes the project (`runs/p26`, 2026-09-01)

**MEASURED, pre-registered verdict E2 SUPPORTED.** Permuting the model term only
among rows sharing the same (subject, object) category — destroying image
content, preserving pair identity — costs **−0.114 ± 0.265** Pareto points, with
`full` ahead on **1 of 5** partitions. Destroying pair identity instead costs
**+3.163**.

| arm (5 partitions, nested, out-of-fold) | Pareto mean ± sd | R@50 | floor |
|---|---|---|---|
| `full` (prior + model) | +1.911 ± 1.056 | 66.656 ± 0.257 | 4/5 |
| `pair_matched_null` (image destroyed) | **+2.026 ± 0.977** | 66.596 ± 0.236 | 4/5 |
| `shuffled_model` (identity destroyed) | −1.137 ± 0.838 | 65.856 ± 0.278 | 0/5 |

Two structural facts explain it:

- **VF** `ensemble_alpha = 0.0` — the "model term" is **100% the CLIP text
  branch**; the visual classifier head contributes **exactly zero**. Everything
  the C′ work attributed to "the model" is the text path.
- **MR** **86.87%** of the term's variance is *between* (subject, object) groups;
  only 13.13% is within-group, and within-group is the only place image
  conditioning can live.

Effect-size bound: image conditioning ≲ **0.2** Pareto points, point estimate
wrong-signed.

## 3. Current bottleneck

**The signal being converted is a pair prior, not visual evidence.**
Superseded reading (pre-`p26`): "a representation bottleneck for ranking, with a
decision-formulation bottleneck for conversion". That was correct about the
mechanism and wrong about the source.

MEASURED: `model_only`, fitted optimally inside the candidate set, is negative in
all four cells and fails the R floor at tau=0.05 — the representation cannot rank
on its own. It carries a stable **+0.68 to +0.79 R points** over prior-derived
features, cross-fitted and null-controlled. `runs/p26` establishes that this
increment is **(subject, object) identity expressed in text-embedding space**,
not image content.

So the bottleneck is no longer "how do we convert the model's visual evidence".
There is no visual evidence in this term to convert. What remains is a better
*pair-conditioned* predicate distribution than the frequency prior supplies, and
the open question is whether it can be distilled without vision at all.

## 4. Currently active hypothesis

**EXPLORATORY, screening only.** A candidate-restricted, class-reweighted
decision rule converts ~3.4× more of the model's complementary information into
Pareto movement than the additive arm, at the same R@50 floor.

With beta selected inside the training folds only, **resampled over 5
independent fold partitions** (`runs/p25`, tau=0, k=5). The single-partition
number (`runs/p22`, +2.894) was a favourable draw and is superseded:

| arm | R@50 mean ± sd | Pareto gap mean ± sd | Pareto min | clears floor |
|---|---|---|---|---|
| achieved additive C' | 67.474 | +0.861 | — | 5/5 |
| prior_only (no vision) | 65.861 ± 0.265 | **−1.205 ± 0.793** | −1.937 | **0/5** |
| shuffled-model null | 65.856 ± 0.278 | **−1.137 ± 0.838** | −1.922 | **0/5** |
| **full (prior + model)** | **66.656 ± 0.257** | **+1.911 ± 1.056** | **+0.244** | **4/5** |

Two findings of opposite sign, both load-bearing:

- **WEAKENED.** The absolute gap is partition-dependent (+0.244…+2.941) and the
  arm fails the R@50 floor on 1 of 5 partitions. It is **not** yet a reliably
  usable operating point.
- **STRENGTHENED.** The *separation* from both nulls is ≥ **+1.84** Pareto
  points on **every** partition (`full − prior_only` min +1.904,
  `full − shuffled` min +1.842). The separation is the defensible quantity; the
  absolute level is not.
- **A learned calibration does not even match tau.** The no-vision arms average
  −1.2 Pareto points — *below* the tau frontier — and clear the floor 0/5. Tuning
  tau beats learning a per-class rule.

This is a 3,000-image screening result and is **not** a headline.

## 4b. Full-split confirmation of that hypothesis (`runs/p29`, 2026-09-01)

**MEASURED, pre-registered verdict CONFIRMED.** The section-4 screening result
re-run unchanged on the full 10,401-image cache. Prior-only baseline on this
split: R@50 **66.593** / mR **22.304**.

Registered partition (salt 0), tau=0, k=5:

| arm | R@50 | mR@50 | Pareto gap | floor 66.5 |
|---|---|---|---|---|
| `prior_only` (no vision) | 65.357 | 27.178 | +0.378 | **FAIL** |
| **`full`** | **66.760** | 25.049 | **+2.745** | **ok** |
| `shuffled_model` | 65.348 | 27.157 | +0.352 | **FAIL** |
| `pair_matched_null` (image destroyed) | 66.704 | 25.009 | **+2.705** | **ok** |

All three registered conditions pass: floor 66.760 >= 66.5; gap +2.745 > +1.5;
separation +2.368 over `prior_only` and +2.394 over `shuffled_model`, both > +1.0.

Resampled over 5 partitions, `full` = **+2.947 ± 0.190**, floor **5/5** — against
screening's +1.911 ± 1.056 and 4/5. The sd falls 5.6x. **The screening
instability was a small-sample artifact, not a fragile effect**, and the
pre-registration's addendum was wrong in the conservative direction.

**And the criterion could not test what mattered.** `pair_matched_null` passes
every registered condition as well. Paired over partitions,
`full − pair_matched_null` = **+0.031 ± 0.188** (negative on 1 of 5; t-based 95%
interval [−0.20, +0.26]). This replicates `p26`'s −0.114 ± 0.265 at 3.48x the
rows with a tighter bound: **image conditioning contributes <= ~0.26 Pareto
points, point estimate indistinguishable from zero.**

Correct statement: *the decision rule reliably converts the checkpoint's
contribution, and that contribution is (subject, object) identity in
text-embedding space.* Incorrect: *the visual model adds +2.9 mR* — an arm with
every image association destroyed adds +2.9 too.

`full` also buys head/body movement at a **tail cost**: tail mR 9.89 vs
`prior_only`'s 14.49. It is not a long-tail gain.

Detail: `docs/FULL_VALIDATION_RESULT.md`.

## 5. Experiments completed this cycle

| run | question | verdict |
|---|---|---|
| `p14`/`p15` | oracle ceiling at the tie-break fix | superseded by p17 |
| `p17` | oracle ceiling, canonical ordering | PREREG SUCCESS 27/27 (vacuous); REALIZABLE **EXHAUSTED** 27/27 |
| `p18` | learned candidate scorer, plain CE | **EXHAUSTED** 4/4 on ΔR |
| `p19` | same, fully class-balanced | R collapses to 25–31; mR reaches 42.2 |
| `p20` | beta frontier | superseded by p21 (×100 Pareto bug) |
| `p21` | beta frontier, corrected, with null | full off-frontier at beta=0.20 |
| `p22` | nested beta selection | operating-point-free, but ONE partition |
| `p23` | throughput pilot, workers 7 | 0.95x -- GPU-bound, more workers cannot help |
| `p25` | nested selection resampled over 5 partitions | magnitude weakened, separation confirmed |
| `p26` | pair-matched null (image destroyed, identity kept) | **E2 SUPPORTED -- the effect is pair identity** |
| `p24` | full-validation cache extension (GPU, frozen forward pass) | COMPLETE, cache **VALID 12/12** |
| `p27` | pair-prior distillation | **WITHDRAWN** -- pair arms blindfolded on 45% of rows |
| `p28` | oracle ceiling on the full split | REALIZABLE **EXHAUSTED** 9/9; tau-restoration falsified again |
| `p29` | nested scorer on the full split | **PREREG CONFIRMED 3/3** -- and the image-destroying null passes too |
| `p30` | audit of pair-arm fold coverage | 45.4% (3k) / 33.2% (full) of rows get NO pair information |
| `p31` | learned R/mR frontier on the full split | `full` has a usable region beta in [0.10,0.20], peak **+3.334**; no-model arms have none |

## 6. Experiments pending

- **Corrected pair-prior distillation (CPU, no GPU).** `runs/p27` was withdrawn;
  the estimable-subset design is specified in
  `docs/PAIR_PRIOR_DISTILLATION_RESULT.md` §6. ~88.6k rows of the full `p24`
  cache — larger than the entire 3k screening set. Must recompute the prior-only
  baseline and tau frontier **on the subset**, and must report that the ~33%
  singleton-pair rows remain unaddressable by any pair-conditioned estimator.
  Open question: can a vision-free pair-conditioned model reproduce C′?

(The full-validation confirmation that was listed here is complete — see §4b/§6b.)

## 6b. Full-validation confirmation — COMPLETE

`runs/p24` finished (exit 0, 12,423 s, 10,401 images, 132,556 rows, cache
validated 12/12). All three registered analyses have now run: `p28` (oracle
ceiling), `p29` (nested scorer — carries the criterion), `p31` (frontier).

**Verdict: CONFIRMED**, 3/3 registered conditions on the registered partition.
See section 4b and `docs/FULL_VALIDATION_RESULT.md`.

Its reduced weight after `p26` stands, and the full-split data reinforces it
rather than softening it: the image-destroying `pair_matched_null` passes every
registered condition too, at `full − null = +0.031 ± 0.188`. A CONFIRMED verdict
here does not resurrect H4.

Note on execution: the first `p29` attempt was killed by a VM shutdown at
12:06 UTC with no `result.json` written. It produced no number; it was archived
as `runs/p29_full_scorer_nested_KILLED_BY_VM_REBOOT/` with a `STOPPED.md` and
relaunched with byte-identical argv. Nothing was re-run that had completed.

## 7. Experiments explicitly abandoned

- **Candidate-restricted learned reranker (GPU).** Pre-registered criterion
  returned EXHAUSTED 4/4 on ΔR and the raw score is negative as a ranker. Not
  built.
- **The visual-architecture branch as currently posed** (`runs/p26`). The
  quantity being converted is a pair prior in text-embedding space; the visual
  head is at weight zero and is untrained in this checkpoint. Building an
  architecture to exploit it would be building a second frequency prior.
- **Architecture scaling / added visual capacity.** No measured failure mode
  demands it. `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md`.
- **Additive appearance scoring on frozen CLIP.** Falsified.

## 8. Historical claims that remain unverified or superseded

- **The +2.29 mR headline is not "the model adds +2.29 mR".** tau contributes
  most of that movement. Any table quoting it must use the operating-point
  separation required by the directive's section 13.
- **README's historical performance claims** predate the leak-free prior and the
  GT-extraction fix (`220c5c2e`). They have not been re-derived under the current
  evaluator and should not be cited without one.
- **`docs/CPRIME_MECHANISM_REPORT.md` §3 margin-decile table** was computed with
  the `topk` tie-break and is affected; its load-bearing claim (0.00% of rows
  above the 5th decile change) cannot move, because tied rows have margin 0 and
  live entirely in decile 1.

## 9. Protocol corrections made this cycle

1. The oracle's pre-registered R floor was applied to an arm that cannot lose
   recall, making the gate non-falsifiable. Floor now also binds on the
   realizable arm; both verdicts reported. (`a75aa03`)
2. The candidate-scorer null condition was written against the wrong baseline
   (`null > achieved` rather than `null > prior_only`). Corrected; both
   computed and stored. (`docs/CANDIDATE_SCORER_RESULT.md` §2)

Neither correction changed a pre-registered threshold.
