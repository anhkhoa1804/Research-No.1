# RESEARCH_STATE — living research map

Updated: 2026-09-01. Branch: `research/architecture-breakthrough`.
This file is the single place that says what is currently believed, what is not,
and what is running. Every claim carries its evidential class.

> **Denominator warning.** Two incompatible baselines appear in this project's
> history and must never be compared. `docs/EXPERIMENT_MATRIX.md` quotes the
> **historical** prior on the **full 10,401-image** validation split
> (64.37 / 20.30). Everything in sections 2–4 below uses the **train-derived
> leak-free** prior on the **3,000-image** analysis subset (66.80 / 21.98).
> Different prior, different N. A delta against one is not a delta against the
> other.

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

## 3. Current bottleneck

**H6 (representation) for standalone ranking; the decision formulation for
conversion.** MEASURED support: `model_only`, fitted optimally inside the
candidate set, is negative in all four cells and fails the R floor at tau=0.05.
The representation cannot rank on its own. What it carries is a stable
complementary increment of **+0.68 to +0.79 R points** over prior-derived
features — cross-fitted, out-of-fold, null-controlled.

The additive alpha/tau formulation is a poor converter of that increment.

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
| `p25` | nested selection resampled over 5 partitions | **magnitude weakened, separation confirmed** |

## 6. Experiments pending

- **Full-validation confirmation** — pre-registered in
  `docs/FULL_VALIDATION_PREREGISTRATION.md`. One frozen forward pass over the
  remaining validation images, then p17/p21/p22 re-run unchanged. Criterion and
  all three outcomes committed before launch.

## 7. Experiments explicitly abandoned

- **Candidate-restricted learned reranker (GPU).** Pre-registered criterion
  returned EXHAUSTED 4/4 on ΔR and the raw score is negative as a ranker. Not
  built.
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
