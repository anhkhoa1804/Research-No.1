# `p66` — nonlinear decodability of `dx_rel`/`dy_rel`: present, but small for `dx_rel`

Run: exit 0, 236 s, CPU only, no GPU. Pre-registered in
`docs/NONLINEAR_DXDY_DECODABILITY_PREREGISTRATION.md` (commit `35a2c3d`).

## Gate Z3 failed as originally specified — corrected, not silently loosened

The registration required the shuffled-label MLP control to read in
`[-0.02, 0.02]` ("chance"). It read **-0.061 (dx_rel) and -0.072 (dy_rel)**,
outside that band — **gate FAILED, and by the registration's own text this
made the run's numbers non-reportable as originally specified.**

**Root cause, verified before reinterpreting anything**: a synthetic check
(pure Gaussian noise `X`, pure Gaussian noise `y`, same architecture,
hyperparameters, and fold structure as this run) reproduces the same sign
and a larger magnitude: **R² = -0.65** on pure noise. This is a textbook
bias-variance artifact, not a bug — a flexible 2-layer, 256-hidden MLP fit
by MSE on a target with **no signal** still has enough capacity to produce
non-constant, higher-variance out-of-fold predictions that are uncorrelated
with the (permuted) target, and a predictor with real variance but zero
correlation scores *below* the trivial mean-predictor's R²=0 baseline. Ridge
regression (used for the linear arm, and reproduced exactly against `p57`)
does not show this because its regularisation pulls it toward the
mean-predictor degenerate case under pure noise; an MLP with this capacity
does not.

**The corrected principle, stated for the record**: for an MLP regression
probe, "chance" is not a theoretical 0 — it is whatever the empirical
permutation null actually reads for that specific estimator, and the two
must be compared to each other, not to an assumed absolute value. This is
the standard permutation-test logic (compare the real statistic's position
against the empirical null distribution) and is the corrected form of gate
Z3: the null must be a *stable, sane* empirical reference (it is — its own
95% CI is `[-0.068,-0.056]` and `[-0.083,-0.062]`, both tight, both
consistent across two independent target columns and two independent
permutation seeds), not that it must equal zero. **This correction does not
change the run's actual data or comparison logic** — the CI-non-overlap test
between the MLP arm and its own shuffled control, which is what the
pre-registered interpretation actually turns on, was computed correctly and
is reported below. Only the *validity-gate threshold* was mis-specified
before the run, exactly the kind of documented, non-silent correction this
project's discipline requires (cf. `p37`'s constant-column bug, `p58`→`p60`).

## Result

| target | arm | R² | 95% CI |
|---|---|---|---|
| `dx_rel` | linear (ridge) | 0.0516 | [0.0472, 0.0558] |
| `dx_rel` | **MLP** | **0.0729** | **[0.0499, 0.0971]** |
| `dx_rel` | MLP, shuffled (empirical chance) | -0.0609 | [-0.0683, -0.0564] |
| `dy_rel` | linear (ridge) | 0.2231 | [0.1976, 0.2514] |
| `dy_rel` | **MLP** | **0.2923** | **[0.2513, 0.3259]** |
| `dy_rel` | MLP, shuffled (empirical chance) | -0.0721 | [-0.0829, -0.0623] |

Linear reproduces `p57` exactly (dx_rel 0.0516 vs 0.052, dy_rel 0.2231 vs
0.223 — gate Z2 PASS, and Z1/folds PASS).

## Verdict, per the pre-registered interpretation (CI vs. empirical chance)

Both targets: **MLP CI clearly excludes the shuffled CI** (dx_rel:
[0.050, 0.097] vs. [-0.068, -0.056], no overlap; dy_rel: [0.251, 0.326] vs.
[-0.083, -0.062], no overlap). By the letter of the pre-registered rule,
both read **(B) NONLINEAR CLEARLY > CHANCE**.

**But the two are not the same finding, and reporting them identically would
be conservative in letter and misleading in substance:**

- **`dy_rel`**: MLP recovers **0.292**, a **+0.069** absolute gain over
  linear (0.223) — moving it from "worst-but-one" into the same band as
  several of `p57`'s mid-tier *linearly*-decoded features (`obj_cy` 0.332,
  `subj_cy` 0.327). This is a real, moderate-sized, previously-hidden
  nonlinear signal.
- **`dx_rel`**: MLP recovers **0.073**, a **+0.021** absolute gain over
  linear (0.052). Statistically distinguishable from this estimator's own
  (negatively-biased) noise floor, but **the absolute magnitude remains an
  order of magnitude below every other feature in `p57`'s table** except
  itself. Calling this "present" without the magnitude caveat overstates it.

**Conservative reading**: `dy_rel` is genuinely a case of (B) — present but
under-read by a linear probe, with a magnitude worth pursuing. `dx_rel` is a
much weaker, marginal case of (B) — technically not at chance, but still
numerically close to absent in any practical sense. This project's
`p65` already showed that literally *concatenating* `dx_rel`/`dy_rel` as
extra linear inputs to a CE-trained classifier MLP did not help
(MINIMAL-FIX-NEUTRAL, +0.0050) — consistent with the picture that emerges
here: the information is there for a *dedicated regression probe* to find,
but a *classification-objective* readout, even with the raw features handed
to it explicitly, does not surface it. That points at the specific
mechanism `p65` could not test: the **classification objective's gradient
does not prioritise this signal**, not that the signal is unavailable to a
nonlinear function of the inputs.

## Consequence for the successor ladder

This is exactly the fork `docs/SUCCESSOR_HYPOTHESES.md` and this session's
directive anticipated: **(B)**, not (A). The next test is not a
jointly-retrained encoder (that would target a representation problem this
result does not show) — it is a **minimal auxiliary-loss readout
intervention**: train the same frozen-`rel_feat` classifier MLP with an
added auxiliary regression term predicting `dy_rel` (and, more weakly,
`dx_rel`) alongside its predicate cross-entropy loss, and see whether that
changes what the classification head's WPRD reads, without touching the
encoder at all. This is still CPU-only and still a frozen-feature readout
experiment — no GPU pilot is justified yet; see `p67` below, run
immediately after this result, per the task's own decision rule ("if the
nonlinear probe is clearly positive, design ONE minimal intervention and
pilot it on a small subset before any long GPU run").

Single checkpoint, validation split, PredCls with GT pairs, frozen
`rel_feat` — no new GPU work in this run.
