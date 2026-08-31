# Pre-registration — Experiment D: does appearance add anything ON TOP of calibration?

Registered **before** any arm of this experiment was run.
Written at commit `a46fbad` + the uncommitted C' pathway.
CPU-only. No GPU. No new encoding: it reuses the fp32 cache produced by the
pre-registered experiment B run (`runs/p6_appearance_probe_l14_336_fp32/`).

**This does not modify, supersede or reinterpret experiment B.**
`tools/appearance_probe.py`, `docs/APPEARANCE_PROBE_L14_PREREGISTRATION.md` and
`docs/APPEARANCE_PROBE_L14_RESULT.md` stand exactly as recorded. D asks a
different question that B was not designed to answer.

---

## 1. Why B does not settle the question it is being read as settling

B measured: `s = log P(p|s,o) + lambda * appearance`, sweeping `lambda`, and
found the captured oracle headroom to be **-1.2 %** at the held-out-selected
lambda. It is being read as "a stronger frozen encoder does not convert
appearance into predicate recovery".

Two facts, both measured after B was designed, make that reading unsafe:

**(i) The lambda = 0 baseline is not on the achievable frontier.**
`runs/p7_prior_temperature_sweep/` shows a zero-information scalar

        log P(p|s,o)  ->  log P(p|s,o) - tau * log P(p)

moves mR@50 from 21.98 to 26.00 (+4.03) for -0.64 R@50, and to 38.15 (+16.18)
at tau = 0.5. B's denominator is anchored at tau = 0, a point well inside the
frontier a parameter-free transformation already reaches.

**(ii) The lambda-only family cannot express "calibrate AND add appearance".**
The prior's top-5 ranking is extremely peaked (mean entropy 0.557 nats,
`runs/p7_prior_temperature_forensics/`). In `s = prior + lambda * appearance`
the appearance term is the only term that can flatten it, so lambda must pay
for calibration out of the same budget it uses to express appearance. The
family does not contain the calibrated operating point. A null in a family that
excludes the point of interest does not license a claim about that point.

**(iii) Measured while preparing this registration:** PCA-48 retains only
**55-57 %** of each 768-d block's variance (`svdvals` on the same cache:
subj 56.8 %, obj 56.7 %, union 55.2 %, glob 55.3 %). "The representation was
truncated" is therefore a live alternative to "the representation is
uninformative", and B could not distinguish them.

---

## 2. Question

> Does frozen ViT-L/14-336 appearance improve the predicate decision **beyond
> what a zero-information recalibration of the same prior already achieves**?

## 3. Hypotheses

- **H0 (null):** the best `(tau, lambda>0)` point lies **on or inside** the
  `(tau, lambda=0)` curve on the (R@50, mR@50) plane. Appearance adds nothing
  that calibration does not already give. Branches E/J of the bottleneck list.
- **H1:** the best `(tau, lambda>0)` point lies **strictly outside** that curve
  by more than the shuffled-appearance null band. Appearance carries
  convertible signal that B's 1-D family could not express. Branches A/B.

## 4. Design

| | |
|---|---|
| **CONTROL** | the `lambda = 0` tau curve, scored through the **identical** candidate/decision path as every fitted arm |
| **VARIABLE** | `lambda` (appearance weight), crossed with `tau` (calibration), crossed with `PCA_DIM` |
| **NULL ARM** | shuffled appearance, **refit** at the selected `(tau, lambda)`, 5 seeds — so the null absorbs "adding any scalar-weighted noise term perturbs a peaked ranking" |
| **METRIC** | Pareto gap: `mR@50` of the point minus the `lambda=0` curve's `mR@50` linearly interpolated **at the same R@50**. Reported in mR points. |

Grid: `tau in {0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}`,
`lambda in {0.1, 0.25, 0.5, 1.0, 2.0}`, `PCA_DIM in {48, 128, 256}`.

Held fixed from B: cache, image-level 80/20 held-out split, top-5 candidate set
taken from the **raw (tau=0)** prior, optimiser, 30 epochs, mR definition,
head/body/tail bucketing from train counts.

`log P(p)` is estimated on the **train-fit rows only**, never on validation, so
tau remains leak-free.

## 5. Selection discipline

`tau`, `lambda` and `PCA_DIM` are selected on the **held-out-from-train** split
by held-out mR, exactly as B selected lambda. **Validation is never used for
selection.** The full surface is reported regardless, so the cherry-picked
upper bound is visible alongside the honestly-selected point.

## 6. Success / failure criterion — REGISTERED BEFORE RUNNING

Per `POST_B_PRIOR_CALIBRATION_ANALYSIS.md` section 6.2, an mR-only criterion is
inadmissible here because mR is movable by an information-free transformation.
The criterion is therefore Pareto and null-referenced:

> **H1 is accepted iff**, at the held-out-selected `(tau, lambda)`:
> 1. the Pareto gap is **> 0**, and
> 2. it exceeds the shuffled-appearance null mean by **>= 2 standard deviations**
>    of that null.
>
> Otherwise **H0 is supported.**

An `INDETERMINATE` verdict is recorded, not silently resolved, if the selected
point's R@50 falls outside the tau curve's R range (the comparison would be an
extrapolation).

## 7. Co-registered defect check — Experiment D0

Found while building D, registered here before its result was read:

`tools/appearance_probe.py` computes its reported baseline with
`Pva.argmax(-1)` but predicts every scored arm through
`topk(K=5)` then `gather`-argmax. On tied prior rows the two select different
predicates, so **the additive arms do not reduce to the reported P0 at
lambda = 0** and `headroom_pct` compares across two decision rules.

`tools/probe_tiebreak_forensics.py` measures the disagreement and restates B's
additive sweep against the like-for-like baseline. It is **read-only**: it
re-reads B's recorded metrics and recomputes only the baseline. B's result JSON
is preserved verbatim.

Registered expectation: the correction is **small** and its direction is
**not predicted**; whichever way it moves, it is reported.

## 8. What decision this enables

| Outcome | Decision |
|---|---|
| **H1** | The bottleneck is the decision rule / composition (branch B), not the representation or the encoder. B's headline reading is withdrawn as over-broad. Next: C' with a calibrated composition. |
| **H0 at every PCA_DIM** | Appearance genuinely does not convert on this protocol. B's conclusion stands and strengthens. Stop buying encoder capacity; the remaining live branches are C/D/F/I. |
| **H0 at 48 but H1 at 128/256** | **Representation truncation** was the binding constraint (branch A). B's conclusion is withdrawn as an artifact of PCA-48. |

## 9. Cost

CPU only, ~2.5 s per fit measured; ~35 fits per PCA dim plus 5 null refits.
Estimated **< 10 minutes total**, no GPU. The single L4 stays free for C'.
