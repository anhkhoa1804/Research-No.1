# The principle: composition overwrites the evidence

`runs/p44`. The simplest statement that accounts for every result in this cycle,
and it moves the bottleneck from where the programme had been looking.

---

## The prediction that was tested

If the visual pathway had merely *learned the prior* — "prior absorption" — then
its own output would be roughly as determined by (subject, object) as the ground
truth is. If it had *over-absorbed* the prior, its output would be **more**
pair-determined than reality.

Measured on the 75,366 decidable rows, two ways: within-group label entropy, and
the accuracy of the best pair-**constant** predictor at reproducing each arm's
own output. A pure function of (s,o) is 100% reproducible that way.

| arm | within-group entropy (macro / wtd) | pair-constant predictability |
|---|---|---|
| **GROUND TRUTH** | 0.7391 / 0.7513 | **69.23%** |
| prior only | 0.0000 / 0.0000 | 100.00% *(by construction)* |
| **prior + MODEL — the deployed system** | 0.0630 / 0.0527 | **97.45%** |
| prior + GEOMETRY (w=1) | 0.0359 / 0.0283 | 98.62% |
| prior + MODEL + GEOMETRY | 0.0795 / 0.0653 | 96.94% |
| **MODEL term alone (argmax)** | 0.5660 / **0.8030** | **68.97%** |

## The result, and it is not what was expected

**Prior absorption is FALSIFIED at the representation level.** The model term on
its own is **68.97%** pair-constant-predictable against reality's **69.23%** —
a difference of 0.26 points. By this measure it carries *approximately the right
amount* of within-pair variation. Its weighted within-group entropy (0.8030) is
in fact *higher* than the ground truth's (0.7513).

**The collapse happens at composition.** The deployed system —
`3.75·(prior − τ·logP) + model_term` — is **97.45%** predictable from object
identity alone, against reality's 69.23%. Its within-group entropy is **0.0630**
versus reality's **0.7391**: composition destroys about **91%** of the
within-pair variation that actually exists in the data.

## The principle

> **PRIOR OVERWRITE.** The checkpoint's image-conditioned variation is roughly
> correct in *magnitude* and only weakly correct in *direction*. Almost all of it
> is then discarded at decision time by a prior weight (`alpha = 3.75`) tuned to
> maximise average recall — because the prior is right on two thirds of rows, so
> deferring to it is optimal on average and catastrophic on exactly the third
> where it is wrong.

This is a **decision-rule** failure sitting on top of a much smaller
**representation** failure, and the two are separable and separately sized:

| failure | size | fixable how |
|---|---|---|
| the model's within-pair variation is only weakly aligned with truth | WPRD 0.554 vs geometry's 0.596 — a 0.042 gap | retraining / better objective |
| **composition discards ~91% of whatever within-pair variation exists** | **entropy 0.566 → 0.063** | **decision rule only — no retraining** |

**The second is far larger than the first.**

## Why every earlier result follows from it

| observation | explained by |
|---|---|
| model adds only +0.575 R (`p24`) | its variation is overwritten |
| 7.9% of prior-adversarial rows fixed (`p43`) | ditto — the prior wins the argmax |
| model wins the composed metric but loses on WPRD (`p41`) | the composed metric rewards *surviving* composition, WPRD measures what composition discards |
| WPRD reads only 14.2% of the term's variance (`p42`) | that 14.2% is the part composition suppresses |
| mR rises with tau, not with grounding (`p29`, `p35`) | tau reweights classes; it does not restore within-pair variation |
| tail grounding indistinguishable from chance in the evaluated head (`p35`) | the smaller, genuine representation failure |

## What it predicts, and what to do next

**Prediction (testable on the existing cache, no GPU):** a composition that
preserves within-pair variation — e.g. applying the prior *between* groups but
not *within* them, or gating the prior by its own margin — should recover a
disproportionate share of the 44,283 prior-adversarial rows *without* the
retraining that the representation failure would require.

That is the smallest intervention targeting the largest measured failure, and it
is the natural successor experiment. It must be pre-registered with a null
(the same intervention applied to a shuffled model term) before it is run.

## Caveats

- "Approximately the right amount of variation" is **not** "the right
  variation". WPRD says the model term's within-pair signal is only weakly
  aligned with truth (0.554 against a 0.5 floor). The two statements are
  compatible and both are needed.
- The macro and weighted entropies disagree in sign for the model term alone
  (0.5660 < 0.7391 macro; 0.8030 > 0.7513 weighted). The pair-constant
  predictability measure, which is unambiguous, is the one the claim rests on.
- `alpha = 3.75` is the historical protocol's value, not an optimum this project
  chose. The principle is about that operating point.
- One checkpoint, PredCls with GT pairs, validation split.
