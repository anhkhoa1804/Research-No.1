# Pre-registration — `p67`, auxiliary `dx_rel`/`dy_rel` loss on the frozen-`rel_feat` readout

Status: **PRE-REGISTERED**, committed before the tool exists and before any
number exists. CPU only, frozen `rel_feat` (`p36`'s cache) — no GPU, no
encoder retraining.

## Why

`p66` (`docs/NONLINEAR_DXDY_DECODABILITY_RESULT.md`) found `dy_rel`
nonlinearly decodable from `rel_feat` at R²=0.292 (up from linear's 0.223),
clearly above this estimator's own empirical permutation null, with `dx_rel`
weakly so (0.073). `p65` had already shown that *concatenating*
`dx_rel`/`dy_rel` explicitly as extra inputs to a CE-trained classifier MLP
does not help (MINIMAL-FIX-NEUTRAL). Put together: the signal is available
to a nonlinear function of `rel_feat`, but a classification-only readout
does not surface it even when handed the raw numbers directly. This is the
(B) branch of the directive's decision rule ("information exists but is
poorly organised/read out ⇒ design a minimal nonlinear/spatial readout
intervention before changing the encoder"), and the intervention it
specifies is a change to the **readout's training signal**, not to its
inputs — an auxiliary loss forcing the shared representation to retain what
the classification objective alone does not prioritise.

## Design

Same frozen `rel_feat` (`p36`), same 5 image-level folds
(`objective_ablation_relfeat.folds_of`), same WPRD/CI reporting path as
`p60`/`p65`/`p66`. A single shared-trunk MLP
(`Linear(769,256)→ReLU→Linear(256,256)→ReLU`, identical trunk to every MLP
probe in this line) feeds **two heads**:

- classification head: `Linear(256, 51)`, predicate cross-entropy (the
  existing objective, unchanged)
- regression head: `Linear(256, 2)`, MSE against standardised `dx_rel`,
  `dy_rel`

Total loss: `CE + lambda * MSE`. `lambda` is **not tuned on WPRD** — it is
swept over a small, fixed-in-advance grid `{0.0, 0.5, 1.0, 2.0}` chosen so
the two loss terms are of comparable magnitude at initialisation (CE over
51 classes ≈ ln(51) ≈ 3.9; MSE over 2 unit-variance targets ≈ 2), and every
value is reported — none is selected as "the" answer. `lambda=0.0`
reproduces `A_relfeat` from `p60`/`p65` and is this run's gate.

## Validity gates

| gate | requirement |
|---|---|
| **W1** | folds identical to `p37`/`p57`/`p60`/`p65`/`p66` |
| **W2** | `lambda=0.0` arm reproduces `p60`'s `A_relfeat` = 0.5732 ± 0.005 |
| **W3** | prior control (`P_prior`) reads exactly 0.5000 |
| **W4** | shuffled-label control (classification labels shuffled, `lambda=0`) within [0.49, 0.51] |

## Criteria — fixed here, before any number exists

`delta_aux = max(WPRD at lambda in {0.5, 1.0, 2.0}) - WPRD at lambda=0.0`

| verdict | condition |
|---|---|
| **AUX-LOSS-WORKS** | `delta_aux >= +0.02` |
| **AUX-LOSS-NEUTRAL** | `\|delta_aux\| < 0.02` |
| **AUX-LOSS-HARMFUL** | `delta_aux <= -0.02` |

(Threshold set at the same ±0.02 magnitude this line of work has used
throughout — `p58`, `p65` — for consistency with the existing decision
scale, not tuned to this data.)

## What each outcome means

- **AUX-LOSS-WORKS**: the readout, not the representation or the encoder,
  was the fixable part of this specific deficit. A GPU pilot retraining the
  actual classifier head (not just a frozen probe) with this auxiliary loss
  becomes justified, scoped to a small subset first per the task's own
  policy.
- **AUX-LOSS-NEUTRAL or HARMFUL**: even forcing the shared trunk to retain
  `dy_rel` information does not help predicate discrimination — meaning the
  information, while nonlinearly recoverable, is not usefully positioned for
  the classification task in the way this intervention assumes. This would
  close the "readout fix" branch too, at which point every additive,
  frozen-readout, and now readout-training-signal intervention this
  programme can cheaply test has failed, and a joint-training pilot on the
  actual encoder is the only untested class left — still not automatically
  justified, since "the readout can't use it either" is itself informative
  against that pilot's likely payoff.

## What this cannot settle

This is still a frozen-`rel_feat` probe; the encoder that produced `rel_feat`
is never touched. A negative result here says the deficit is not fixable by
retraining a *shallow readout* with this specific auxiliary signal — it
does not rule out an encoder-level fix. Single checkpoint, validation split,
PredCls with GT pairs.
