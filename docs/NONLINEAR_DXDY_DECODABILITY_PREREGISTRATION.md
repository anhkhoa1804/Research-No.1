# Pre-registration — nonlinear decodability of `dx_rel`/`dy_rel` from `rel_feat`

Status: **PRE-REGISTERED**, committed before the tool exists and before any
number exists. Directive given inline by the user before this run, quoted
verbatim as the registration text. CPU only. Reads
`runs/p36_relfeat_cache/pair_logits_relfeat.pt` (already on disk).

## Why

`p57` (`docs/GEOMETRY_DECODABILITY_RESULT.md`) measured **linear**
out-of-fold R² of `rel_feat` predicting each of 19 box-geometry features.
`dx_rel` (pair-relative horizontal displacement) was the single worst,
R² = 0.052; `dy_rel` (vertical) was 0.223. `p57`'s own pre-registration
named this limitation up front: "R² measures linear decodability... A
nonlinear decoder might recover more, and that is untested." `p65`
(`docs/MINIMAL_FIX_AND_CLEAN_FUSION_RESULT.md`) then found that handing a
readout MLP `dx_rel`/`dy_rel` explicitly as extra linear inputs did not
meaningfully help (MINIMAL-FIX-NEUTRAL) — but that is a different question
from whether the information is nonlinearly *already present* in `rel_feat`
itself, unextracted.

## Question

Is relative position genuinely absent from `rel_feat`, or merely not
linearly decodable?

## Design

Same 5 image-level validation folds as `p37`/`p57`/`p60`/`p65`
(`[26483, 26856, 27190, 26586, 25441]`), same standardised `rel_feat`
(768-d + bias) as every prior probe in this line. Targets: `dx_rel`,
`dy_rel` — columns 4 and 5 of `wprd_geometry_control._geom`'s 19-feature
output, standardised the same way `p57` standardises its targets.

Three arms per target, all cross-fitted out-of-fold:

1. **Linear** (ridge, `l2=1e-2`) — reproduces `p57`'s D1 numbers for these
   two columns exactly; serves as this run's own reproduction gate.
2. **MLP** — a small 2-layer ReLU network (hidden=256), AdamW, MSE loss,
   20 epochs, identical architecture family to every other MLP probe this
   programme uses (`objective_ablation_relfeat.mlp`, adapted to a
   single-unit regression output).
3. **Shuffled-label control** — same MLP architecture, row-permuted targets
   (must read R² ≈ 0; this is the chance reference the interpretation below
   is measured against).

Reported with 95% bootstrap CIs (row-level resampling of the out-of-fold
residuals, 200 resamples, matching this project's standard bootstrap
convention).

## Validity gates

| gate | requirement |
|---|---|
| **Z1** | folds identical to `p37`/`p57`/`p60`/`p65` |
| **Z2** | linear R² for `dx_rel`/`dy_rel` reproduces `p57`'s 0.052/0.223 within ±0.01 |
| **Z3** | shuffled-label MLP control R² is within [−0.02, 0.02] for both targets |

## Interpretation — fixed here, exactly as directed, before any number exists

Judged on the **MLP** arm against its own shuffled-label control and 95% CI,
for **each of `dx_rel` and `dy_rel` separately** (the two targets are not
pooled into one verdict, since `p57` already found they differ by 4x
linearly and there is no reason to assume they'd behave identically
nonlinearly):

- **(A) NONLINEAR ~ CHANCE**: the MLP arm's 95% CI includes the shuffled
  control's range (effectively, R² not clearly distinguishable from 0) —
  **the representation genuinely lacks usable spatial information** for that
  coordinate. Consequence: a jointly retrained spatial representation
  becomes the strongest remaining successor hypothesis for that axis.
- **(B) NONLINEAR CLEARLY > CHANCE**: the MLP arm's 95% CI excludes 0 with
  daylight (and, for magnitude context only, is compared against `p57`'s own
  0.30 "partial" boundary) — **the information exists but is poorly
  organised or read out**, not absent. Consequence: a minimal nonlinear
  readout intervention is design-worthy and should be piloted on a small
  subset before touching the encoder.

No new numeric threshold is invented beyond what is stated above; the
CI-vs-chance comparison is the criterion, exactly as the directive specified.

## What this cannot settle

An MLP with this specific architecture and training budget is not proof of
the ceiling of nonlinear decodability — a larger or differently-regularised
probe could in principle recover more or less. It is a reasonable, cheap,
CPU-only next step past the linear probe, not a definitive nonlinearity
ceiling. Single checkpoint, validation split, PredCls with GT pairs, frozen
`rel_feat` — no new GPU work.
