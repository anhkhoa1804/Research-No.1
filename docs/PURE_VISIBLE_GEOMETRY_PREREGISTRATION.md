# `p69` preregistration — is the geometry PURE cannot see worth anything?

Committed before `tools/pure_visible_geometry.py` is run. CPU only, no GPU.
Reads the `p36` cache. Reuses `p60`'s exact estimator, folds and reporting
path, so every number is comparable to `p60`/`p65` by construction.

## Motivation

`p68` (`docs/GEOMETRY_INPUT_DEGENERACY_RESULT.md`) established, bit-exactly,
that 6 of the 8 geometry channels reaching PURE's `forward_pairs` are
constant: only frame-relative `dx`, `dy` vary. The tempting inference is
"therefore fixing the units will help." **That inference is not licensed**,
and this experiment exists to try to kill it before any GPU is spent.

The 19-number geometry probe (`B_geometry` = 0.5976 under `p60`'s estimator)
includes sizes, areas, aspect ratios, IoU and containment — channels PURE
never receives. If a probe **restricted to what PURE actually receives**
already reaches the same WPRD, then the zeroed channels carry no
discriminative content this metric can see, the units defect is
**scientifically inert**, and the C1 retraining pilot is not justified.

## Arms (all cross-fitted, 5 folds, salt 0, `p60` estimator, hidden 256)

| arm | features |
|---|---|
| `A_relfeat` | 768-d `rel_feat`, standardised (`p60` anchor 0.5732) |
| `B_geometry` | all 19 geometry numbers (`p60` anchor 0.5976) |
| `P_pure_visible` | **cols 6,7 only** — `(ocx−scx)/W`, `(ocy−scy)/H`, the image-scale offsets, the only channels PURE receives non-degenerately |
| `Q_visible_plus_sizes` | cols 6,7 **+** cols 8–11 (`sw/W, sh/H, ow/W, oh/H`) — isolates whether *box size specifically* is the missing ingredient |
| `N_shuffled` | `rel_feat` against permuted labels (null) |
| `P_prior` | prior control (must read exactly 0.5000) |

## Primary quantity

```
delta_missing = WPRD(B_geometry) − WPRD(P_pure_visible)
```

i.e. how much of geometry's discriminative advantage lives **specifically in
the channels the units defect destroys**.

## Pre-registered decision rule

- **INERT — hypothesis killed, no GPU:** `delta_missing < +0.01`.
  The zeroed channels add nothing over what PURE already receives. `p68`
  remains a true and reportable defect, but it is not a lever, the C1
  retraining pilot is **not** justified, and Track C's remaining route closes.
- **MATERIAL — C1 justified:** `delta_missing >= +0.03`.
  The zeroed channels carry substantial discriminative content that PURE is
  structurally denied. This is the first *measured* mechanism in this
  programme that predicts a specific intervention.
- **AMBIGUOUS:** `+0.01 <= delta_missing < +0.03`. Report as weak support;
  decide on cost, do not auto-escalate to GPU.

Secondary, reported but carrying no decision:
`delta_size = WPRD(Q_visible_plus_sizes) − WPRD(P_pure_visible)`, which
attributes the gap to size channels specifically rather than to
IoU/containment/aspect.

## Validity gates (all must pass or the run is void)

- **G1** `A_relfeat` reproduces `p60`'s 0.5732 within ±0.005.
- **G2** `B_geometry` reproduces `p60`'s 0.5976 within ±0.005.
- **G3** fold sizes equal the registered `26483,26856,27190,26586,25441`.
- **G4** `P_prior` reads exactly 0.5000.
- **G5** `N_shuffled` lies within noise of 0.5.
- **G6** every arm scores every GT row, all finite.

## Known caveat, stated in advance

`P_pure_visible` uses `(ocx−scx)/W` with `W` the per-image object extent,
whereas PURE divides by the fixed 336 frame. These are not identical: the
per-image normalisation is mildly *more* informative than a global constant.
So `P_pure_visible` is, if anything, an **upper bound** on what PURE's two
surviving channels supply. That biases the test **against** the MATERIAL
verdict, which is the conservative direction for the conclusion this
experiment would license. It does not bias the INERT verdict, which is
therefore the more trustworthy of the two outcomes.

This arm is a *proxy for PURE's input*, not a reconstruction of PURE's
pipeline. It bounds information content; it does not simulate the encoder.
