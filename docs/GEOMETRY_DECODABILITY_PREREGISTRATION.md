# Pre-registration — `p57`, does `rel_feat` encode the layout that beats it?

Status: **PRE-REGISTERED**, committed before the tool exists and before any
number exists.

## Why

After `p55` and `p56`, **H6 (representation bottleneck) survives only by
elimination**: the readout route is dead (`p37`, `p55`), the objective route is
refuted on both channels (`p48`, `p55`), and the supervision route is refuted
(`p56`). "Everything else failed" is not a mechanism, and a hypothesis held that
way cannot direct an architecture.

This gives H6 positive content or takes it away. A linear probe on **19 numbers
from two rectangles** reaches WPRD 0.5961, beating every probe on the 768-d
`rel_feat` (best 0.5732). There are exactly two ways that can happen:

- **(A) The layout information is ABSENT from `rel_feat`.** The encoder discards
  spatial configuration. That is a *named, actionable* representational deficit
  and it tells an architect precisely what to add.
- **(B) The layout information is PRESENT in `rel_feat` but unused.** Then the
  deficit is not representational at all, and since the readout and objective
  routes are already closed, the research map has a genuine contradiction that
  must be resolved before anything is built.

The experiment simply asks which.

## Design

`rel_feat` is 768-d, cached for the 132,556 validation GT rows (`p36`).
Geometry is the same 19 scale-invariant box features
(`wprd_geometry_control._geom`) used by every geometry arm in this programme.

Cross-fitted over the **same 5 image-split validation folds** as `p37`/`p55`.

| arm | quantity |
|---|---|
| `D1` | out-of-fold R² of `rel_feat` -> each of the 19 geometry features, and the mean |
| `D2` | out-of-fold R² of `rel_feat` -> the geometry probe's 50 **logits** (the actual decision-relevant projection) |
| `D3` | reverse: out-of-fold R² of geometry -> `rel_feat` (reported for orientation, already ~18% in `p42`) |
| `N1` | shuffled-row control for D1 (must be ~0) |
| `N2` | R² of a random 768-d Gaussian -> geometry (must be ~0) |

## Validity gates

| gate | requirement |
|---|---|
| **X1** | `N1` and `N2` mean R² < 0.02 |
| **X2** | `rel_feat` is (132556, 768), finite |
| **X3** | folds identical to `p37` (`[26483, 26856, 27190, 26586, 25441]`) |

## Criteria — fixed here, before any number exists

Primary quantity: `R2_geom` = mean out-of-fold R² of `rel_feat` predicting the
19 geometry features (`D1`).

| verdict | condition | meaning |
|---|---|---|
| **LAYOUT-ABSENT** | `R2_geom < 0.30` | the encoder discards spatial configuration — H6 gains a named mechanism |
| **LAYOUT-PARTIAL** | `0.30 <= R2_geom < 0.70` | partially retained; deficit is degradation, not omission |
| **LAYOUT-PRESENT** | `R2_geom >= 0.70` | the information is there and unused — the map has a contradiction |

Reported alongside, not criteria: the per-feature R² table (which *kinds* of
spatial fact survive — position, size, offset, IoU, containment), and `D2`.

## What this cannot settle

R² measures **linear** decodability. Low R² does not prove the information is
absent in an information-theoretic sense, only that it is not linearly
available — which is nonetheless the relevant sense here, because the geometry
arm that beats `rel_feat` is itself linear. This limitation is registered up
front and will be restated with the result.
