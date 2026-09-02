# Pre-registration — `p58`, matched-fitting geometry / `rel_feat` fusion

Status: **PRE-REGISTERED**, committed before the tool exists and before any
number exists.

## Why this run is owed

`p37` concluded that "adding the learned representation to boxes **destroys**
information", from `R9_relfeat+geometry` = 0.5735 < `R8_geom` = 0.5961. Those
two arms differ in **three** ways simultaneously:

| | `R8_geom` (0.5961) | `R9` fusion (0.5735) |
|---|---|---|
| fitting data | **1,046,427 TRAIN rows** | **132,556 VALIDATION rows**, cross-fitted |
| loss | **LBFGS softmax cross-entropy** | **ridge on one-hot targets** (squared error) |
| input dimensionality | **20** | **788** |
| regularisation | `1e-4 * (W*W).sum()` on CE | fixed `l2=1e-4` ridge, never tuned |

A 788-dimensional probe cross-fitted on 132k rows against a 20-dimensional probe
fitted on 1.05M rows is not a controlled comparison, and `p55` has since shown
this probe family moves by ~0.013 on hyperparameters alone. **The `p37` R9
conclusion is therefore not supportable as stated, and this run replaces it.**

This matters because R9 is the one result standing between `p57`'s named deficit
(the encoder discards relative position, `dx_rel` R² = 0.052) and a concrete
successor architecture. If fusion helps under matched fitting, the deficit is
additively fixable and a successor is specified. If it does not, that
architectural route closes on a clean measurement instead of a confounded one.

## Design — one regime, applied identically to every arm

`rel_feat` exists **only for the validation split** (`p36`); there is no
train-split `rel_feat` cache and building one is ~30 GPU-hours. Therefore the
only regime in which all arms are comparable is **cross-fitting on the
validation folds**, and every arm uses it. No arm is train-fitted.

Held identical across all arms: the 5 image-split folds (`p37`/`p55`/`p57`
construction, salt 0), the estimator family (ridge to one-hot targets), the
**nested regularisation search** (below), the standardisation, and the metric.

### The capacity control, which is the point of the run

Input dimensionality cannot be equalised across arms — that *is* the variable.
It is handled two ways:

1. **Nested regularisation selection.** For every arm and every outer fold, the
   ridge strength is chosen from a shared grid
   `lambda in {1e-4, 1e-3, 1e-2, 1e-1, 1e0}` by inner-fold **multiclass
   log-loss** (a proper scoring rule) on a held-in 20% of the training rows,
   then refitted on all training rows. WPRD is never consulted during
   selection. This gives every arm its best-regularised form, so a
   high-dimensional arm is not penalised merely for being high-dimensional.
2. **Arm F, the dimensionality null.** `random-768 + geometry` has *exactly*
   arm C's dimensionality and *exactly* arm B's information. It isolates
   "does padding geometry with 768 uninformative dimensions degrade it?"

### Arms

| arm | features | dims | role |
|---|---|---|---|
| `A_relfeat` | `rel_feat` | 769 | representation alone |
| `B_geometry` | 19 box features | 20 | layout alone |
| **`C_fusion`** | `rel_feat` + geometry | 788 | **the primary question** |
| `D_relfeat_plus_dxdy` | `rel_feat` + `dx_rel`, `dy_rel` | 771 | the minimal `p57`-motivated fix |
| `E_relfeat_plus_shuffled_geom` | `rel_feat` + row-shuffled geometry | 788 | null for geometry's contribution |
| **`F_random_plus_geometry`** | random-768 + geometry | 788 | **dimensionality null** |
| `G_prior` | the prior | — | must read exactly 0.5000 |
| `H_shuffled_labels` | `rel_feat`, shuffled labels | 769 | chance control |

Reported as an **anchor, not an arm**: `p37`/`p39`'s train-fitted geometry CE
probe at **0.5961**. It is in a different regime and is quoted only so the two
regimes can be compared, never contrasted arm-to-arm.

## Validity gates — if any fails, no number is reportable

| gate | requirement |
|---|---|
| **Y1** | `G_prior` WPRD is **exactly 0.5000** (deviation < 1e-6) |
| **Y2** | `H_shuffled_labels` within [0.49, 0.51] |
| **Y3** | folds identical to `p37`: `[26483, 26856, 27190, 26586, 25441]` |
| **Y4** | every arm produces an out-of-fold score for every one of the 132,556 rows |
| **Y5** | the selected lambda is reported per arm per fold, and no arm is pinned to a grid endpoint on every fold (which would mean the grid is too narrow) |

## Criteria — fixed here, before any number exists

### PRIMARY — is fusion better than layout alone, under matched fitting?

`delta_fuse = C_fusion − B_geometry`

| verdict | condition | consequence |
|---|---|---|
| **FUSION-GAIN** | `delta_fuse >= +0.02` | the representation adds to layout; `p37`'s R9 was a fitting artifact; **a successor is specified** |
| **FUSION-NEUTRAL** | `\|delta_fuse\| < 0.02` | the representation adds nothing to layout it does not already have |
| **FUSION-HARMFUL** | `delta_fuse <= -0.02` | `p37`'s R9 survives correction; the representation actively interferes |

### SECONDARY — does the minimal `p57`-motivated fix work?

`delta_min = D_relfeat_plus_dxdy − A_relfeat`

**MINIMAL-FIX-WORKS** if `delta_min >= +0.02`. This asks whether handing the
encoder *only the two displacement numbers it fails to encode* recovers the gap,
which is the cheapest possible architectural intervention and the one `p57`
directly motivates.

### TERTIARY — the dimensionality confound

`delta_dim = F_random_plus_geometry − B_geometry`

If `delta_dim <= -0.02`, then padding with uninformative dimensions degrades
geometry on its own, the PRIMARY contrast `C − B` is confounded by dimension,
and the **dimension-matched contrast `C_fusion − F_random_plus_geometry`** is
reported as the corrected primary. Both are computed and both are reported
regardless of outcome.

## What this cannot settle

All arms are **readout probes on a frozen encoder**, cross-fitted on validation.
A FUSION-GAIN would show the two feature sets are complementary *to a linear
readout*; it would not by itself show that a jointly-trained encoder achieves
it. A FUSION-NEUTRAL/HARMFUL result closes the *additive* route only, not every
possible architecture.

Single checkpoint, validation split, PredCls with GT pairs.
