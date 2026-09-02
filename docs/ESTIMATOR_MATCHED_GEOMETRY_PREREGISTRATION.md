# Pre-registration — `p60`, estimator-matched geometry vs `rel_feat`

Status: **PRE-REGISTERED**, committed before the tool exists and before any
number exists.

## Why this run is owed

The load-bearing quantitative claim of H6/H7 across this programme is that the
768-d learned representation sits **below 19 numbers from two rectangles**. Every
statement of it contrasts a `rel_feat` probe against **G = 0.5961**.

Those two quantities have never been measured under the same estimator:

| | `G = 0.5961` (`p37` R8) | `rel_feat` probes (`p37` R3/R4, `p55` A) |
|---|---|---|
| fitting data | **1,046,427 TRAIN rows** | **132,556 VALIDATION rows**, 5-fold cross-fitted |
| estimator | **LBFGS, softmax CE, linear** | ridge-on-one-hot (`p37` R3) / **AdamW MLP + CE** (`p55` A) |
| dimensionality | 20 | 769 |

`p58` supplied the first matched-regime measurement and it is not reassuring for
the headline: under **ridge cross-fitted on validation**, geometry reads
**0.5655** and `rel_feat` reads **0.5600** — a gap of **0.0055**, not 0.036, with
overlapping CIs. But `p58` matched the arms by *handicapping both* with a poor
estimator (ridge on one-hot targets is not a classifier), and `p58` additionally
**failed its own gate Y5**, so none of its numbers are reportable.

The missing cell is therefore precise and singular:

> **geometry, under softmax CE, cross-fitted on the validation folds** —
> matched exactly to `p55`'s `A_ce_all` (`rel_feat` = 0.5732).

Until that cell exists, "the representation is below geometry" is confounded
with "the geometry probe was fitted on 8x more data with a better loss", and the
central quantitative claim of H6 is not supportable as stated.

## Design — one estimator, one data regime, applied identically

The estimator is **`p55`'s `fit_ce`, imported and reused unmodified**: AdamW MLP
(hidden 256, 2 hidden layers, ReLU), `lr=2e-3`, `weight_decay=1e-4`, 20 epochs,
batch 8192, seed 0. The data regime is the 5 image-split validation folds,
salt 0, identical to `p37`/`p55`/`p57`/`p58`. Every arm gets both, unchanged.
Features are standardised and bias-augmented identically.

### Arms

| arm | features | dims | role |
|---|---|---|---|
| `A_relfeat` | `rel_feat` | 769 | **must reproduce `p55` A_ce_all = 0.5732** |
| **`B_geometry`** | 19 box features | 20 | **the missing cell** |
| `C_fusion` | `rel_feat` + geometry | 788 | complementarity |
| `D_geometry_linear` | 19 box features, **linear** (no hidden layer) | 20 | the CE analogue of R8's linear form, cross-fitted |
| `N_shuffled` | `rel_feat`, shuffled labels | 769 | chance control |
| `P_prior` | the prior | — | must read exactly 0.5000 |

**Anchors, quoted and never contrasted arm-to-arm** (different regimes):
train-fitted geometry linear **0.5961** and MLP **0.6149** (`p37`/`p39`);
ridge-CV geometry **0.5655** and `rel_feat` **0.5600** (`p58`, gate-failed).

## Validity gates — if any fails, no number is reportable

| gate | requirement |
|---|---|
| **G1** | `P_prior` is **exactly 0.5000** (deviation < 1e-6) |
| **G2** | `N_shuffled` within [0.49, 0.51] |
| **G3** | fold sizes are `[26483, 26856, 27190, 26586, 25441]` |
| **G4** | `A_relfeat` reproduces `p55`'s `A_ce_all` = 0.5732 to within **0.005**. Same estimator, same seed, same rows — a larger deviation means the estimator is not the one `p55` used and nothing here is comparable to `p55`. |
| **G5** | every arm produces an out-of-fold score for every one of the 132,556 rows |

## Criteria — fixed here, before any number exists

### PRIMARY — does geometry's advantage survive estimator matching?

`delta_est = B_geometry − A_relfeat`

| verdict | condition | consequence |
|---|---|---|
| **GEOMETRY-ABOVE** | `delta_est >= +0.020` | the H6/H7 headline survives. The representation genuinely sits below boxes and the "below geometry" framing stands. |
| **COMPARABLE** | `\|delta_est\| < 0.020` | **the 0.036 headline gap was substantially a fitting-regime artifact.** The "below geometry" framing must be withdrawn from every document and H6's quantitative claim restated as a *parity* claim, not a *deficit* claim. |
| **GEOMETRY-BELOW** | `delta_est <= -0.020` | the representation is above boxes under matched fitting; H7 is reversed for discrimination as well as for the composed metric. |

### SECONDARY — is the 0.5961 anchor regime-sensitive?

`regime_gap = 0.5961 − B_geometry`

If `regime_gap >= 0.020`, then **0.5961 is a train-fitting artifact of the same
kind** and it is barred, programme-wide, from being contrasted against any
cross-fitted probe. Both `p37`'s "BELOW GEOMETRY" secondary verdict and every
document quoting the 0.023–0.036 gap must be corrected.

### TERTIARY — complementarity

`delta_fuse = C_fusion − max(A_relfeat, B_geometry)`

**COMPLEMENTARY** if `delta_fuse >= +0.020` — the two feature sets carry
different information and a fusion successor is specified. Otherwise
**REDUNDANT**, and the additive-fusion route closes on a properly-matched
measurement (which `p58` was not).

## What this cannot settle

Every arm is a **readout probe on a frozen encoder**, cross-fitted on
validation, one checkpoint, PredCls with GT pairs. It cannot show what a jointly
trained encoder would do. It also cannot rule out that BOTH quantities would rise
under train-fitting — that comparison is impossible because no train-split
`rel_feat` cache exists (~30 GPU-hours to build). This run makes the two
quantities **comparable**, which is the only thing currently in dispute.
