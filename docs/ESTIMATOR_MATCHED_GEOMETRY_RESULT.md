# `runs/p60` — GEOMETRY-ABOVE, REGIME-STABLE, REDUNDANT. The headline survives.

Pre-registered in `docs/ESTIMATOR_MATCHED_GEOMETRY_PREREGISTRATION.md` (commit
`bb2796f`), before the tool existed. **This run was designed to break the
programme's central quantitative claim. It failed to, and that is the result.**

## Why it was run

Every statement of H6/H7 contrasts a `rel_feat` probe **cross-fitted on 132,556
validation rows** against **G = 0.5961**, a geometry probe **train-fitted on
1,046,427 rows with LBFGS softmax CE**. Those are different estimators on
different data. `p58` had already hinted the gap might be regime-driven (under
ridge-on-one-hot: geometry 0.5655 vs `rel_feat` 0.5600, a gap of 0.0055 rather
than 0.036) — though `p58` failed its own gate Y5 and none of its numbers are
citable.

## Validity gates — ALL PASS

| gate | requirement | observed |
|---|---|---|
| **G1** | prior exactly 0.5000 | **0.500000** |
| **G2** | shuffled in [0.49, 0.51] | **0.4992** |
| **G3** | folds `[26483, 26856, 27190, 26586, 25441]` | identical |
| **G4** | `A_relfeat` reproduces `p55`'s `A_ce_all` = 0.5732 ± 0.005 | **0.5732**, Δ = **+0.0000** |
| **G5** | every arm scores all 132,556 rows | 8 arms × 132,556 |

**G4 is the load-bearing gate and it is exact.** The estimator here *is* `p55`'s
estimator, so `B_geometry` and `A_relfeat` are comparable by construction.

## Result — one estimator (AdamW MLP + softmax CE), one regime (5-fold CV on validation)

| arm | dims | WPRD | 95% CI | head–head | body–body | tail–tail |
|---|---|---|---|---|---|---|
| `A_relfeat` | 769 | 0.5732 | [0.5674, 0.5782] | 0.5823 | 0.6064 | 0.4859 |
| **`B_geometry`** | **20** | **0.5976** | **[0.5930, 0.6027]** | 0.5988 | 0.6200 | 0.5820 |
| `C_fusion` | 789 | 0.5922 | [0.5871, 0.5974] | 0.6009 | **0.6427** | 0.4759 |
| `D_geometry_linear` | 20 | 0.5741 | [0.5689, 0.5794] | 0.5818 | 0.6219 | 0.5935 |
| `N_shuffled` | — | 0.4992 | [0.4940, 0.5043] | ✓ | | |
| `P_prior` | — | **0.5000** | [0.5000, 0.5000] | ✓ | | |

```
PRIMARY    delta_est  = B - A            = +0.0244  ->  GEOMETRY-ABOVE   (threshold +0.020)
SECONDARY  regime_gap = 0.5961 - B       = -0.0015  ->  REGIME-STABLE    (threshold +0.020)
TERTIARY   delta_fuse = C - max(A,B)     = -0.0054  ->  REDUNDANT        (threshold +0.020)
```

## The three findings

### 1. The geometry advantage is NOT a fitting-regime artifact

Cross-fitted on validation with the identical estimator that gives `rel_feat`
0.5732, geometry reads **0.5976** — within **0.0015** of the train-fitted
0.5961. The CIs are **disjoint** ([0.5930, 0.6027] vs [0.5674, 0.5782]).

**The 0.5961 anchor is vindicated**, and the "below geometry" framing may keep
being used. This was the single most dangerous confound in the programme and it
is now measured rather than assumed.

### 2. `p58` was wrong about the mechanism, and its gate was right to void it

`p58` matched the two arms and got a 0.0055 gap. It attributed this to fitting
regime. `p60` shows the actual cause was **the estimator, not the regime**:
ridge-on-one-hot is a poor classifier surrogate and it handicapped the *low
dimensional* arm far more than the high dimensional one. Under a real classifier
the gap returns in full. `p58`'s numbers were correctly declared non-reportable
by gate Y5; they should also not be quoted as a *qualitative* hint, and this
document is the correction.

### 3. Capacity, not data, is what geometry needs — and the fusion route is closed

`D_geometry_linear` (linear, CE, cross-fitted) reads **0.5741**, well below
`B_geometry`'s MLP at 0.5976. So a *linear* geometry probe genuinely does gain
from train-fitting (0.5741 CV → 0.5961 train), but a *nonlinear* one recovers
the same value from validation alone. The box channel's information is
**nonlinear in the 19 features**, which no previous run isolated.

And `C_fusion` (0.5922) is **below `B_geometry` alone** (0.5976). Under a matched
good estimator, concatenating the 768-d representation onto the boxes **does not
help and slightly hurts** — `p37`'s R9 finding survives its own correction.
Combined with `p58`'s failed minimal fix, **the additive-fusion successor route
is closed on a clean measurement.**

## What this does not settle

Readout probes on a frozen encoder, cross-fitted on validation, one checkpoint,
PredCls with GT pairs. It cannot say what a jointly trained encoder would do. It
also cannot compare both quantities *train-fitted*, because no train-split
`rel_feat` cache exists (~30 GPU-hours). What it establishes is that the
comparison the programme has been making is **sound**, which is what was in
dispute.
