# `runs/p45` — WEAK. And it corrects the principle that motivated it.

Registered verdict: **WEAK**, against thresholds committed in
`docs/WITHIN_PAIR_COMPOSITION_PREREGISTRATION.md` (commit `d6494a1`) before the
tool was written. No threshold moved.

---

## The prediction, and what happened

`p44` proposed **PRIOR OVERWRITE** and inferred a prescription: since composition
destroys ~91% of the within-pair variation, *restoring* it should recover a
disproportionate share of the 44,283 prior-adversarial rows without retraining.

| arm | λ | adversarial fixed | prior-correct kept | net rows | R@50 | floor |
|---|---|---|---|---|---|---|
| **real** | **1.0** (deployed) | 3,502 (**7.91%**) | 85,533 (96.90%) | **+762** | 67.168 | ok |
| real | 1.5 | 3,723 (8.41%) | 85,064 (96.36%) | +514 | 66.981 | ok |
| **real** | **2.0** | 3,975 (**8.98%**) | 84,470 (95.69%) | **+172** | 66.723 | ok |
| real | 3.0 | 4,354 (9.83%) | 82,808 (93.81%) | −1,111 | 65.755 | FAIL |
| real | 12.0 | 5,364 (12.11%) | 66,183 (74.98%) | −16,726 | 53.975 | FAIL |

Best floor-holding gain: **+1.07 points** (7.91% → 8.98%). The registered
SUPPORTED threshold was **+2.0**. → **WEAK**.

Nulls behaved: `null_shuffled` **cannot hold the floor at any λ** (it fails even
at λ=1), and `null_pair_matched`'s best floor-holding gain is **+0.49** against
the real term's +1.07 — so the gain is not purely pair identity, though the pair
identity null reproduces roughly **46%** of it.

## What this corrects in `p44`

`p44`'s **descriptive** finding stands and is unaffected: the deployed system is
97.45% pair-constant-predictable against reality's 69.23%, and the model term
alone is 68.97%. Composition really does discard ~91% of the within-pair
variation.

`p44`'s **prescriptive inference does not.** I wrote that the composition failure
was "far larger" than the representation failure and "fixable by decision rule
only". `p45` tests exactly that and the trade is close to **one-for-one**:
going from λ=1 to λ=2 fixes 473 more adversarial rows and breaks 1,063
prior-correct ones, so net rows fall from +762 to +172 and R@50 falls from
67.168 to 66.723.

**The corrected reading:** composition discards the within-pair variation
*because that variation is mostly wrong*. `alpha = 3.75` is close to a rational
response to a weak signal, not a mistake to be undone. The 91% figure measures
how much is discarded, **not** how much is available.

> The binding constraint is the **quality** of the within-pair signal
> (WPRD 0.554, barely above a 0.5 floor), not the composition that suppresses it.

The two failures identified in `p44` are therefore **not separable in the way it
claimed**. The composition failure is downstream of, and bounded by, the
representation failure.

## Where this leaves the successor question

It moves it **back to the representation and the training objective**, and
therefore makes `p36`/`p37` the decisive experiment rather than a
nice-to-have — which is where the pre-registration already put it.

A decision-rule-only successor is **not** justified on this evidence. The
smallest intervention targeting the largest failure is no longer a composition
change; it is whatever `p37` says about `rel_feat`.

## Secondary observation worth keeping

The **geometry** arm degrades far more gracefully: it holds the floor out to
λ=3.0 and its best net is **+920 at λ=1.5**, the highest net of any arm tested
here — above the deployed system's +762. Consistent with `p43`, and consistent
with geometry's higher WPRD: a more accurate within-pair signal survives
amplification better. That is the mechanism this experiment actually
demonstrates.

## Limitations

- **Transductive**, as registered: group means read other validation rows. This
  is a headroom diagnostic, and a deployable version would score no better.
  Since the verdict is WEAK, the transductive advantage makes the result
  *conservative in the direction that matters* — a deployable version would be
  weaker still.
- One λ family, one composition form. A different intervention (margin-gating,
  per-class λ) is not excluded by this, but it would need its own registration.
- tau = 0, PredCls with GT pairs, validation split, one checkpoint.
