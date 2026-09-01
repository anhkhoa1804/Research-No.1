# Pre-registration — does preserving within-pair variation recover adversarial rows?

Status: **PRE-REGISTERED**, committed before the tool is written.
Run: `p45_within_pair_composition`. CPU only. No retraining. No GPU.

## Hypothesis

`runs/p44` established **PRIOR OVERWRITE**: the model term alone is 68.97%
pair-constant-predictable against reality's 69.23%, but the composed system is
**97.45%**, destroying ~91% of the within-pair variation the data contains. If
that is the dominant failure, then *amplifying the within-group component of the
model term at decision time*, changing nothing else, should recover a
disproportionate share of the 44,283 prior-adversarial rows.

## The intervention — one scalar

Split the model term by (s,o) group and reweight only the within part:

```
term(lambda) = between_group(model) + lambda * within_group(model)
score        = 3.75 * (prior - tau*logP) + term(lambda)
```

`lambda = 1` is exactly the deployed system. Sweep
`lambda ∈ {1, 1.5, 2, 3, 5, 8, 12}`.

**No labels are used anywhere in this construction.** The group mean is a
label-free function of the model term. It is, however, **transductive** — it
reads other validation rows' model terms. This is therefore a **headroom
diagnostic**, not a deployable method, and will be reported as one. A
fold-restricted variant (group means from training folds only, estimable rows
only) is reported alongside as the non-transductive check.

## Arms

| arm | purpose |
|---|---|
| `real` | the model term |
| `null_shuffled` | model term permuted across all rows — must NOT benefit |
| `null_pair_matched` | permuted within groups — destroys image content, keeps pair identity |
| `geometry` | the same amplification applied to the geometry probe |

## Primary criterion

On prior-adversarial rows at tau = 0, with the R@50 floor of the deployed system
(**66.593**, the prior-only baseline) required to hold:

- **SUPPORTED**: some `lambda > 1` raises adversarial-fixed share by **≥ +2.0
  percentage points** over `lambda = 1` (7.91% → ≥ 9.91%) while overall R@50
  stays ≥ 66.593, **and** exceeds `null_shuffled`'s best by ≥ 1.0 point.
- **WEAK**: gain in [+0.5, +2.0) points under the same constraints.
- **REFUTED**: gain < +0.5 points, or the floor fails at every `lambda` that
  gains, or `null_shuffled` matches within 1.0 point.

Thresholds: +2.0 points is roughly a quarter of the checkpoint's entire
adversarial yield (7.91%), so it is a materially large effect; +0.5 is the
smallest gain that would exceed the ~0.3-point run-to-run spread seen across
tau in `p43`. The 1.0-point null margin is the project's existing
`NULL_MARGIN_PTS`.

## Interpretation fixed in advance

- **SUPPORTED** ⇒ the dominant failure is the decision rule, and a successor
  should change composition before touching the encoder. The transductive
  caveat means the *deployable* version still has to be built and re-tested.
- **REFUTED** ⇒ the within-pair variation is too weakly aligned to exploit even
  when preserved, and the representation failure dominates after all. That
  would move the successor question back to training, and would partly
  rehabilitate the reading `p44` argues against.
- If `null_pair_matched` benefits as much as `real`, the recovered rows are
  explained by pair identity rather than image content, and no successor follows
  from this line regardless of the headline number.
