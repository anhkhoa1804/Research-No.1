# Corrected pair-prior distillation (`runs/p32`) — result

Run: exit 0, 5,351 s (89 min, budget 2 h). Estimable subset of the `p24` full
cache, **88,596 rows mean** over 5 partitions (66.9% of 132,556) — more than
twice the entire 3,000-image screening set that `p27` used.

Pre-registered in `docs/PAIR_PRIOR_DISTILLATION_PREREGISTRATION.md` (commit
`91d33b8`), **before the tool that runs it was written**. No threshold moved.

## 0. Gates — all pass, on every partition

| gate | requirement | observed |
|---|---|---|
| G1 outer fallback | exactly 0 | **0.000%** on all 5 |
| G2 residual cosine | < 0.70 | **0.458–0.459** (p30 independently measured 0.459) |
| G4 rows | ≥ 80,000 | 88,390–88,722 |
| G5 `null_shuffled` ≤ frontier | — | −1.569 ✓ |

Inner-loop fallback 4.1–4.4%, reported and not gated (it affects only which beta
is chosen, not the scores that are read). Compare `p27`'s **45.4%** outer
fallback — the defect that withdrew it is gone.

Subset prior-only baseline: R@50 **72.160** / mR **26.069**, recomputed on the
subset as registered. It is 5.6 R points above the full-split baseline, which is
exactly why inheriting the full-split number was forbidden.

## 1. The registered verdict, reported verbatim

> **VERDICT: NOT EXPLAINED**
> `P_G` (G_model) = −2.456 · `P_V` (best floor-eligible vision-free) = −0.751 ·
> gates all pass.

**This label is correct by the registered rule and misleading about the
direction of the result.** The rule reads:

> restricted to arms that clear the registered floor on ≥ 4 of 5 partitions. If
> no vision-free arm is eligible, ... the verdict is NOT EXPLAINED **regardless
> of its magnitude**.

**No arm clears the floor on ≥4/5 partitions — including `G_model` itself
(0/5).** So the eligibility set is empty and the rule fires. The floor
(`subset prior − 0.30 pts`) was calibrated from the full-population floor's own
construction, and on this subset every learned arm loses more than 0.30 R.
That is a mis-calibration of my floor for a changed population, and it is
recorded rather than repaired after the fact.

## 2. What the numbers actually show

| arm | Pareto | floor | R@50 | mR@50 |
|---|---|---|---|---|
| `F_pair_foldfit` | **−0.751** | 0/5 | 71.182 | 30.977 |
| `D_pair` | −1.441 | 1/5 | 71.628 | 29.357 |
| `E_backoff` | −1.564 | 0/5 | 71.580 | 29.508 |
| `null_shuffled` | −1.569 | 1/5 | 71.641 | 29.205 |
| `G_residual` | −1.962 | 1/5 | 71.702 | 28.742 |
| `G_pairmean` | −2.068 | 0/5 | 71.186 | 29.656 |
| `null_pair_matched` | −2.327 | 0/5 | 71.401 | 29.049 |
| **`G_model`** | **−2.456** | **0/5** | 71.452 | 28.838 |
| `B_subject` | −2.579 | 0/5 | 71.458 | 28.704 |
| `C_object` | −3.364 | 0/5 | 71.286 | 28.198 |
| `A_global` | −4.367 | 0/5 | 71.561 | 26.669 |

**The checkpoint's model term is beaten by every pair-conditioned statistic
tested**, and by both of its own nulls:

| contrast | value | reading |
|---|---|---|
| `G_model − F_pair_foldfit` | **−1.706** | a fold-fitted pair frequency table beats the checkpoint |
| `G_model − D_pair` | **−1.015** | the *existing* train-derived pair prior beats it |
| `G_model − E_backoff` | **−0.892** | hierarchical smoothing beats it |
| `G_model − null_shuffled` | **−0.887** | destroying the term's pair identity *improves* it |
| `G_model − null_pair_matched` | −0.129 | image content ≈ 0 — replicates `p26`/`p29` a third time |

So the honest statement is **stronger than "explained"**: on frequent-pair rows
the checkpoint is not merely reproducible by vision-free pair statistics, it is
**worse than them**. The pair-conditioned information it carries is a degraded
copy of a table you can build by counting.

The `−0.887` against the shuffled null is the strangest number and has a
mundane reading: the design matrix already contains explicit prior features, so
the model term's pair content is redundant with them and adds noise on this
subset; shuffling turns it into pure noise the ridge penalty can suppress, while
the real term actively misleads.

## 3. The registered interpretation rule does not fit, and is not applied

The pre-registration says:

> **MOSTLY / NOT EXPLAINED** ⇒ a residual exists. It is pair-conditioned but not
> reproducible from these statistics.

That rule assumed the residual would be **positive** — that NOT EXPLAINED would
mean the model has something the statistics lack. Here the sign is reversed:
there is no residual to characterise, because `G_model` is below every
pair-conditioned arm. Applying the rule as written would assert the opposite of
the measurement, so it is **not applied**, and this paragraph records why.

## 4. Scope, stated as registered

Frequent-pair rows only. The ~33% singleton-pair rows are unaddressable by any
pair-conditioned estimator; that is a property of VG150's pair distribution, not
an estimator detail. Nothing here speaks for them.

## 5. Consistency with the rest of the cycle

This is the same picture `p38`/`p39`/`p40` reach from a completely different
direction:

- vision-free **pair statistics** beat the checkpoint on frequent-pair rows
  (`p32`);
- vision-free **box geometry** beats the checkpoint on within-pair relational
  discrimination, and by more in the tail (`p39`, `p40`).

Two independent trivial baselines, two independent metrics, same conclusion.
