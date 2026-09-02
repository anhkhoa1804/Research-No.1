# Pre-registration — `p55`, the objective ablation ON `rel_feat`

Status: **PRE-REGISTERED**, committed before the tool that runs it exists and
before any number in it exists.

## Why this run is owed

`runs/p48` compared plain cross-entropy against a within-group contrastive
objective and returned **REFUTED** (contrastive 0.6020 vs CE 0.6163, paired
−0.0143 [−0.0198, −0.0094]). That run has **two** limitations, both recorded in
`docs/OBJECTIVE_ABLATION_RESULT.md` and `docs/SUCCESSOR_HYPOTHESES.md` at the
time:

1. **Wrong channel.** It ran on the 19 box-geometry features, not on `rel_feat`.
   `p48`'s own docstring says the comparison "transfers to rel_feat once
   runs/p36 lands". `p36` landed. This is that transfer.
2. **A supervision-supply confound.** The contrastive arm can only train on
   groups with >= 2 distinct predicates — **19.0%** of train groups (`p50`) —
   while the CE arm used **every** row. B saw ~235k pairs against A's 1.05M
   rows. A loss comparison in which one arm sees 5x less usable data is not a
   loss comparison.

This run fixes both. **`p48`'s verdict is not restated, reused, or extended by
this registration; it stands as a result about the box channel only.**

## The design

`rel_feat` exists only for the **validation** split (`p36`, 132,556 x 768 fp16).
There is no train-split `rel_feat` cache and generating one is ~30 GPU-hours, so
every arm here is **cross-fitted over 5 validation folds split by IMAGE**, the
identical fold construction `p37` used (`candidate_scorer_probe.fold_of_image`,
salt 0). Each row's score comes from a model that never saw that row; WPRD is
then computed once over all out-of-fold scores.

Held fixed across every fitted arm: the same `rel_feat` features, the same folds,
the same capacity (2-layer MLP, hidden 256), the same optimiser, the same epochs,
the same seed. **Only the objective and the sampling change.**

### Arms

| arm | features | objective | training rows |
|---|---|---|---|
| `A_ce_all` | `rel_feat` | cross-entropy | all training-fold rows |
| `B_contr_decidable` | `rel_feat` | within-group contrastive | decidable groups only |
| **`C_ce_matched`** | `rel_feat` | cross-entropy | **only rows in decidable groups** |
| `D_ce_pairbal` | `rel_feat` | cross-entropy, pair-balanced sampling | all |
| `E_contr_pairbal` | `rel_feat` | contrastive, pair-balanced group sampling | decidable groups |
| `F_ce_groupcentred` | **group-centred** `rel_feat` | cross-entropy | all |
| `N_shuffled` | `rel_feat` | cross-entropy on shuffled labels | all |
| `P_prior` | — | the prior itself | — |

`C_ce_matched` is the confound fix: it is `A` restricted to exactly the
population `B` is allowed to learn from, so `B - C` isolates **the objective**
with the information budget matched.

`F_ce_groupcentred` tests whether `p37`'s R5 finding — that removing the (s,o)
group mean *before* the readout raises WPRD from 0.5601 to 0.5807 — survives a
trained nonlinear readout, or was an artifact of the linear probe.

The contrastive loss is `p48`'s, unchanged. For two training rows i, j in one
(s,o) group with `y_i != y_j`:

```
margin = (f_i[y_i] - f_i[y_j]) - (f_j[y_i] - f_j[y_j])
loss   = softplus(-margin)
```

The prior cancels in that double difference exactly as it does in WPRD, so the
objective cannot be satisfied by learning P(p|s,o).

## Validity gates — if any fails, no number in the run is reportable

| gate | requirement |
|---|---|
| **V1** | `P_prior` WPRD reads **exactly 0.5000** (deviation < 1e-6) |
| **V2** | `N_shuffled` WPRD within [0.49, 0.51] |
| **V3** | `rel_feat` is (132556, 768), finite, `missing_rel_feat == 0` |
| **V4** | every fitted arm produces an out-of-fold score for every row |

## Criteria — fixed here, before any number exists

### PRIMARY — the confound-free objective test

`Δ_obj = B_contr_decidable − C_ce_matched`

| verdict | condition |
|---|---|
| **OBJECTIVE-LIMITED** | `Δ_obj >= +0.020` |
| **WEAK** | `+0.005 <= Δ_obj < +0.020` |
| **REFUTED** | `Δ_obj < +0.005` |

The +0.020 threshold is inherited from `p48` unchanged (the classifier-vs-text
head gap of 0.0186 rounded up). **It is not renegotiated here.**

### SECONDARY — the supervision-supply test (H8)

`Δ_sup = C_ce_matched − A_ce_all` and `Δ_bal = D_ce_pairbal − A_ce_all`

**SUPERVISION-SENSITIVE** if `max(|Δ_sup|, |Δ_bal|) >= 0.020`; otherwise
**SUPERVISION-INSENSITIVE**. A supervision-insensitive result means reweighting
the supervision that exists cannot move within-pair discrimination on these
features, which weakens H8 as an explanation *at fixed representation* — it does
**not** speak to what a differently-trained encoder could do.

### TERTIARY — the ceiling test (H6)

`G = 0.5961`, the train-fitted geometry linear probe from `p37`/`p39`.

**BEATS GEOMETRY** if any fitted arm's macro WPRD `>= G + 0.02`;
**REACHES GEOMETRY** if within 0.02; **BELOW GEOMETRY** otherwise. Any arm
clearly exceeding G would falsify the representation-limited reading of `p37`
and is the single outcome that would most change the research map.

## Reported alongside (not criteria)

training rows per arm, contrast count per arm, pair-support distribution,
macro and weighted WPRD, 95% CI by cluster bootstrap over cells, and
head/body/tail buckets. R@50 / mR@50 are **not** reported for these arms: they
are cross-fitted validation probes, not deployable scorers, and composing them
against the prior would invite exactly the discrimination/calibration conflation
this programme exists to separate.

## What this run cannot settle

The encoder is **frozen**. Every arm changes only the readout's objective on a
fixed representation. A REFUTED primary therefore means *"the objective is not
the limit given these features"* — it does **not** mean an encoder trained from
scratch under a within-pair objective would fail. That experiment is not
affordable here and is not claimed.
