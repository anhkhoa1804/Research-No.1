# `runs/p56` — H8 (supervision scarcity) is REFUTED as the explanation

Run: exit 0, 338 s. Pre-registered in
`docs/SUPERVISION_SCALING_PREREGISTRATION.md` (commit `25b1450`), committed
before the tool existed. No threshold moved.

This was the leading hypothesis entering the cycle. It does not survive.

## Validity gates — all PASS

| gate | requirement | observed |
|---|---|---|
| **W1** | prior control **exactly 0.5000 in every support bin** | `[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]` |
| **W2** | random control at chance | 0.5047 |
| **W3** | bins partition the cells exactly once | 20,016 cells |
| **W4** | train->validation group-key join | 26,852 / 45,607 groups (58.9%) |

W1 is the load-bearing one: the prior-cancellation property holds **within every
stratum**, so no bin's number can be prior contamination.

## WPRD by the training supervision its (s,o) pair received

`n_train_contrasts` = sum over predicate pairs of `n_a * n_b` for that (s,o) in
`train.jsonl` — the quantity that actually supplies within-pair learning signal.

| arm | 0 | 1-9 | 10-99 | 100-999 | 1k-10k | >=10k | all |
|---|---|---|---|---|---|---|---|
| text head (**evaluated**) | 0.5412 | 0.5582 | 0.5481 | 0.5637 | 0.5461 | 0.5613 | 0.5542 |
| classifier head (discarded) | 0.5478 | 0.5524 | 0.5662 | 0.5705 | 0.5801 | **0.5817** | 0.5728 |
| geometry linear | 0.5516 | 0.5978 | 0.6042 | 0.5995 | 0.5962 | 0.6027 | 0.5961 |
| **geometry MLP** | 0.5693 | 0.6125 | 0.6009 | 0.6097 | 0.6244 | **0.6302** | 0.6149 |
| prior (control) | **0.5000** | **0.5000** | **0.5000** | **0.5000** | **0.5000** | **0.5000** | 0.5000 |
| random (control) | 0.5183 | 0.5162 | 0.4844 | 0.5068 | 0.5104 | 0.5002 | 0.5047 |
| n cells | 1,604 | 965 | 2,479 | 4,261 | 5,567 | 5,140 | 20,016 |

## The registered statistic

`Spearman(log1p(n_train_contrasts), cell WPRD)`, permutation p, 2000 draws,
n = 20,016 cells:

| arm | rho | p | |
|---|---|---|---|
| **text head (evaluated)** | **−0.0085** | 0.2169 | **ns** |
| classifier head | +0.0053 | 0.4438 | ns |
| geometry linear | −0.0102 | 0.1404 | ns |
| geometry MLP | +0.0111 | 0.1119 | ns |
| prior | +0.0000 | 1.0000 | ns |
| random | −0.0017 | 0.8091 | ns |

**No arm shows a significant cell-level relationship between how much a pair was
taught and how well it is discriminated.**

## Verdict

```
PRIMARY   rho_model -0.0085 (p 0.2169), top bin 0.5613 vs G 0.5961  -> H8-REFUTED
SECONDARY rho_perpair +0.0108 (p 0.1869)                            -> NOT A LEVER
TERTIARY  rho_model - rho_geometry = +0.0016
```

### Part C — what per-pair supervision actually buys

Fitting a discriminator on **that pair's own training rows** (geometry — a
channel known to carry relational signal), 3,835 pairs, 15,069 cells, mean
per-pair WPRD 0.5693:

`rho(log1p(n_train_rows of pair), per-pair WPRD) = +0.0108, p = 0.1869, ns`

Even with the representation held constant and known-adequate, giving a pair
more of its own within-pair supervision does **not** improve within-pair
discrimination for it.

## The decisive argument, from the exploratory panel

The bin means do drift upward, and the registered cell-level Spearman does not
see it (cell AUCs come from very few rows and are extremely noisy). A
cell-weighted trend across bin means was added **after** the pilot and is
labelled EXPLORATORY, NOT A CRITERION — it decides no verdict:

| arm | slope (WPRD per bin) | top minus bottom |
|---|---|---|
| text head (evaluated) | +0.00219 | +0.0202 |
| classifier head | +0.00689 | +0.0339 |
| geometry linear | +0.00580 | +0.0511 |
| **geometry MLP** | **+0.01051** | **+0.0609** |
| prior | +0.00000 | +0.0000 |
| random | −0.00116 | −0.0181 |

**This kills H8 rather than rescuing it.** Geometry is fitted **globally** over
19 shared features and accumulates **no per-pair capacity**, so it cannot
benefit from a pair's supervision at all — yet it shows the **largest** trend
(+0.0609 and +0.0511) while the checkpoint's heads show the **smallest**
(+0.0202, +0.0339).

The only consistent reading: **the upward drift is the population getting
easier at high pair support, not supervision being converted into
discrimination.** And the checkpoint converts that easiness *less* well than two
rectangles do.

## A second finding, unregistered and worth stating

**41.1% of validation groups appear in no training row at all.** On those rows
the checkpoint's supervision for the pair is exactly zero — and both heads still
read **above chance** (text 0.5412, classifier 0.5478, both clear of the
random control's 0.5183 and the prior's exact 0.5000).

Whatever relational signal the checkpoint has is therefore **not** pair-specific
memorisation. It generalises to pairs never seen. That is a point *in the
checkpoint's favour* and it is the first one this programme has produced.

## What this cannot settle

Registered in advance: the checkpoint is **frozen and already trained**. This is
a **correlation** between supervision received and discrimination achieved. Pair
support is confounded with pair frequency, predicate entropy and object
commonness. **No causal claim is made or licensed.** A REFUTED verdict here
removes the *observational* support for H8; it does not prove that a retrain
under balanced within-pair supervision would fail. That intervention is a
retrain and was explicitly out of scope.

Taken with `runs/p55` — where pair-balanced sampling and a within-pair
contrastive objective both *lost* to plain CE at a matched budget — the
supervision route has now failed from two independent directions.
