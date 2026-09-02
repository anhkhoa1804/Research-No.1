# Pre-registration — `p56`, does within-pair discrimination scale with supervision supply?

Status: **PRE-REGISTERED**, committed before the tool that runs it exists and
before any number in it exists.

## The question this decides

`p37` returned **REPRESENTATION-LIMITED**: no probe on `rel_feat` matches the
classifier head already attached to it, and all of them sit below a linear model
on 19 box numbers. That is a measurement of the representation **as it exists**.
It does not say **why** it is that way, and two explanations survive it:

- **H6 representation bottleneck** — the encoder cannot represent within-pair
  relational structure, and more supervision would not change that.
- **H8 supervision scarcity** — the encoder was never *taught* within-pair
  discrimination, because VG150 barely contains the contrast: only **19.0%** of
  raw-name train groups (58.1% VG150-restricted) have >= 2 distinct predicates,
  and tail-tail contrasts are **0.04%** of the supply (`p50`, `p52`).

Nothing run so far distinguishes them. This does.

## The discriminating signal

If H8 is true, then **discrimination should improve for exactly those (s,o)
pairs the training set taught most**. The checkpoint saw the train split; for a
pair with hundreds of training rows spanning several predicates it had ample
within-pair signal, and for a singleton pair it had none. So under H8 the
checkpoint's WPRD should **rise with that pair's training supervision**.

Under H6 it should not: if the representation cannot carry the distinction, more
examples of a pair do not help, and WPRD is flat in training support.

**Geometry is the control that makes this interpretable.** The geometry probe is
fitted **globally** over 19 shared features, so it does not accumulate per-pair
capacity. If the checkpoint's WPRD tracks per-pair supervision and geometry's
does not, the difference is attributable to supervision rather than to the
population being easier at high support.

## Design

For every (subject, object) pair appearing in `datasets_vg150_clean/train.jsonl`
compute its **training supervision supply**:

```
n_train_rows            rows with this (s,o)
n_train_predicates      distinct predicates for this (s,o)
n_train_contrasts       sum over predicate pairs of n_a * n_b   <- the WPRD-relevant quantity
```

Attach that to every validation WPRD cell via its group key, and report, per
support bin and per arm: macro WPRD, cell count, and the bucket split.

Arms: evaluated text head, discarded classifier head, geometry linear
(train-fitted), geometry MLP (train-fitted), prior control, random control.

Support bins, fixed here: `n_train_contrasts` in
`{0}, [1,10), [10,100), [100,1e3), [1e3,1e4), [1e4,inf)`.

### Part C — what supervision actually buys, measured directly

Independently of the checkpoint: for each (s,o) pair with sufficient training
rows, fit a **per-pair** predicate discriminator on **that pair's train rows
only** (geometry features, the channel known to carry signal) and evaluate WPRD
on **that pair's validation rows**. This is an empirical answer to *"what is the
achievable WPRD given the within-pair supervision that pair actually has?"* and
gives the supervision-scaling curve with the representation held constant and
known-adequate.

## Validity gates — if any fails, no number in the run is reportable

| gate | requirement |
|---|---|
| **W1** | prior control reads **exactly 0.5000** in **every** support bin |
| **W2** | random control within [0.49, 0.51] overall |
| **W3** | support bins partition the cells exactly once, no cell dropped or double-counted |
| **W4** | group keys join train->validation on the same raw-name convention `Groups` uses |

## Criteria — fixed here, before any number exists

### PRIMARY — H8 on the deployed checkpoint

`rho_model = Spearman(log1p(n_train_contrasts), cell WPRD)` for the **evaluated
text head**, over all decidable validation cells, permutation p with 2000 draws.

| verdict | condition |
|---|---|
| **H8-SUPPORTED** | `rho_model >= +0.15`, `p < 0.05`, **and** the top support bin's model WPRD `>= 0.5961` (the geometry reference) |
| **H8-WEAK** | `rho_model >= +0.05` and `p < 0.05`, but the top bin stays below 0.5961 |
| **H8-REFUTED** | `rho_model < +0.05`, or `p >= 0.05` |

### SECONDARY — is supervision a lever at all?

From Part C: `rho_perpair = Spearman(log1p(n_train_rows_of_pair), per-pair WPRD)`.

**LEVER** if `rho_perpair >= +0.15` with `p < 0.05`; **NOT A LEVER** otherwise.
A NOT-A-LEVER result would mean that even on a channel that demonstrably carries
relational signal, adding within-pair supervision for a pair does not improve
within-pair discrimination for it — which would weaken H8 substantially and
independently of the checkpoint.

### TERTIARY — the differential

`rho_model - rho_geometry`. Reported with a paired permutation interval. A
positive differential is the signature H8 predicts and geometry cannot produce.

## What this cannot settle

The checkpoint is **frozen and already trained**. This measures a *correlation*
between how much a pair was taught and how well that pair is discriminated. Pair
support is confounded with pair frequency, predicate entropy and object
commonness, and no causal claim is registered. A positive result licenses
"consistent with H8 and worth an intervention"; it does **not** license
"supervision causes discrimination". The intervention that would is a retrain,
which is not affordable here and is not claimed.
