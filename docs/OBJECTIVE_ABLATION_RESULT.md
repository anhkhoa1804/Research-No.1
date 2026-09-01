# H2 — REFUTED on the box channel, in the opposite direction (`runs/p48`)

Criterion pre-registered inline before the run: SUPPORTED ≥ +0.02, WEAK ≥ +0.005.

| arm | WPRD | weighted | 95% CI |
|---|---|---|---|
| **A — plain cross-entropy** | **0.6163** | 0.5560 | [0.6108, 0.6218] |
| **B — within-group contrastive** | **0.6020** | 0.5496 | [0.5970, 0.6071] |
| prior (control) | 0.5000 | 0.5000 | [0.5000, 0.5000] |

Paired: **B − A = −0.0143** [−0.0198, −0.0094], **excludes 0**.
**Verdict: REFUTED.**

`A` reproduces `p46`'s independently-run MLP (0.6163 vs 0.6153), which is the
consistency check that this comparison is on the same footing.

## What was held fixed, and what varied

Everything except the loss: the same 19 box-geometry features, the same 2-layer
256-unit MLP, the same 1,046,427 train rows, the same WPRD on the same
validation cells.

`B`'s loss is the AUC surrogate WPRD itself measures — for train rows i, j in
one (s,o) group with `y_i ≠ y_j`,

```
margin = (f_i[y_i] − f_i[y_j]) − (f_j[y_i] − f_j[y_j])
loss   = softplus(−margin)
```

The prior cancels in that double difference exactly as it does in WPRD, so the
objective **cannot** be satisfied by learning P(p|s,o). It trained properly —
the contrastive loss fell 0.5748 → 0.5029 over 30 epochs.

## The result

**Directly optimising the metric produced a worse score on it than plain
cross-entropy did.** That is not a bug; it is the finding.

The most likely mechanism is in the run's own output: only **40,470 of 212,981**
train groups (**19%**) have ≥2 distinct predicates and are usable for a
within-group contrastive pair. `B` therefore trains on a small, structurally
biased slice — frequent, ambiguous pairs — while `A` uses every row. The
supervision that teaches within-pair discrimination is **scarce in VG150 by
construction**, and an objective restricted to it trades away the coverage that
plain CE gets for free.

## Consequence for the successor programme

**H2 (objective bottleneck) is REFUTED on the box channel**, and that matters
beyond geometry: it says plain CE is already near-optimal at *extracting*
within-pair discrimination from a given feature set. If extraction is not the
limitation, then a low WPRD is more likely a property of the **features** than
of the loss — which points at **H4 (representation)**.

Three of four successor hypotheses are now weakened or refuted before any
architecture was written:

| hypothesis | status |
|---|---|
| H1 readout | **PENDING** — `p37` |
| H2 objective | **REFUTED on the box channel** (must be re-run on `rel_feat`) |
| H3 prior/evidence entanglement | **WEAKENED** — `p45` returned +1.07 vs a registered +2.0 |
| H4 representation | **PENDING** — `p37`, and now the leading candidate |

## Limitations — this is one channel, not a general law

- **Box features, not `rel_feat`.** The identical comparison must be re-run on
  `p36`'s cache before H2 is called refuted for PURE. The tool is written to
  take any feature matrix, so that re-run is a flag change.
- One loss form. A different within-pair objective (e.g. mixing CE and
  contrastive, or upweighting rather than restricting) is not excluded — but it
  now carries the burden of beating a baseline that a purpose-built objective
  failed to beat.
- The 19% coverage figure suggests the honest reframing: this may be a
  **supervision-scarcity** result rather than an objective result. VG150 simply
  does not contain many groups where within-pair discrimination is teachable.
  That is a statement about the dataset, and it is the strongest argument yet
  that a *dataset* contribution might eventually be warranted.
