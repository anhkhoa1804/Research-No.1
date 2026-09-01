# Pair-prior distillation (`runs/p27`) — result, and why it is INCONCLUSIVE

Status: **INCONCLUSIVE, cause identified and measured.** Not a refutation, not a
confirmation. The arms that were supposed to carry pair information could only
carry it on part of the data, and the audit below measures how much.

Run: `runs/p27_pair_prior_distillation`, exit 0, 1331 s, 3,000-image cache
(`runs/p10`), 38,053 GT rows, tau=0, k=5, 5 fold partitions.
Audit: `runs/p30_pair_fold_coverage`, `tools/audit_pair_fold_coverage.py`.

## 1. What p27 reported

Mean Pareto gap over 5 partitions, prior-only baseline R@50 66.802 / mR 21.976:

| arm | what it is | Pareto | floor |
|---|---|---|---|
| `A_global` | P(p) | −2.697 | 0/5 |
| `B_subject` | P(p \| s) | −1.905 | 0/5 |
| `C_object` | P(p \| o) | −2.551 | 0/5 |
| `D_pair` | P(p \| s,o), train-derived | −2.581 | 0/5 |
| `E_backoff` | hierarchical smoothing | −1.275 | 2/5 |
| `F_pair_foldfit` | P(p \| s,o) re-fitted on this cache's training folds | −0.294 | 0/5 |
| **`G_model`** | **the C′ model term** | **+2.112** | **4/5** |
| `G_pairmean` | model term → its training-fold (s,o) group mean | −0.510 | 0/5 |
| `G_residual` | model term − that group mean | +0.024 | 3/5 |
| `null_shuffled` | identity destroyed | −2.590 | 0/5 |
| `null_pair_matched` | image destroyed, pair kept | +2.000 | 4/5 |

The tempting reading: *no pair-conditioned statistic comes near `G_model`, so
the checkpoint carries something beyond a pair prior.* That reading does not
survive the audit.

## 2. Why the ladder cannot answer the question as posed

`Distill._fold_pair_mean` is leak-free by construction, and says so:

> Groups with no training row fall back to the training global mean, so no
> held-out row ever contributes to the statistic used to score it.

That is the correct thing to do, and it has a consequence the p27 table does not
show. A (subject, object) group that occurs **once** in the cache can never
appear in the training rows of the fold that holds it out. Such a row is scored
by a "pair-conditioned" arm that gives it the **global** mean — no pair
information whatsoever — while still being counted as evidence about pair
conditioning.

Measured with the same `Distill` object, the same fold rule and the same pair
ids (`tools/audit_pair_fold_coverage.py`):

| cache | rows with NO pair information out-of-fold |
|---|---|
| `p10`, 3,000 images, 38,053 rows | **45.4%** (mean over 5 partitions) |
| `p24`, 10,401 images, 132,556 rows | **33.2%** (mean over 5 partitions) |

By bucket on the 3k cache: head 44.9%, body 52.8%, tail 47.9% — the dilution is
*worst in the body*, and is substantial everywhere. It is not a tail artifact.

So `D_pair`, `E_backoff`, `F_pair_foldfit` and `G_pairmean` were asked to
reproduce `G_model` while being blindfolded on ~45% of the rows they were
scored on. Their negative Pareto gaps are **not** clean evidence that pair
statistics are insufficient.

## 3. The decomposition does not decompose

The stronger claim in the p27 design was mechanical: `G_pairmean` + `G_residual`
should split the model term into its pair-conditioned and within-group parts.
On the fallback rows that split degenerates. If the group mean is the global
mean, then

    residual = md − global_mean

is just a recentred copy of the whole model term. Measured, comparing
`G_residual` against the REAL model term row-wise (class-centred cosine):

| rows | 3k cache | full cache |
|---|---|---|
| fallback (no pair info) | **+0.938** | **+0.936** |
| estimable (pair info present) | +0.419 | +0.459 |

On roughly a third to a half of all rows, `G_residual` **is** the model term to a
cosine of 0.94. It is therefore not a within-group residual, and `G_residual`'s
+0.024 cannot be read as "the within-group component contributes nothing". Nor
can `G_model − G_pairmean = +2.622` be read as "the within-group part is worth
+2.6": both quantities are contaminated by the same fallback rows.

This also explains the otherwise odd non-additivity — `G_pairmean` −0.510 and
`G_residual` +0.024 separately, but `G_model` +2.112 jointly — without needing
any hypothesis about extra-pair content.

## 4. Blast radius

Deliberately bounded, and checked rather than assumed:

- **`runs/p27` — affected.** Every A–F and G_pairmean/G_residual conclusion is
  withdrawn. No claim from it enters `RESEARCH_STATE.md`.
- **`runs/p26` — NOT affected.** Its `pair_matched_null` permutes the model term
  *within* (s,o) groups across the whole analysis set. It never estimates an
  out-of-fold group statistic, so the fallback path is not on its code path. Its
  known conservatism (31.7% singleton rows keep their real term) was
  pre-registered and reported, and is a *different* limitation.
- **`runs/p24`, `p28`, `p29` — NOT affected.** They do not use `_fold_pair_mean`.
- **`p25`, `p22`, `p21`, `p17` — NOT affected.** Same reason.

No previously reported number changes. What changes is that p27 is not evidence
for anything, in either direction.

## 5. What the audit does NOT license

It does not license "pair statistics *do* explain C′". That is equally
unmeasured. The honest state is that the question is open and the instrument was
too blunt to answer it.

## 6. The corrected experiment

Restrict the comparison to rows whose (subject, object) group **is** estimable
out-of-fold, where `G_pairmean`/`G_residual` genuinely decompose, and report the
estimable subset as its own population with its own prior-only baseline and its
own tau frontier. On the full `p24` cache that subset is ~67% of 132,556 rows
(~88.6k rows), which is larger than the entire 3k screening set — so the
corrected analysis is better powered than the original, and needs no GPU.

Two things must be reported alongside it, or the restriction becomes a new
shortcut:

1. The estimable subset is **not** a random subsample — it over-represents
   frequent pairs by construction. Its prior-only baseline must be recomputed on
   the subset, never inherited from the full split.
2. Whatever holds on the estimable subset is a statement about *frequent-pair
   rows only*. The ~33% singleton-pair rows remain unaddressed by any
   pair-conditioned estimator, and that is a property of VG150's pair
   distribution, not a fixable estimator detail.

