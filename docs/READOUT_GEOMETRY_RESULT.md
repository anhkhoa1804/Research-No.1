# `runs/p51` — the readout-geometry hypothesis is REFUTED

I proposed it in `docs/SUPERVISION_STRUCTURE_RESULT.md` §3 and promoted it to a
first-class arm of `p37`. It does not survive its own test. Recorded here so the
promotion is reversed explicitly rather than quietly dropped.

## The hypothesis

The text head is a cosine against **fixed, never-trained** CLIP predicate
embeddings, so its ability to separate `a` from `b` should be bounded by the
angle between `pred_emb[a]` and `pred_emb[b]`. Predicted: near-collinear
predicate pairs get low WPRD **for the text head specifically**, and the learned
classifier head should escape that bound.

Tested via two proxies computable from the `p24` cache (196 predicate pairs with
≥20 cells): the **column correlation** of the stored logits (a proxy for the
embedding Gram matrix) and the **contrast variance** `var(logits[:,a] −
logits[:,b])`, the discriminative *budget* the readout has for that pair.

## The result — backwards

| | Spearman(col corr, WPRD) | Spearman(log contrast var, WPRD) |
|---|---|---|
| **text head** | **−0.073, p = 0.294 — ns** | +0.142, p = 0.039 |
| **classifier head** | **−0.291, p = 0.0005** | **+0.281, p = 0.0005** |

The geometry association is **stronger for the learned head**, which is the
opposite of the prediction. For the text head — the one the hypothesis was about
— column collinearity is **uncorrelated** with WPRD.

Two further facts kill it outright:

- On the 196 shared pairs the classifier head has **less** discriminative budget
  (log₁₀ contrast variance **−0.473**, roughly 3× smaller) and **better** WPRD.
  Budget is not the binding constraint.
- `above` vs `under` has one of the text head's **smallest** budgets (0.1345) and
  its **highest** WPRD (**0.8002**). A readout with almost no contrast variance
  on that pair still separates it nearly perfectly.

**Conclusion: the text head's weakness is not explained by its embedding
geometry.** The hypothesis is withdrawn and removed from `p37`'s planned arms;
`p37` will still report the `pred_emb` Gram matrix as a descriptive, non-criterion
quantity, since `p36` stores it and it costs nothing.

## A confound this surfaced, checked and dismissed

The lowest-budget pairs are dominated by **semantic near-duplicates** where the
ground-truth label is essentially arbitrary: `near`/`next to`, `wearing`/`wears`
(both collapsed by the cache's own 50→48 alias map), and `laying on`/`lying on`
(not in that map). WPRD on those pairs *should* be ~0.5.

Measured:

| | all cells | alias-pair cells | non-alias cells |
|---|---|---|---|
| text head | 0.5542 (20,016) | **0.5117** (621, 3.10%) | 0.5556 (19,395) |
| classifier head | 0.5728 | **0.5112** (621) | 0.5748 |

**The confound is real, small, and self-validating.** WPRD reads ~0.511 on pairs
whose labels are arbitrary — which is the correct behaviour and an independent
check that the metric is not manufacturing signal. Excluding them moves the
headline by **+0.0014 / +0.0020**. **No conclusion in this programme changes.**

## What this leaves standing

The `p50` dissociation is unexplained: the evaluated text head's WPRD tracks
within-pair supervision supply (Spearman +0.657) while the discarded classifier
head's does not (−0.086), and neither embedding geometry nor discriminative
budget accounts for it. That remains an open mechanism question, and it is now
more clearly a question about **what the two readouts learned**, not about their
structural form.

`p37` is unaffected — its registered arms and thresholds are unchanged.
