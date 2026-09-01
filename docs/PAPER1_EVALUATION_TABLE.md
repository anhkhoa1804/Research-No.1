# Paper 1 — the evaluation table, and the rank inversion in it

`runs/p47`. The artifact the diagnostic paper is built around. Full `p24` cache,
tau=0, identical cells for every arm.

---

## The claim being tested

Standard SGG evaluation conflates four things:

1. the object-pair prior P(p | s,o)
2. calibration (the tau / class-reweighting axis)
3. candidate selection
4. **relational discrimination**

`R@50`, `mR@50` and the Pareto gap mix all four. **WPRD isolates (4)**: it
conditions on the (s,o) group, which makes (1) *exactly* non-informative and
cancels (2) in the same double difference.

## The table

| arm | R@50 | mR@50 | Pareto | **WPRD** | 95% CI | head–head | body–body | tail–tail |
|---|---|---|---|---|---|---|---|---|
| PURE text head (**evaluated**) | 67.168 | **23.204** | **+0.901** | 0.5542 | [0.5495, 0.5592] | 0.5612 | 0.5746 | 0.5126 |
| PURE classifier head (discarded) | 67.215 | 22.362 | +0.059 | 0.5728 | [0.5681, 0.5779] | 0.5701 | 0.6167 | 0.5677 |
| geometry linear (train-fitted) | **67.273** | 18.576 | **−3.728** | **0.5934** | [0.5895, 0.5994] | 0.5919 | **0.6404** | **0.6249** |
| pair prior only | 66.593 | 22.304 | 0.000 | **0.5000** | [0.5000, 0.5000] | 0.5000 | 0.5000 | 0.5000 |
| random null | 64.886 | 21.817 | −5.247 | 0.5046 | [0.4993, 0.5094] | 0.5067 | 0.4891 | 0.5715 |

Two control rows validate the instrument: the pair prior reads **exactly
0.5000** and the random null ~0.5.

## The rank inversion

| arm | Pareto | rank | WPRD | rank |
|---|---|---|---|---|
| PURE text head | +0.901 | **4** (best) | 0.5542 | **2** |
| PURE classifier head | +0.059 | 3 | 0.5728 | 3 |
| geometry linear | −3.728 | **1** (worst) | 0.5934 | **4** (best) |
| pair prior only | 0.000 | 2 | 0.5000 | 1 (worst) |

```
Spearman(R@50,   WPRD) = +1.000
Spearman(mR@50,  WPRD) = -0.400
Spearman(Pareto, WPRD) = -0.400
```

**The arm with the best relational discrimination has the worst Pareto gap and
the worst mR@50 — and the best R@50.**

## The caveat, stated before the interpretation

**n = 4 arms.** A Spearman coefficient on four points is very low-powered.
`ρ = −0.400` is *not* statistically distinguishable from zero, and even
`ρ = +1.000` on n=4 is only p≈0.042 one-tailed. **This is an observation about
four arms, not an established correlation.** It is exactly the reason the
cross-model study matters, and it must not be quoted as a general result about
SGG metrics until more models populate the table.

## What it suggests, mechanistically

The pattern is not arbitrary, which is why it is worth pursuing:

- **`mR@K` rewards moving mass to rare predicates.** Calibration does that with
  no image access. Geometry is strongly *head-biased* in its calibration — it
  gets the **best R@50 (67.273)** and the **worst mR@50 (18.576)** — while
  having the **best** within-pair discrimination. Discrimination and calibration
  are separable, and `mR@K` reads the calibration axis.
- **`R@50` tracks WPRD perfectly here** (ρ = +1.000). If that survives more
  arms, the uncomfortable implication is that the metric the field introduced to
  fix long-tail bias is the one *least* aligned with relational grounding, while
  plain recall is the one most aligned.

## The tail column is the sharpest single row

For the head the checkpoint actually runs, `tail–tail` WPRD is **0.5126** — and
the **random null reads 0.5715 in the same column**, because tail–tail rests on
only **61 cells** and is dominated by noise. That is not evidence the model is
worse than random; it is evidence **the tail column is underpowered and must not
be read as a point estimate**. Geometry's 0.6249 there is likewise uncertain.

Fixing this needs more tail cells: the test split, more checkpoints, or
Haystack's rare-predicate annotations.

## Interface for the cross-model study

`tools/sgg_evaluation_table.py --extra name=path.pt` accepts any model's
per-pair predicate logits in the cache's row order and vocabulary:

```
{"model_term": Tensor(132556, 50)}                  # GT-row aligned
{"per_image_logits": [Tensor(n_pairs_i, 50), ...]}  # cache order per image
```

So the study is **not blocked on any one codebase** — only on obtaining logits
from one more model. That is now the single highest-value action for Paper 1.
