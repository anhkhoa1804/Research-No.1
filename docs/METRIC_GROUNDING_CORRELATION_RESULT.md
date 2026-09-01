# `runs/p49` — mR@50 is anti-correlated with relational grounding

Twelve scoring functions, all composed and evaluated identically on the full
`p24` cache at tau=0. Addresses the n=4 power objection to `p47`.

| arm | R@50 | mR@50 | Pareto | **WPRD** |
|---|---|---|---|---|
| pair prior only | 66.593 | 22.304 | 0.000 | **0.5000** |
| random null | 64.883 | 21.837 | −5.228 | 0.5041 |
| PURE text (α=0) | 67.168 | **23.204** | +0.901 | 0.5542 |
| PURE ens α=0.25 | 67.291 | 23.203 | +0.899 | 0.5596 |
| PURE ens α=0.5 | 67.337 | 22.866 | +0.562 | 0.5656 |
| PURE ens α=0.75 | 67.297 | 22.675 | +0.371 | 0.5706 |
| PURE classifier (α=1) | 67.215 | 22.362 | +0.059 | 0.5728 |
| geometry linear | 67.273 | 18.576 | −3.728 | 0.5934 |
| geometry MLP | 67.335 | 20.345 | −1.959 | 0.6123 |
| text + geometry | 67.685 | 18.952 | −3.352 | 0.5925 |
| classifier + geometry | 67.590 | 18.358 | −3.945 | 0.6026 |
| classifier + geoMLP | **67.685** | 20.255 | −2.049 | **0.6163** |

```
Spearman(R@50,   WPRD) = +0.741   p = 0.0080   significant
Spearman(mR@50,  WPRD) = -0.650   p = 0.0205   significant
Spearman(Pareto, WPRD) = -0.371   p = 0.2359   NOT significant
```
(permutation p, 2000 draws)

## The finding

**Across this family, the metric the SGG field adopted specifically to address
long-tail bias — mean Recall — moves in the *opposite* direction to prior-free
relational grounding.** Plain R@50 moves *with* it.

The mechanism is visible in the table and is not mysterious. The arms with the
best WPRD are geometry-based (0.593–0.616) and have the **worst** mR@50
(18.4–20.3), because geometry is head-biased in its calibration: it knows which
relation holds but concentrates its mass on frequent predicates. The arms with
the best mR@50 are PURE's (22.4–23.2) and have the **worst** WPRD among the
non-null arms. `mR@K` reads the calibration axis; WPRD reads the discrimination
axis; `p41`/`p42` already showed these are separable.

Note the Pareto gap is **not** significant. The claim is specifically about
`R@50` and `mR@50`, not about the Pareto construction this project uses
internally.

## The limitation that matters, stated plainly

**The twelve arms are not independent.** They cluster into roughly three
families:

1. prior / random (2)
2. the PURE ensemble-α sweep (5) — a smooth one-parameter family
3. geometry-containing arms (5)

Within a family the variation is smooth, so the *effective* sample size is much
closer to 3 than to 12, and the correlation is driven by **between-family
ordering**: the PURE family sits at WPRD 0.554–0.573 with mR 22.4–23.2, the
geometry family at WPRD 0.593–0.616 with mR 18.4–20.3. The two families are
cleanly separated on both axes, in opposite directions.

So the permutation p-values are optimistic. What `p49` genuinely establishes is
that the inversion seen in `p47` is **not an accident of which four arms were in
that table** — it holds across a deliberately widened family and has a coherent
mechanism. It does **not** establish a general law about SGG metrics.

**These are scoring functions built from one cache, not published SGG models.**
The cross-model study remains the gate on any general claim, and
`tools/sgg_evaluation_table.py --extra` exists so that study needs only logits.

## Why this is the most consequential result of the cycle

If it survives published checkpoints, it says something uncomfortable and
useful: **a decade of SGG progress measured by mR@K may have been selecting
against relational grounding**, because mR@K is improved most cheaply by
recalibrating toward rare predicates — which requires no image — while genuine
within-pair discrimination *costs* mR@K unless it is accompanied by matching
calibration.

That is a claim about **evaluation**, not about any one model, and it is the
form of result that would change how the field reads its own numbers. It is
also exactly the claim that must not be made until more models are in the table.
