# C7 — the prior-adversarial subset (`runs/p43`)

Run: exit 0. Full `p24` cache. This is the decision-level counterpart of WPRD,
on the field's own accuracy scale, and it is the most directly interpretable
number this programme has produced.

---

## The population

The prior is **train-derived**, so `argmax_p prior[row]` is a pure function of
(subject, object) and the training split. A row is **prior-adversarial** when
its GT differs from that argmax — i.e. following P(p|s,o) is *wrong* there. The
population is therefore defined without touching validation labels.

| tau | prior-adversarial rows | share |
|---|---|---|
| 0.00 | 44,283 | **33.4%** |
| 0.05 | 44,594 | 33.6% |
| 0.10 | 45,017 | 34.0% |

**A third of VG150 validation is prior-adversarial.** That is the real headroom.
The prior scores 0% there by construction.

## The result (tau = 0.0)

| arm | adversarial fixed | prior-correct kept | net rows | R@50 |
|---|---|---|---|---|
| prior only | 0 (0.00%) | 88,273 (100%) | 0 | 66.593 |
| **+ MODEL** | 3,502 (**7.91%**) | 85,533 (96.90%) | **+762** | 67.168 |
| + GEOMETRY (w=1.0) | 3,012 (6.80%) | 86,162 (97.61%) | **+901** | 67.273 |
| **+ MODEL + GEOMETRY (w=1.0)** | 4,662 (**10.53%**) | 85,059 (96.36%) | **+1,448** | 67.685 |

At tau = 0.1 the contrast is sharper still: MODEL nets **+182**, GEOMETRY nets
**+1,680**, MODEL+GEOMETRY nets **+2,081** and fixes **11.50%**.

**Independent cross-check:** the +762 net at tau=0 reproduces `runs/p28`'s
separately computed "model's actual net flips = +762" exactly. Two different
tools, two different code paths, same number.

## What it says

> **On the third of VG150 where the frequency prior gives the wrong answer, the
> 79.9M-parameter checkpoint fixes 7.9% of cases. A linear probe on two bounding
> boxes fixes a comparable share at lower cost. Together they reach 10.5–11.5%.
> Roughly 89% of prior-contradicting relations are recovered by nothing
> available.**

Three things follow, and the third is the actionable one:

1. **The headroom is enormous and almost entirely unclaimed.** 44,283 rows are
   available; the checkpoint claims 3,502 of them and breaks 2,740 others.
2. **The benefit must always be reported with the cost.** The model's +3,502 is
   a net +762 because it destroys 2,740 correct decisions. Any method reporting
   only the fixed count is reporting half a result.
3. **Model and geometry are complementary at the decision level, not just the
   discrimination level.** MODEL+GEOMETRY nets roughly double MODEL alone at
   tau=0 and eleven times at tau=0.1. `p40` says why: geometry wins the
   spatially decidable contrasts, the model wins the functional ones.

## Relation to the rest of the cycle

| finding | source |
|---|---|
| the prior is 85.8% of the model term's variance | `p42` |
| WPRD reads only the other 14.2% | `p42` |
| geometry out-discriminates the model on that 14.2% | `p39` |
| the model still wins the composed metric at tau ≤ 0.05 | `p41` |
| **but on prior-adversarial rows both fix < 8%, and jointly ~11%** | **`p43`** |

`p43` is the one that says the *size of the problem*: not "the model is bad" but
"the mechanism everything is competing over covers a third of the data and
nobody has claimed a tenth of it."

## Limitations

- PredCls with GT pairs, validation split, one checkpoint.
- `w` is swept, not selected out-of-fold; the geometry column is therefore
  optimistic. The MODEL column has no free parameter and is not.
- "Prior-adversarial" inherits VG's incomplete annotation: some rows counted as
  adversarial may have an unannotated relation for which the prior's argmax was
  right. `p35`'s same-instance control shows this class of artifact is
  conservative for WPRD; it has **not** been separately quantified here.
