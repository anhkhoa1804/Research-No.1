# Appearance probe — findings and re-scope recommendation

Terminal experiment of the candidate-reranking investigation. Reproduce with
`python tools/appearance_probe.py --extract --score`.

**Verdict: APPEARANCE SIGNAL WEAK — RE-SCOPE.** The signal is real and lands
exactly where theory predicts, but converts **0.6 %** of the available
headroom.

---

## 1. What the investigation established, in order

| Phase | Question | Answer |
|---|---|---|
| Prior control | How much does the model add over a lookup table? | **+0.50 R@50 / +0.34 mR@50** over a train-derived prior (66.59 / 22.30) |
| Headroom | Is there room for vision? | Yes: `H(p|s,o)` = 3.69 bits; oracle@5 mR **63.80** vs prior **22.30** |
| Generator half | Is the prior a good candidate generator? | **Yes, decisively.** 89.4 % coverage @5 (tail 53.0 %) |
| Ranker half — geometry | Can a cheap scorer rank within candidates? | **No.** 3 scorer families, 3 objectives, 2 decision rules: **0.0 %** captured |
| Ranker half — appearance | Can frozen CLIP? | **Barely.** Gate passes, **0.6 %** captured |

---

## 2. Appearance probe results

**Setup.** Frozen CLIP ViT-B/32 (never fine-tuned), 1,200 train + 1,200
validation images, 70,709 deduplicated crops, 30,026 instances. Only a tiny
candidate scorer is trained, over the prior's top-5.

### Visual ablation gate — **PASSES**

| arm | val R | val mR@50 | tail |
|---|---:|---:|---:|
| REAL appearance | 42.81 % | **17.00 %** | 5.5 % |
| SHUFFLED appearance | 35.50 % | 13.79 % | 7.5 % |
| ZERO appearance | 45.53 % | 11.09 % | 0.6 % |

`real > shuffled > zero` holds cleanly. **CLIP appearance carries genuine
predicate information.**

### Additive composition — the honest test

`score = log P(p|s,o) + λ · f(appearance)`, where λ=0 is exactly P0, so any
gain is attributable to appearance alone.

| λ | val R | val mR@50 | headroom captured |
|---:|---:|---:|---:|
| 0.00 (P0) | 67.38 % | 20.98 % | 0.0 % |
| 0.10 | 67.66 % | 21.08 % | 0.2 % |
| 0.25 | 67.41 % | 21.15 % | 0.4 % |
| **0.50** | 66.35 % | **21.24 %** | **0.6 %** |
| 1.00 | 65.63 % | 20.89 % | −0.2 % |
| 2.00 | 63.01 % | 20.63 % | −0.8 % |

**Best: +0.26 mR@50, i.e. 0.6 % of a 43.95-point headroom.**

### Per-predicate — the mechanism is exactly right

| predicate | n | P0 rank-1 | additive | Δ |
|---|---:|---:|---:|---:|
| `walking on` | 23 | 0.0 % | 30.4 % | **+30.4** |
| `eating` | 22 | 40.9 % | 63.6 % | **+22.7** |
| `riding` | 49 | 10.2 % | 20.4 % | **+10.2** |
| `standing on` | 81 | 1.2 % | 11.1 % | **+9.9** |
| `holding` | 308 | 64.9 % | 70.8 % | +5.8 |
| `under` | 185 | 20.5 % | 22.2 % | +1.6 |

Appearance improves precisely the action and pose predicates it should. The
mechanism is not in doubt; only the magnitude is.

---

## 3. Why the aggregate barely moves

The gains are concentrated on predicates with 20–80 validation instances,
while `on` (5,601), `has` (2,219) and `in` (1,767) dominate. Under
class-averaged mR@50 a +30-point gain on a 23-instance class contributes
`30/50 = 0.6` points; a −5-point slip on a large class cancels it. The
predicates where appearance helps are exactly the ones where the metric has
the least leverage per unit of effort.

---

## 4. A control that was necessary, not decorative

A smoke test on synthetic data with **planted** signal showed the naive
scorer — per-predicate linear over raw 512–3072-dim CLIP features on ~10k
instances — overfits so badly that **ZEROED appearance outscored REAL
appearance**. Run as-is, the real experiment would have produced an
uninterpretable false negative.

Fixed with PCA-48 (train-only), an **image**-level held-out split for epoch
selection, and weight decay. The corrected pipeline was then validated
two-sided on synthetic data:

- **100.0 %** of headroom captured when signal is planted in `union`
- **−33 to −37 %** for pure-noise features

That two-sided validation is what makes the real negative interpretable
rather than ambiguous.

---

## 5. Caveats — the negative is weaker than a positive would have been

1. **ViT-B/32, not ViT-L/14-336.** Forced: L/14-336 measured at **0.3
   crops/s** on this CPU-only machine (3.1 s/crop), infeasible for ~70k
   crops. B/32 is weaker, so this is a **lower bound**. Confirming on GPU
   with L/14-336 is the single experiment that could overturn this verdict.
2. **1,200-image subsample.** Tail predicates are thin, so per-predicate
   deltas are noisy even where large.
3. **PCA-48** may discard signal.
4. **Held-out mR (~49 %) ≫ val mR (~21 %).** Part of that gap is that the
   train-derived prior was built from the full train split and has already
   seen the held-out train images, so held-out-from-train numbers are
   optimistically biased. **The validation numbers are the honest ones.**

---

## 6. Re-scope recommendation

Do **not** build a neural reranker on this evidence. Three cheap
architectures have now failed to convert the headroom (additive
prior-residual, shared linear reranking, per-confusion tournament), and the
appearance probe captures under 1 %.

Ranked options:

**A — Confirm the bound on GPU (cheapest decisive step).** Re-run
`tools/appearance_probe.py` with ViT-L/14-336 on an L4, full validation
split. One run. If L/14-336 also captures <5 %, the appearance thesis is
settled and the project should stop pursuing it.

**B — Re-scope the contribution to the negative result.** The strongest
defensible claim this repository currently supports is *methodological*, not
architectural:

> On VG150 PredCls under GT pairs, a pair-conditioned co-occurrence prior
> built from the training split alone achieves R@50 66.59 / mR@50 22.30 —
> within 0.5 / 0.34 of a 79.9M-parameter CLIP-based model. Oracle reranking
> of that prior's top-5 would reach mR@50 63.80, yet neither box geometry nor
> frozen CLIP appearance converts more than 1 % of that headroom. The
> protocol is prior-saturated.

That is a real, reproducible, and useful finding, supported by five
committed tools. It is worth more than another architecture iteration.

**C — Change the task, not the model.** If the goal is to demonstrate visual
relational reasoning, the metric and protocol have to reward it. Options:
report on the decidable-predicate subset explicitly; evaluate under the
multi-predicate protocol (`predcls_multi`, already implemented); or move to
SGCls/SGDet where object identity is not given and co-occurrence is
correspondingly weaker.

**Not recommended:** a relation transformer, a mixture of specialists, or a
larger encoder. Nothing measured suggests capacity is the constraint.
