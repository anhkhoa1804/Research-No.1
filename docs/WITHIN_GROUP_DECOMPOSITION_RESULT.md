# Within-group decomposition (`runs/p42`) — the four quantities, separated

Run: exit 0, 101 s, full `p24` cache. This is the analysis the `p32`
interpretation needs, and it settles a question that was being assumed rather
than measured: **within-group ≠ visual reasoning.**

---

## 1. A structural fact, not a measurement

**WPRD is invariant to anything constant within a (subject, object) group.** It
compares `(score[i,a]−score[i,b])` against `(score[j,a]−score[j,b])` for i, j in
the same group, so subtracting that group's mean from every row changes nothing.

Verified by recomputing WPRD on the group-centred term:

| head | raw | within-group-centred | \|diff\| |
|---|---|---|---|
| text | 0.554195 | 0.554195 | 1.3e-08 |
| classifier | 0.572806 | 0.572806 | 0.0 |

*(The tool printed FAIL for the text head against a 1e-9 tolerance. That
tolerance is mis-set for float32 — 1.3e-08 is epsilon-scale for a sum over
132,556 rows, and the classifier head returns exactly 0. The invariance holds;
my threshold was too tight. Disclosed rather than silently loosened.)*

**Consequence:** the between-group share of the model term contributes
**exactly zero** to WPRD. WPRD and R@50 read *different halves of the same
tensor*. That is why `p39` (geometry wins on WPRD) and `p41` (the model wins on
R@50/mR@50 at tau ≤ 0.05) are both true and not in tension.

## 2. The variance split

| head | between-group (pair identity) | within-group |
|---|---|---|
| text (evaluated) | **85.77%** | 14.23% |
| classifier (discarded) | **91.17%** | 8.83% |

The classifier head is *more* pair-dominated and yet discriminates *better*
(0.5728 vs 0.5542). It uses a smaller within-group budget more effectively.

## 3. Splitting the within-group part: layout vs the rest

Box geometry, cross-fitted out-of-fold on the project's own folds (sizes
`[26483, 26856, 27190, 26586, 25441]`, matching `p30` salt 0), regressed on the
group-centred term:

| | text head | classifier head |
|---|---|---|
| out-of-fold R² of geometry on the within-group term | **17.61%** | **18.67%** |
| WPRD of the whole within-group term | 0.5542 [0.5495, 0.5592] | 0.5728 [0.5681, 0.5779] |
| WPRD of its **layout-predictable** part | 0.5450 [0.5401, 0.5501] | **0.5730** [0.5680, 0.5779] |
| WPRD of the **residual (non-layout)** part | 0.5343 [0.5294, 0.5401] | 0.5477 [0.5422, 0.5524] |

**Both intervals exclude chance in both heads.** So:

- The checkpoint's within-pair discrimination is **not purely layout** — the
  non-layout residual independently discriminates above chance (text 0.5343,
  classifier 0.5477). There is genuine non-geometric image-conditioned evidence.
- For the **classifier head**, the layout-predictable component alone reaches
  **0.5730**, statistically identical to the whole (0.5728). Its discriminative
  power is essentially *fully expressed* by the geometry-predictable 18.67% of
  its within-group variance.
- For the **text head**, layout alone (0.5450) is *below* the whole (0.5542), so
  its non-layout residual is contributing something the layout part does not.

## 4. The four quantities, as percentages of the text head's total variance

| quantity | share of total variance | WPRD it supports |
|---|---|---|
| 1. between-group — **pair prior** | **85.77%** | **0.5000 by construction** |
| 2. within-group, layout-predictable | 14.23% × 17.61% = **2.51%** | 0.5450 |
| 3. within-group, non-layout residual | 14.23% × 82.39% = **11.72%** | 0.5343 |
| 4. (2)+(3) jointly = all within-group | 14.23% | 0.5542 |

**The headline of this table:** ~2.5% of the model term's variance — the part a
linear map from two rectangles can reconstruct — supports nearly as much
relational discrimination as the 11.7% that cannot. Most of the within-group
variance is not discriminative. It is not noise-free evidence waiting to be
decoded; it is mostly not about the predicate at all.

## 5. What this licenses, and what it does not

**Licensed:**
- p32's "pair prior" quantity is the 85.8% between-group share, and it is
  invisible to WPRD by construction.
- Non-layout image-conditioned evidence **exists** in this checkpoint's readouts
  (both residual CIs exclude 0.5).

**Not licensed:**
- *"All within-group variance is visual."* It is not — 82% of it supports
  discrimination barely above what the layout 18% supports alone.
- *"The grounding is just geometry."* The residual excludes chance in both heads.
- *"Tail grounding is zero."* `p35` shows tail intervals that **contain** chance
  for the evaluated head; containing chance is not being zero, and the
  classifier head's body–tail interval excludes it.

## 6. What it implies for the readout-vs-representation question (`p36`/`p37`)

The heads spend 85.8% / 91.2% of their output variance on pair identity and
leave 14.2% / 8.8% for everything image-conditioned. If `rel_feat` turns out to
carry substantially more within-group predicate structure than that, the
bottleneck is the **readout**, and this table says where the compression
happens. If it does not, the representation is the limit. `p36` is still
running; the criterion is registered and unchanged.
