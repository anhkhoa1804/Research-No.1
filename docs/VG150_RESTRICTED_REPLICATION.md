# `runs/p53` — the headline replicates on the STANDARD VG150 subset, and strengthens

Prompted by discovering mid-cycle that `datasets_vg150_clean` retains raw Visual
Genome object names (`docs/DATASET_IDENTITY_OBJECT_VOCAB.md`), which made every
prior number a statement about a **non-standard** population. This re-runs `p49`
restricted to GT rows whose subject **and** object are in the standard
150-category vocabulary — **37,121 of 132,556 rows (28.0%)**, 31,125 decidable.

`R@50`, `mR@50`, the tau frontier **and** WPRD are all recomputed on the
restricted population, so the comparison is internally consistent.

---

## The table (standard VG150 population, tau=0)

| arm | R@50 | mR@50 | Pareto | **WPRD** |
|---|---|---|---|---|
| pair prior only | 70.666 | 21.250 | 0.000 | **0.5000** |
| random null | 70.039 | 20.826 | −5.897 | 0.4980 |
| PURE text (α=0, **the deployed head**) | 70.319 | **22.464** | **−2.496** | 0.5553 |
| PURE ens α=0.25 | 70.580 | 22.296 | +0.587 | 0.5621 |
| PURE ens α=0.5 | 70.707 | 22.017 | +0.767 | 0.5710 |
| PURE ens α=0.75 | 70.825 | 21.847 | +0.597 | 0.5782 |
| PURE classifier (α=1) | 70.860 | 21.526 | +0.276 | 0.5815 |
| geometry linear | 70.898 | 18.565 | −2.685 | 0.6168 |
| **geometry MLP** | **71.022** | 20.124 | −1.126 | **0.6452** |
| text + geometry | 70.779 | 18.380 | −2.870 | 0.6063 |
| classifier + geometry | 70.928 | 17.885 | −3.365 | 0.6242 |
| classifier + geoMLP | **71.105** | 19.614 | −1.636 | 0.6435 |

## The correlations replicate and strengthen

| | raw-name (`p49`) | **standard VG150 (`p53`)** |
|---|---|---|
| Spearman(**R@50**, WPRD) | +0.741, p = 0.0080 | **+0.951, p = 0.0005** |
| Spearman(**mR@50**, WPRD) | −0.650, p = 0.0205 | **−0.629, p = 0.0255** |
| Spearman(Pareto, WPRD) | −0.371, ns | −0.112, ns |

**The dataset-identity problem does not explain the finding.** On the standard
VG150 population, `R@50` rank-orders these twelve scoring functions almost
exactly as prior-free relational grounding does (ρ = **+0.951**), while `mR@50`
orders them **backwards** (ρ = −0.629).

The same non-independence caveat applies as in `p49`: the twelve arms cluster
into ~3 families, so effective n is nearer 3 than 12 and the permutation p-values
are optimistic. And these remain scoring functions from one cache, not published
models.

## Two things that change on the standard subset

1. **Everything discriminates better.** Geometry MLP rises 0.6153 → **0.6452**,
   PURE classifier 0.5728 → 0.5815. The standard population has larger (s,o)
   groups (mean 9.66 vs 2.91 on validation), so there is more within-pair
   structure to find. The **gap** between geometry and PURE also widens: +0.061
   → **+0.090**.
2. **PURE's composed advantage largely disappears.** The deployed head's Pareto
   gap is **−2.496** here, versus **+0.901** on the full split. On the standard
   VG150 population the α=0 operating point sits *below* the tau frontier, and
   the best PURE variant is α=0.5 (+0.767) — the discarded classifier mixed in.
   Stated carefully: α=3.75 and α=0 were tuned on the non-standard population,
   so this is a statement about a *transferred* operating point, not evidence
   that PURE is worse than the prior in a like-for-like tuning.

## Standing

This substantially de-risks the dataset-identity concern for the **diagnostic**
claims. It does **not** make the numbers comparable to published VG150 leaderboards
— the detector, the GT construction and the evaluation protocol still differ —
and no number here should be placed beside published results.

The cross-model gate is unchanged and remains the binding one.
