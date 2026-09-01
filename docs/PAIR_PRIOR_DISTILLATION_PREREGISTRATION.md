# Pre-registration — corrected pair-prior distillation (estimable subset)

Status: **PRE-REGISTERED.** Committed before the corrected tool is written and
before any number from it exists.
Branch: `research/architecture-breakthrough`
Supersedes: `runs/p27`, withdrawn — see `docs/PAIR_PRIOR_DISTILLATION_RESULT.md`.
CPU only. **No GPU.** The `p24` cache is read-only input.

---

## 1. The question

`runs/p29` confirmed at full validation scale that the checkpoint's contribution
converts into +2.947 ± 0.190 Pareto points, and that destroying image content
while preserving (subject, object) identity costs **+0.031 ± 0.188** — nothing.
So the contribution is pair-conditioned.

That leaves the question `p27` was built to answer and could not:

> **Can a vision-free, purely statistical pair-conditioned model reproduce the
> checkpoint's surviving contribution?**

If yes, the diagnosis is that the checkpoint's usable output is a *learned pair
prior*, and the architecture branch stays closed. If no, there is a residual
that is pair-conditioned but not reproducible from pair statistics, and the next
step is to characterise it — **not** to assume it is visual.

## 2. Why `p27` could not answer it, in one line

`Distill._fold_pair_mean` falls back to the training **global** mean for any
(s,o) group with no training row. Singleton groups are always in that state, so
the pair-conditioned arms carried no pair information on **45.4%** of rows (3k)
/ **33.2%** (full cache), worst in the body at 52.8%. Verified by reading the
source and by `runs/p30`.

## 3. Population — fixed here, before the numbers

**Per-partition estimable subset.** For fold partition `salt` and outer fold
`f`, a held-out row `r` is *estimable* iff its `pair_id` has at least one row
outside fold `f`. Only estimable rows enter the analysis, for every arm equally.

Three properties of this restriction, registered as facts rather than discovered
later:

1. It is **not label leakage.** The mask depends only on `pair_id` and on the
   fold assignment, which is a hash of image id and salt. It never touches `y`,
   the model term, or any arm's score.
2. It is **not a random subsample.** It over-represents frequent pairs by
   construction. Therefore the prior-only baseline **and the tau frontier are
   recomputed on the subset** for each partition. Inheriting the full-split
   baseline (66.593 / 22.304) would be a category error and is forbidden.
3. It **cannot** speak for the ~33% singleton-pair rows. Any result here is a
   statement about frequent-pair rows only. That is a property of VG150's pair
   distribution, not a fixable estimator detail, and must be stated wherever the
   result is quoted.

Expected size ≈ 88.6k rows per partition (measured at salt 0 by `runs/p30`),
larger than the entire 3,000-image screening set.

## 4. Everything held identical to `p29`

Folds (5, by image, `SEED=0`), fold salts 0–4, candidate set (`k=5`), tau
handling (tau=0, prior rows raw, tau applied CPU-side), nested beta selection
inside training folds only, evaluation semantics, denominator convention,
`pareto_gap` definition, `l2=1e-4`. The registered partition is **salt 0**;
salts 1–4 are the secondary resampled read, as in `p29`.

## 5. R@50 floor on a changed population

`R_FLOOR = 0.665` is an absolute constant set against the 3k prior-only baseline
of 66.802, i.e. **prior − 0.302 points**. The subset has a different baseline,
so the constant cannot transfer.

**Registered primary floor:** `floor_subset = (prior-only R@50 on that
partition's estimable subset at tau=0) − 0.30 points`, computed per partition.
This reproduces the original floor's own construction on the new population.

The absolute 66.5 is reported as a secondary column. Neither is adjusted after
the numbers exist.

## 6. Arms

Vision-free ladder (your §7 A–F):

| arm | definition |
|---|---|
| `A_global` | P(p) |
| `B_subject` | P(p \| subject) |
| `C_object` | P(p \| object) |
| `D_pair` | P(p \| subject, object), train-derived prior file |
| `E_backoff` | hierarchical smoothing over global/subject/object/pair |
| `F_pair_foldfit` | P(p \| s,o) re-fitted on this cache's training folds |

Reference and decomposition:

| arm | definition |
|---|---|
| `G_model` | the C′ model term — the quantity to be reproduced |
| `G_pairmean` | model term → its training-fold (s,o) group mean |
| `G_residual` | model term − that group mean |
| `null_shuffled` | model term permuted across all rows (identity destroyed) |
| `null_pair_matched` | model term permuted within (s,o) groups (image destroyed) |

On the estimable subset `G_pairmean` and `G_residual` are genuine for the first
time, so the decomposition is interpretable here and was not in `p27`.

## 7. Primary criterion

Let `P_G` = `G_model`'s Pareto gap, mean over the 5 partitions.
Let `P_V` = the best mean Pareto gap among {`A_global`, `B_subject`,
`C_object`, `D_pair`, `E_backoff`, `F_pair_foldfit`}, restricted to arms that
clear the registered floor on **≥ 4 of 5** partitions. If no vision-free arm is
eligible, `P_V` is recorded as the best ineligible value and the verdict is
**NOT EXPLAINED** regardless of its magnitude.

- **EXPLAINED**: `P_V ≥ P_G − 0.50`
- **MOSTLY EXPLAINED**: `P_V ∈ [P_G − 1.50, P_G − 0.50)`
- **NOT EXPLAINED**: `P_V < P_G − 1.50`

Thresholds are inherited from constants already in this codebase rather than
chosen for this run: `0.50` is 2× `CALIBRATION_EPS` (0.25, the project's
"adds nothing beyond a decision rule" epsilon) and `1.50` is `INCONCLUSIVE_PTS`.

**All three outcomes will be reported with equal prominence.** NOT EXPLAINED is
not a licence to call the residual visual — see §9.

## 8. Validity gates — if any fails, no number is reported

| gate | requirement |
|---|---|
| G1 fallback eliminated | fallback fraction on the restricted subset is **exactly 0** |
| G2 decomposition genuine | class-centred cosine of `G_residual` vs the real model term < **0.70** (p30 measured 0.46 on estimable rows, 0.94 on fallback rows) |
| G3 model-term identity | the existing 3.576e-06 recomposition gate passes |
| G4 power | subset ≥ **80,000** rows on every partition |
| G5 sanity | `null_shuffled` lands at or below the subset's tau frontier, as it does on every prior run |

G5 is a check on the harness, not a hypothesis test. If `null_shuffled` comes
out *above* the frontier, the instrument is wrong and the run is void.

## 9. Interpretation rules, fixed in advance

- **EXPLAINED** ⇒ the diagnosis is *the checkpoint's usable contribution on
  frequent-pair rows is a learned pair prior, not image-conditioned relational
  reasoning*. The architecture branch stays **closed**. Next work is
  generalisation across predicates / buckets / support levels.
- **MOSTLY / NOT EXPLAINED** ⇒ a residual exists. It is **pair-conditioned but
  not reproducible from these statistics**. It is **not** thereby visual. No
  architecture work follows. The next step is the cheapest discriminating test
  from the directive's §10 (role swap, image-preserving/pair-shuffling null,
  frozen visual feature test), and only a positive result there would license
  considering a model.
- No result here reopens the oracle/reranking branch, which is closed on a
  separate and independent ground (`oracle_R ≥ prior_R` by construction).

## 10. Compute budget and stop rule

CPU only, 8 cores. `p27` took 1,331 s on 38,053 rows with 11 arms; this is
~88.6k rows, so ~50 min is expected and **2 h is the budget**. If it exceeds the
budget the run is killed and reported as incomplete rather than trimmed to fit.
No GPU job will be launched on any outcome of this experiment.
