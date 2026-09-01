# Full-validation confirmation (`runs/p24` → `p28`/`p29`/`p31`) — RESULT

Status: **CONFIRMED on the registered criterion.** And the same table shows the
confirmed quantity is not image-conditioned. Both halves are reported here with
equal prominence, as `docs/FULL_VALIDATION_PREREGISTRATION.md` requires.

No threshold in the pre-registration was altered before, during, or after the
numbers existed. The criterion is quoted verbatim in §2.

---

## 0. Provenance of the cache the verdict rests on

`runs/p24_full_val_cache`, exit 0, 12,423 s (3 h 27 m, inside the 5 h budget).

| | |
|---|---|
| images | **10,401** — the entire validation split |
| GT rows | 132,556; GT triplets whose (s,o) is in the pair list **100.00%** |
| predicates | all **50** present in GT; alias map collapses 50→48 via `near`, `wears` |
| vocabulary | order identical to `predicates.json` (50 fg + 1 background) |
| tau | `eval_freq_bias_tau = 0.0` — prior rows stored RAW, tau applied CPU-side |
| alpha | `ensemble_alpha = 0.0`; `freq_bias_alpha = 3.75` recorded, NOT baked in |
| checkpoint | `8845c3af…a442`, recomputed and matched against the manifest |
| commit | `9ab094d`, clean tree — includes the `cap_batches` sentinel fix, so **no duplicated full-split sweep** (runtime confirms one pass, not two) |
| validation | `tools/validate_pair_dump.py` **12/12 PASS**, verdict `CACHE VALID` |

The 3,000-image screening set was `--eval_batches 250` of this same split. The
full split is **3.48×** more rows.

## 1. The registered partition (salt 0), tau=0, k=5

Prior-only baseline on the full split: R@50 **66.593**, mR **22.304**.
Achieved additive C′: R@50 67.168, mR 23.204 (ΔR **+0.575**).

| arm | R@50 | mR@50 | Pareto gap | floor 66.5 | head | body | tail |
|---|---|---|---|---|---|---|---|
| `prior_only` (no vision) | 65.357 | 27.178 | +0.378 | **FAIL** | 41.43 | 26.00 | 14.49 |
| **`full` (prior + model)** | **66.760** | 25.049 | **+2.745** | **ok** | 42.08 | 23.65 | 9.89 |
| `shuffled_model` (identity destroyed) | 65.348 | 27.157 | +0.352 | **FAIL** | 41.44 | 26.00 | 14.41 |
| `pair_matched_null` (image destroyed) | 66.704 | 25.009 | **+2.705** | **ok** | 42.05 | 23.21 | 10.37 |

## 2. Adjudication against the criterion, quoted verbatim

> **CONFIRMED**: full clears the floor, its Pareto gap > +1.5, and it exceeds
> both `prior_only` and `shuffled_model` by > +1.0 Pareto points.

| condition | required | observed | verdict |
|---|---|---|---|
| `full` clears R@50 floor | ≥ 66.5 | **66.760** | PASS |
| `full` Pareto gap | > +1.5 | **+2.745** | PASS |
| `full` − `prior_only` | > +1.0 | **+2.368** | PASS |
| `full` − `shuffled_model` | > +1.0 | **+2.394** | PASS |

**Verdict: CONFIRMED.** Three of three registered conditions, on the registered
partition, with the registered thresholds.

The pre-registration's addendum predicted WEAKENED was as likely as CONFIRMED
because the screening mean was +1.911 ± 1.056. That prediction was wrong in the
conservative direction: at full scale the gap is **larger and far more stable**
than screening suggested (§3).

## 3. Secondary — resampled over 5 independent fold partitions

Salt 0 is the registered read; salts 1–4 are the secondary robustness read.

| arm | Pareto mean ± sd | min | R@50 mean ± sd | floor held |
|---|---|---|---|---|
| `prior_only` | −1.167 ± 1.101 | −2.498 | 65.787 ± 0.357 | **0/5** |
| **`full`** | **+2.947 ± 0.190** | **+2.736** | 66.668 ± 0.057 | **5/5** |
| `shuffled_model` | −1.111 ± 1.074 | −2.533 | 65.788 ± 0.347 | **0/5** |
| `pair_matched_null` | **+2.915 ± 0.142** | +2.705 | 66.636 ± 0.038 | **5/5** |

Against the 3k screening (`runs/p25`): `full` was +1.911 ± **1.056**, floor
**4/5**. At full scale it is +2.947 ± **0.190**, floor **5/5** — the standard
deviation falls by 5.6×. **The screening instability was a small-sample
artifact, not a fragile effect.** The separation from both no-model nulls, which
the addendum identified as the robust half, holds on every partition
(`full − prior_only` min **+2.368**, `full − shuffled_model` min **+2.394**).

## 4. The result the criterion could not test

`pair_matched_null` — the model term permuted only among rows sharing the same
(subject, object) category, so image content is destroyed and pair identity is
preserved — **passes every registered condition too**: floor 66.704, gap +2.705,
+2.327 over `prior_only`, +2.353 over `shuffled_model`.

Paired per-partition, `full` minus `pair_matched_null`:

| salt | `full` | `null` | difference |
|---|---|---|---|
| 0 (registered) | 2.745 | 2.705 | +0.041 |
| 1 | 2.736 | 3.019 | **−0.283** |
| 2 | 3.110 | 3.014 | +0.096 |
| 3 | 3.048 | 2.829 | +0.219 |
| 4 | 3.095 | 3.010 | +0.085 |

**full − null = +0.031 ± 0.188** (paired, n=5), negative on one partition.
A t-based 95% interval on the paired mean is **[−0.20, +0.26]** Pareto points.

This **replicates `runs/p26`** (3k: −0.114 ± 0.265) at 3.48× the rows and with a
29% tighter sd, and it tightens the bound: **image conditioning contributes
≲ 0.26 Pareto points, with a point estimate indistinguishable from zero.**

This arm did not exist when the criterion was written — it was added by `p26`
afterwards — so it is **not** part of the criterion and does not change the
CONFIRMED verdict. What it establishes is what the verdict *means*.

## 4b. Analysis 3/3 — the operating-point frontier (`runs/p31`)

Reported, not a criterion. Exit 0, 345 s, same cache, tau=0, k=5, betas swept.
Prior R@50 66.593 / mR 22.304; achieved additive C′ Pareto +0.900.

| beta | `prior_only` | | `full` | | `shuffled_model` | |
|---|---|---|---|---|---|---|
| | Pareto | floor | Pareto | floor | Pareto | floor |
| 0.00 | −1.872 | ok | −2.015 | ok | −1.870 | ok |
| 0.05 | −1.333 | ok | −0.655 | ok | −1.374 | ok |
| 0.10 | −0.133 | FAIL | **+0.685** | **ok** | −0.175 | FAIL |
| 0.15 | −1.213 | FAIL | **+2.223** | **ok** | −1.126 | FAIL |
| 0.20 | −1.058 | FAIL | **+3.334** | **ok** | −1.129 | FAIL |
| 0.25 | +0.381 | FAIL | +1.100 | FAIL | +0.348 | FAIL |
| 0.30 | +2.146 | FAIL | +2.783 | FAIL | +2.081 | FAIL |

The structural fact this adds to the nested read: `full` has a **contiguous
region** of usable operating points — beta ∈ [0.10, 0.20], three points that are
simultaneously above the tau frontier and above the R@50 floor, peaking at
**+3.334** (R@50 66.515, mR 25.919) at beta=0.20. Neither no-model arm has a
single such point: wherever they clear the floor their Pareto gap is negative,
and wherever their gap turns positive they have already failed the floor. That
is exactly the signature of "everything without the model term is calibration".

Screening (`runs/p21`, 3k) read +2.86 at beta=0.20. The full split reads
**+3.334** — consistent, slightly stronger, same argmax beta.

**Coverage limitation.** `--frontier` predates `p26` and does not carry the
`pair_matched_null` arm, so this table cannot separate pair identity from image
content. Only the nested run (§4) can, and it puts that difference at
+0.031 ± 0.188. Nothing here contradicts that; it simply does not test it.

## 5. How to state this result

Correct: *the candidate-restricted decision rule reliably converts the
checkpoint's contribution into Pareto movement over the tau frontier, at full
validation scale, on every fold partition; and that contribution is
(subject, object) identity expressed in text-embedding space, not image
content.*

Incorrect: *the visual model adds +2.9 mR points.* An arm with every image
association destroyed adds +2.9 too.

Two structural facts already on record explain why, and are unchanged by this
run: `ensemble_alpha = 0.0` means the model term is **100% the CLIP text
branch** with the visual classifier head at exactly zero weight, and **86.87%**
of that term's variance is *between* (s,o) groups.

Note also that `full` buys its head/body movement at a **tail cost**: tail mR
9.89 vs `prior_only`'s 14.49 on salt 0. The gain is not a long-tail gain.

## 6. What this does and does not license

- It **does** close the full-validation confirmation as CONFIRMED, and retires
  the "3,000 images is too small to trust" objection in both directions.
- It **does not** reopen the visual-architecture branch. `p26` closed that on
  source, and this run reproduces `p26`'s contrast at full scale with a tighter
  bound. A CONFIRMED magnitude for a pair-conditioned effect is still a
  pair-conditioned effect.
- It **does not** answer whether a vision-free pair-conditioned estimator can
  reproduce the effect. `runs/p27` attempted that and was withdrawn
  (`docs/PAIR_PRIOR_DISTILLATION_RESULT.md`); the corrected estimable-subset
  study is the next experiment and needs no GPU.
