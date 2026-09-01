# WPRD literature audit — classified, not conflated

The directive's question, answered exactly:

> **Does any existing SGG evaluation explicitly condition predicate
> discrimination on the subject-object category pair so that P(p|s,o) becomes
> non-informative?**

**Answer: no.** The audit below is classified into OBSERVATION / METHOD /
METRIC / BENCHMARK / DATASET, because conflating those is how false novelty
claims are made. Several rows are things this project **rediscovered** and must
credit.

---

## The five categories

| | definition |
|---|---|
| **OBSERVATION** | a stated fact about the data or about model behaviour |
| **METHOD** | a training/inference technique that changes a model |
| **METRIC** | a scalar computed from predictions and labels |
| **BENCHMARK** | a metric + protocol + population, intended for comparison |
| **DATASET** | new images or new annotations |

## The audit

| work | category | what it is | conditions on (s,o) pair? |
|---|---|---|---|
| **Neural Motifs** (Zellers 2018) | OBSERVATION + METHOD | the `FREQ` baseline; prior dominance in VG | **no** |
| **Plesse et al.** (WACV 2020) | **OBSERVATION** + METHOD | states the confound outright: *"the majority relation for each object category pair often represents from 50% to 75% of the examples … the evaluation does not reflect that"* | **no** — a relevance model, not a conditioned metric |
| **TDE / Unbiased SGG** (Tang, CVPR 2020) | METHOD + METRIC | causal Total Direct Effect; the SGG diagnosis toolkit; mR@K popularised; S2G retrieval | **no** — causal *adjustment*, estimated, not conditioning |
| `R@K` | METRIC | all pairs, top predicate | **no** |
| `ngR@K` | METRIC | all predicates per pair | **no** |
| `mR@K` | METRIC | stratifies by **predicate** | **no** |
| `ng-mR@K` | METRIC | as above, no graph constraint | **no** |
| `zR@K` / `ng-zR@K` | METRIC | conditions on the **triplet** being unseen in training | **no** — nearest miss; a zero-shot triplet still sits in a group whose prior is fully informative about its other rows |
| `A@K` | METRIC | GT pairs only | **no** |
| **Pair Recall** (Lorenz, CVPRW 2024) | METRIC | strips the predicate; pools across all (s,o) | **no** |
| **Predicate Rank** (Lorenz, CVPRW 2024) | METRIC | rank of the correct predicate given a **correct** pair | **no** — conditions on the pair being *right*, not on its *identity* |
| **Predicate ROC-AUC** | METRIC | per-predicate ROC across all pairs | **no** — a model scores well by ranking (person,shirt) above (car,road) for *wearing*: pure object identity |
| **VG-OOD** | BENCHMARK (re-split) | redistributes data to reduce frequency bias | **no** — changes the distribution, not the conditioning |
| **Haystack** (ICCVW 2023) | **DATASET** + METRIC | rare-predicate PSG data with explicit **negative** annotations | **no** — addresses tail *measurability* |
| **SpatialSense** (ICCV 2019) | **BENCHMARK + DATASET** | adversarial crowdsourcing against 2D and language cues; reports **language-only** and **2D-only** baselines | **no** — reduces prior informativeness *empirically by curation*; binary relation verification, not predicate ranking |
| **Winoground / ARO / SugarCrepe** | BENCHMARK | image–text matching over constructed caption negatives; SugarCrepe drives **blind text** models to chance | **no** — controls *text* plausibility, not the (s,o) prior; not a per-pair predicate decision |

## What this project must credit, not claim

1. **The observation.** Prior dominance is Zellers'; the 50–75% per-pair
   majority figure is **Plesse et al.'s**. This project independently measured
   69.23% and 48.7% — *rediscovery, not discovery*.
2. **The prior-only baseline** and **the 2D/geometry-only baseline** are
   **SpatialSense's** (language-only and 2D-only). Reconstructed here
   independently; **not novel**.
3. **The goal** — prior-robust relational evaluation — is shared by SpatialSense,
   VG-OOD and the entire debiasing line.

## What appears to survive

A **METRIC**, not a benchmark and not a dataset:

> Conditioning predicate discrimination on the (s,o) category pair, which makes
> P(p|s,o) **exactly** non-informative rather than approximately less
> informative, computed on existing annotations, verified by a control that
> reads **0.5000 with CI [0.5000, 0.5000] in every stratum**.

Every other approach in the table attacks the confound by **curation**
(SpatialSense), **re-splitting** (VG-OOD), **predicate stratification** (mR@K),
or **causal adjustment** (TDE). None conditions. Conditioning is the only route
that yields an exact analytic zero rather than an empirical reduction, and the
exactness is what makes the control checkable.

Secondary, and possibly the more transportable idea: the
**discrimination-vs-calibration decomposition** (`p41`, `p47`), which showed
these can rank arms in *opposite* orders. Not SGG-specific.

## Novelty status — NOT YET CLAIMABLE

Three gates, none of which is passed:

1. **One checkpoint.** WPRD has been computed for PURE and for baselines. Until
   it runs on published models it measures PURE, not SGG.
2. **No formal related-work pass.** This audit is a working map from targeted
   searches, not a systematic survey. A submission needs the latter.
3. **The rank inversion rests on n = 4 arms** (`p47`) and is not statistically
   established.

**Nothing here should be described as novel in writing until gate 1 is passed.**
