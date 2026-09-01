# Hypothesis decision matrix — post-p26

Adjudicated 2026-09-01, after `runs/p25` (resampling) and `runs/p26`
(pair-matched null). `runs/p24` (full validation) was **still executing** when
this was written; no row depends on it, and the rows it will bear on are marked.

Evidential classes: **VF** verified fact · **MR** measured result ·
**INF** inference · **HYP** hypothesis.

---

## H1 — Candidate-generation bottleneck

**FALSIFIED.**

- **For:** none surviving.
- **Against:** **MR** GT is inside the prior's top-5 for **89.7%** of GT rows
  (top-3 85.5%, top-2 80.6%) (`runs/p17`). A learned scorer restricted to those
  candidates cannot convert the remainder (`runs/p18`, EXHAUSTED 4/4).
- **Unresolved:** nothing material.
- **Cheapest discriminating test:** already run (`runs/p17` coverage column).

## H2 — Prior calibration bottleneck

**SUPPORTED as the dominant axis, but calibration is already at its ceiling.**

- **For:** **MR** τ moves mR far more than the model does (21.98 → 26.00 over
  τ ∈ [0, 0.1]); a no-vision recalibration reproduced the model's entire mR gain
  (`f004fe2`).
- **Against (as a *remaining* bottleneck):** **MR** a *learned* per-class rule
  does not merely fail to beat τ, it fails to **match** it — Pareto
  **−1.205 ± 0.793**, clearing the R@50 floor on **0 of 5** partitions
  (`runs/p25`). There is no headroom left in calibration alone.
- **Unresolved:** whether a non-linear or per-pair calibration differs. Weakly
  motivated: the linear class-bias family already spans τ.
- **Cheapest discriminating test:** already run (`runs/p25` `prior_only` arm).

## H3 — Score-scale mismatch

**CONTRADICTED.**

- **Against:** **MR** the α → ∞ limit (`model_rerank`) is strictly worse than the
  additive composition in every cell (best +0.423 vs +0.673) and costs up to
  −15.6 R points (`runs/p17`). **VF** a monotone rescale cannot change an argmax
  inside a fixed candidate set, so no rescaling recovers more.
- **Cheapest discriminating test:** already run (`runs/p17` α-limit arm).

## H4 — Image-conditioned complementary information

**REFUTED at the current operating point.** *(Primary discriminator: `runs/p26`.)*

- **For:** **MR** the model-bearing arm separates from identity-destroying nulls
  by ≥ +1.84 Pareto points on every partition (`runs/p25`). Historically, +256
  net beneficial top-1 flips, ΔR +0.673, bootstrap CI [+0.403, +0.963], positive
  in 100.0% of 2,000 image-resamples.
- **Against:** **MR** destroying image content while preserving (subject, object)
  identity costs **−0.114 ± 0.265** Pareto points, with `full` ahead on **1 of 5**
  partitions (`runs/p26`). Effect-size bound: image conditioning ≲ **0.2** points,
  point estimate wrong-signed. **VF** `ensemble_alpha = 0.0` → the term is 100%
  the CLIP **text** branch; the visual classifier head contributes exactly zero.
  **MR** **86.87%** of the term's variance is *between* pair groups.
- **Unresolved:** whether a *trained* visual classifier head, or a different
  representation, would carry image-conditioned signal. **`p26` closes the
  current branch, not the question.**
- **Cheapest discriminating test:** re-dump the cache at `ensemble_alpha > 0` and
  repeat `p26` — but the checkpoint's classifier head is untrained (1.41% / 0.94%),
  so this measures an untrained head. **Not currently worth GPU.**

## H5 — Subject/object identity leakage or pair-prior reparameterisation

**SUPPORTED — this is the surviving explanation.**

- **For:** **MR** `pair_matched_null` ≥ `full` on 4 of 5 partitions and tracks it
  including the floor failure on salt 3 (`runs/p26`). **MR**
  `pair_matched_null − shuffled_model` = **+3.163** — destroying pair identity is
  what costs. **MR** 86.87% between-group variance. **VF** text-branch-only
  composition.
- **Against:** **MR** the term still adds +3.16 Pareto points over the prior's own
  features, so it is **not redundant** with the frequency prior — it is a *better*
  pair-conditioned predicate distribution, expressed in text-embedding space.
- **Unresolved:** whether that improvement is distillable into the prior itself
  (a cheaper, vision-free artifact).
- **Cheapest discriminating test:** fit the prior's features plus a learned
  (subject, object)-group embedding, no model term, and see whether it recovers
  the +3.16. **CPU-only, on the existing cache.** This is the recommended next
  experiment.

## H6 — Global-ranking failure with sparse top-1 correction

**CONFIRMED, and now re-attributed.**

- **For:** **MR** every ranking metric degrades — mean GT rank 1.826 → 2.782,
  MRR 0.7998 → 0.7815, R@2 84.68 → 81.70, R@5 95.72 → 91.69, R@10 98.81 → 95.07.
  Yet +256 net beneficial top-1 flips. Rescues concentrate at prior GT rank 2
  (10.64% rescue rate, 679 of 1,042 rescues); on rescued rows the model ranks GT
  top-1 68.14% of the time against 38.86% overall. The rank-21+ bucket grows
  60 → 1,022 rows.
- **Against:** nothing — this is the best-established mechanism in the project.
- **Re-attribution (new):** the tie-breaker is real, but `p26` shows *what it
  breaks ties with* is pair identity, not appearance.
- **Cheapest discriminating test:** already run (`docs/CPRIME_MECHANISM_REPORT.md`).

## H7 — Evaluation artifact

**LARGELY CLOSED; one residue.**

- **For (historically):** five evaluator defects found and fixed with regression
  tests — GT-triplet index misalignment (`220c5c2e`), `eval_batches=0` evaluating
  zero images (`74c54a2`), `topk`/`argmax` tie-break on 522 GT rows (`7d4a658`),
  unspecified tie at the k-th candidate slot (`889483e`), and
  `min(25, eval_batches)` inverting the unlimited sentinel (`9ab094d`).
- **Against (as a live explanation):** **VF** historical artifacts match their
  recorded SHA256; splits verified disjoint (83,249 / 10,401 / 10,403) and pinned
  by `tests/test_split_separation.py`; the cache's predicate ordering is identical
  to the canonical vocabulary; `p24` ran at the sentinel fix with a clean tree.
- **Unresolved:** the `README` results table has never been re-derived under the
  current evaluator and remains `UNKNOWN` provenance.
- **Cheapest discriminating test:** re-derive it from the `p24` cache — no extra
  GPU.

---

## Ranking of surviving explanations

1. **H5** — pair-prior reparameterisation. Directly measured, mechanistically
   explained by two independent structural facts.
2. **H6** — tie-breaker mechanism. Confirmed, but now understood to be breaking
   ties on pair identity.
3. **H2** — calibration dominates the axis, and is exhausted.
4. **H4** — refuted at this operating point; open only for an untested
   representation.
