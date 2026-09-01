# Bottleneck assessment — H1…H9 against the evidence already obtained

Every row is adjudicated on measurements that already exist. No hypothesis below
required a new experiment to reach its verdict, which is the point: the cheapest
discriminating experiment is the one already run.

Evidential classes: **VF** verified fact · **MR** measured result ·
**INF** inference · **HYP** hypothesis.

---

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | Candidate generation | **FALSIFIED** | **MR** GT is inside the prior's top-5 for **89.7%** of GT rows, top-3 for 85.5%, top-2 for 80.6% (`runs/p17`). A learned scorer restricted to those candidates still cannot convert it. Generation is not what is missing. |
| **H2** | Prior calibration | **SUPPORTED but BOUNDED** | **MR** tau moves mR far more than the model does (21.98 → 26.00 across tau ∈ [0, 0.1]) and a no-vision recalibration reproduced the model's entire mR gain (`f004fe2`). **But** with beta chosen out-of-fold, calibration-only arms land at Pareto gap **+0.059** and their null at **+0.224** — i.e. *on* the tau frontier (`runs/p22`). Calibration dominates the movement and cannot leave the frontier. |
| **H3** | Score-scale mismatch | **CONTRADICTED** | **MR** The alpha → ∞ limit (`model_rerank`) is strictly worse than the additive composition in every cell (best +0.423 vs +0.673), so no rescaling of the model term recovers more. A monotone rescale cannot change an argmax inside a candidate set; only re-ranking can, and that is H4. |
| **H4** | Candidate ranking | **EXHAUSTED on R, LIVE on the Pareto axis** | **MR** Pre-registered ΔR criterion: EXHAUSTED 4/4, `full − achieved` +0.21…+0.73 against a +2.0 bar (`runs/p18`). **MR** But on the frontier, candidate-restricted + class-reweighted reaches Pareto **+2.894** vs the additive arm's **+0.861** at the same R floor (`runs/p22`). Ranking is not where the gain is; the *decision rule over the candidates* is. |
| **H5** | Global-softmax normalisation | **PARTIALLY CONFIRMED, already banked** | **VF** `cprime_analysis.Bench.ensemble_term` sliced to 50 foreground columns and *then* standardised, while the evaluator standardises over all 51 and suppresses background afterwards; max abs difference 1.61e-01. Fixed in `Mech.fixed_ensemble`, identity-checked at 3.6e-06 at every tool start. Not a live bottleneck — it is a closed defect. |
| **H6** | Representation | **SUPPORTED — the primary bottleneck for ranking** | **MR** `model_only`, fitted optimally inside the candidate set and cross-fitted, is **negative in all four cells** (−0.089…−0.134) and fails the R floor at tau=0.05 (`runs/p18`). **MR** The raw score as ranker costs up to −15.6 R points (`runs/p17`). The representation cannot rank predicates on its own. What it carries is a stable **+0.68…+0.79 R points** of complementary signal. |
| **H7** | Insufficient visual conditioning | **CONTRADICTED as posed** | **MR** Frozen CLIP ViT-L/14-336 appearance signal is real but adds nothing beyond calibration (`docs/APPEARANCE_TAU_INTERACTION_RESULT.md`, `de88772`), and PCA-48 was not the cause. More visual conditioning of the same additive kind is not indicated. |
| **H8** | Pairwise relational reasoning | **UNRESOLVED — and deliberately not pursued** | **HYP** No measurement in this project isolates relational reasoning from the (subject, object) co-occurrence prior, which already supplies +28.09 R / +18.30 mR over the global marginal. Testing it needs a new objective, not a new head. Not pursued because no *measured failure mode* demands it (directive §17). |
| **H9** | Evaluation artifact | **LARGELY CLOSED, one residue** | **VF** Four evaluator defects found and fixed with regression tests: GT-triplet index misalignment (`220c5c2e`), `eval_batches=0` evaluating zero images (`74c54a2`), the `topk`/`argmax` tie-break on 522 rows (`7d4a658`), and `min(25, eval_batches)` inverting the unlimited sentinel (`9ab094d`). **Residue:** the historical README claims predate several of these and have never been re-derived under the current evaluator. |

---

## The single most supported diagnosis

**MR + INF.** The bottleneck is **not** candidates (H1 falsified), **not** score
scale (H3), and **not** visual conditioning of the additive kind (H7). It is
two things at once:

1. **H6 — the representation cannot rank.** Given a perfect candidate set and an
   optimal linear fit, the model score alone loses to the prior's argmax.
2. **H4′ — the decision formulation, not the ranking, is the convertible axis.**
   The model carries ~+0.75 R points of genuine complementary information, and
   the additive alpha/tau composition converts about a third of what a
   candidate-restricted, class-reweighted rule converts (+0.861 vs +2.894 Pareto
   points at the same R floor).

This is deliberately *not* the most sophisticated hypothesis on the list. H8 is
more interesting and is unresolved; it is not chosen, because nothing measured
points at it.

## Cheapest discriminating experiment per unresolved row

| Hypothesis | Cheapest next test | Cost |
|---|---|---|
| H2 vs H4′ | already run — the frontier comparison with an out-of-fold beta (`runs/p22`) separates them | done, CPU |
| H4′ magnitude | full-validation confirmation (`runs/p24`), pre-registered | 1 frozen GPU pass |
| H4′ stability | resample the fold partition (`runs/p25`) | CPU, minutes |
| H6 | a linear probe for predicate identity on the frozen pair features, against a prior-conditioned control | CPU on the existing cache |
| H8 | a subject/object-shuffled control: how much of the model term survives destroying the pair identity? | CPU on the existing cache |
| H9 residue | re-derive the README numbers under the current evaluator, or retract them | 1 GPU pass, shares `runs/p24` |
