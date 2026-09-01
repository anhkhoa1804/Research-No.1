# Session — the oracle ceiling, and where the model's information actually goes

Date: 2026-09-01. Branch: `research/architecture-breakthrough`.
Host: `research-no-1`, NVIDIA L4 24 GB, 8 vCPU, 31 GB RAM, 81 GB free.
Environment: Python 3.10.12, torch 2.9.1+cu129, transformers 5.15.1, numpy 2.2.6,
driver 580.173.02, CUDA 13.0.

---

## Question

Where, and how, can model evidence be converted into useful long-tail predicate
decisions — given that the model does not improve global ranking?

## Hypotheses carried in

H1 candidate generation · H2 prior calibration · H4 candidate ranking ·
H6 representation · plus the tie-breaker mechanism from
`docs/CPRIME_MECHANISM_REPORT.md`.

## Protocol

Four stages, each pre-registered before its numbers existed, all on the frozen
C' cache (`runs/p10_model_recalibration/pair_logits.pt`, 3,000 images,
38,053 GT rows, 50 predicates), leak-free train-derived prior, CPU only:

1. oracle tie-breaker ceiling — k ∈ {2,3,5} × budgets {∞, 5.08, 0.93} × tau
2. learned candidate-restricted scorer — 4 arms, 5-fold cross-fit **by image**
3. beta-swept R/mR frontier vs the prior-only tau frontier
4. nested beta selection — beta chosen on inner folds only

Then one GPU pass to extend the cache to the full validation split, against a
criterion committed before launch.

## Measured results

**Stage 1 (`runs/p17`).** The pre-registered gate returned SUCCESS 27/27, and
that is a property of the gate, not of the data: the oracle never decides
wrongly, so `oracle_R >= prior_R` identically and the floor cannot bind. Its
"22-point headroom" at k=5 is 89.7% candidate coverage restated. **H1 falsified.**
The realizable arm — raw model score as ranker inside the candidates — is
EXHAUSTED 27/27, costing up to −15.6 R points, and the additive arm (+0.673)
already beats the best pure reranking (+0.423).

Also falsified: this project's own "the model restores recall that tau spent".
ΔR *falls* across tau ∈ [0, 0.1] (+0.673 → +0.205) and rises only at tau ≥ 0.2
where the prior has already lost 4.8–55.0 R points.

**Stage 2 (`runs/p18`).** EXHAUSTED 4/4 on the pre-registered ΔR criterion.
`model_only` is negative in every cell — the representation cannot rank alone
(**H6**). But `full − prior_only` is +0.68 to +0.79, stable across all four
cells: a cross-fitted, null-controlled measurement of the model's genuine
complementary contribution, which C' itself never had.

**Stage 3 (`runs/p19`, `p21`).** Plain CE and class-balanced CE are two ends of
one scalar: mR@50 19.9 ↔ 42.2 while R@50 goes 67.7 ↔ 28.2. So ΔR alone measures
one end of a dial. The right comparison is between frontiers, since only one of
them has seen an image.

**Stage 4 (`runs/p22`), the load-bearing result.** With beta chosen inside the
training folds only — no constant anywhere selected by looking at the answer:

| arm | R@50 | mR@50 | Pareto gap | floor 66.5 |
|---|---|---|---|---|
| achieved additive C' | 67.474 | 22.837 | +0.861 | ok |
| prior_only (no vision) | 65.651 | 26.354 | +0.059 | FAIL |
| shuffled-model null | 65.495 | 26.608 | +0.224 | FAIL |
| **full** | **66.673** | **25.161** | **+2.894** | **ok** |

The two arms with no model information land *on* the tau frontier. That is what
"everything is calibration" looks like, and it is now measured. The model-bearing
arm is the only one clearing the floor, and the only one off the frontier.

## Bugs found

1. **`topk` vs `argmax` tie-break** (`7d4a658`, carried in from the prior
   session) — 522 tied rows; shifted every oracle baseline and gap.
2. **Non-falsifiable pre-registered gate** (`a75aa03`) — floor applied to the one
   arm that cannot lose recall.
3. **Candidate-0 overwrite** (`889483e`) — duplicated a column on tied rows,
   shrinking the candidate set to k−1; coverage read 85.2% vs the true 85.5%.
4. **Unspecified tie at the k-th candidate slot** (`889483e`) — two independently
   written tools disagreed on 4 rows at k=2. Fixed once in `canonical_topk`.
5. **Pareto gap scaled ×100** — the achieved arm read +86.079 instead of +0.861.
   Caught because +0.861 is exactly its ΔmR at the frontier's lower-bound branch.
6. **`min(25, eval_batches)` destroys the "0 = whole split" sentinel**
   (`9ab094d`) — would have made the full-validation pass do a second full pass.
   Found while costing that run, before paying for it.

## Corrections to protocol

- The oracle floor now also binds on the realizable arm; both verdicts reported.
- The null condition was pre-registered against the wrong baseline
  (`null > achieved` rather than `null > prior_only`) and fired for the wrong
  reason. Corrected; both computed and stored in the artifact.

Neither moved a pre-registered threshold.

## Decision

**No reranker. No architecture scaling.** The pre-registered criteria closed
both. The live direction is the *decision formulation*: a candidate-restricted,
class-reweighted rule converts ~3.4× more of the model's small complementary
signal into Pareto movement than the additive alpha/tau composition, at the same
R@50 floor.

Screening only. Not a headline.

## Next experiment

Full-validation confirmation (`runs/p24`), pre-registered in
`docs/FULL_VALIDATION_PREREGISTRATION.md` before launch, with CONFIRMED /
WEAKENED / REFUTED all committed in advance and REFUTED committed to being
reported as prominently as CONFIRMED.
