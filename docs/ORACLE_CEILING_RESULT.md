# Result — the oracle tie-breaker ceiling

Pre-registration: commit `ed6113b` (criteria as constants in the tool, before any number)
Runs: `runs/p17_oracle_ceiling_canonical` (authoritative)
Superseded: `runs/p13` (pre-fix tie-break), `runs/p14`, `runs/p15` (pre-canonical ordering)
Compute: CPU only. The C' cache is read-only. No GPU.

Classification of every claim below: **MEASURED RESULT** unless marked otherwise.

---

## 1. The headline is about the criterion, not about the ceiling

The pre-registered gate returned **SUCCESS in 27 of 27 cells**. That is not
evidence of headroom. It is evidence that the gate cannot fail.

**VERIFIED FACT**, and the reason is structural rather than a coding slip:

> The oracle arm substitutes GT only on rows where GT is already inside the
> prior's top-k. It therefore never makes a decision that is wrong, so
> `oracle_R >= prior_R` holds identically. The pre-registered R@50 floor is
> consequently satisfied in every cell whenever the *prior* clears it, and
> `gap_oracle_minus_achieved` grows with k only because P(GT in prior top-k)
> grows with k.

So the oracle arm measures **candidate coverage** — a property of the prior —
and carries no information about any scorer that would have to choose among
those candidates. Its 22-point "headroom" at k=5 is 89.7% coverage restated.

The floor was pre-registered precisely to stop a degenerate operating point from
masquerading as headroom. It was applied to the one arm in the ladder that
cannot be degenerate. The corrected gate binds it on `model_rerank`, the only
realizable arm, and both verdicts are now reported side by side.

`tests/test_cprime_oracle_ceiling.py` pins the structural property rather than
the verdict, so "SUCCESS in 27/27" cannot be re-read as a result later.

## 2. What the realizable arm measures

`model_rerank` = argmax of the model score inside the prior's top-k, on rows
inside the margin budget. It is the alpha → ∞ limit of the restricted arm.

At tau=0, 38,053 GT rows over 3,000 images, prior baseline R@50 66.802:

| k | budget | eligible | GT-in-top-k | rerank R@50 | ΔR | floor 66.5 | oracle R@50 | PREREG | REALIZABLE |
|---|---|---|---|---|---|---|---|---|---|
| 2 | unrestricted | 100.0% | 80.6% | 58.997 | −7.805 | **FAIL** | 80.609 | SUCCESS | EXHAUSTED |
| 2 | median 5.08 | 50.1% | 72.4% | 64.807 | −1.995 | **FAIL** | 77.261 | SUCCESS | EXHAUSTED |
| 2 | decile1 0.93 | 10.0% | 67.2% | 67.225 | **+0.423** | ok | 69.937 | SUCCESS | EXHAUSTED |
| 3 | unrestricted | 100.0% | 85.5% | 55.233 | −11.568 | **FAIL** | 85.497 | SUCCESS | EXHAUSTED |
| 3 | median 5.08 | 50.1% | 78.9% | 63.572 | −3.230 | **FAIL** | 80.517 | SUCCESS | EXHAUSTED |
| 3 | decile1 0.93 | 10.0% | 75.8% | 67.075 | **+0.273** | ok | 70.804 | SUCCESS | EXHAUSTED |
| 5 | unrestricted | 100.0% | 89.7% | 51.205 | −15.597 | **FAIL** | 89.699 | SUCCESS | EXHAUSTED |
| 5 | median 5.08 | 50.1% | 84.5% | 61.635 | −5.166 | **FAIL** | 83.323 | SUCCESS | EXHAUSTED |
| 5 | decile1 0.93 | 10.0% | 83.6% | 66.765 | −0.037 | ok | 71.584 | SUCCESS | EXHAUSTED |

Achieved additive arm (the real C' arm, alpha=3.75): R@50 67.474, ΔR **+0.673**.

Three readings, all MEASURED:

1. **The raw model score is not a ranker.** Allowed to decide inside the
   candidate set, it destroys recall in every cell that reaches more than 10% of
   rows — up to −15.6 points. Only inside the tightest margin budget, where it
   touches 10% of rows, does it become mildly positive.
2. **The additive composition already beats the best pure reranking.** +0.673
   against a best-cell +0.423. Composition is not the thing that is failing.
3. **Candidate generation is not the bottleneck.** GT is inside the prior's
   top-5 for 89.7% of rows. **H1 is falsified.**

## 3. Falsification of this session's own prior framing

`docs/CPRIME_MECHANISM_REPORT.md` described the model as "restoring recall that
tau spent". If that were causal, the model's ΔR would *grow* with tau, because
there would be more to restore.

| tau | 0.0 | 0.01 | 0.025 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| model ΔR | +0.673 | +0.854 | +0.759 | +0.612 | +0.205 | +1.222 | +2.276 | +3.958 | +4.983 |

Across the operating region (tau ≤ 0.1) ΔR **falls**, from +0.673 to +0.205. It
grows only at tau ≥ 0.2, where the prior has already lost 4.8 to 55.0 R points
and no one would operate.

**The "restoring" framing is falsified in the operating region.** What grows
with tau at tau ≥ 0.2 is the model's ability to rescue a prior that has been
destroyed, which is a different and uninteresting claim.

## 4. Decision

**EXHAUSTED on the realizable arm, 27/27 cells.** No reranker is justified by
this table, and no GPU time is bought by it.

This did **not** close the branch, for a reason stated in the same commit:
`model_rerank` bounds the *raw score*, not the *learnable capacity* of the
cached features. That is the pre-registered follow-up
(`docs/CANDIDATE_SCORER_PREREGISTRATION.md`), and it changed the conclusion —
see `docs/CANDIDATE_SCORER_RESULT.md`.

## 5. Bugs found and fixed in this stage

1. **`topk` vs `argmax` tie-break** (`7d4a658`). 522 GT rows tie for the prior's
   maximum. `argmax` is the evaluator's convention; `topk(2).indices[:,0]` is
   not. Moved the prior baseline 66.8016 → 66.7543 and shifted every oracle gap.
   Caught by a disagreement inside a single run's own output.
2. **Non-falsifiable gate** (`a75aa03`). Documented in §1.
3. **Unspecified tie at the k-th candidate slot** (`889483e`). `torch.topk`
   leaves it unordered; the oracle tool and the scorer probe disagreed on 4 GT
   rows at k=2 and 2 rows at k=3. Fixed once in `Mech.canonical_topk` — stable
   argsort on the negated score, a total order — and shared by both tools.
   Blast radius ≤ 1.1e-4 in coverage; runs re-executed rather than patched.
