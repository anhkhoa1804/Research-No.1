# Phase 4 — scientific reassessment on measured L4 evidence

Everything below was **measured on this machine today** unless explicitly
labelled otherwise. Run artifacts are under `runs/`; each carries
`provenance.json` with git SHA, environment, GPU and artifact hashes.

Epistemic labels are used strictly: **VERIFIED FACT** (checked in code or by
cryptographic identity), **MEASURED** (produced by a run here), **INFERENCE**,
**HYPOTHESIS**, **CLAIM** (asserted elsewhere, not checked here).

---

## 0. The headline

> **A one-parameter recalibration of a co-occurrence lookup table, using no
> image data at all, beats the 79.9M-parameter CLIP-based model's mR@50 on
> the identical evaluation subset — at zero cost in R@50.**

MEASURED, same 240 validation images, identical GT triplet count (3,093):

| arm | R@50 | mR@50 | vision? | params |
|---|---:|---:|:---:|---:|
| train-derived prior, τ=0 | 65.37 | 20.58 | no | 0 |
| **train-derived prior, τ=0.1** | **65.44** | **27.70** | **no** | **1** |
| historical checkpoint + historical prior | 66.28 | 24.40 | yes | 79.9M |
| historical prior, τ=0 | 63.17 | 19.70 | no | 0 |
| historical prior, τ=0.1 | 62.85 | 28.05 | no | 1 |

`runs/p4_decision_rule_sweep/`, `runs/p3c_pilot20/`, `runs/p3c_control_*_240/`.

This is a **control**, not a contribution. Its value is that it removes an
interpretation, not that it wins a benchmark.

---

## 1. Is the historical 67.09 / 22.64 result reproducible?

**Answer: not as a like-for-like claim, and the question is less important
than it looked.**

- VERIFIED FACT: the checkpoint, the historical frequency prior and
  `demo_config.env` are byte-identical to `historical_checkpoint_v1.yaml`
  (SHA256 confirmed, `runs/provenance/machine_provenance_v1.yaml`).
- VERIFIED FACT: the protocol resolves exactly as the manifest specifies. The
  canary passes all 14 configuration checks plus vocabulary size **and
  ordering** (`runs/p3c_historical_canary/canary_verdict.txt`).
- MEASURED (240 images): R@50 66.28 / mR@50 24.40.
- **Full-split run: IN PROGRESS** (`runs/p3c_historical_full_val/`).

But the manifest itself records that the evaluation split, the number of
images, the aggregation, and `FREQ_BIAS_ENABLED` were **never recorded** for
the original run. A match would therefore not confirm reproduction, and a
mismatch is explainable by any of four unknowns. This is bounded, not settled,
and no amount of GPU time on this checkpoint can settle it.

## 2. How much does the checkpoint contribute over the correct prior?

**Answer: on mR@50, nothing that a single scalar cannot supply without vision.**

The earlier claim (`docs/APPEARANCE_PROBE_FINDINGS.md`) was **+0.50 R@50 /
+0.34 mR@50** over a train-derived prior. That comparison is invalid: it
compares a full-split prior measurement against the *self-reported, never
verified* 67.09 / 22.64. It compares a measurement to a claim.

Measured like-for-like on the same 240 images, the model's advantage over the
train-derived prior is **+0.91 R@50 / +3.82 mR@50** — much larger than
documented. But recalibrating that same prior with τ=0.1 yields **+0.06 R@50 /
+7.13 mR@50**, nearly double the model's mR gain, with no visual input.

**INFERENCE, and the central conclusion of this phase:** the model's mR@50
advantage over the prior is not evidence of visual understanding. It is
consistent with the model acting as an implicit recalibrator of the prior's
class distribution — something a scalar does better and for free.

## 3. Is the appearance-probe result reproducible with the real ViT-L/14-336?

**PENDING.** Pre-registered in
`docs/APPEARANCE_PROBE_L14_PREREGISTRATION.md` (thresholds fixed before
running, λ selected on held-out-from-train, 4–6 % declared inconclusive).
The probe was ported to GPU with fp16 batching; three defects were fixed
first, including a hardcoded VERDICT string that printed "signal is REAL but
far too WEAK" regardless of what was measured.

## 4. Is the flat 50-way prediction formulation the bottleneck?

**Answer: partly — but the decision rule is the larger, cheaper part.**

Three measurements, all on this machine:

1. **The information exists.** `runs/p4_predicate_discriminability/`: balanced
   binary discrimination on 8 geometry features reaches AUC 0.878
   (`walking on` vs `on`), 0.865 (`above` vs `on`), 0.850 (`riding` vs `on`).
   10/22 confusions are strongly separable. The two that are not — `part of`
   vs `of` (0.559) and `near` vs `next to` (0.580) — are exactly the
   annotation-style pairs, which is the control that makes the rest credible.

2. **Argmax under the natural prior discards it.** MEASURED: τ=0.1 converts
   +4.11 mR@50 on the full split with a −0.56 R@50 cost. Nothing was learned
   and nothing was seen; only the decision changed.

3. **Most of the apparent headroom is not real.** `runs/p3e_headroom_train_derived/`:
   of 33.41 R@50 points of total headroom, only **8.84 are recoverable**.
   64.2 % of the prior's errors are generic→generic (`on`↔`in`↔`of`), and only
   **12.6 %** of GT triplets carry a decidable predicate.

Point 3 matters for how every prior result should be read: the "0.6 % of
headroom captured" figure uses a denominator that is ~74 % unreachable by
construction. It understates whatever appearance does contribute.

## 5. Are the earlier A1 and candidate-reranking failures robust?

**The measurements are robust; one stated conclusion is not.**

`tools/candidate_reranking_analysis.py` reports 0.0 % headroom captured by 200
pairwise probes. Its own docstring asserts the failure "is not a capacity
artifact" because the reranker and the flat model share features and parameter
shape. That reasoning holds for *capacity* but not for the *decision rule*:
both were trained by cross-entropy on the natural distribution, where the
loss-minimising behaviour under 45 % head dominance is to agree with the
prior. The shared control does not separate those.

The τ sweep is the missing arm, and it is positive. So: reranking failed, but
"the signal cannot be converted" was **too strong a conclusion** from that
evidence.

**Caveat, stated plainly:** none of the earlier runs' artifacts were
transferred to this machine (`runs/` did not exist). Their numbers are
documentation claims here, not checkable measurements. The two I re-ran
(prior baseline, discriminability) both reproduced exactly, which is
reassuring about the rest but does not verify it.

## 6. Any hidden data or protocol issue that changes the interpretation?

**Four found today. Two would have silently corrupted results.**

| # | Issue | Severity |
|---|---|---|
| 1 | `--eval_batches 0` evaluates **zero** batches. The documented full-run entrypoint (`eval_historical_checkpoint.sh --full`, commented "0 = the whole split") would have reported all-zero metrics with **exit code 0**. | **Critical, silent** |
| 2 | `verify_canary.py`'s zero-image guard read top-level keys the evaluator never writes. It **PASSED** the zero-image run. The guard passed its own tests because only the synthetic fixtures used that layout. | **Critical, silent** |
| 3 | `verify_canary.py` asserted the runtime vocabulary was 50. It is 51 (+1 synthetic `relation` background class, documented in `vg150.yaml`). Every correctly-configured run failed the canary. | Blocking, loud |
| 4 | Strict preflight fails on three artifacts. PROVEN to be CRLF-vs-LF only: the manifest's expected digest equals this machine's content rendered with CRLF. | Provenance, benign |

Evidence for #1 is preserved at `runs/p3c_historical_full_val_ZEROBUG/`
(n=0, num_gt=0, all metrics 0.0000, exit 0).

Also relevant to interpretation: under GT-pairs the headline `predcls R@K`
emits **one predicate per pair**, making K inert — which is why R@50 = R@100
in every prior baseline. The literature-comparable `predcls_multi` numbers are
computed but not headlined. MEASURED at 240 images: multi_mR@50 **53.45** vs
headline mR@50 24.40. Any comparison to published VG150 numbers must use the
`multi_` fields.

## 7. Which experiment has the highest information gain per GPU-hour?

Ranked on evidence, after the fact for the first two:

| rank | experiment | GPU cost | status |
|---|---|---|---|
| 1 | **Decision-rule τ sweep** | **0 (CPU, 22 s)** | Done. Overturned the central interpretation. |
| 2 | Predicate discriminability | 0 (CPU, 36 s) | Done. Reproduced exactly. |
| 3 | Appearance probe at ViT-L/14-336 | ~0.5 h | Pre-registered, pending |
| 4 | Full historical eval | ~3.1 h | Running; bounded value (§1) |
| 5 | Any retraining | ≥10 h | **Not justified by current evidence** |

The two highest-value experiments of this session cost **58 seconds of CPU
between them**. That is the finding about method: the expensive run was not
the informative one.

---

## 8. What the strongest defensible claim now is

Not the one in `docs/APPEARANCE_PROBE_FINDINGS.md` §6B. That claim —
"a prior achieves within 0.5 / 0.34 of a 79.9M-parameter model" — rests on
comparing a measurement to an unverified self-report, and is superseded:

> On VG150 PredCls under GT pairs, a pair-conditioned co-occurrence prior with
> a **single** class-balancing parameter (τ=0.1, no image data, no training)
> reaches R@50 66.04 / mR@50 26.42 on the full validation split — **exceeding
> the mR@50 both of the 79.9M-parameter CLIP model measured on a matched
> subset (24.40) and of the historical self-reported claim (22.64), at a cost
> of 0.56 R@50 points.**
>
> Meanwhile 64.2 % of that prior's residual errors are generic→generic
> confusions no visual evidence can resolve, and only 12.6 % of ground-truth
> triplets carry a decidable predicate. The protocol is not merely
> prior-saturated; its headline metric is **recalibration-saturated**.

This is reproducible from committed tools on committed data in under a minute,
which the superseded claim is not.

## 9. What would have to be true for the architecture to be worth pursuing

Falsifiable, and worth stating so it can be checked rather than assumed:

- appearance at ViT-L/14-336 must capture ≥5 % of headroom (§3, pending); **and**
- the model's contribution must survive **after** the prior is recalibrated —
  i.e. model + τ-adjusted prior must beat τ-adjusted prior alone. **This has
  never been measured** and is the single most informative remaining
  experiment. It requires the model's per-pair scores, hence GPU.

If both fail, the defensible contribution is methodological, and the
architecture should be abandoned rather than iterated.
