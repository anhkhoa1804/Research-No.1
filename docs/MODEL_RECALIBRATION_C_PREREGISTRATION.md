# Pre-registration — Experiment C′: does the trained model carry information the prior does not?

Registered **before** the GPU pass. Committed before `runs/p10_model_recalibration/`
contains any measurement.

Supersedes the *design* of the never-run "experiment C" sketched in
`docs/research_sessions/POST_B_SCIENTIFIC_REASSESSMENT.md` §H. That sketch is
preserved unchanged; §1.2 below states why its criterion was inadmissible.

**Nothing about experiments A, B or D is modified by this document.**

---

## 1. Question

> **Does the trained model contain complementary predicate information beyond the
> long-tail decision frontier already achievable by the leak-free frequency prior?**

"Complementary" is the operative word. The question is **not** whether
model + prior scores higher than prior alone on some metric — a zero-information
scalar already does that (`runs/p7_prior_temperature_sweep/`: +4.03 mR@50). The
question is whether the model supplies information the prior does not already
contain.

### 1.1 Why this is the right experiment now

- Experiment D closed branch A: frozen-CLIP appearance through a linear probe
  adds nothing beyond calibration at any capacity (`runs/p9_appearance_tau_interaction/`).
- The remaining leading candidate is branch B — the decision rule / composition.
- The trained model is neither frozen, nor linear in CLIP features, nor
  restricted to the prior's top-5. It is the one untested source of visual
  information, and it is the **last of the two falsifiable conditions**
  `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §9 set for the architecture.

### 1.2 Why the inherited criterion is rejected

The earlier sketch used `Δ mR@50 ≥ +3.0`. τ = 0.05 clears that bar with
**negative** information content and an accuracy loss. Any criterion satisfiable
by an information-free transformation cannot be evidence of information. It is
replaced by §4.

---

## 2. Source trace — where model signal can be suppressed

Verified by reading `openvocab_rel/evals.py` at this commit, not from
documentation.

**MODEL.** `_relation_predicate_logits`, mode `ensemble`:

```
text_logits = out.text_predicate_logits(rel_feat, pred_emb)                 # CLIP-text branch
cls_logits  = out.calibrated_predicate_logits(rel_feat, pred_log_prior)     # trained classifier
cls_logits  = cls_logits - tau_adj * log P(p)         # _apply_eval_logit_adjustment; inert here
norm(x)     = (x - mean(x)) / max(std(x), 1e-4)                             # per row, over P
score_m     = a * norm(cls_logits)/T_cls  +  (1 - a) * norm(text_logits)/T_text
              where a = eval_sgg_predicate_ensemble_alpha
```

**PRIOR.** `_frequency_bias_for_pairs` → a row per pair, with fallback
`pairs[s||o]` → `0.5*(subjects[s] + objects[o])` → `subjects[s]` → `objects[o]`
→ `global`:

```
score_p(p | s, o) = log P(p | s, o)
```

**TAU.** `_apply_freq_bias_tau` (added for C′, `b8ce06b`; τ = 0 returns the
tensor by identity):

```
score_p_tau(p | s, o) = log P(p | s, o)  -  tau * log P(p)
```

**COMBINED.** `_apply_frequency_bias`, then masking and ranking:

```
score_combined = score_m + alpha * score_p_tau            alpha = freq_bias_alpha = 3.75
score_combined[:, background] = -1e4                      _mask_background_logits
probs          = softmax(score_combined)
prediction     = argmax over the 50 foreground columns
```

### 2.1 The four suppression points, in order of severity

| # | Where | Effect |
|---|---|---|
| **S-1** | `a = eval_sgg_predicate_ensemble_alpha = 0.0` | `score_m` **= the text branch alone**. The trained predicate classifier, adaptive calibration included, is multiplied by exactly zero and discarded. This is the historical protocol. |
| **S-2** | `norm(·)` per-row standardisation | whatever survives is rescaled to zero mean / unit std per pair, destroying any calibrated magnitude; only the within-row *shape* reaches the sum |
| **S-3** | `alpha = 3.75` against a unit-variance model term | the prior term is ~3.75× the model term's scale. `runs/p6_prior_dominance_margin/` measures ~2/3 of the prior's errors as unreachable by the visual term at this α. |
| **S-4** | `_apply_eval_logit_adjustment` | applied to `cls_logits`, which S-1 has already annihilated. Provably inert (`runs/p8_tau_path_bug/`). |

**S-1 is the reason C′ dumps both branches.** A null on the composed term at
`a = 0.0` would say nothing about the model's classifier, only about CLIP text
similarity. Distinguishing "the model has no information" from "the protocol
discards the model's information" requires the discarded tensor, and getting it
without a second GPU pass requires caching it. See
`runs/p10_model_recalibration/cache_schema.md` §2.

### 2.2 Contrast with the experiment-D (frozen CLIP) path

| | D (`tools/appearance_probe.py`) | C′ |
|---|---|---|
| features | frozen CLIP crops, PCA'd | trained model's `rel_feat` |
| head | linear, fit on 80 % of 1200 train images | trained end-to-end, historical checkpoint |
| candidates | prior top-5 | all 50 |
| composition | `prior + λ·appearance` | `score_m + α·score_p_tau` |
| prior | leak-free train-derived | leak-free train-derived (same file) |

D and C′ therefore differ in representation, head, candidate set and
composition. A null in D does not predict a null in C′, and vice versa.

---

## 3. Design

**One GPU pass**, dumping `pair_logit_dump_v2` (schema doc committed with this
registration), then every arm below on CPU.

| | |
|---|---|
| **CONTROL** | prior-only τ frontier, `score_p_tau` alone, scored through the identical ranking/masking code as every other arm |
| **VARIABLE** | presence of the model term, and which branch of it; crossed with τ and α |
| **METRIC** | Pareto gap: arm mR@50 minus the prior-only τ frontier's mR@50 linearly interpolated **at matched R@50**, in mR points |
| **NULL** | see §5 |
| **DENOMINATOR** | one predicate vocabulary for every arm — see §6 |

### Arms

- **A.** prior-only, τ ∈ {0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}
- **B.** model + prior, same τ grid, α = 3.75 (the historical value)
- **B′.** classifier-branch + prior, same τ grid — the branch S-1 discards
- **C.** model-only (α = 0), text branch and classifier branch separately
- **D.** model + prior over α ∈ {0, 0.5, 1.0, 2.0, 3.75, 7.5}, τ held at 0 and at the
  frontier-matched value. α is a **protocol constant** being varied to bound
  S-3, not a hyperparameter being tuned; the whole grid is reported.

### Held fixed

Checkpoint, dataset split and membership, image set (the same first 3,000
validation images as `runs/p5_model_vs_leakfree_prior/`), GT-pair construction,
background masking, ranking code, and the leak-free train-derived prior file.

---

## 4. Primary criterion — REGISTERED BEFORE ANY RESULT IS READ

A model contribution counts as evidence of **complementary information** only if
**all five** hold:

1. **Pareto.** model + prior yields a **positive** Pareto gap over the prior-only
   τ frontier at matched R@50.
2. **Null margin.** the gap exceeds the matched null's mean by **≥ 2 SD** of that
   null (§5).
3. **Strata reproducibility.** the gain is positive in **≥ 2 of 3** frequency
   strata (head / body / tail) and is not carried by a single predicate — no
   single predicate may account for **> 50 %** of the total mR gain.
4. **Denominator invariance.** the sign of the gap is unchanged under **both**
   predicate schemes (raw-50 and eval-48, §6).
5. **No selection leak.** τ and α are **not** selected on validation. Selection,
   where required, is on a held-out-from-train split; the full grid is reported
   regardless, so any cherry-picked upper bound is visible next to the
   honestly-selected point.

Failing any of 1–5 ⇒ **not** evidence of complementary information.

### Secondary evidence (reported always, decisive never on its own)

prior top-5 coverage; model rescue of prior top-5 misses; model destruction of
prior top-5 hits; GT rank under prior / model / model+prior; head/body/tail;
prior-entropy strata; wrong→right vs right→wrong flip accounting; model-only
performance.

---

## 5. The null — defined before results

Two nulls, both passing through the **identical** pipeline, denominator, τ/α
grid, ranking and selection as the real arm. A fitted real arm is never compared
against an unfitted null.

- **N1 — pair-shuffled model.** Within each image, permute the model's per-pair
  logit rows across pairs. Preserves the model's marginal output distribution and
  its per-row shape exactly; destroys only the correspondence between a pair and
  its score. This is the matched null for "does the model's score depend on
  *which* pair it is scoring".
- **N2 — image-shuffled model.** Permute model rows across the whole split
  (subject to equal pair counts where possible). Destroys pair *and* image
  correspondence; bounds N1 from the other side.

5 seeds each. The null is **refit/rescored under every τ and α** the real arm
uses, and its Pareto gap is computed against the same prior-only frontier.

Registered expectation, from D: **the null's mean Pareto gap will not be zero.**
Adding any small perturbation to a peaked prior nudges mR at matched R@50 (D
measured +0.43 ± 0.17 for shuffled appearance). A test assuming a zero null
would read noise as signal.

---

## 6. Denominator unification (blocker 2)

**Traced at source.** `_build_vg_aliases` is predominantly an *object*-vocabulary
normaliser (`man/woman/boy/girl→person`, `bike→bicycle`, `sofa→couch`, …). Its
predicate entries are morphological/free-text variants
(`wear`/`wears`/`wearing`, `ride`/`riding`, `upon`/`on top of`/`on`) — the
signature of an open-vocabulary text-matching helper. Applied to closed-vocabulary
VG150 PredCls it fires on exactly **two** of the 50 predicates:

```
near  -> next to
wears -> wearing
```

giving **48** classes. Measured, not asserted:
`{'near': 'next to', 'wears': 'wearing'}`, 50 → 48.

**Classification: accidental implementation artifact** in the closed-vocabulary
path.
- Not a *historical protocol requirement*: `eval_sgg_use_vg_aliases` is **not**
  among the 14 flags the historical canary checks
  (`docs/research_sessions/POST_B_CLAIM_AUDIT.md` 4.2), so the historical
  protocol never pinned it.
- Not a mere *convenience*: it changes reported mR@50 by ~0.7–0.9 points and
  silently departs from the literature-standard 50-class VG150 denominator.

**Action, per the standing constraint not to alter the historical reproduction:**

- `eval_sgg_use_vg_aliases` keeps its default (`True`). The historical evaluator
  is **untouched**; pinned by the existing suite (300 passed).
- The C′ comparison is **entirely CPU-side** and applies **one** scheme to **all**
  arms. Primary: **raw-50**, the literature-standard VG150 denominator.
  Secondary robustness column: **eval-48**, the evaluator's own scheme.
- This is possible only because the cache stores **raw** GT plus the alias map;
  the collapse is lossy and one-directional.
- Regression tests: `tests/test_pair_logit_dump.py` (raw GT survives, alias map
  reproduces 48, ordering preserved, redundancy checked) and
  `tests/test_cprime_analysis.py` (identical predicate sets and identical mR
  denominator across arms).

---

## 7. Information tests — registered questions

| | Question | Instrument |
|---|---|---|
| Q1 | Does the model help when the prior is already correct? | flip accounting on prior-correct pairs |
| Q2 | Does the model rescue cases where the prior is wrong? | rescue rate on prior-top-1-wrong pairs |
| Q3 | Does the model improve ranking inside the prior's top-5? | GT rank within the prior's top-5, prior vs model+prior |
| Q4 | Is the model useful mainly at high prior uncertainty? | Pareto gap within prior-entropy terciles |
| Q5 | Does the model improve tail predicates? | head/body/tail mR at matched R@50 |
| Q6 | Pareto improvement, or only movement along the τ frontier? | the primary criterion |
| Q7 | Does model information survive the ranking/normalisation procedure? | arm B′ (discarded classifier branch) vs arm B |

## 8. Decision — exactly one, from the complete evidence

- **A. COMPLEMENTARY INFORMATION CONFIRMED** — criterion §4 met.
- **B. COMPLEMENTARY INFORMATION EXISTS BUT THE CURRENT DECISION RULE DESTROYS IT**
  — §4 fails for arm B, but arm B′ or a different α meets it.
- **C. MODEL INFORMATION IS INDISTINGUISHABLE FROM THE PRIOR/NULL** — §4 fails
  for every arm, including B′ and every α.
- **D. PROTOCOL STILL DOES NOT PERMIT A CLEAN CONCLUSION** — a validation gate
  fails, or the strata are too sparse to evaluate §4.3.

Registered prediction (recorded so it can be wrong): **C or B**, because the
model is already Pareto-dominated at τ = 0 and pushes toward head predicates
while τ pushes toward the tail. Prediction is not measurement.

## 9. Cost and resource policy

- **One** GPU pass. Measured baseline: `runs/p5_model_vs_leakfree_prior/` did the
  identical 3,000-image forward at bs=12 in **3,582 s**. The dump adds a second
  head evaluation on cached `rel_feat` plus disk writes.
- Everything after that is CPU. Repeated full model passes are forbidden.
- One L4; exactly one GPU experiment at a time. `nvidia-smi` checked immediately
  before launch.

## 10. Post-C′ constraint

If C′ is null, the deliverable is a **design memo only**.
**No architecture is to be implemented or trained** on the strength of this
result.
