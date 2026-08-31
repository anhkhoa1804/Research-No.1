# Post-B: what τ actually buys, and whether experiment C is still necessary

CPU-only, read-only. No GPU experiment was launched, no architecture changed,
no registered probe touched, no historical artifact or dataset modified, and
no earlier result report rewritten.

Mechanism, source trace and equations: **`docs/PRIOR_TEMPERATURE_FORENSICS.md`**.
This document covers the comparison to appearance (Phase 5) and the decision on
experiment C (Phase 6).

---

## 1. The result in one table

MEASURED this session, all on the A′ 3,000-image subset (38,053 GT triplets,
all 50 predicates present):

| | what it is | ΔR@50 | ΔmR@50 |
|---|---|---:|---:|
| τ = 0.1 recalibration | one scalar, **no vision, no training** | **−0.64** | **+4.03** |
| net top-1 decisions it gets right | | **−245 of 38,053** | |
| frozen ViT-L/14-336 appearance (experiment B) | 304M-param encoder | −0.52 | −0.51 |

**τ improves the class-averaged metric by 4 points while getting 245 more
answers wrong.** That single sentence contains the whole finding, and
everything below is the accounting behind it.

---

## 2. Why a scalar beats CLIP: four measured reasons

### 2.1 They act on different terms, and only one of them has leverage

The score is `log P(p|s,o)` composed with something. τ modifies the
**class-frequency term** `−τ·log P(p)`. Appearance modifies the
**pair-conditional term**. `mR@50` is an unweighted mean over 50 classes, so:

```
gain:  +2.1318 summed class-recall  /  376 instances  =  +0.005670 / instance
cost:  −0.1178 summed class-recall  /  621 instances  =  −0.000190 / instance
                                        leverage ratio = 29.9×
```

The 33 classes τ improves hold 11,971 GT instances (mean 362.8 each); the 6 it
degrades hold 25,286 (mean 4,214.3). **A hit moved into a rare class is worth
~30× what a hit lost from a common class costs.** Any transformation that
systematically redistributes toward rare classes wins this metric. A
transformation that improves accuracy uniformly does not.

Appearance does not redistribute by class frequency — it re-ranks within the
pair-conditional distribution. It is playing on the axis with no leverage.

### 2.2 τ adds no information; appearance adds some, but not enough

| | adds information? | evidence |
|---|---|---|
| τ = 0.1 | **No — provably** | net top-1 accuracy **falls** by 245 pairs (−0.644 pp). An information-adding transformation raises accuracy. |
| ViT-L/14-336 appearance | **Yes** | ablation gate passes: real 16.90 > shuffled 12.17 > zero 8.58 mR@50 |

But information is not what this metric rewards most. Appearance's standalone
ranking (16.90 mR@50) is **worse than the prior's own** (20.98), so adding it
at any positive weight in the pre-registered λ grid degrades the composition —
while τ, adding nothing, gains 4 points.

### 2.3 In the probe's own pre-registered units, τ clears the bar B failed

This is the sharpest comparison available, because it uses **experiment B's own
primary metric on the same oracle@5 denominator** rather than two different
tables. MEASURED, `runs/p7_tau_vs_oracle_headroom/`:

On the A′ subset: coverage@5 = 89.70 %, prior mR@50 = 21.98 %, oracle@5 mR@50 =
64.35 %, headroom = **42.38 mR points**.

| arm | captured headroom |
|---|---:|
| **experiment B** — frozen ViT-L/14-336, selected λ | **−1.2 %** |
| τ = 0.05 | **+5.3 %** |
| **τ = 0.10** | **+9.5 %** |
| τ = 0.20 | +15.2 % |
| τ = 0.50 | +38.2 % |

**A single no-information scalar captures 9.5 % of the exact quantity a 304M-
parameter frozen encoder captured −1.2 % of.** And τ = 0.05 alone clears the
pre-registered **≥ 5 %** threshold that experiment B was designed to test.

Two conclusions, kept separate because they point in opposite directions:

- **B's negative is strengthened.** Appearance did not even match what a free
  scalar achieves on the same denominator.
- **B's threshold was necessary but not sufficient.** Because a
  zero-information transformation clears 5 %, *passing* that threshold would
  never have demonstrated visual information. This is a defect in the
  criterion, not in the execution — and it is a defect that only became
  visible once both experiments existed. See §5.

### 2.4 They help the same predicates — and τ mostly helps them more

τ's gains land on precisely the action/pose predicates appearance was supposed
to own.

⚠ **Not a controlled comparison.** The appearance column is from 1,200
validation images, the τ column from 3,000; the class counts differ, so this
table is *indicative of direction*, not a matched measurement.

| predicate | appearance Δ (n) | τ = 0.1 Δ (n) |
|---|---:|---:|
| `walking on` | +4.3 (23) | **+26.8** (56) |
| `riding` | +8.2 (49) | **+27.1** (140) |
| `eating` | +18.2 (22) | **+22.8** (57) |
| `standing on` | +6.2 (81) | +5.0 (238) |
| `from` | +4.3 (23) | +4.5 (66) |
| `with` | +4.1 (414) | +2.7 (1,127) |
| `under` | +2.2 (185) | +1.9 (474) |
| `in front of` | +1.5 (136) | +1.4 (360) |
| `holding` | **+4.9** (308) | +0.3 (712) |
| `behind` | **+0.3** (356) | −3.8 (885) |

Appearance wins on `holding` and `behind`; τ wins overwhelmingly on
`riding`, `walking on` and `eating`. **INFERENCE:** those three were never
recognition failures. `runs/p7_tau_vs_oracle_headroom/` shows the truth already
sits inside the prior's top-5 for 82.1 % of `riding` and 71.4 % of `walking on`
GT pairs, while the prior's top-1 recall on them is 18.6 % and 0.0 %. The
answer was already in the prior; only the decision rule was hiding it, and a
scalar surfaces it without looking at a single pixel.

Across the split, **8,715 GT pairs (22.90 %)** have the truth inside the
prior's top-5 but not its top-1. That is the pool both interventions are
competing for.

### 2.5 Is mR@50 unusually sensitive to calibration? **Yes, and measurably**

| | value |
|---|---:|
| pairs whose argmax changed (τ=0 → 0.1) | 1,584 (4.16 %) |
| pairs that changed *correctness* | **997 (2.62 %)** |
| resulting ΔR@50 (pooled accuracy) | **−0.64** |
| resulting ΔmR@50 | **+4.03** |
| ratio ΔmR / ΔR | **−6.3×** |

2.6 % of decisions move the class-averaged metric 6.3× further than they move
pooled accuracy, **in the opposite direction**. mR@50 under a 45 %-head
distribution is a calibration-sensitive metric far more than an
information-sensitive one.

Supporting control (`docs/PRIOR_TEMPERATURE_FORENSICS.md` §1.4): changing the
softmax temperature α from 3.75 to 1.0 alters mean entropy 4.6× and moves
R@50 and mR@50 by **exactly zero**. Only the argmax matters — because top-K is
provably inert here (no image has more than 48 candidates against K=50), so
`mR@50` *is* class-averaged top-1 accuracy.

---

## 3. The model is Pareto-dominated by the zero-parameter curve

The τ sweep traces the entire achievable (R@50, mR@50) trade-off **without any
model**. Placing the model on it — both sides on `evals.py`'s own 48-class
alias scheme, so the comparison is like-for-like
(`runs/p7_tau_curve_eval_scheme/`):

| arm | params | vision | R@50 | mR@50 |
|---|---:|:---:|---:|---:|
| **model + historical prior** | **79.9M** | **yes** | **66.90** | **21.16** |
| prior, τ = 0.00 | 0 | no | **67.93** | **22.68** |
| prior, τ = 0.05 | 1 | no | **67.61** | **25.00** |
| prior, τ = 0.10 | 1 | no | **67.31** | **26.86** |
| prior, τ = 0.15 | 1 | no | 64.81 | 28.22 |
| prior, τ = 0.20 | 1 | no | 63.12 | 29.36 |
| prior, τ = 0.30 | 1 | no | 57.84 | 34.60 |

**The model is strictly dominated on both axes by the prior at τ = 0, τ = 0.05
and τ = 0.10.** It is not on the frontier; it is inside it.

Two caveats, both working *against* the model, so the domination is
conservative: the model arm uses the historical prior, which the manifest says
cannot be verified leak-free (any leak inflates the model), and the τ curve
uses the leak-free train-derived prior.

---

## 4. Limitations of this analysis

1. Single split, single 3,000-image subset. The full-split τ result
   (+4.11 mR@50) is consistent but separately measured.
2. The τ curve uses the train-derived prior; the model arm uses the historical
   prior. Not a matched ablation — see §3's caveat on direction.
3. §2.4's predicate table compares two different subsets and is indicative
   only.
4. Rare-class deltas (n < 25) are noisy; the aggregate accounting does not rely
   on them.
5. Nothing here measures how τ interacts with the model's visual term. That
   has never been run — see §6.
6. `mR@50 = class-averaged top-1 accuracy` holds for **this** protocol (GT
   pairs, ≤48 candidates/image, one predicate per pair). It fails under
   `predcls_multi` and under SGDet.

---

## 5. A metric finding that outranks the experiment

Stated plainly because it changes how every future arm must be judged:

> **On this protocol, mR@50 can be moved +4.03 points by a transformation that
> adds no information and reduces accuracy. The τ sweep is therefore not a
> baseline — it is the achievable frontier that any informative arm must beat
> on *both* axes.**

Consequences:

- Reporting an mR@50 gain without its R@50 cost is uninformative here.
- The appearance probe's "≥ 5 % of oracle headroom" criterion is clearable by
  τ = 0.05. It is a **necessary but not sufficient** test for visual
  information. B failing it remains decisive; B passing it would not have been.
- Any future arm must be placed on the (R@50, mR@50) plane against the τ curve
  of §3, not against a single τ point.

---

## 6. Phase 6 decision — is experiment C necessary?

**Experiment C as currently conceived:** "model under recalibrated
train-derived prior", i.e. model + τ-adjusted prior versus τ-adjusted prior
alone.

### Verdict: **C — necessary, but it needs a modified control.** With **D** as a co-finding.

Not (A) *necessary and decisive as designed*, and not (B) *redundant*. Three
findings force the redesign.

#### 6.1 C cannot be run through the existing flag — the path is provably inert

VERIFIED by source reading (`docs/PRIOR_TEMPERATURE_FORENSICS.md` §1.6):

- `_apply_eval_logit_adjustment` (`evals.py:1074-1080`) is called at
  `evals.py:1256` on **`cls_logits` only**;
- the historical protocol sets `eval_sgg_predicate_ensemble_alpha = 0.0`, so
  `evals.py:1273` returns `(0.0 · cls_norm) + (1.0 · text_norm)` — the adjusted
  tensor is multiplied by **exactly zero**;
- and `eval_logit_adj_tau = −1.0` falls back to `logit_adj_tau = 0.0` anyway,
  firing the `τ ≤ 0` early return.

**So `--eval_logit_adj_tau 0.1` on the historical protocol yields results
byte-identical to omitting it.** A naive run of C would burn ~60 GPU-minutes,
return exactly the τ=0 numbers, and invite the conclusion "recalibration does
not help the model" — which would be an artifact of a dead code path, not a
measurement. This is the fifth silent-null defect found in this repository and
would have been the most expensive.

The faithful implementation subtracts `τ · log P(p)` from the **frequency-bias
rows** before `_apply_frequency_bias` (`evals.py:1178-1193`) — i.e. the exact
analogue of `decision_rule_probe.py:116`. That is a change to evaluation
semantics and must be pre-registered, not slipped in.

#### 6.2 C's success criterion is inadmissible as stated

The pre-registration draft in `POST_B_SCIENTIFIC_REASSESSMENT.md` §H used
`Δ mR@50 ≥ +3.0` (inherited from the A′ registration). §5 shows that criterion
can be satisfied by class rebalancing with **negative** information content.
The criterion must become **Pareto**: the model + τ point must lie strictly
outside the τ curve of §3 — better on R@50 at matched mR@50, or better on
mR@50 at matched R@50.

#### 6.3 The predicted outcome is not the reason to redesign, but it is relevant

The model pushes toward head predicates (measured: `pose` predicted at 0.41× GT
frequency, `contact` 0.58×, `possession` 1.10×); τ pushes toward the tail.
Composed, they partially cancel. The model is already Pareto-dominated at τ=0
(§3). A null or negative C is strongly predicted — but prediction is not
measurement, and C remains the last of the two falsifiable conditions
`PHASE4` §9 set for the architecture. It should be run, once, correctly.

### The recommended replacement: C′

**One GPU pass that caches the model's per-pair pre-composition logits**, then
every downstream question on CPU:

- τ sweep on the model arm — the actual C question, at every τ, not one point;
- α sweep — how much of the model's contribution the 3.75 weighting suppresses
  (`runs/p6_prior_dominance_margin/` says ~2/3 of prior errors are unreachable
  at α=3.75);
- Pareto placement against the τ curve;
- the appearance-probe question asked of the *real model* instead of frozen
  CLIP: does the model's ranking beat the prior's *within* the prior's top-5?

C′ costs the **same single GPU pass** as C and answers the whole family instead
of one point. It also removes the need to choose a τ implementation before
knowing what the logits look like.

### What must be fixed before C′ runs

| # | Requirement | Why |
|---|---|---|
| 1 | Pre-register the τ-on-frequency-bias implementation | §6.1 — the existing path is inert; this is an evaluation-semantics change |
| 2 | Pareto success criterion against the τ curve | §6.2 — mR@50 alone is movable without information |
| 3 | One alias scheme for both arms; report the mR class count | `POST_B_SOURCE_FORENSICS.md` §3 |
| 4 | Add `eval_sgg_use_vg_aliases` to the canary's checked flags | same |
| 5 | Cache per-pair logits | makes every later composition question CPU-only |
| 6 | Regression test pinning that `eval_logit_adj_tau` is inert at `ensemble_alpha = 0` | so §6.1 cannot be rediscovered the expensive way |

**None of this has been launched.** Items 1–6 are a preparation list for a
human decision, not a plan in motion.

---

## 7. Artifacts

| Path | What |
|---|---|
| `runs/p7_prior_temperature_sweep/` | 11-point τ sweep, **unmodified** `decision_rule_probe.py`; reproduces +4.03 exactly |
| `runs/p7_prior_temperature_forensics/` | top-K inertness, flip accounting, leverage, entropy, per-predicate table; validation gate PASS |
| `runs/p7_tau_vs_oracle_headroom/` | τ in the appearance probe's own captured-headroom units; per-predicate top-5 coverage vs top-1 recall |
| `runs/p7_tau_global_control/` | τ without pair-conditioning — does nothing, as predicted |
| `runs/p7_tau_curve_eval_scheme/` | τ curve under both mR schemes, for the Pareto placement |
| `tools/prior_temperature_forensics.py` | new, CPU-only; self-checks against `decision_rule_probe.py` |
| `tools/prior_temperature_vs_oracle.py` | new, CPU-only |
| `tools/decision_rule_probe.py` | **unmodified** |
| `tools/appearance_probe.py` | **unmodified** |

---

## 8. Stopping point

Stop here. No GPU experiment launched, no architecture change, no new
experiment started. The next action requiring a human decision is whether to
pre-register **C′** with the six requirements in §6. That decision is not taken
here.
