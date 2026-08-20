# The prior-residual hypothesis

Research design for architecture candidate **A1**. Implementation:
`openvocab_rel/prior_residual.py`. Controls: `tools/frequency_prior_baseline.py`,
`tools/headroom_analysis.py`.

**No novelty is claimed here.** Prior-conditioned residual learning, logit
adjustment and frequency baselines all exist in the literature. This document
states a hypothesis about *this* system and how it will be falsified.

---

## 1. Problem

A pair-conditioned co-occurrence table, built **only from the training
split**, scores this on VG150 validation with no model at all:

| System | R@50 | mR@50 |
|---|---:|---:|
| Train-derived prior alone (`frequency_prior_train.json`) | **66.59 %** | **22.30 %** |
| Historical model + prior (single-source claim) | 67.09 % | 22.64 % |
| **Implied model contribution** | **+0.50** | **+0.34** |

> **Retraction.** An earlier figure in this branch put the model's
> contribution at +2.72 / +2.34. That used the *historical* prior (251,126
> relationships, from a training set that no longer exists). The correct
> control is the leak-free prior built from the current training split
> (1,046,427 relationships). **The model's measured contribution is smaller
> than previously stated, not larger.**

A 79.9M-parameter model with a frozen CLIP ViT-L/14-336 backbone adds roughly
half a recall point over a JSON file.

**Diagnosis (`VERIFIED FROM CODE`).** The cause is structural, not tuning.
Plain cross-entropy rewards a 50-way classifier for reproducing
$P(p \mid s, o)$ — precisely what the table stores. Capacity is spent
re-deriving statistics the system is handed for free at evaluation time,
where the prior then outweighs the model roughly 17:1.

## 2. Information-theoretic motivation

Measured (`tools/headroom_analysis.py`, train-derived prior, full split):

- $H(p \mid s, o) = 2.554$ nats = **3.69 bits** of 3.912 max. The label pair
  removes only **34.7 %** of predicate uncertainty. High top-1 accuracy and
  high residual entropy simultaneously — the distribution is peaked but
  heavy-tailed.
- **Oracle-rerank curve.** A perfect visual reranker over the prior's top-K:
  @1 = 66.59 %, @2 = 80.27 %, @5 = 89.48 %, @10 = 92.50 %. The task is
  *"reorder a handful of plausible predicates"*, not *"search 50"*.
- **Not all headroom is real.** VG's vocabulary mixes 23 *generic*
  interchangeable terms (`on/in/of/has/with/wearing/near` — "wheel of bus"
  vs "wheel on bus" is annotator style) with 27 *decidable* predicates whose
  truth is fixed by the image. **64.2 % of the prior's errors are
  generic→generic** and unwinnable. Total headroom 33.41 points; **recoverable
  headroom 8.84 points**.
- **But the decidable subset is where the prior fails and vision can win:**

| On the 27 decidable predicates | |
|---|---:|
| Prior rank-1 | **29.91 %** |
| Perfect rerank over prior's top-5 | **71.74 %** |
| Perfect rerank over top-10 | 81.59 % |

Decidable predicates are **54 % of the vocabulary by class count but only
12.6 % of instances** — so they dominate `mR@50` and are nearly invisible to
`R@50`. An independent bound agrees: on pair types seen ≥5×, the best
possible *label-only* accuracy is 76.1 %, so 23.9 % of those instances
require vision.

## 3. Hypothesis

> Once pair-conditioned co-occurrence is supplied **explicitly during
> training**, the learned model allocates capacity to image-dependent
> predicate disambiguation, especially for predicates whose truth cannot be
> inferred from $(s, o)$ alone.

Measurable consequences: prior-only stays strong on generic predicates;
model-only may be weak; prior+residual improves *decidable* predicates;
visual ablation materially reduces the gain; a shuffled prior removes the
prior's advantage while leaving some learned signal; gains concentrate in
decidable/tail predicates.

## 4. Objective

$$z = \alpha \log P(p \mid s,o) + f_\theta(x), \qquad \mathcal{L} = \mathrm{CE}(z, y)$$

At the optimum $\mathrm{softmax}(z) = P(p \mid x,s,o)$, hence

$$f_\theta^*(x) = \log P(p \mid x,s,o) - \alpha \log P(p \mid s,o) + c$$

At $\alpha = 1$ this is exactly the pointwise **log-likelihood ratio** — the
information the image adds over co-occurrence.

### Gradient interpretation

$$\frac{\partial \mathcal{L}}{\partial f_\theta} = \mathrm{softmax}(z) - \mathbf{1}_y$$

The prior is *inside* the softmax, so the gradient is proportional to what
the prior gets wrong. **Verified numerically, not assumed** — sum $|\partial
\mathcal{L}/\partial z|$, true class fixed:

| Setting | prior RIGHT | prior WRONG | ratio |
|---|---:|---:|---:|
| A0 (no prior in graph) | 1.9600 | 1.9600 | 1.0× |
| **A1, α = 1.0** | **0.0120** | **1.9998** | **166×** |

A0 spends identical gradient whether or not the prior already knows the
answer. A1 concentrates it on the errors.

### Why α = 1.0, not the evaluation-time 3.75

Overturning a confident prior needs a logit swing of ~9.0 nats at α = 1 but
~33.8 at α = 3.75, while a randomly-initialised linear head emits $O(1)$.
A large α also forces $f_\theta$ to spend capacity cancelling
$2.75\log P(p\mid s,o)$ before expressing anything visual.

### Variants, and what they are not

| | Form | Note |
|---|---|---|
| A0 | $f_\theta(x)$ | control |
| **A1** | $\mathrm{sg}(\alpha\log P) + f_\theta(x)$ | **primary** |
| A2 | $\alpha\log P + r_\theta(x)$ + residual-weighted loss | differs from A1 *only* in the loss weighting — the stop-gradient itself is a **mathematical no-op** while the prior is a fixed parameterless table |
| A3 | $\alpha\log P + g_\theta(x)\,r_\theta(x)$ | **not implemented.** $\partial\mathcal{L}/\partial r = (p-y)g$, so $g\to 0$ is a self-reinforcing trap: the gate collapses, $r$ stops receiving gradient, and the system silently degenerates to prior-only |

Stop-gradient is implemented and configurable anyway, because it stops being
a no-op the moment the prior is composed with the learned calibration gate
(`adaptive_calibration_enabled`).

## 5. What this does **not** establish

A1 removes the *incentive* to relearn the prior. It does not make it
impossible — $f_\theta$ still sees subject and object appearance and could
re-derive a label-conditioned term. **Only the visual ablation can settle
this**, which is why it is a hard gate rather than a formality.

## 6. Ablations

| Arm | Configuration |
|---|---|
| **P0** | prior only, no model |
| **C0** | model only (A0) |
| **A1** | prior + residual |
| A1-shuffled-prior | predicate labels permuted inside the prior |
| A1-weak-prior | α reduced |
| A1-zero-vision | `visual_ablation_mode=zero` |
| A1-shuffle-vision | `visual_ablation_mode=shuffle` |
| A1-decidable / A1-generic | evaluated on each predicate group |

Report `R@20/50/100`, `mR@20/50/100`, `image_mean_R@50`, `multi_R@K`,
decidable vs generic recall, head/body/tail, pair-proposal recall, and
`residual_to_prior_ratio`.

**The central comparison is `A1 − P0`, never `A1 − 67/22`.**

## 7. Falsification criteria

The hypothesis is **FALSIFIED** if any holds:

1. `none ≈ shuffle ≈ zero` on visual ablation — the model is not using the
   image, whatever the headline metric says. **This overrides all others.**
2. `residual_to_prior_ratio ≈ 0` — the residual is numerically dead and A1
   has degenerated into P0.
3. A1 does not beat P0's **22.30 mR@50** by more than seed variance.
4. Gains appear only on generic predicates — i.e. the model learned
   annotation style, not visual semantics.

It is **SUPPORTED** only if all of: training is stable; residual gradients
are non-zero and non-negligible; `none > shuffle ≥ zero`; decidable-predicate
recall improves over 29.91 %; A1 > P0 on raw mR@50; and shuffled-prior
behaves differently from the exact prior.

A clean falsification is a publishable result about calibrated SGG
evaluation and should be reported, not buried.

## 8. Relationships

**Long-tail SGG.** This is not another reweighting trick. Logit adjustment
(`tail_logit_adjustment`) corrects the *marginal* $P(p)$; the prior here is
*pair-conditioned* $P(p \mid s,o)$. They are composable but distinct, and
stacking them uncontrolled is exactly the four-way conflict recorded in
`docs/known_issues.md`. Run one at a time.

**CLIP text scoring.** Currently measured to collapse onto two classes
(`in` + `has` = 93 % of predictions, `runs/text_path_gate/`). A1 does not fix
that and does not depend on it — it operates on the classifier head. A
prototype/metric head remains a *backup* candidate, viable only after text
alignment is repaired.

**Calibration.** A1 moves the prior from *evaluation* into *training*. If it
works, evaluation-time `freq_bias` becomes redundant rather than load-bearing
— which is itself a testable prediction: A1 should gain **less** from
evaluation-time calibration than C0 does.

## 9. Known limitations

- Prior lookup is a Python dict hit per pair (~480/step). Negligible against
  a CLIP forward, unmeasured on GPU.
- 26.3 % of validation triplets miss the exact pair row and fall back to
  subject/object marginals; the residual target is weaker there.
- The generic/decidable split is **my judgment**, not annotation. It is
  declared in `tools/headroom_analysis.py:GENERIC_PREDICATES` and results
  should be reported with it stated.
- The `use_all_pairs=true` negative distribution is untested at scale here.
- **No GPU measurement exists.** VRAM, step time and throughput are `UNKNOWN`.

## 10. Artifact provenance

| | |
|---|---|
| Training prior | `datasets_vg150_clean/frequency_prior_train.json` — built by `tools/build_vg150_frequency_prior.py --train_jsonl datasets_vg150_clean/train.jsonl`, which accepts **no** val/test input |
| Relationships | 1,046,427 · 212,981 pair rows · 50 predicates · smoothing 1.0 |
| Evaluation prior | `checkpoints/demo_best/frequency_prior.json` (historical, 251,126 relationships) |

The two are **deliberately distinct artifacts**. Training raises if
`prior_residual_path` is empty and warns if it equals `freq_bias_path`, so the
residual and the calibration can never silently become the same experiment.
