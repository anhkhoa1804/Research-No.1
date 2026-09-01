# The diagnosis

**What long-tail SGG performance actually measures, and what PURE actually learned.**

Status: **MEASURED**, on the full 10,401-image VG150 validation split
(`runs/p24`, cache validated 12/12), with every claim carrying its run and its
limitation. Two rows are **PENDING** on `runs/p32` and `runs/p36`/`p37`.

---

## The founding question

> Why does long-tail SGG performance appear better than the underlying
> relational reasoning?

## The answer, in one paragraph

Because the two are measured by quantities that are almost independent.
`mR@K` rises when probability mass moves toward rare predicates, and a
temperature on the frequency prior does that **without looking at the image**.
Meanwhile the checkpoint's actual image-conditioned ability to tell two
relations apart — holding the object pair fixed, so the prior cancels exactly —
is **0.554 AUC against a 0.5 floor**, is **statistically absent in the tail**,
and is **beaten by a 19-feature logistic regression on two bounding boxes**.
The metric and the mechanism are decoupled, and nothing in the standard
protocol reveals it.

---

## The instrument: WPRD

Within one (subject, object) **category** group the train-derived prior row is
constant — measured `max deviation 9.441e-05`. So in the double difference

```
( score[i,a] − score[i,b] ) − ( score[j,a] − score[j,b] )
```

the prior cancels **exactly**, and so does any per-class calibration, any tau,
any logit adjustment and any global temperature. WPRD is the AUC of
`s(a,b) = score[·,a] − score[·,b]` separating rows whose GT is `a` from rows
whose GT is `b`, **within the same group**. 0.5 = no image-conditioned
relational information. No operating point, no free parameters.

**It validates itself:** the prior, pushed through the identical code path,
reads **exactly 0.5000**, CI [0.5000, 0.5000], in every support band and every
predicate bucket.

56.9% of GT rows (75,366) are *decidable* — their group holds ≥2 distinct GT
predicates. The ceiling for any pair-constant predictor there is 69.23%.

---

## The five findings

### 1. The frequency prior dominates — known, and confirmed at full scale
Prior alone: R@50 **66.59** / mR **22.30**. The whole trained model adds
**+0.575 R**. *(`p24`, `p28`. Not novel — Neural Motifs established this.)*

### 2. mR movement is calibration, not grounding
tau alone moves mR 22.30 → 26.42 with no image access. A *learned* per-class
rule cannot even match tau: Pareto −1.167 ± 1.101, clears the R@50 floor on
**0 of 5** partitions. *(`p29`)*

### 3. Image-conditioned relational signal exists, and is weak
WPRD text head **0.5542** [0.5495, 0.5592]; classifier head **0.5728**
[0.5681, 0.5779]; random null 0.5046. The CI excludes 0.5 by nine half-widths.
*(`p33`)*

**This overturned the project's previous position.** `p26`/`p29` measured
`full − pair_matched_null = +0.031 ± 0.188` and it was read as "no image
conditioning". That reading was wrong for a measurable reason: **43.1% of rows
are structurally inert to a within-group permutation** (23.3% singleton groups,
19.8% multi-row groups with a constant GT). Those runs correctly bounded a
diluted quantity. *(`docs/AUDIT_TEXT_BRANCH_IS_IMAGE_CONDITIONED.md`)*

### 4. The grounding is absent exactly where the field claims progress
For the head the checkpoint actually runs:

| predicates compared | WPRD | contains chance? |
|---|---|---|
| head–head | 0.5612 [0.5549, 0.5665] | no |
| head–tail | 0.5273 [0.5113, 0.5458] | no, barely |
| **body–tail** | 0.5230 [0.4822, 0.5570] | **yes** |
| **tail–tail** | 0.5126 [0.4174, 0.6246] | **yes** (61 cells — low power) |

*(`p35`)* The long-tail metric improves while long-tail discrimination is
indistinguishable from chance.

### 5. Box geometry beats the trained model
A logistic regression on **19 scale-invariant numbers from two rectangles** —
no pixels, no CLIP, no `rel_feat`, no predicate embeddings — trained on the same
VG150 training split and applied to validation:

| arm | WPRD |
|---|---|
| **geometry, TRAIN-FITTED** | **0.5961** [0.5921, 0.6014] |
| classifier head (discarded) | 0.5728 |
| text head (**evaluated**) | 0.5542 |

Paired: text **−0.0419** [−0.0489, −0.0354]; classifier **−0.0232**
[−0.0292, −0.0163]. *(`p38`, `p39`)*

**PURE's image-conditioned relational *discrimination* is worse than what two
bounding boxes give you for free.**

**Corrected by `runs/p41`, and the correction matters.** On the field's own
metric (R@50/mR@50 under the evaluator's composition) the model *beats*
geometry at tau = 0 and 0.05 — its actual operating region — by +1.95 and +1.72
Pareto points. Discrimination is not calibration: geometry separates tail
predicates better and still lowers tail mR, because its scores are not scaled to
compete with the prior for the argmax. The defensible claim is therefore
narrower than "geometry beats the model":

> The checkpoint's advantage on the field's metric does not come from relational
> discrimination — rectangles discriminate better, especially in the tail. What
> it supplies that geometry does not is a term already calibrated against the
> prior.

At tau >= 0.1 the best arm measured anywhere in this cycle is **model +
geometry** (+4.051 at tau=0.2), so the two carry partially complementary
information. See `docs/GEOMETRY_SGG_BASELINE_RESULT.md`.

---

## What PURE actually learned

**MR + INF.** A decomposition consistent with every measurement above:

| component | share / strength | image-conditioned? |
|---|---|---|
| frequency prior P(p \| s,o) | decides most rows; 82.6% of model-term variance is between-group | no |
| calibration (tau) | moves mR 22.3 → 26.4 | no |
| spatial layout | ≈0.596 WPRD available from boxes alone | yes |
| **what the model adds over layout, on WPRD** | **negative** (−0.023 to −0.042) | — |
| what the model adds over layout, on R@50/mR@50 at tau<=0.05 | **positive** (+1.7 to +1.9 Pareto) | — |
| appearance / semantic relational evidence | **no evidence of any** | — |

The 79.9M-parameter visual pathway is real, runs on real images, and produces a
feature both heads read. What it contributes, after the prior is removed, is a
weak layout-like signal whose *discriminative* quality a 20-parameter-per-class
linear model on rectangles exceeds. Its value at the operating point comes from
being *calibrated against the prior*, not from discriminating relations
(`runs/p41`).

## Corollary: the checkpoint runs the worse of its two readouts

`ensemble_alpha = 0.0` selects the text-cosine head and discards the trained
classifier head. The discarded head is better in **13 of 13** strata, and is the
only one with measurable tail grounding (body–tail 0.5918 [0.5559, 0.6339], CI
clear of 0.5). Mixing it back in raises ΔR over the prior at every tau tested.
*(`p33`, `p35`, `p34` — with the caveat that `p34`'s alphas were read off
validation with no held-out selection, so "better grounded" is robust and
"α=0.5 is correct" is not established.)*

---

## What this does NOT establish

- **Not** that the prior's dominance is a new observation. It is not.
- **Not** that PURE is uniquely bad. **WPRD has been run on exactly one
  checkpoint.** Whether this is a property of the field or of this checkpoint is
  the single most important open question, and it is cheap to answer.
- **Not** that geometry is "the right model". The geometry probe is a *lower
  bound on triviality*, not a proposal.
- **Not** that the encoder is incapable — only that its *readouts* are behind
  geometry. `p36`/`p37` test whether `rel_feat` itself holds more. **PENDING.**
- **Not** anything about SGDet or SGCls. All of this is PredCls with GT pairs.
- **Not** anything about the test split. All of this is validation.

## Limitations that cut against the finding, recorded

- VG annotates one instance pair with several predicates; those rows tie at 0.5
  in WPRD. Excluding them **raises** every estimate (text 0.5542 → 0.5592), so
  the headline numbers are conservative.
- `tail–tail` rests on 61 cells. Its interval is wide and it is the weakest cell
  in the table.
- WPRD measures *discrimination*, not *calibration* or *ranking*. A model could
  be useful in ways WPRD does not see.
