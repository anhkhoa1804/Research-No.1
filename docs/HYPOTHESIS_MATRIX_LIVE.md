# Live hypothesis matrix

Updated **2026-09-01, after `p42`–`p45`**. `p36`/`p37` still executing; rows
depending on them are marked **PENDING**. Supersedes the post-`p26` matrix and
the post-`p35` revision, both retained as history.

Classes: **VF** verified fact · **MR** measured · **INF** inference · **HYP** hypothesis.

Changed this revision: **H4, H9, H12 (new), H13 (new)**. **H12 was raised and
then partly withdrawn by its own test within the cycle.**

---

| # | Hypothesis | Evidence FOR | Evidence AGAINST | Status | Next test |
|---|---|---|---|---|---|
| **H1** | Candidate generation is the bottleneck | — | **MR** GT in prior top-5 for 89.5% of rows; scorer EXHAUSTED 9/9 (`p28`) | **FALSIFIED** | settled |
| **H2** | Frequency-prior dominance | **MR** prior alone 66.59 R@50; model adds +0.575. **MR** 85.8% of the model term's variance is between-group (`p42`) | — | **ESTABLISHED** | settled |
| **H3** | Calibration/decision artifact explains the mR headline | **MR** tau moves mR 22.3→26.4 with no image access | **MR** a learned per-class rule cannot match tau (`p29`) | **ESTABLISHED for the mR headline**, not for the model increment | settled |
| **H4** | Image-conditioned relational information exists | **MR** WPRD text 0.5542 [0.5495,0.5592], classifier 0.5728; prior control exactly 0.5000 (`p33`). **MR** the non-layout residual excludes chance in both heads (`p42`) | **MR** converts to ≈0 at the additive operating point (`p29`) | **ESTABLISHED BUT WEAK** | `p37` **PENDING** |
| **H5** | The term is a lexical/text lookup | — | **VF** `text_logits = normalize(rel_feat)·normalize(pred_emb)ᵀ` — a cosine against an image feature | **FALSIFIED AS STATED** | `pred_emb` Gram matrix from `p36` |
| **H6** | Global ranking bottleneck | **MR** worsens mean GT rank 1.83→2.78; −16 R as a ranker (`p28`) | — | **ESTABLISHED** | settled |
| **H7** | Candidate reranking has headroom | — | **MR** EXHAUSTED 9/9 (`p28`) | **FALSIFIED** | settled |
| **H8** | Annotation artifact manufactures the finding | **MR** VG multi-labels one instance pair | **MR** excluding same-instance rows *raises* WPRD 0.5542→0.5592 (`p35`) | **PRESENT BUT CONSERVATIVE** — cuts against us | quantify per bucket |
| **H9** | Representation / objective bottleneck | **MR** both heads cap at ≈0.57 vs geometry's 0.5961 (`p39`). **MR** evaluated head's tail intervals contain chance (`p35`). **MR** `p45` shows the composition failure is *bounded by* this one | **MR** the two readouts differ by 0.0186 with disjoint CIs, so some loss is readout | **OPEN, NOW THE PRIMARY CANDIDATE** — promoted by `p45` | **`p37` is exactly this test** |
| **H10** | The checkpoint runs a suboptimal readout | **MR** classifier head wins 13/13 strata; ΔR higher at α=0.5 for every tau (`p34`) | **caveat** α read off validation, no held-out selection | **SUPPORTED for "better grounded"**, not for "α=0.5 is right" | nested α selection |
| **H11** | Grounding is absent specifically in the tail | **MR** evaluated head body–tail [0.4822,0.5570] and tail–tail [0.4174,0.6246] contain chance (`p35`); geometry's edge grows to +0.123 at tail–tail (`p40`) | **MR** tail–tail rests on 61 cells | **SUPPORTED, POWER-LIMITED** | WPRD on published checkpoints |
| **H12** *(new)* | **Prior absorption** — the model over-absorbed the prior | — | **MR** the model term alone is 68.97% pair-constant-predictable vs reality's 69.23%; its weighted within-group entropy is *higher* than GT's (`p44`) | **FALSIFIED at the representation level** | settled |
| **H13** *(new)* | **Prior overwrite** — composition discards usable within-pair evidence | **MR** deployed system 97.45% pair-determined vs reality 69.23%; ~91% of within-pair variation destroyed (`p44`) | **MR** restoring it gains only +1.07 pts vs a registered +2.0 threshold, at a ~1:1 cost in prior-correct rows (`p45`) | **DESCRIPTIVELY TRUE, PRESCRIPTIVELY WEAK** — composition discards the variation *because it is mostly wrong* | a different composition form would need its own registration |

---

## Current scientific model

**MR + INF, and it is now constrained from both sides.**

1. The prior decides most rows and is 85.8% of the model term's variance.
2. The model term's *within-pair* variation is roughly the right **magnitude**
   (68.97% vs reality's 69.23% pair-constant predictability) but only weakly
   correct in **direction** (WPRD 0.5542 vs a 0.5 floor and geometry's 0.5961).
3. Composition at `alpha=3.75` discards ~91% of that variation — and `p45` shows
   this is close to rational, because restoring it trades adversarial fixes for
   prior-correct breaks at ~1:1.
4. On the 33.4% of rows where the prior is wrong, the checkpoint fixes 7.9%,
   geometry 6.8%, both together 10.5%. **~89% is recovered by nothing.**
5. `mR@K` rises with tau, which needs no image; the evaluated head has no
   measurable tail discrimination.

**The binding constraint is the quality of the within-pair signal.** Not its
suppression (H13, tested and weak), not its absence (H4, falsified), not prior
over-absorption (H12, falsified).

## What would overturn this

- `p37`: a probe on `rel_feat` reaching WPRD ≫ 0.6 ⇒ readout story, H9 down,
  H10 up, and a cheap successor becomes justified.
- WPRD ≈ 0.5 on published checkpoints ⇒ we measured a PURE quirk, and the
  benchmark claim collapses.
- WPRD ≫ 0.5 in the tail on some other checkpoint ⇒ H11 is PURE-specific.
