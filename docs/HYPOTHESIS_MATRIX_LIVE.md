# Live hypothesis matrix

Updated **2026-09-01, after `p33`/`p34`/`p35`** (WPRD) and the source audit
`docs/AUDIT_TEXT_BRANCH_IS_IMAGE_CONDITIONED.md`. `p32` and `p36` were still
executing; rows depending on them are marked **PENDING**.

Evidential classes: **VF** verified fact · **MR** measured result ·
**INF** inference · **HYP** hypothesis.

Supersedes `docs/HYPOTHESIS_MATRIX.md` (post-p26), which is retained as the
historical record. **H4 and H9 changed status.**

---

| # | Hypothesis | Evidence FOR | Evidence AGAINST | Status | Next test |
|---|---|---|---|---|---|
| **H1** | Candidate generation is the bottleneck | none surviving | **MR** GT in prior top-5 for 89.5% of rows, top-3 85.2% (`p28`, full split). Learned scorer restricted to those candidates: EXHAUSTED 9/9 | **FALSIFIED** | none — settled |
| **H2** | Frequency-prior dominance | **MR** prior alone = R@50 66.59 / mR 22.30 on full split; model adds only +0.575 R. **MR** tau moves mR 22.3→26.4 with no image access | — | **ESTABLISHED** | none — settled |
| **H3** | Calibration/decision artifact explains the model-bearing effect | **MR** tau reproduces most mR movement | **MR** a *learned* per-class rule fails to even match tau: Pareto −1.167 ± 1.101, floor 0/5 (`p29`). **MR** `full` separates from both no-model arms by ≥ +2.37 on every partition | **PARTIALLY FALSIFIED** — calibration explains the *mR headline*, not the model-bearing increment | none — settled |
| **H4** | Image-conditioned relational information exists | **MR** WPRD text head **0.5542** [0.5495, 0.5592]; classifier head **0.5728** [0.5681, 0.5779]; prior control exactly **0.5000**; random null 0.5046 (`p33`). **VF** both heads are functions of image-derived `rel_feat` | **MR** at the additive operating point it converts to ≈0: `full − pair_matched_null` = +0.031 ± 0.188 (`p29`) | **STATUS CHANGED → ESTABLISHED BUT WEAK.** Previously "not established". The nulls that said 0 were diluted: 43.1% of rows are structurally inert to a within-group permutation | `p37` — is the weakness readout or representation? **PENDING** |
| **H5** | Text-semantic prior (the term is a lexical lookup) | **MR** 82.6% of model-term variance is *between* (s,o) groups | **VF** `text_logits = normalize(rel_feat) @ normalize(pred_emb).T` — a cosine against an image-derived feature, not a lookup. **MR** WPRD > 0.5 decisively | **FALSIFIED AS STATED.** The term is image-conditioned; it is *dominated* by pair identity, which is a different claim | `pred_emb` Gram matrix from `p36` — is the readout geometry degenerate? |
| **H6** | Global ranking bottleneck | **MR** model worsens mean GT rank (1.83→2.78); raw model score as a ranker costs up to −16.0 R (`p28`) | — | **ESTABLISHED** | none — settled |
| **H7** | Candidate reranking has headroom | — | **MR** REALIZABLE EXHAUSTED 9/9 on the full split (`p28`); oracle criterion non-falsifiable by construction | **FALSIFIED / branch closed** | none — settled |
| **H8** | Annotation/data artifact | **MR** 48.7% of multi-row (s,o) groups have a CONSTANT GT; VG labels one instance pair with several predicates (those tie at 0.5 in WPRD) | **MR** excluding same-instance rows *raises* WPRD (0.5542→0.5592), so the artifact is diluting the signal, not creating it | **PRESENT BUT CONSERVATIVE** — it works against our finding | quantify multi-label rate per predicate bucket |
| **H9** | Representation bottleneck | **MR** WPRD ceiling of both heads is ≈0.57 against a 1.0 ceiling. **MR** evaluated head has *no* measurable tail grounding (body–tail CI [0.4822, 0.5570] contains 0.5) | **MR** the two readouts differ by 0.0186 with non-overlapping CIs, so at least *some* of the loss is readout, not representation | **STATUS CHANGED → OPEN AND NOW DECIDABLE.** Previously conflated with H4 | **`p37` is exactly this test.** READOUT-LIMITED if a cross-fitted probe on `rel_feat` beats the classifier head by ≥0.03 |
| **H10** *(new)* | The checkpoint runs a suboptimal readout | **MR** classifier head beats text head in 13/13 strata; ΔR over prior higher at α=0.5 for every tau; Pareto +0.864→+4.126 at tau=0.1 (`p34`) | **caveat** those α were read off validation with no held-out selection | **SUPPORTED for "better grounded"; NOT ESTABLISHED for "α=0.5 is right"** | nested α selection with held-out folds |
| **H11** *(new)* | Grounding is absent specifically in the tail | **MR** text head body–tail 0.5230 [0.4822, 0.5570] and tail–tail 0.5126 [0.4174, 0.6246] both contain chance, while head–head 0.5612 [0.5549, 0.5665] does not | **MR** tail–tail has only 61 cells — wide interval, low power | **SUPPORTED, POWER-LIMITED in tail–tail** | more tail cells: run WPRD on the *test* split too, and on published checkpoints |

---

## The current best explanation

**MR + INF.** PURE's relational encoder produces a genuinely image-conditioned
feature, and both predicate heads read it. But 82.6% of the resulting term's
variance is between (subject, object) groups, and the image-conditioned
remainder supports only ≈0.55–0.58 within-pair AUC — concentrated on head
predicates and **statistically absent from the tail in the head actually used**.

Composed additively with a prior that already decides most rows, that weak
signal converts to +0.575 R and essentially zero Pareto movement beyond what
pair identity alone achieves. `mR@K` rises anyway, because tau moves mass to
rare predicates without looking at the image.

So: **the metric improves for reasons unrelated to the mechanism it is taken to
certify.** That is the diagnosis, and WPRD is the instrument that separates them.

## What would overturn it

- `p37` showing a probe on `rel_feat` reaching WPRD ≫ 0.6 ⇒ the encoder is fine
  and this is a readout story, not a grounding story.
- WPRD ≈ 0.5 on published checkpoints ⇒ our metric is measuring a PURE-specific
  quirk rather than a field-wide property.
- WPRD ≫ 0.5 in the tail for some other checkpoint ⇒ H11 is PURE-specific.
