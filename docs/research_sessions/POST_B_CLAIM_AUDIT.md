# Claim audit — every improvement/superiority statement in the repository

Read-only audit produced while experiment B's fp32 control arm was running.
**Nothing outward-facing has been rewritten.** This is the inventory that a
rewrite would be based on, not the rewrite.

Classification:

- **SUPPORTED** — a measurement on this machine, under a recorded protocol,
  reproducible from committed tools.
- **PARTIALLY SUPPORTED** — the measurement exists but the claim overstates
  its scope, or a confound is unbounded.
- **WITHDRAWN** — explicitly retracted in-repo, by its own author.
- **UNVERIFIED** — asserted, never independently checked here.

---

## 1. Model improvement / architecture superiority

| # | Claim | Where | Status | Basis |
|---|---|---|---|---|
| 1.1 | "Adapt-light + fixed prior `fa375`: R@50 67.09 / mR@50 22.64, best calibrated mR@50" | `README.md:55`, `notes/current_status.tex:379` | **UNVERIFIED** | Self-report from `demo_config.env`. `data/manifests/historical_checkpoint_v1.yaml` labels it `HISTORICAL_EVIDENCE / UNREPRODUCED / SINGLE_SOURCE`; four protocol fields were never recorded. **Correctly caveated in place** — `README.md:48-50` says "Do not cite any of them as a current result." |
| 1.2 | "The model contributes +0.50 R@50 / +0.34 mR@50 over a train-derived prior" | `docs/APPEARANCE_PROBE_FINDINGS.md:16,139` | **WITHDRAWN** | Retracted in `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §2: it compared a measurement against the unverified 67.09/22.64 self-report. Comparing a measurement to a claim. |
| 1.3 | "The model is +3.82 mR@50 ahead of the leak-free prior" (240-image pilot) | `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §0/§2, since headed `SUPERSEDED IN PART` | **WITHDRAWN** | Overturned by its own author in `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` §3. At N=240 only 47/50 predicates occur, so mR@50 averaged over 47 classes with 1–2 instances in the rarest. At N=3,000 the same comparison is **−0.82**. |
| 1.4 | "The model does not beat the leak-free prior: Δ = +0.10 R@50 / −0.82 mR@50" | `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` §2 | **PARTIALLY SUPPORTED** | The measurement is sound and pre-registered, but the two arms use **different mR denominators** (48 aliased vs 50 unaliased). MEASURED this session (`runs/p6_alias_control_3000/`): like-for-like Δ = **−1.03 R@50 / −1.52 mR@50**. The correction moves *against* the model, so the NEGLIGIBLE verdict stands and strengthens. See `POST_B_SOURCE_FORENSICS.md` §3. |
| 1.5 | "The model is a more confident version of the prior's dominant mode, not a source of new discriminative information" | `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` §4 | **SUPPORTED** | head mR 42.5→54.1, tail 12.8→5.8. Independently corroborated this session by the predicate-group exposure diagnostic: the model under-predicts `pose` by 59 % and `contact` by 42 % relative to GT while over-predicting `possession` (+10 %) and `spatial` (+2.7 %). |
| 1.6 | "Three cheap architectures have failed to convert the headroom" | `docs/APPEARANCE_PROBE_FINDINGS.md` §6 | **PARTIALLY SUPPORTED** | The three failures are real, but `PHASE4` §5 already notes the stated conclusion ("the signal cannot be converted") was too strong: all three were trained by cross-entropy on the natural distribution, where agreeing with the prior is loss-minimising. Capacity was controlled; the decision rule was not. |

## 2. Appearance usefulness

| # | Claim | Where | Status | Basis |
|---|---|---|---|---|
| 2.1 | "Frozen CLIP appearance carries genuine predicate signal (`real > shuffled > zero`)" | `docs/APPEARANCE_PROBE_FINDINGS.md` §2; reconfirmed in `docs/APPEARANCE_PROBE_L14_RESULT.md` §3 | **SUPPORTED** | MEASURED at both encoders. At L/14-336 (`runs/p6_appearance_probe_l14_336/`): 16.90 > 12.17 > 8.58 mR@50, a **larger** margin than the documented B/32 one. |
| 2.2 | "Appearance converts only 0.6 % of the oracle headroom (ViT-B/32)" | `docs/APPEARANCE_PROBE_FINDINGS.md` §2 | **UNVERIFIED (as a number), SUPPORTED (as a direction)** | No B/32 run artifact exists on this machine; the number is documentation only. Two *mutually inconsistent* B/32 records exist in-repo: the findings doc (λ=0.50, 0.6 %) and `tools/appearance_probe.py`'s docstring (λ=0.25, 0.8 %). The direction is now independently confirmed at L/14-336. |
| 2.3 | "ViT-B/32 is a lower bound; confirming on GPU with L/14-336 is the single experiment that could overturn this verdict" | `docs/APPEARANCE_PROBE_FINDINGS.md` §5 caveat 1 | **DISCHARGED** | Experiment B ran it. H0 supported. This caveat should now be marked resolved rather than left standing as an open qualifier. |
| 2.4 | "Appearance at ViT-L/14-336 captures < 5 % of headroom" | `docs/APPEARANCE_PROBE_L14_RESULT.md` §9 | **SUPPORTED** | MEASURED, pre-registered, gate passes: −1.2 % at the held-out-selected λ, +0.2 % at the cherry-picked λ, +1.5 % on the decidable-only denominator. All far below threshold. |
| 2.5 | "Nothing measured suggests capacity is the constraint" | `docs/APPEARANCE_PROBE_FINDINGS.md` §6 | **SUPPORTED, and now much more strongly** | B raised encoder capacity (512→768 proj dim, 224→336 px, ~88M→~304M vision params), the measured gate margin *rose* (5.91→8.32), and conversion did not follow. |

## 3. Prior contribution / decision rule

| # | Claim | Where | Status | Basis |
|---|---|---|---|---|
| 3.1 | "A pair-conditioned prior with one parameter (τ=0.1) reaches R@50 66.04 / mR@50 26.42 on the full validation split" | `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §8 | **SUPPORTED** | `runs/p4_decision_rule_sweep/`. Reproducible on CPU in seconds. |
| 3.2 | "τ=0.1 gains +4.03 mR@50 over the leak-free prior at N=3,000 for −0.64 R@50" | `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` §2 | **SUPPORTED** | `runs/p5_arm3_trainprior_tau01_3000/`. Confirmed independently this session by `runs/p6_alias_control_3000/` (`raw50`, τ=0.1: 66.16 / 26.00). |
| 3.3 | "A single scalar with no image data outperforms the entire visual system on mR@50 by +4.84" | `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` §2 | **SUPPORTED, and understated** | Like-for-like on the evaluator's own alias scheme the gap is **+5.70 mR@50**, and τ=0.1 also beats the model on R@50 (+0.41). |
| 3.4 | "The τ result shows the decision rule was discarding recoverable signal" | `tools/decision_rule_probe.py` verdict, `PHASE4` §4 | **PARTIALLY SUPPORTED** | True as measured — but τ has **only ever been applied in `tools/decision_rule_probe.py`, never inside `openvocab_rel/evals.py`**. `_apply_eval_logit_adjustment` is a verified no-op in every recorded run. "The decision rule is the lever" is supported for the prior-only path; it is untested for the model path. |
| 3.5 | "64.2 % of the prior's errors are generic→generic; only 12.6 % of GT triplets carry a decidable predicate" | `runs/p3e_headroom_train_derived/`, `PHASE4` §4 | **SUPPORTED** | Full-split measurement. Used as the secondary denominator in B, where it moved the answer from −1.2 % to +1.5 % — still far below threshold. |
| 3.6 | "The historical prior cannot be independently verified as leak-free" | `data/manifests/historical_checkpoint_v1.yaml` | **SUPPORTED (as a statement of ignorance)** | `num_relationships = 251,126` does not match `train.jsonl` (1,046,427); the generating corpus no longer exists. Correctly treated as an upper-bound asymmetry that works *against* the model. |

## 4. Historical reproduction

| # | Claim | Where | Status |
|---|---|---|---|
| 4.1 | "The checkpoint, historical prior and `demo_config.env` are byte-identical to the manifest" | `PHASE4` §1 | **SUPPORTED** — re-verified this session before B launched (SHA256 all match). |
| 4.2 | "The protocol resolves exactly as the manifest specifies (canary: 14 flags + vocab size and order)" | `runs/p3c_historical_canary/` | **PARTIALLY SUPPORTED** — true for the 14 flags checked. `eval_sgg_use_vg_aliases`, which measurably changes reported mR@50 by 0.71 points, is **not** among them. |
| 4.3 | "67.09 / 22.64 is reproducible" | `docs/GCP_EXPERIMENT_PROTOCOL.md:8-9` frames it as the goal | **UNVERIFIED** — never attempted at full split; `PHASE4` §1 argues a match would not confirm reproduction anyway, because four protocol fields were never recorded. |
| 4.4 | "No PURE checkpoint exists anywhere in this environment" | `docs/GT_EXTRACTION_BUG_TRIAGE.md` §8 | **SUPPORTED** |

## 5. Zero-shot / open-vocabulary benefit

| # | Claim | Where | Status |
|---|---|---|---|
| 5.1 | "Open-vocabulary predicate scoring … extension paths that can be enabled and reported separately" | `README.md:118` | **UNVERIFIED** — a capability statement, not a result. No open-vocabulary benefit is measured anywhere in `runs/`. |
| 5.2 | zR@20/50/100 (zero-shot recall) | `runs/p5_model_vs_leakfree_prior/latest_metrics.json` | **MEASURED = 0.0** at all K. `eval_zs_predicates` was empty, so the field is vacuous, not a zero-shot failure. Should not be cited in either direction. |

## 6. Long-tail improvement

| # | Claim | Where | Status |
|---|---|---|---|
| 6.1 | "Tail improves across epochs" (`Adapt-light raw e5`) | `README.md:58` | **UNVERIFIED** — from the same uncorroborated historical block. |
| 6.2 | "The model is substantially worse on the tail (−7.0 mR@50)" | `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` §4 | **PARTIALLY SUPPORTED** — the direction is robust and corroborated by the group-exposure diagnostic, but head/body/tail buckets are recomputed from the evaluated split's own GT counts (`evals.py:1310`), so the *magnitude* is partly a bucketing artifact when comparing across different N. See `POST_B_SOURCE_FORENSICS.md` §5 issue #4. |
| 6.3 | "τ=0.1 raises tail mR@50 from 12.8 to 19.6" | `runs/p5_arm3_trainprior_tau01_3000/` | **SUPPORTED** — both arms share one tool and one bucketing, so this comparison is internally consistent. |
| 6.4 | "Balanced binary discrimination on 8 geometry features reaches AUC 0.878 (`walking on` vs `on`)" | `runs/p4_predicate_discriminability/` | **SUPPORTED** — with the built-in control that the two annotation-style pairs (`part of` vs `of`: 0.559; `near` vs `next to`: 0.580) are *not* separable, which is what makes the rest credible. |

## 7. Methodological claims

| # | Claim | Status |
|---|---|---|
| 7.1 | "Under GT pairs the headline `predcls R@K` emits one predicate per pair, making K inert" | **SUPPORTED** — verified at source (`_make_triplet_predictions`), and R@50 = R@100 in every recorded arm. |
| 7.2 | "Any comparison to published VG150 numbers must use the `predcls_multi` fields" | **SUPPORTED** — arm 1 multi_R@50 84.21 / multi_mR@50 47.37 vs headline 66.90 / 21.16. |
| 7.3 | "`--eval_batches 0` evaluated zero batches and exited 0" | **SUPPORTED** — evidence preserved at `runs/p3c_historical_full_val_ZEROBUG/`; fixed at `74c54a2`. |
| 7.4 | "The protocol is recalibration-saturated" | **SUPPORTED** — and strengthened this session: at α=3.75 roughly two thirds of the prior's errors are unreachable by the visual term regardless of encoder (`runs/p6_prior_dominance_margin/`). |

---

## Summary

| Status | Count |
|---|---:|
| SUPPORTED | 14 |
| PARTIALLY SUPPORTED | 7 |
| WITHDRAWN | 2 |
| UNVERIFIED | 6 |
| DISCHARGED (was an open caveat) | 1 |

**No claim in this repository currently asserts that the architecture beats a
leak-free baseline.** Every such claim has been withdrawn by its own author or
is labelled unverified in place. That is unusual and worth stating plainly: the
documentation's epistemic hygiene is the repository's strongest asset.

**The two items most in need of action** (neither performed here):

1. `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md` should gain a *"corrected in
   part"* header pointing at the alias finding — its Δ is real but not
   like-for-like, and the correction is in the direction of its own
   conclusion.
2. `docs/APPEARANCE_PROBE_FINDINGS.md` §5 caveat 1 and
   `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §3 ("PENDING") should point at
   `docs/APPEARANCE_PROBE_L14_RESULT.md`. The caveat that qualified every
   appearance conclusion in this repository is now discharged.
