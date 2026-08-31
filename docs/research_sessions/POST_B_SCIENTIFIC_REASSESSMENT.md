# Post-B scientific reassessment

Written after experiment **B** (the pre-registered ViT-L/14-336 appearance
probe) completed on the L4. Companions:

- `docs/APPEARANCE_PROBE_L14_RESULT.md` — B's result report
- `docs/research_sessions/POST_B_SOURCE_FORENSICS.md` — source-level signal path, and the code/reproducibility audit
- `docs/research_sessions/POST_B_CLAIM_AUDIT.md` — every improvement claim, classified

Epistemic labels are strict and used throughout: **VERIFIED FACT** (checked in
code or by cryptographic identity), **MEASURED** (produced by a run recorded
under `runs/` on this machine), **HISTORICAL SELF-REPORT**, **INFERENCE**,
**HYPOTHESIS**, **NOT ESTABLISHED**.

Corrections are recorded, never silently applied. No historical artifact, raw
dataset, manifest, or pre-registered criterion was modified.

---

# A. Current evidence

## A.1 The consolidated ledger

All rows are PredCls under GT pairs. "N" is the number of validation images.
Rows A–H are the labels used in the session brief.

| | Result | R@50 | mR@50 | N | vision | params | Status |
|---|---|---:|---:|---:|:---:|---:|---|
| **A** | Historical claim (`demo_config.env`) | 67.09 | 22.64 | ? | yes | 79.9M | **HISTORICAL SELF-REPORT** — single-source, unreproduced, four protocol fields never recorded |
| **B** | Historical prior alone, τ=0 | 64.46 | 20.08 | 3,000 | no | 0 | **VERIFIED MEASUREMENT** `runs/p5_arm2b_histprior_3000/` |
| **C** | Leak-free train-derived prior, τ=0 | 66.80 | 21.98 | 3,000 | no | 0 | **VERIFIED MEASUREMENT** `runs/p5_arm2_trainprior_3000/`; reproduced to 10 d.p. this session |
| **C′** | …same prior, evaluator's alias scheme | **67.93** | **22.68** | 3,000 | no | 0 | **VERIFIED MEASUREMENT** `runs/p6_alias_control_3000/` — the like-for-like baseline for row E |
| **D** | Leak-free prior, τ=0.1 | 66.16 | 26.00 | 3,000 | no | **1** | **VERIFIED MEASUREMENT** `runs/p5_arm3_trainprior_tau01_3000/` |
| **D′** | …same, evaluator's alias scheme | 67.31 | 26.86 | 3,000 | no | 1 | **VERIFIED MEASUREMENT** `runs/p6_alias_control_3000/` |
| **E** | Model + historical prior | 66.90 | 21.16 | 3,000 | yes | 79.9M | **VERIFIED MEASUREMENT** `runs/p5_model_vs_leakfree_prior/`, canary PASS, 59.7 min |
| **F** | ViT-B/32 appearance probe, best λ | — | 21.24 | 1,200 | yes | tiny | **NOT ESTABLISHED here** — documentation only; no artifact on this machine; two inconsistent in-repo records |
| **G** | **ViT-L/14-336 appearance probe, selected λ** | 66.87 | **20.47** | 1,200 | yes | tiny | **VERIFIED MEASUREMENT** `runs/p6_appearance_probe_l14_336/`, 20.6 min, gate PASS |
| **H** | fp32 control of G, selected λ | 66.85 | **20.57** | 1,200 | yes | tiny | **VERIFIED MEASUREMENT** `runs/p6_appearance_probe_l14_336_fp32/`, 67.0 min, gate PASS, **same verdict, same selected λ**; `captured_total` −0.94 % vs −1.16 % ⇒ fp16 inert |

Supporting measurements, all **VERIFIED MEASUREMENT**:

| Quantity | Value | Source |
|---|---:|---|
| oracle@5 mR@50 (probe subsample) | 64.60 | `runs/p6_appearance_probe_l14_336/` |
| coverage@5 | 89.90 % | same |
| prior argmax mass on `on` | 45.13 % | `runs/p5_arm2_trainprior_3000/` |
| top-5 predicates' share of prior argmax | 87.5 % | same |
| recoverable share of R@50 headroom | 8.84 of 33.41 pts | `runs/p3e_headroom_train_derived/` |
| generic→generic share of prior errors | 64.2 % | same |
| decidable share of GT triplets | 12.6 % | same |
| geometry AUC, `walking on` vs `on` | 0.878 | `runs/p4_predicate_discriminability/` |
| geometry AUC, `near` vs `next to` (control) | 0.580 | same |
| median swing needed to correct a prior error at α=3.75 | 4.44 z-units | `runs/p6_prior_dominance_margin/` |
| prior errors reachable within a 3-unit visual swing | 35.6 % | same |
| arm E, literature-comparable `multi_mR@50` | 47.37 | `runs/p5_model_vs_leakfree_prior/` |

## A.2 Correction history, preserved explicitly

Nothing below is rewritten. Each correction is recorded with its cause.

| # | Original claim | Status | Cause |
|---|---|---|---|
| 1 | Model contributes **+0.50 R@50 / +0.34 mR@50** over a train-derived prior | **WITHDRAWN** — NOT ESTABLISHED | Compared a measurement against the unverified 67.09/22.64 self-report. A measurement was compared to a claim. |
| 2 | Model is **+3.82 mR@50** ahead of the leak-free prior (N=240 pilot) | **WITHDRAWN** — sampling artifact, must not be used as headline evidence | At N=240 only 47/50 predicates occur; mR@50 is an unweighted class mean, so classes with 1–2 instances dominated it. |
| 3 | A′ at N=3,000: model **−0.82 mR@50** vs the leak-free prior | **STANDS as measured; NOT like-for-like** | Arm E uses the evaluator's 48-class aliased mR; arm C uses a 50-class unaliased mR. Correction below. |
| 3′ | **NEW this session:** like-for-like Δ = **−1.03 R@50 / −1.52 mR@50** | **VERIFIED MEASUREMENT of the scheme gap; INFERENCE for the Δ** | `evals.py:430` merges `near`→`next to`, `wears`→`wearing`. Worth +1.13 R@50 / +0.71 mR@50 to the prior (`runs/p6_alias_control_3000/`, validation gate reproduces arm C to 10 d.p.). Moves **against** the model; the pre-registered NEGLIGIBLE verdict is unchanged and strengthened. |
| 4 | τ=0.1 gains **+4.03 mR@50** over the leak-free prior at N=3,000 | **STANDS** | Reconfirmed independently this session. Like-for-like versus the model the gap is **+5.70 mR@50**, and τ=0.1 also wins on R@50 (+0.41). |
| 5 | B: ViT-L/14-336 **passes** the visual-ablation gate but captures **−1.2 %** of total headroom under the pre-registered scoring criterion | **STANDS** | `runs/p6_appearance_probe_l14_336/`. Verdict **H0 SUPPORTED**, applied mechanically. |

---

# B. What B established

**MEASURED.**

1. Frozen ViT-L/14-336 appearance carries genuine predicate information. The
   ablation gate passes cleanly: real 16.90 > shuffled 12.17 > zero 8.58 mR@50.
2. The gate margin (real − zero) is **8.32 points**. The two documented
   ViT-B/32 margins disagree — **5.91** (`APPEARANCE_PROBE_FINDINGS.md`) and
   **7.92** (`tools/appearance_probe.py` docstring, an earlier run) — so the
   defensible statement is that the stronger encoder registers **at least as
   much** appearance signal, not that it registers dramatically more.
3. Under the pre-registered additive composition, headroom captured is
   **−1.2 %** at the λ selected on held-out-from-train, **+0.2 %** at the
   cherry-picked validation-argmax λ, and **+1.5 %** on the decidable-only
   secondary denominator. Every point of the λ grid, under either selection
   rule and either denominator, lies between −1.2 % and +1.5 %.
4. The mechanism reproduces exactly where theory predicts: `eating` +18.2,
   `riding` +8.2, `standing on` +6.2, `holding` +4.9, `walking on` +4.3
   rank-1 points — the same predicate set the B/32 run identified.
5. Appearance **alone** (the replacement arm, no prior term) reaches mR@50
   16.90, which is **worse than the prior alone** (20.98).

8. **Mixed precision is inert.** The pre-registered fp32 control arm
   (`runs/p6_appearance_probe_l14_336_fp32/`, 67.0 min, exit 0) reproduces the
   verdict, the gate outcome, the selected λ (0.5) and the cherry-picked λ
   (0.25). The baselines and the ZERO arm are bit-identical; the whole
   additive curve moves by ≤ 0.10 mR points; `captured_total` moves +0.23 pp
   (−1.16 % → −0.94 %). The one methodological shortcut taken for speed
   changed nothing.
9. **An empirical noise floor.** That 0.23 pp spread, and the ~0.2 pp spread
   between the two documented B/32 runs, put this protocol's sensitivity an
   order of magnitude below the ~5 pp distance from the measured value to the
   decision threshold.

**INFERENCE, well supported.**

6. Encoder capacity is not the binding constraint. This is the strongest form
   of the claim available: the confound was named in advance as decisive,
   removed, and the outcome did not change — while the *measurable signal*
   moved in the direction capacity predicts.
7. The failure is a **conversion** failure, not a **perception** failure.
   Point 5 is the sharpest statement of it: a scorer built on frozen CLIP
   appearance is a *weaker ranker of predicates than label co-occurrence is*,
   and their errors are correlated enough that no positive mixing weight in
   the pre-registered grid beats the prior alone.

---

# B.2 Deep reading of B: what the gate proves, and the ten alternatives

## B.2.1 Line-by-line: does the implementation match the pre-registration?

Checked against `tools/appearance_probe.py` at HEAD, which
`git diff 91b24c7 HEAD` shows is **byte-identical to its registered version**.

| Pre-registered (§3/§4) | Implementation | Match |
|---|---|---|
| `openai/clip-vit-large-patch14-336`, frozen, never fine-tuned | `CLIPModel.from_pretrained(...).eval()`, whole extraction under `@torch.no_grad()`; no optimizer ever sees encoder params | ✓ |
| Prior = `frequency_prior_train.json` | `DEFAULT_PRIOR`, and the command passes it explicitly | ✓ |
| Eval split = `validation.jsonl` | ✓ | ✓ |
| Candidate set = prior top-K, K=5 | `K = 5`; `cand_va = topk(Pva, k=K)` | ✓ |
| `score = log P(p|s,o) + λ·f` | `s = Pva.gather(1, cand_va) + lam * s`, where `Pva` is the log-prior minus its **per-row mean** — a per-row constant, so within-row ranking is unchanged | ✓ (equivalent) |
| λ grid `0, 0.1, 0.25, 0.5, 1.0, 2.0` | loop over `(0.1, 0.25, 0.5, 1.0, 2.0)`; λ=0 reported as P0 | ✓ |
| PCA-48, 30 epochs, image-level held-out split, weight decay | `PCA_DIM = 48`, `EPOCHS = 30`, hold-out over `unique(TR["img"])` at 20 %, `wd = 3e-3` | ✓ |
| λ selected on held-out-from-train, never validation | `fit()` returns held-out mR; `selected` tracks `hm`, not `m` | ✓ |
| Thresholds 5 % / inconclusive band 4–6 % | `THRESHOLD_LOW_PCT = 4.0`, `THRESHOLD_HIGH_PCT = 6.0`; verdict derived in code | ✓ |
| fp16 autocast on CUDA, fp32 control arm | `use_amp = amp and dev.type == "cuda"`; `--no_amp` arm run separately | ✓ |
| "Same splits" as B/32, ~70k crops | 1,200 + 1,200 images → **70,709 crops, 30,026 instances** — *identical* to the documented B/32 counts | ✓ (strong) |

**One discrepancy found, and it matters for how the headline is worded.**

The **ablation gate** arms are fitted on `"subj+obj+union"` — appearance only.
The **additive** arms are fitted on `"subj+obj+union+geom"` — appearance **plus
8 box-geometry features**. So the headline "captured −1.2 %" is properly
"appearance **and geometry**, composed additively with the prior, captured
−1.2 %".

This is **not** a protocol deviation: it is exactly what the B/32 run did, and
the pre-registration says "unchanged from the B/32 run except for the encoder
and device". It is faithful. But the two arms are not measuring the same
feature set, and the result document should say so. Prior work measured
geometry alone as converting **0.0 %** of headroom
(`tools/candidate_reranking_analysis.py`, three scorer families), so the
geometry block is unlikely to be carrying the negative — but that is an
INFERENCE from a different experiment, not something B controlled internally.

## B.2.2 What the ablation gate proves

**It proves the instrument works.** Three arms differ *only* in what happens to
the appearance block; the prior, the candidate set, the scorer family, the
optimiser, the epochs and the held-out split are identical.

- **ZERO** (8.58 mR@50) is the floor: the scorer sees a constant. Whatever it
  scores is what per-predicate bias alone buys.
- **SHUFFLED** (12.17) breaks the *correspondence* between features and
  instances while preserving the feature distribution exactly — same marginals,
  same covariance, same norms. Anything above ZERO here is what a scorer can
  extract from feature *statistics* without any instance-level information
  (it can still learn "this feature block has high variance", and the shuffle
  leaves per-class frequency intact).
- **REAL** (16.90) adds only one thing: the features belong to the instance
  being classified.

`REAL > SHUFFLED` is therefore a paired test of **instance-level
correspondence**, and it is the reason the ordering is evidence rather than
decoration. A feature block that carried no predicate information would score
the same shuffled as unshuffled — the B/32 investigation showed exactly that
failure mode on synthetic noise (−33 to −37 % captured), and showed the
opposite (100 % captured) with planted signal. The gate is two-sided
validated.

`SHUFFLED > ZERO` shows the scorer also exploits distributional structure, which
is why the *middle* arm is necessary: without it, `REAL > ZERO` alone could be
explained by capacity rather than correspondence.

**MEASURED: the margin is 8.32 points at L/14-336 versus a documented 5.91 at
B/32.** The instrument is not merely working; it is registering *more* signal
at the stronger encoder.

## B.2.3 What the gate does NOT prove

- **Not** that the signal is large. The gate is an ordering test with no
  effect-size threshold. REAL (16.90) is still **below the prior alone**
  (20.98).
- **Not** that the signal is *complementary* to the prior. Ordering says
  nothing about whether appearance errors and prior errors are decorrelated —
  and the additive arms show they largely are not.
- **Not** that the signal is appearance-specific in the additive arm, which
  also carries geometry (§B.2.1).
- **Not** that the signal survives the metric. The gate is measured on mR@50
  of a *replacement* scorer; the headline is measured on mR@50 of a
  *composition*. They are different quantities.

## B.2.4 Why low captured headroom means the formulation fails to convert

This is the crux, and it has a sharper reading than "the signal is too weak".

**MEASURED, three facts that only fit together one way:**

1. Appearance alone ranks **worse** than the prior alone: 16.90 vs 20.98 mR@50.
2. Increasing the appearance weight monotonically **hurts**: λ = 0.25 → +0.2 %,
   0.5 → −1.2 %, 1.0 → −1.2 %, 2.0 → −2.5 %.
3. Yet appearance improves specific predicates by large margins: `eating`
   +18.2, `riding` +8.2, `standing on` +6.2 rank-1 points.

A linear combination `a·P + b·A` can only beat `max(P, A)` when the two
rankers' errors are decorrelated. Facts 1 and 2 together say they are not
decorrelated enough: adding *any* positive amount of the weaker ranker drags
the stronger one down faster than the complementary cases lift it. Fact 3 says
the complementary cases are real but confined to classes that carry almost no
weight in a 50-class unweighted mean.

**So the correct statement is not "frozen CLIP cannot see predicates".** It is:
*frozen CLIP appearance, decoded linearly, is a worse predicate ranker than
label co-occurrence, and is not complementary enough to it for additive
composition to help — on a metric that gives its successes almost no leverage.*

That is a statement about the **formulation** (linear decoder, additive
composition, class-averaged metric), and it is why §F ranks the protocol and
the decision rule above perception.

## B.2.5 The ten alternative explanations, classified

**VERIFIED** = checked in code or measured here. **PLAUSIBLE** = consistent
with the evidence and not excluded. **SPECULATIVE** = no evidence either way.

| # | Explanation | Verdict | Basis |
|---|---|---|---|
| 1 | **Crop construction** (subject/object pad 0.1, union pad 0.05, union = joint bounding box) | **PLAUSIBLE, low prior** | Not varied in either run, so not excluded. But the gate passes strongly on exactly these crops, so they demonstrably carry instance-level predicate information. A crop defect would have to suppress *conversion* while preserving *detectability*, which is a narrow failure mode. |
| 2 | **PCA dimensionality (48)** | **PLAUSIBLE — the strongest surviving one** | Named in the pre-registration §6 and retained for comparability. 768→48 here vs 512→48 at B/32 is a harsher ratio. *Partial counter-evidence:* the gate is computed on those same 48 components and got **stronger**, so PCA-48 does not suppress L/14-336's extra signal to zero. Free to test (**N3**). |
| 3 | **Linear scorer** | **PLAUSIBLE** | Per-predicate linear over 48 PCs. Untested against a non-linear decoder at either encoder. Free to test (**N3**). |
| 4 | **Candidate restriction (K=5)** | **VERIFIED NOT the explanation** | Restriction *creates* the headroom being measured: coverage@5 = 89.90 %, oracle@5 mR@50 = **64.60** against a prior of 20.98. The 10.1 % of instances outside the top-5 cap the ceiling, not the floor. |
| 5 | **λ selection** | **VERIFIED NOT the explanation** | Both selection rules are reported. Held-out-selected: −1.2 %. Validation-argmax (the optimistic upper bound, selected on the split it is reported on): **+0.2 %**. Both far outside the 4–6 % band. |
| 6 | **Predicate imbalance** | **VERIFIED — but as a real mechanism, not a measurement artifact** | `on` has ≈5,600 validation instances against `eating`'s 22. Under an unweighted 50-class mean, +18.2 points on `eating` contributes 0.36 aggregate points. This is why the aggregate does not move, and it is hypothesis **H7**, not an error in B. |
| 7 | **Insufficient sample size** | **VERIFIED NOT decisive for the verdict; VERIFIED relevant for the per-predicate table** | 15,280 validation instances. The documented B/32 run-to-run spread is ~0.2 pp of captured headroom; the distance to the threshold is 4–6 pp — an order of magnitude larger. But tail classes are thin (`eating` n=22, `walking on` n=23), so the *per-predicate deltas* in §B.4 are noisy and must not be over-read. |
| 8 | **Preprocessing mismatch** | **VERIFIED NOT** | The processor's `crop_size` was independently checked to be `height=336, width=336`; the `preprocess_resolution: null` in the cache metadata is a `SizeDict`-vs-`dict` reporting artifact of transformers 5.x, not a resolution error. The precision half of this concern is exactly what the fp32 control arm tests (§H). |
| 9 | **Frozen encoder limitation** | **PLAUSIBLE — explicitly out of scope** | Pre-registration §6 states this bounds *frozen* features only. Not falsified, and not falsifiable by any variant of this experiment. |
| 10 | **Objective mismatch** | **PLAUSIBLE — and under-weighted so far** | The scorer is trained by cross-entropy over the top-5 candidates **on the natural distribution**, where agreeing with the prior is loss-minimising under 45 % head dominance. This is the same objection `PHASE4` §5 raised against the A1 and reranking conclusions, and it applies to B unchanged. It is *not* tested by varying the decoder family alone, so **N3 must include a class-balanced / tail-weighted objective arm** as well as a capacity arm. |

**No explanation was invented to be fashionable.** Items 1–10 are the ones the
pre-registration itself named plus the objective-mismatch objection this
repository already raised against its own earlier work. Two are VERIFIED as
excluded (4, 5), two are VERIFIED as excluded for the verdict though real in
other respects (6, 7), one is VERIFIED excluded on inspection (8), and the four
that survive (2, 3, 9, 10) are exactly the four the pre-registration and
`PHASE4` had already flagged. Three of the four — 2, 3 and 10 — are testable
on the **already-cached features with zero GPU time**.

---

# C. What B falsified

| Falsified | Evidence |
|---|---|
| **H1 of the pre-registration**: "the weak B/32 appearance signal is an artifact of encoder capacity" | Capacity raised (512→768 projection dim, 224→336 px, ~88M→~304M vision params); the gate margin did **not** shrink (8.32 vs B/32's 5.91 and 7.92); captured fell +0.6 %→−1.2 %. The encoder detects at least as much and converts less. |
| "A larger frozen encoder is a promising direction" | Directly measured against, twice. |
| "The appearance thesis is untested at the repository's own encoder" | It is now tested, at exactly `openai/clip-vit-large-patch14-336`. |
| The standing caveat in `docs/APPEARANCE_PROBE_FINDINGS.md` §5.1 — "confirming on GPU with L/14-336 is the single experiment that could overturn this verdict" | **Discharged.** The B/32 negative should now be read as a real negative, not a lower bound. |

**What B did NOT falsify**, stated so it cannot be read into the above:

- That a **fine-tuned** encoder could do better. B bounds what is linearly
  decodable from *frozen* features.
- That **PCA-48** is innocent. It compresses 768-dim blocks here against
  512-dim at B/32 — a harsher ratio, retained deliberately for comparability.
  *Partial counter-evidence:* the gate is computed on those same PCA-48
  features and got **stronger**, so PCA-48 demonstrably does not suppress
  L/14-336's extra signal to zero. It may still discard some.
- Anything about **SGCls/SGDet**, where object identity is not given.
- That the current architecture is **validated**. A negative here removes one
  candidate explanation for its failure; it supplies no evidence in favour.
- That **vision is useless for SGG**. B measures one protocol, one metric, one
  decoder family.

---

# D. What remains unresolved

1. **Does the model's contribution survive recalibration of the prior?**
   `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` §9 names this as the second of two
   falsifiable conditions. The first is now settled negatively by B. This one
   has **never been measured**. τ has only ever been applied inside
   `tools/decision_rule_probe.py`; `_apply_eval_logit_adjustment` in
   `evals.py` is a **VERIFIED no-op in every recorded run**.
2. **Is the frozen + linear + PCA-48 decoder the binding limit?** Untested.
   The cached L/14-336 features (359 MB, on disk) make this testable with
   **zero new GPU time**.
3. **Is the residual tool-matching gap in the A′ correction material?** The
   leak-free prior has not been scored through `evals.py` itself, only through
   the prior tool under the evaluator's alias merge.
4. **What does the protocol look like under the literature-comparable
   `predcls_multi` fields?** Arm E has them (multi_mR@50 47.37 vs headline
   21.16 — more than double). The prior arms do not. No comparison exists at
   all under the protocol the field actually publishes.
5. **Is the historical 67.09 / 22.64 reproducible?** Deferred, and bounded:
   the manifest records that four determining protocol fields were never
   written down, so a match would not confirm reproduction.
6. **SGCls / SGDet.** Entirely unmeasured.

---

# E. Combined A′ + B interpretation

The two experiments look like the same negative. **They are not**, and
conflating them would be the main interpretive error available here.

## E.1 They fail through different mechanisms

| | A′ (model, `evals.py`) | B (appearance probe) |
|---|---|---|
| Composition | `z-score(cosine) + 3.75 · log P` | `log P + λ · f(appearance)`, λ ≤ 2.0 |
| Effective prior : visual weight | **3.75 : 1** | 1 : λ — up to **1 : 2** |
| Is the visual term suppressed by the composition? | **Yes, measurably** | **No** — B explored weightings up to 7.5× more favourable to vision |
| What the negative shows | The composition cannot express the correction | The appearance ranker is genuinely weaker than the prior |

**MEASURED (A′ side), `runs/p6_prior_dominance_margin/`:** at α = 3.75, the
median swing needed to correct a prior error is **4.44 z-units**, while the
visual term is a difference of two unit-variance z-scores (swing std ≈ 1.41).
Only **35.6 %** of the prior's errors are reachable within a 3-unit (2.1 σ)
budget. **Roughly two thirds of the prior's errors are mathematically
unreachable by the model's visual term, whatever the encoder sees.**

**MEASURED (B side):** increasing the appearance weight *monotonically hurt* —
λ = 0.25 → +0.2 %, λ = 0.5 → −1.2 %, λ = 1.0 → −1.2 %, λ = 2.0 → −2.5 %. And
the pure-appearance replacement arm scores 16.90 mR@50 against the prior's
20.98. **Giving appearance more weight makes things worse, because its ranking
within the candidate set is worse than the prior's.**

So: A′ fails because the *composition* muzzles a term that might have helped.
B fails because the *term itself* does not rank better than co-occurrence,
even when un-muzzled. Fixing the composition would not rescue B; fixing the
encoder would not rescue A′.

## E.2 What the model actually does

**MEASURED**, arm E vs arm C (bucketing caveat in §F, H6.2):

| | head mR@50 | tail mR@50 |
|---|---:|---:|
| leak-free prior | 42.5 | 12.8 |
| model + historical prior | **54.1** | **5.8** |
| prior + τ=0.1 | 42.7 | **19.6** |

Corroborated independently by the predicate-group exposure diagnostic (GT
count → model's predicted count, arm E):

| group | GT | predicted | ratio |
|---|---:|---:|---:|
| `pose` | 687 | 283 | **0.41** |
| `contact` | 2,388 | 1,392 | **0.58** |
| `attribute` | 106 | 26 | 0.25 |
| `possession` | 11,270 | 12,386 | 1.10 |
| `spatial` | 22,744 | 23,354 | 1.03 |

The model **systematically under-emits exactly the visually-decidable groups**
(pose, contact) and over-emits the co-occurrence-driven ones (possession,
spatial). It preserves R@50 because R@50 is dominated by the head; it loses
mR@50 because mR@50 averages over classes and the classes it suppresses are
the numerous, small ones.

**INFERENCE:** the model is a *sharpener of the prior's dominant mode*. That is
precisely what §E.1's arithmetic predicts a unit-variance perturbation on a
3.75×-scaled prior would produce.

## E.3 The joint conclusion

> **VERIFIED FACT + MEASURED:** on VG150 PredCls under GT pairs, (i) the
> 79.9M-parameter CLIP model does not beat a leak-free co-occurrence prior —
> like-for-like it is behind on **both** R@50 (−1.03) and mR@50 (−1.52) — and
> (ii) a stronger frozen encoder does not recover the gap, capturing −1.2 % of
> oracle headroom while demonstrably carrying *more* raw appearance signal
> than the weaker one.
>
> **INFERENCE:** the binding constraints are the **decision rule and the
> evaluation protocol**, not perception and not capacity. A one-parameter
> recalibration with no image data beats the whole visual system by +5.70
> mR@50, and at the composition weight the evaluator actually uses, two thirds
> of the prior's errors cannot be corrected by any visual term of that scale.

---

# F. Bottleneck ranking

Ranked by **measured evidence**, not by intuition or by what would be
interesting to build.

### Rank 1 — H7: the evaluation protocol masks useful visual information
**Confidence: HIGH.**

*For (all MEASURED):* mR@50 is an unweighted class mean, so +18.2 rank-1
points on a 22-instance class contributes 0.36 aggregate points while a small
slip on `on` (n≈13,900) cancels it. 64.2 % of prior errors are
generic→generic and unresolvable by any encoder; only 12.6 % of GT triplets
carry a decidable predicate. K is inert under GT pairs (R@50 = R@100 in every
arm), so the headline is top-1 accuracy, not a ranking metric. The
literature-comparable `multi_mR@50` is 47.37 against a headline 21.16 — the
protocols disagree by more than a factor of two. And the alias merge silently
moves mR@50 by 0.71 points and the class count by two.

*Against:* it is the protocol the field uses; changing it is a scope change,
not a fix, and cannot be done to make a number look better.

*Cheapest decisive experiment:* **N0** — recompute every existing arm under
`predcls_multi` and under the decidable-only subset. **CPU, minutes, zero GPU.**

*Pursue?* **Yes — first.** It is nearly free and it reframes every other
result.

### Rank 2 — H5 + H4: calibration/objective mismatch and prior dominance
**Confidence: HIGH.** These are one mechanism seen from two sides — the scale
of the prior term, and the scale of the correction that would be needed.

*For (all MEASURED):* τ=0.1, a single scalar with no image data and no
training, converts +4.03 mR@50 at N=3,000 (+4.11 full split) — roughly **five
times** the model's entire contribution, and it beats the model on R@50 too
once the comparison is like-for-like. At α=3.75 the median correction needs a
4.44 z-unit swing and only 35.6 % of errors are reachable at 3 units. The
model's head/tail signature (+11.6 / −7.0) and its group-exposure profile are
exactly the predicted consequence.

*Against:* τ has never been applied inside `evals.py`, so its benefit is
established for the prior-only path, not for the composed system. It is
possible — untested — that the model's sharpening and τ's flattening cancel.

*Cheapest decisive experiment:* **N2** — model + τ-recalibrated prior vs
τ-prior alone. This is `PHASE4` §9's second condition verbatim. **One GPU
pass**, and if that pass **caches per-pair model logits**, every subsequent
composition question (α sweep, τ sweep, restricted softmax) becomes CPU.

*Pursue?* **Yes — this is the decisive experiment.**

### Rank 3 — H2 (refined): the frozen + linear + PCA-48 decoder is the limit
**Confidence: MEDIUM.**

*For:* the appearance replacement arm (16.90) is worse than the prior (20.98);
`captured_decidable` is only 1.5 %.

*Against:* the gate passes decisively at both encoders, and gains land exactly
on action/pose predicates. Geometry *alone* — 8 features, weaker than
appearance — reaches AUC 0.878 on `walking on` vs `on`. The information
demonstrably exists; the question is only whether a linear scorer over 48
principal components can reach it.

*Cheapest decisive experiment:* **N3** — a non-linear scorer and a
higher-dimensional PCA on the **already-cached** L/14-336 features. **Zero new
GPU time**; the 359 MB cache is on disk. Falsifies or confirms the "frozen +
linear + PCA-48" qualifier that every appearance conclusion currently carries.

*Pursue?* **Yes** — it is free, and it is the only remaining way to make B's
negative unconditional rather than qualified.

### Rank 4 — H8: implementation bug
**Confidence: MEDIUM (as a base rate, not a specific defect).**

*For:* five defects found so far — `--eval_batches 0` evaluating zero images
at exit 0; the canary guard that passed that run; the vocabulary-size
assertion; the CRLF provenance gap; and, this session, the alias drift that
made A′ not like-for-like. Two of those five would have silently corrupted
results.

*Against:* no *current* result is known to be wrong, and this session's find
strengthened rather than reversed the conclusion it touched.

*Cheapest decisive experiment:* **N1** (close the residual tool-matching gap,
CPU, ~1 min) plus the regression tests in
`POST_B_SOURCE_FORENSICS.md` §5 #8.

*Pursue?* **Yes, as hygiene** — not as a research direction.

### Rank 5 — H3: flat 50-way predicate normalization
**Confidence: LOW that it is the binding constraint.**

*For:* softmax over 51 classes with `on` holding 45.1 % of the prior's argmax
mass; a tail class must out-score all 50 alternatives simultaneously.

*Against, and this is close to decisive:* **experiment B already tested a
restricted decision.** Its whole design scores within the prior's top-5, not
over the flat 50 — and it captured nothing. Oracle@5 mR is 64.60 against the
prior's 20.98, so restriction *creates* enormous headroom and the scorer still
failed to convert it. Restricting normalization is not sufficient.

*Pursue?* **Low priority.** Largely answered by B's design; would only be
worth revisiting inside `evals.py` after N2.

### Rank 6 — H6: candidate-generation formulation — **REJECTED**
Coverage@5 = 89.90 %, oracle@5 mR = 64.60 vs prior 20.98. The prior is an
excellent *generator*; the failure is entirely in *ranking within* the
candidates. Generation is not the bottleneck. **Do not pursue.**

### Rank 7 — H1: visual encoder capacity — **REJECTED**
Directly falsified by B: capacity raised, measurable signal did not fall,
conversion did. **Do not pursue.** Specifically: do not propose a larger encoder, a
relation transformer, or a mixture of specialists on the basis of these
results — nothing measured suggests capacity is the constraint, and B is
direct evidence against it.

---

# G. Experiment matrix

Ranked by information gain per unit cost, on the evidence above. **None of
these has been launched.**

| id | Question | Control | Independent variable | Dependent variable | N | GPU | Wall time | Info gain | Confounders | Falsification criterion | Stopping criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **N0** | Does the picture change under the literature-comparable protocol and on decidable predicates only? | Every existing arm, unchanged inputs | reporting protocol (`predcls` → `predcls_multi`, all → decidable) | multi_R/mR@50, decidable mR@50 | 3,000 | **none** | **~5 min CPU** | **High** — reframes every result; arm E already has multi fields, no arm else does | Prior arms need a multi-hypothesis ranker written to match `_make_multi_triplet_predictions` exactly; a mismatch invalidates the comparison | If the model↔prior ordering is unchanged under `multi_`, H7 is weakened as an *explanation for the model's failure* (it remains true about the metric) | One pass; no tuning |
| **N1** | Is the like-for-like A′ correction exact, or is there residual tool-matching drift? | arm E, untouched | scoring implementation (prior tool vs `evals.py`) | R@50, mR@50 of the leak-free prior | 3,000 | **none** | **~1 min CPU** | Medium — closes the one open gap in correction 3′ | Requires running the prior through `evals.py` with the model term zeroed, or `--aliases` in the prior tool | If `evals.py` gives the prior ≠ 67.93 ± 0.05, the −1.52 estimate must be revised | Single number; done |
| **N2** | **Does the model's contribution survive recalibration of the prior?** (`PHASE4` §9 condition 2) | τ-recalibrated prior **alone**, same images, same τ | presence of the model term | Δ mR@50, Δ R@50, head/body/tail | 3,000 | **yes, 1 pass** | **~60 min** | **Highest** — the last named falsifiable condition; settles whether to abandon or iterate | τ interacts with `freq_bias_alpha`; the historical prior's possible leakage inflates the model arm (conservative direction); bucket instability across N | Δ mR@50 < +1.0 ⇒ the model adds nothing even after recalibration ⇒ **abandon the architecture** | One pass at N=3,000. Do **not** sweep τ on the model arm — τ is fixed at 0.1 from the pre-registered CPU sweep |
| **N2a** | (rider on N2, not a separate run) Sweep α and τ offline | — | composition weights | same | 3,000 | **none, if N2 caches per-pair logits** | minutes | High | Caching must record pre-composition logits, not post | — | — |
| **N3** | Is B's negative an artifact of the linear PCA-48 decoder **or of the training objective**? | B's own λ grid and gate, unchanged | scorer family (linear/PCA-48 → MLP; PCA-48 → 128/full) **× objective (natural CE → class-balanced / tail-weighted)** | headroom captured | 1,200 | **none** — reuses the 359 MB cache | ~10–20 min CPU | High — removes three of the four surviving alternatives (§B.2.5 items 2, 3, 10) | Higher capacity will overfit; the B/32 run already showed a naive scorer letting ZERO beat REAL. The image-level held-out split and the two-sided synthetic validation must be kept | If a non-linear scorer still captures < 5 %, B's negative becomes unconditional for frozen features | Fixed grid, decided in advance; report the whole grid |
| **N4** | Do the head/tail deltas survive a fixed, train-derived bucketing? | arm E and arm C metrics, unchanged | bucket definition (eval-split counts → frozen train counts) | head/body/tail mR@50 | 3,000 | none | ~2 min CPU | Medium — bounds a known artifact in a headline table | Needs recomputation from per-predicate recalls, which arm C does not currently emit | If the +11.6/−7.0 signature shrinks materially, §E.2's magnitude claim must be softened | One pass |
| **N5** | Restricted-softmax / candidate-shortlist formulation inside `evals.py` | flat 51-way softmax | decision support set | mR@50 | 3,000 | yes | ~60 min | **Low** — B already tested a restricted decision (top-5) and it failed | Shares B's failure mode | — | **Not recommended before N2** |
| **N6** | Pairwise / binary predicate decision | flat classification | decision formulation | mR@50 on confusable pairs | — | yes | hours | Low–medium — `tools/candidate_reranking_analysis.py` already measured 0.0 % captured by 200 pairwise probes | Same CE-on-natural-distribution objection that weakened that conclusion | — | **Not recommended before N2** |
| **N7** | Prior-residual / tail-focused objective | flat CE | training objective | mR@50 | — | **retraining, ≥10 h** | — | **Low** — A1 prior-residual already falsified (`d845468`) | — | — | **Rejected on current evidence** |
| **N8** | SGCls / SGDet | PredCls | task | R/mR@50 | — | yes, large | many hours | Medium — genuinely untested, and co-occurrence is weaker there | A different protocol; not a fix for this one | — | **Scope change, not a next step** |

---

# H. Recommended next experiment

**N0 and N1 first** (both CPU, both under 10 minutes together), because they
are nearly free and they change how N2's result must be read.

**Then N2, and only N2, on GPU.**

> Does the model's contribution survive after the prior is recalibrated?
> Model + τ=0.1-adjusted prior, versus the τ=0.1-adjusted prior alone, on the
> identical 3,000 validation images.

It is the last of the two falsifiable conditions `PHASE4` §9 set for the
architecture being worth pursuing. B settled the first, negatively. Every
model-vs-prior comparison so far has used the *un-recalibrated* prior; since
recalibration adds +4.03 mR@50 to the baseline for free, N2 raises the bar the
model must clear rather than lowering it.

**It must be pre-registered the way B was** — thresholds, τ, sample size and
verdict fixed in writing before the run — and it should **cache per-pair
model logits**, which converts every subsequent composition question (N2a: α
sweep, τ sweep, restricted softmax) from a GPU experiment into a CPU one.

**N3 in parallel**, since it needs no GPU and cannot contend with N2.

Preparation for all of these — exact commands, run directories, provenance
requirements, success criteria and a pre-registration draft for N2 — is in
§Preparation below. **Nothing has been launched.**

---

# I. Why the rejected alternatives were rejected

| Rejected | Why — on measured evidence, not preference |
|---|---|
| **A larger or fine-tuned visual encoder** | B raised capacity, the gate margin did not shrink (8.32, against B/32 records of 5.91 and 7.92), and conversion fell (+0.6 %→−1.2 %). Capacity is direct evidence against itself here. Fine-tuning remains formally untested, but proposing it now would be architecture-shopping: no measurement points at it. |
| **A relation transformer / mixture of specialists** | Same reason. These are capacity-and-inductive-bias proposals, and capacity has been falsified as the constraint. Nothing measured motivates either. |
| **A neural reranker over the prior's candidates** | Four formulations have now failed: additive prior-residual (A1), shared linear reranking, per-confusion tournament (0.0 % captured over 200 pairwise probes), and B's appearance scorer. B is the strongest of these because it passed its validity gate. |
| **Changing the candidate generator** | Coverage@5 is 89.90 % and oracle@5 mR is 64.60 vs a prior of 20.98. Generation is not where the loss is. |
| **Retraining anything** | ≥10 GPU-hours, and `PHASE4` §7 already ranked it "not justified by current evidence". Nothing since has changed that; two further negatives have accumulated. |
| **Restricted softmax (N5) as the *next* step** | B's design *is* a restricted decision (prior top-5) and it converted nothing. Testing restriction again before N2 would spend GPU on a question B has largely answered. |
| **Sweeping τ or α on the model arm to find a better number** | This would be tuning on the test split. τ=0.1 is fixed by the pre-registered CPU sweep; α=3.75 is the historical protocol value. Any sweep must be offline on cached logits and reported in full. |
| **Reporting `multi_mR@50 = 47.37` as the headline** | It is the literature-comparable field and *should* be reported — but switching headline metrics after seeing that it is more than double the current one would be exactly the goalpost move this project has avoided so far. N0 reports it for **every** arm, or not at all. |

---

# J. Exact stopping point

**Stop here.**

Completed this session, all recorded under `runs/` with full provenance:

- Experiment **B** (pre-registered), plus its **fp32 control arm** — the
  pre-registered scope, in full. Both exit 0; 20.6 min and 67.0 min on the L4;
  identical verdicts (**H0 SUPPORTED**).
- Its secondary decidable-denominator diagnostic, proven inert bit-for-bit.
- Four CPU-only, zero-GPU analyses that did not touch the running experiment:
  the alias-scheme control, the prior-dominance margin measurement, the full
  test suite (288 passed, 1 skipped), and the GT class-count audit.
- Five documents: B's result report, this reassessment, the source forensics
  and code audit, and the claim audit.

**Not started, deliberately:**

- N0, N1, N2, N2a, N3, N4 — designed, costed and pre-registered in outline
  only.
- No architecture change, no retraining, no second GPU experiment after B.
- No historical artifact, manifest, raw dataset or split membership modified.
- No pre-registered criterion altered.
- `docs/MODEL_VS_LEAKFREE_PRIOR_RESULT.md`, `docs/APPEARANCE_PROBE_FINDINGS.md`
  and `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md` **not edited** — their needed
  corrections are recorded here and in the claim audit, for a human to apply.

The next action requiring a decision is whether to pre-register and run **N2**.
That decision is not taken here.

---

# Preparation (Phase IX) — ready to launch, not launched

## N0 — protocol re-frame (CPU, ~5 min)

Needs a small new tool that ranks (pair × predicate) hypotheses from the prior
exactly as `_make_multi_triplet_predictions` does, so the prior arms gain
`multi_*` fields comparable with arm E's.

```
python tools/run_experiment.py --name p7_protocol_reframe_3000 \
  --note "Recompute prior arms under predcls_multi and on the decidable subset" -- \
  python tools/<new>_prior_multi_baseline.py --limit 3000 \
    --prior datasets_vg150_clean/frequency_prior_train.json \
    --taus 0.0,0.1 --predicates_per_pair 10 \
    --out runs/p7_protocol_reframe_3000/metrics.json
```

- **Inputs:** `frequency_prior_train.json` (`54c8c910…`), `validation.jsonl` (`74d99779…`)
- **Outputs:** `multi_R@{20,50,100}`, `multi_mR@{20,50,100}`, decidable-only mR@50, per arm
- **Success:** reproduces the τ=0 headline `R@50 66.80 / mR@50 21.98` in its single-predicate mode — the same self-check the alias control used
- **Failure:** if it cannot reproduce that, the multi numbers are not comparable and must be discarded

## N1 — like-for-like closure (CPU, ~1 min)

Score the leak-free prior through `evals.py` itself with the model term
zeroed, or add an opt-in `--aliases` flag to `tools/frequency_prior_baseline.py`.

- **Success:** `evals.py` gives the prior `R@50 = 67.93 ± 0.05`, confirming
  `runs/p6_alias_control_3000/`
- **Failure:** a larger discrepancy means the two tools differ beyond
  aliasing, and correction 3′ must be re-derived

## N2 — model versus the recalibrated prior (GPU, ~60 min) — **PRE-REGISTRATION DRAFT**

**Question.** Does the model contribute anything over a leak-free prior that
has already been recalibrated by the one parameter known to be worth +4.03
mR@50?

**Arms**, identical first 3,000 validation images, `num_gt` verified equal:

1. model + τ=0.1-recalibrated leak-free prior (GPU)
2. τ=0.1-recalibrated leak-free prior alone (CPU — already have it: D/D′)
3. model + un-recalibrated leak-free prior (GPU, same pass)

**Fixed in advance, not to be renegotiated:**

- τ = **0.1**, from `runs/p4_decision_rule_sweep/`. No τ sweep on the model arm.
- α = **3.75**, the historical protocol value. Any α sweep is offline on cached logits and reported in full.
- Both arms scored under **one** alias scheme, and the effective class count reported beside every mR@50.
- Δ = arm 1 − arm 2.

| Outcome | Criterion | Meaning |
|---|---|---|
| **SUBSTANTIAL** | Δ mR@50 ≥ +3.0 and Δ R@50 ≥ −1.0 | The model carries information beyond a recalibrated leak-free prior. Architecture work becomes justifiable. |
| **MODEST** | +1.0 ≤ Δ mR@50 < +3.0 | Real but small; not on its own a basis for architecture work. |
| **NEGLIGIBLE** | Δ mR@50 < +1.0 | The model adds nothing even after recalibration ⇒ **the defensible contribution is methodological, and the architecture should be abandoned rather than iterated.** |

Thresholds are copied verbatim from
`docs/MODEL_VS_LEAKFREE_PRIOR_PREREGISTRATION.md` so the two experiments
remain comparable. Verdict derived in code, not written by hand.

**Required of the run:** `tools/run_experiment.py` wrapper; clean working tree;
canary PASS on all 14 flags **plus `eval_sgg_use_vg_aliases`**; SHA256 of
checkpoint, both priors, and the split verified before launch; **per-pair
pre-composition logits cached to disk** so N2a needs no GPU.

**Run directory:** `runs/p7_model_vs_recalibrated_prior/`
**Not resumable** (the evaluator writes `metrics.jsonl` only on completion), so
N=3,000 caps the loss from a failure at ~60 min.

## N3 — decoder-family control on cached features (CPU, ~10–20 min, zero GPU)

```
python tools/run_experiment.py --name p7_probe_decoder_family \
  --note "Is B's negative an artifact of linear+PCA-48? Reuses the existing cache; no re-encoding." -- \
  python tools/<new>_appearance_probe_decoder.py --score \
    --cache runs/p6_appearance_probe_l14_336/clip_l14_336_cache.pt \
    --scorers linear,mlp --pca 48,128,full --objectives natural,balanced \
    --out runs/p7_probe_decoder_family/metrics.json
```

- **Inputs:** the existing 359 MB cache — **no re-encoding, no GPU**
- **Must keep:** the image-level held-out split, held-out λ selection, the
  ablation gate, and the two-sided synthetic validation that made B's negative
  interpretable
- **Success:** the linear/PCA-48 cell reproduces B's −1.2 % exactly
- **Falsification:** if a non-linear or higher-dimensional scorer captures
  ≥ 5 %, B's negative is decoder-bound and must be re-scoped
- **Stopping:** one fixed grid, decided before running; the whole grid reported
