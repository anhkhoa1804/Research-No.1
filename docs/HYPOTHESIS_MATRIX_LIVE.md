# Live hypothesis matrix — H1–H10 accounting scheme

Updated **2026-09-02 after `p59`-`p62`**. `p54` (held-out TEST split, GPU) is
**COMPLETE** (exit 0, 3 h 31 m) and its CPU replications `p59`/`p61`/`p62` are
**COMPLETE**: the registered replication verdict is **REPLICATED** on all seven
items. `p60` (estimator-matched geometry) is **COMPLETE**, all gates pass. Numbering follows the
directive's scheme. Note `docs/SUCCESSOR_HYPOTHESES.md` uses an inverted
numbering (its H1 = readout, its H4 = representation); that file is historical.

Classes: **VF** verified fact · **MR** measured · **INF** inference · **HYP** hypothesis.

**Reference scale for every WPRD claim** (0.5 = no image-conditioned relational
information):

```
prior (control)             0.5000   exactly, CI [0.5000,0.5000], EVERY stratum
                                     and (p56) every supervision bin
random null                 0.5047
PURE text head (EVALUATED)  0.5542
best probe on rel_feat      0.5732   (p55 A_ce_all; ties the classifier head)
PURE classifier (discarded) 0.5728
geometry linear             0.5961   train-fitted, LBFGS softmax CE
geometry MLP  (box ceiling) 0.6149   train-fitted
geometry MLP, MATCHED       0.5976   (p60) cross-fitted on val, p55's ESTIMATOR
```

**`p60` settles the estimator confound.** Every "below geometry" statement used
to contrast a validation-cross-fitted `rel_feat` probe against a *train-fitted*
geometry probe. Under `p55`'s estimator and `p37`'s folds applied identically to
both, geometry reads **0.5976** and `rel_feat` **0.5732** (`A_relfeat`
reproduces `p55` to **+0.0000**), CIs **disjoint**, `regime_gap = -0.0015`
-> **REGIME-STABLE**. The comparison is sound and the framing stands.

---

| # | Hypothesis | Supporting evidence | Contradicting evidence | Status | Next test |
|---|---|---|---|---|---|
| **H1** | **Prior dominance** | **MR** prior alone 66.59 R@50, model adds +0.575. **MR** prior is 85.8% of term variance (`p42`). **MR** deployed system 97.45% pair-constant vs reality's 69.23% (`p44`) | — | **ESTABLISHED** (not novel — Zellers; Plesse) | settled |
| **H2** | **Pair identity carries the composed gain** | **MR** `full − pair_matched_null` = +0.031 ± 0.188 (`p29`), replicated by `p26` and `p32` | **MR** WPRD is invariant to the between-group part (`p42`), so this is about the *composed metric*, not the term | **ESTABLISHED for the composed gain** | settled |
| **H3** | **Calibration / decision artifact** | **MR** tau moves mR 22.3→27.2 with no image access. **MR** geometry has best WPRD and worst mR@50. **MR** Spearman(mR@50, WPRD) −0.65 / −0.63 (`p49`,`p53`) | **MR** a learned per-class rule cannot match tau (`p25`,`p29`) | **ESTABLISHED for the mR headline** | settled |
| **H4** | **Image-conditioned signal exists** | **MR** WPRD 0.5542 / 0.5728, CI excludes 0.5 by ~9 half-widths. **MR** non-layout residual excludes chance in both heads (`p42`). **MR NEW** on the **41.1% of validation groups with ZERO training rows** both heads still read above chance (0.5412 / 0.5478) — the signal is **not pair-specific memorisation** (`p56`) | **MR** converts to ~0 at the additive operating point (`p29`); evaluated head has no tail grounding (`p35`) | **ESTABLISHED BUT WEAK**, and now shown to **generalise across pairs** | cross-model |
| **H5** | **Readout bottleneck** | — | **MR** `p37` `P* − C = −0.0127`. **MR** `p55` a better-tuned MLP reaches 0.5732 vs the classifier's 0.5728 — it **ties**, and never exceeds, across two probe families and two hyperparameter settings | **DEAD** (and `p37`'s "cannot even match" prose corrected to "cannot exceed") | none — falsified |
| **H6** | **Representation bottleneck** | **MR** every probe on 768-d `rel_feat` sits **0.023–0.036 below** a linear model on 19 box numbers (`p37`, `p55`). **MR** `R9` rel_feat+geometry (0.5735) < geometry alone (0.5961) — adding the representation **destroys** information. **MR** collapse R² only 18.25%, so it is not an object-identity re-encoding. **STRENGTHENED BY ELIMINATION**: the objective route is now refuted on `rel_feat` itself (`p55`) and the supervision route is refuted (`p56`). **MR `p60` — the confound test the claim most needed.** Under ONE estimator and ONE regime applied identically, geometry 0.5976 vs `rel_feat` 0.5732, `delta_est = +0.0244`, CIs disjoint, all gates pass: the deficit is **not** a fitting artifact. **MR `p60`** `C_fusion` 0.5922 < `B_geometry` 0.5976 — the additive-fusion route is closed on a *matched* measurement. **MR `p59`** the whole verdict replicates on the held-out TEST split (`P*−C = −0.0116`, `P*−G = −0.0320`). **MR `p65` (preregistered): the minimal `dx_rel`/`dy_rel` fix is NEUTRAL** (+0.0050, threshold +0.02) **and fusing the CLEANED (group-centred) `rel_feat` with geometry is actively HARMFUL** (−0.0320, past the −0.02 threshold) — worse than raw fusion, not merely redundant. Four additive-fusion attempts across two estimator families (`p37`, `p60`, `p65`) and the one minimal literal-feature-injection fix (`p65`) have now all failed to beat geometry alone | **MR** `p55` withdraws `p37`'s R5: group-centring gains under a linear probe (0.5807) but **collapses under an MLP** (0.5173), so the "active distractor" reading is a linear-probe property, not a representation property. **MR `p65`** confirms this from the fusion side too — the group-centred representation, expected to be the cleanest candidate, still drags a good MLP+geometry estimator below geometry alone | **SUPPORTED — the only surviving structural account, HELD-OUT REPLICATED (`p59`), and now near-exhausted for every additive/frozen-readout fix tried (`p65`)** | joint-training pilot (untested) or cross-model |
| **H7** | **Geometry shortcut** | **MR** geometry linear 0.5961 / MLP 0.6149 beat every head and every probe, disjoint CIs; margin grows toward the tail (+0.123 tail–tail, `p40`); gap widens to +0.090 on standard VG150 (`p53`); **replicates on TEST 0.5916 vs 0.5446/0.5712 (`p59`)**; **survives estimator matching (`p60`)** | **MR** `p41`: on the field's composed metric the model **beats** geometry at tau ≤ 0.05. Discrimination ≠ calibration. **MR `p64`: this composed-metric reversal REPLICATES on TEST, 4/4 tau winners match `p41` exactly** (not pre-registered). **MR `p64` corrected control**: a disclosed bug in `p41`'s nested-CV weight selection, fixed and run leakage-free, shows MODEL wins 7/8 val+test cells — most of geometry's tau≥0.1 advantage was eval-split leakage in how the mixing weight was chosen | **ESTABLISHED for discrimination; REVERSED for the composed metric at tau≤0.05, HELD-OUT REPLICATED (`p64`); the tau≥0.1 reversal is mostly a selection-leakage artifact under a corrected, non-pre-registered control** | cross-model geometry baseline |
| **H8** | **Supervision scarcity** | **MR** the dataset fact is real and unchanged: 19.0% of raw-name train groups (58.1% VG150-restricted) have ≥2 distinct predicates; tail–tail is 0.04% of contrastive supply (`p50`,`p52`) | **MR `p56` PRIMARY H8-REFUTED**: rho(train contrasts, cell WPRD) = −0.0085, ns. **MR** Part C **NOT A LEVER**: fitting a discriminator on a pair's own train rows, rho = +0.0108, ns. **MR** the decisive one — **geometry, which accumulates no per-pair capacity, shows a LARGER trend across supervision bins (+0.061) than the checkpoint (+0.020/+0.034)**, so the drift is population difficulty, not supervision. **MR `p55`** pair-balanced sampling *loses* to plain CE (−0.0081), and a within-pair contrastive objective loses by −0.0281 at a matched budget | **REFUTED as the explanation** — the dataset fact stands, its causal role does not | a retrain under balanced supervision is the only test that could revive it; not affordable, not claimed |
| **H9** | **Annotation / data artifact** | **MR** VG multi-labels one instance pair; those rows tie at 0.5 | **MR** excluding same-instance rows **raises** WPRD 0.5542→0.5592 — the artifact is **conservative** | **PRESENT BUT CONSERVATIVE** | per-bucket quantification |
| **H10** | **Candidate-generation bottleneck** | — | **MR** GT in prior top-5 for 89.7%; scorer EXHAUSTED 9/9 (`p28`) | **FALSIFIED** | settled |
| **+H11** | **Benchmark metric mismatch** | **MR** Spearman(R@50, WPRD) +0.741 / **+0.951**; Spearman(mR@50, WPRD) −0.650 / −0.629, both p<0.05, replicating across two populations (`p49`,`p53`) | **effective n ≈ 3** (12 arms cluster into ~3 families, one cache, one checkpoint); Pareto correlation ns on both | **SUGGESTIVE, NOT ESTABLISHED** — but now **HELD-OUT**: `p61` on TEST gives +0.914 (p=0.0005) and −0.727 (p=0.0080). The effective-n≈3 objection is untouched. | **cross-model — the gate on Paper B** |

## Retired this cycle

| | |
|---|---|
| **Objective bottleneck** | **REFUTED ON BOTH CHANNELS.** `p48` box geometry −0.0143; `p55` `rel_feat` **−0.0281** with the supervision budget matched and the optimisation budget matched. Directly optimising WPRD's surrogate scores worse on WPRD than plain CE. |
| **Group-centring as a design principle** | **WITHDRAWN.** `p37` R5 = 0.5807 under a linear probe; `p55` `F_ce_groupcentred` = **0.5173** under an MLP. Not robust to the readout family. |
| **"A probe cannot even match the classifier head"** | **CORRECTED.** `p55` `A_ce_all` = 0.5732 vs classifier 0.5728. It ties. `p37`'s registered verdict is unaffected (0.5732 < C + 0.01); only the prose was too strong. |

## Current scientific model

The prior decides most rows. The checkpoint carries a real but weak
image-conditioned relational signal that **generalises to pairs it never saw**,
and that signal sits **below what 19 numbers from two rectangles supply**.

Three routes out of that have now been closed by measurement rather than by
argument: the **readout** (`p37`, `p55` — a tuned probe ties the existing head
and never beats it), the **objective** (`p48`, `p55` — the within-pair surrogate
loses to plain CE on both channels at matched budget), and the **supervision**
(`p56` — discrimination does not track the supervision a pair received, and
geometry, which cannot benefit from per-pair supervision at all, shows a larger
bin trend than the checkpoint does).

**What remains is H6.** It no longer stands by elimination alone. `p57` named a
deficit (the encoder discards relative position: `dx_rel` R² = **0.052**), `p60`
showed the deficit is not an artifact of how the two channels were fitted, and
`p59` showed the whole verdict holds on held-out data. That is a positive,
replicated, confound-tested account.

What is still missing is a **causal** demonstration: no intervention has yet
raised WPRD. `p58` (fusion), `p60` (`C_fusion` 0.5922 < `B_geometry` 0.5976),
`p55` (objective), and now `p65` (the minimal `dx_rel`/`dy_rel` fix, NEUTRAL;
cleaned-fusion, HARMFUL) all failed to. So H6 is well-evidenced as a
**description** and unproven as a **lever** — and a model paper needs the
lever, not the description. Every additive, frozen-readout intervention this
programme has designed has now been tried; the one remaining untested class
is a **jointly retrained** encoder (not a post-hoc probe), which needs an
actual GPU training pilot before H6 can be called exhausted rather than
merely well-evidenced.

## What would overturn this

- **Cross-model.** WPRD ≈ 0.5 on published checkpoints ⇒ we measured a PURE
  quirk; H11 and the benchmark claim collapse. **Still the binding gate.**
- ~~**`p54`** (in flight)~~ **DISCHARGED.** `p59`/`p61`/`p62` meet all seven
  registered criteria on the held-out TEST split. Nothing above is
  split-specific. `p60` additionally removed the estimator confound.
- A retrain under balanced within-pair supervision beating plain CE ⇒ H8 revives
  in a causal form `p56` could not test.
