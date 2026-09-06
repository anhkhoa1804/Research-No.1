# RESEARCH_STATE — living research map

Updated: 2026-09-03. Branch: `research/architecture-breakthrough`.
This file is the single place that says what is currently believed, what is not,
and what is running. Every claim carries its evidential class.

> **Denominator warning.** Three incompatible baselines appear in this project's
> history and must never be compared. `docs/EXPERIMENT_MATRIX.md` quotes the
> **historical** prior on the **full 10,401-image** validation split
> (64.37 / 20.30). Everything in sections 2–4 below uses the **train-derived
> leak-free** prior on the **3,000-image** analysis subset (66.80 / 21.98), and
> section 4b uses that same leak-free prior on the **full 10,401-image** split
> (66.59 / 22.30). Different prior, different N. A delta against one is not a
> delta against the other.

> **PAPER A = FROZEN, READY TO DRAFT** (2026-09-03). Full freeze audit,
> claim-by-claim evidence table, manuscript blueprint, and the binary
> decision: `docs/PAPER_A_FREEZE_AUDIT.md`. Do not add a new Paper A
> scientific hypothesis without first re-opening that audit; documentation
> and manuscript-drafting work may proceed directly from it.

---

## 1. Strongest validated result

**MEASURED.** A leak-free train-derived frequency prior alone reaches
R@50 66.80 / mR@50 21.98 on the 3,000-image analysis set (tau=0), and the
79.9M-parameter historical visual model adds **+0.673 R points** on top of it in
the additive alpha=3.75 formulation.

The model's contribution is real, small, and **not** a ranking improvement: it
worsens mean GT rank (1.83 → 2.78) and worsens more rows than it improves, while
still producing +256 net beneficial top-1 flips. It behaves as a bounded
tie-breaker on an argmax the prior has almost already decided.

## 2. Strongest negative results

1. **Candidate generation is not the bottleneck. H1 falsified.** GT is inside the
   prior's top-5 for 89.7% of rows, top-3 for 85.5%. (`runs/p17`)
2. **The raw model score is not a ranker.** Allowed to decide inside the prior's
   candidate set it costs up to −15.6 R points, and clears the R@50 floor only
   in the 10%-of-rows margin budget. (`runs/p17`)
3. **The additive composition already beats the best pure reranking** (+0.673 vs
   a best-cell +0.423), so composition is not what is failing. (`runs/p17`)
4. **"The model restores recall that tau spent" is falsified in the operating
   region.** ΔR *falls* from +0.673 to +0.205 across tau ∈ [0, 0.1]; it grows
   only at tau ≥ 0.2 where the prior has already lost 4.8–55.0 R points.
   (`runs/p17`)
5. **A learned calibration does not even match tau.** With beta chosen
   out-of-fold and resampled over 5 fold partitions, a learned per-class rule
   with no visual input averages a Pareto gap of **−1.205** and its shuffled
   null **−1.137** — *below* the tau frontier — and neither clears the R@50
   floor on any partition (0/5). Tuning tau beats learning a per-class rule.
   (`runs/p25`; the single-partition `runs/p22` read +0.059 / +0.224 and was
   the optimistic draw.)
6. **Appearance scoring on frozen CLIP L/14-336 adds nothing beyond
   calibration.** (`docs/APPEARANCE_TAU_INTERACTION_RESULT.md`)

## 2b. The result that reframes the project (`runs/p26`, 2026-09-01)

**MEASURED, pre-registered verdict E2 SUPPORTED.** Permuting the model term only
among rows sharing the same (subject, object) category — destroying image
content, preserving pair identity — costs **−0.114 ± 0.265** Pareto points, with
`full` ahead on **1 of 5** partitions. Destroying pair identity instead costs
**+3.163**.

| arm (5 partitions, nested, out-of-fold) | Pareto mean ± sd | R@50 | floor |
|---|---|---|---|
| `full` (prior + model) | +1.911 ± 1.056 | 66.656 ± 0.257 | 4/5 |
| `pair_matched_null` (image destroyed) | **+2.026 ± 0.977** | 66.596 ± 0.236 | 4/5 |
| `shuffled_model` (identity destroyed) | −1.137 ± 0.838 | 65.856 ± 0.278 | 0/5 |

Two structural facts explain it:

- **VF** `ensemble_alpha = 0.0` — the "model term" is **100% the CLIP text
  branch**; the visual classifier head contributes **exactly zero**. Everything
  the C′ work attributed to "the model" is the text path.
- **MR** **86.87%** of the term's variance is *between* (subject, object) groups;
  only 13.13% is within-group, and within-group is the only place image
  conditioning can live.

Effect-size bound: image conditioning ≲ **0.2** Pareto points, point estimate
wrong-signed.

## 3. Current bottleneck

**The signal being converted is a pair prior, not visual evidence.**
Superseded reading (pre-`p26`): "a representation bottleneck for ranking, with a
decision-formulation bottleneck for conversion". That was correct about the
mechanism and wrong about the source.

MEASURED: `model_only`, fitted optimally inside the candidate set, is negative in
all four cells and fails the R floor at tau=0.05 — the representation cannot rank
on its own. It carries a stable **+0.68 to +0.79 R points** over prior-derived
features, cross-fitted and null-controlled. `runs/p26` establishes that this
increment is **(subject, object) identity expressed in text-embedding space**,
not image content.

So the bottleneck is no longer "how do we convert the model's visual evidence".
There is no visual evidence in this term to convert. What remains is a better
*pair-conditioned* predicate distribution than the frequency prior supplies, and
the open question is whether it can be distilled without vision at all.

## 4. Currently active hypothesis

**EXPLORATORY, screening only.** A candidate-restricted, class-reweighted
decision rule converts ~3.4× more of the model's complementary information into
Pareto movement than the additive arm, at the same R@50 floor.

With beta selected inside the training folds only, **resampled over 5
independent fold partitions** (`runs/p25`, tau=0, k=5). The single-partition
number (`runs/p22`, +2.894) was a favourable draw and is superseded:

| arm | R@50 mean ± sd | Pareto gap mean ± sd | Pareto min | clears floor |
|---|---|---|---|---|
| achieved additive C' | 67.474 | +0.861 | — | 5/5 |
| prior_only (no vision) | 65.861 ± 0.265 | **−1.205 ± 0.793** | −1.937 | **0/5** |
| shuffled-model null | 65.856 ± 0.278 | **−1.137 ± 0.838** | −1.922 | **0/5** |
| **full (prior + model)** | **66.656 ± 0.257** | **+1.911 ± 1.056** | **+0.244** | **4/5** |

Two findings of opposite sign, both load-bearing:

- **WEAKENED.** The absolute gap is partition-dependent (+0.244…+2.941) and the
  arm fails the R@50 floor on 1 of 5 partitions. It is **not** yet a reliably
  usable operating point.
- **STRENGTHENED.** The *separation* from both nulls is ≥ **+1.84** Pareto
  points on **every** partition (`full − prior_only` min +1.904,
  `full − shuffled` min +1.842). The separation is the defensible quantity; the
  absolute level is not.
- **A learned calibration does not even match tau.** The no-vision arms average
  −1.2 Pareto points — *below* the tau frontier — and clear the floor 0/5. Tuning
  tau beats learning a per-class rule.

This is a 3,000-image screening result and is **not** a headline.

## 4b. Full-split confirmation of that hypothesis (`runs/p29`, 2026-09-01)

**MEASURED, pre-registered verdict CONFIRMED.** The section-4 screening result
re-run unchanged on the full 10,401-image cache. Prior-only baseline on this
split: R@50 **66.593** / mR **22.304**.

Registered partition (salt 0), tau=0, k=5:

| arm | R@50 | mR@50 | Pareto gap | floor 66.5 |
|---|---|---|---|---|
| `prior_only` (no vision) | 65.357 | 27.178 | +0.378 | **FAIL** |
| **`full`** | **66.760** | 25.049 | **+2.745** | **ok** |
| `shuffled_model` | 65.348 | 27.157 | +0.352 | **FAIL** |
| `pair_matched_null` (image destroyed) | 66.704 | 25.009 | **+2.705** | **ok** |

All three registered conditions pass: floor 66.760 >= 66.5; gap +2.745 > +1.5;
separation +2.368 over `prior_only` and +2.394 over `shuffled_model`, both > +1.0.

Resampled over 5 partitions, `full` = **+2.947 ± 0.190**, floor **5/5** — against
screening's +1.911 ± 1.056 and 4/5. The sd falls 5.6x. **The screening
instability was a small-sample artifact, not a fragile effect**, and the
pre-registration's addendum was wrong in the conservative direction.

**And the criterion could not test what mattered.** `pair_matched_null` passes
every registered condition as well. Paired over partitions,
`full − pair_matched_null` = **+0.031 ± 0.188** (negative on 1 of 5; t-based 95%
interval [−0.20, +0.26]). This replicates `p26`'s −0.114 ± 0.265 at 3.48x the
rows with a tighter bound: **image conditioning contributes <= ~0.26 Pareto
points, point estimate indistinguishable from zero.**

Correct statement: *the decision rule reliably converts the checkpoint's
contribution, and that contribution is (subject, object) identity in
text-embedding space.* Incorrect: *the visual model adds +2.9 mR* — an arm with
every image association destroyed adds +2.9 too.

`full` also buys head/body movement at a **tail cost**: tail mR 9.89 vs
`prior_only`'s 14.49. It is not a long-tail gain.

Detail: `docs/FULL_VALIDATION_RESULT.md`.

## 4c. THE CYCLE'S PRINCIPAL RESULT — a prior-free grounding metric (`p33`–`p41`)

**MEASURED.** Superseding the reading in §2b/§3 that "there is no visual
evidence in this term". See `docs/DIAGNOSIS.md`,
`docs/AUDIT_TEXT_BRANCH_IS_IMAGE_CONDITIONED.md`.

Within one (s,o) group the prior is constant to **9.4e-05**, so in the double
difference it cancels exactly, along with any per-class calibration, tau and any
temperature. WPRD is the AUC of that difference. **Its prior control reads
exactly 0.5000, CI [0.5000, 0.5000], in every stratum.**

| arm | WPRD | 95% CI |
|---|---|---|
| geometry probe, TRAIN-FITTED (19 numbers from 2 boxes) | **0.5961** | [0.5921, 0.6014] |
| classifier head (stored, **discarded** at alpha=0) | 0.5728 | [0.5681, 0.5779] |
| text head (**the evaluated model term**) | 0.5542 | [0.5495, 0.5592] |
| prior (control) | **0.5000** | [0.5000, 0.5000] |

Four findings:

1. **Image-conditioned relational signal EXISTS** (CI excludes 0.5 by 9
   half-widths) and is **weak**. H4 moves from "not established" to
   **ESTABLISHED BUT WEAK**.
2. **`p26`/`p29`'s "+0.031 ± 0.188" was diluted**: 23.3% singleton groups +
   19.8% constant-GT groups = **43.1% of rows structurally inert** to a
   within-group permutation.
3. **The evaluated head has no measurable tail grounding** — body–tail
   [0.4822, 0.5570] and tail–tail [0.4174, 0.6246] both contain chance. The
   discarded head does (body–tail 0.5918, clear of 0.5) and wins **13/13**
   strata.
4. **Box geometry out-discriminates the checkpoint**, and the margin *grows*
   toward the tail (+0.033 head–head → +0.123 tail–tail).

**Corrected by `p41`, which used the FIELD's metric instead of ours.** At
tau ≤ 0.05 — the checkpoint's operating region — prior+MODEL **beats**
prior+GEOMETRY (+1.95, +1.72 Pareto). Discrimination is not calibration.
The narrow, defensible claim: *the checkpoint's advantage on R@50/mR@50 does not
come from relational discrimination; it comes from being calibrated against the
prior.* At tau ≥ 0.1 model+geometry beats both (**+4.051**).

**`p41` REPLICATED on the held-out TEST split (`p64`, 2026-09-03), CPU only,
`p54` cache.** Every winner and every selected mixing weight is identical
between splits at all four tau: MODEL beats GEOMETRY at tau ∈ {0, 0.05} (val
+1.948/+1.716, test +1.615/+1.162), GEOMETRY beats MODEL at tau ∈ {0.1, 0.2}
(val +1.826/+2.281, test +2.956/+1.841). Not pre-registered before running
(no threshold table existed for this specific analysis), so this is a
descriptive replication against `p41`'s own numbers, not a pass against a
committed criterion. A separate, non-pre-registered corrected analysis fixing
a known bug in the nested-CV arm's selection key (`docs/GEOMETRY_SGG_BASELINE_RESULT.md`
limitation 1: it selected on mR alone and always picked w=0) shows that once
the geometry mixing weight is chosen without any eval-split leakage
(cross-fitted, floor-respecting), MODEL wins 7 of 8 val+test tau cells,
including most of the tau ≥ 0.1 region where the disclosed-leakage "best
fixed w" analysis had shown GEOMETRY winning. Detail:
`docs/GEOMETRY_SGG_TEST_REPLICATION_RESULT.md`.

`p32` reaches the same place independently: on the estimable subset the model
term is beaten by every pair-conditioned statistic tested
(`G_model − F_pair_foldfit = −1.706`).

**Answer to the founding question.** mR@K rises when mass moves to rare
predicates, which tau does without looking at the image, while the head actually
run cannot tell tail relations apart at all. The metric and the mechanism are
decoupled.

## 4d. Cross-model WPRD point — IMP+ (`bknyaz/sgg`), decomposed and corrected (2026-09-03)

**MEASURED**, decomposed, and materially corrected from the same-day first
pass. Two claims from the first pass are **retracted** below, not silently
dropped — see `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §10 for full detail.

**Reproduction-gap correction.** The first pass compared this checkpoint's
R@50/mR@50 (76.17/24.03, NoGC, full split) against the wrong row of Knyazev
et al. BMVC 2020 Table 1: 74.8/20.6 is the **"MP" (Message Passing, Xu et
al.)** architecture's baseline, not this checkpoint's. The checkpoint loads
0-missing/0-unexpected against `edge_model="motifs"` — it is the paper's
**"NM" (Neural Motifs)** architecture, whose correct published row is **R@50
80.5 / mR@50 26.9**. Against the *correct* row the gap is a consistent,
modest **shortfall** (−4.33 R@50 / −2.87 mR@50), not an unexplained excess.
Reclassified **UNEXPLAINED → PARTIALLY EXPLAINED**.

**Decomposition (CPU-only, reuses `wprd_pairs_full.pt` + a new alignment-verified
geometry extraction).** Registered gates: model WPRD reproduces 0.620539 to
1e-3 (PASS); prior control exact 0.5000 (PASS).

- **Pair identity (item B) contributes exactly ZERO to WPRD, by algebraic
  proof** — WPRD(raw) == WPRD(within-group-centred) to 6 decimals. This is
  the same invariance `p42` proved for PURE, now proved for IMP+ too, and it
  means the "(s,o) identity via the motifs LSTM" hypothesis floated in the
  first pass is **not viable even in principle** for this metric.
- **Pair-matched null collapses to exact chance** (0.4997, vs model 0.6205),
  so the signal is tied to the specific row/instance, not a group-level
  constant.
- **Geometry (item C) is the dominant identified component.** A linear,
  cross-fitted, pixel-free 19-number box probe alone reaches **0.6229 —
  matching/exceeding the model (0.6205)**. This independently **replicates
  PURE's own `p38`/`p39` finding** ("box geometry out-discriminates the
  checkpoint") in a second model, second codebase, different split — **2/2
  models now show geometry ≥ the learned head on WPRD**, the most robust
  finding of this line of work to date.
- Non-geometry residual (item A) is real but modest: WPRD of the
  geometry-unpredictable part of the within-group signal = 0.5641, clearly
  above chance but well below the layout-predictable part's 0.5959.
- Item D (calibration/score construction): not a confound — both sanity nulls
  read within noise of exactly 0.5.
- Head/body/tail: same head>tail gradient as PURE; tail-tail is underpowered
  (147 cells, wide CI) exactly as PURE's was — do not read it as a point
  estimate.

**Corrected PURE comparison.** The first pass compared IMP+ (standard
150-category vocabulary) against PURE's `p49` numbers, which use PURE's
**raw, unrestricted** object-name vocabulary (16,929 categories) — a
population mismatch. On the correct like-for-like population (`p53`, PURE
restricted to the standard 150-category vocabulary), PURE's best arm
(geometry-MLP) reaches WPRD **0.6452** — **higher** than IMP+'s 0.6205.
**Retracted:** "IMP+ has the highest WPRD ever measured" and "IMP+
simultaneously beats every PURE arm on both axes at once" (the latter also
compared R@K/mR@K numbers across two different, uncontrolled evaluators,
which WPRD is specifically designed to avoid — R@K/mR@K are not
cross-codebase portable the way WPRD is).

**Re-answering the three cross-model questions:** (a) WPRD generalizes — yes,
more strongly than before (every gate, on every arm, passed cleanly on a
second codebase). (b) The mR@K↔WPRD inversion — **untested, not refuted**;
the only valid cross-model axis (WPRD) places IMP+ as an unremarkable
mid-family point, not a counter-example, and no valid mR@K comparison exists
across codebases. (c) Prior/geometry relationship — **resolved**: pair
identity is provably invisible to WPRD, and geometry is confirmed (not just
suspected) as the dominant driver, in 2 independent models now.

Full detail: `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §10.

## 4e. `p68` — the geometry input is 6/8 bit-exactly constant (2026-09-06)

**MEASURED**, in-situ over 17,196 real validation pairs from a live forward
pass with the historical checkpoint. Detail:
`docs/GEOMETRY_INPUT_DEGENERACY_RESULT.md`.

Six of the eight geometry features reaching `forward_pairs` — `rw`, `rh`,
`ar1`, `ar2`, `a1`, `a2` — have **exactly one distinct value each (zero)**.
Only `dx`, `dy` vary, in whole-frame rather than subject-box units.

**Cause (VERIFIED FACT, source).** `relational_model.py:725` normalises boxes
to `[0,1]` (`/img_res`) before calling `geom_feats_torch`, whose
`clamp_min(1.0)` was written for pixel-space boxes. Every clamp binds,
returning exactly 1.0, so all six log-ratio/log-area channels collapse to
`log(1) = 0`. A units-contract violation at the call site. No test covers
this regime — `tests/test_geometry.py` uses only pixel-scale boxes.

**Consequence.** The model has **no access to box scale at all** through the
geometry path. This partially, mechanically explains the programme's most
robust finding (geometry out-discriminates the checkpoint): the comparison was
never "learned encoder vs raw geometry" but "encoder given 2 frame-relative
offsets vs probe given 19 real box numbers". `p57`'s `dx_rel` R² = 0.052 is
explained rather than merely observed — `dx_rel` needs subject width, which is
one of the zeroed channels.

**Also:** `geom_alpha` is **dead code** in this checkpoint (bit-identical to
its 0.1 init; the saved cfg's `vector_fusion_gate=True` means that branch never
executes). The live `fusion_gate` is architecturally input-dependent but
**empirically constant** at 0.50155 ± 0.00013. Geometry is *not* gated off — it
enters at half weight; the gate's *adaptivity* is what collapsed.

**No measured number in this project changes. No frozen Paper A claim is
invalidated** — flagged for re-read before submission, not reopened. Whether
*fixing* the units raises WPRD is untested by `p68`; `p69`
(`docs/PURE_VISIBLE_GEOMETRY_PREREGISTRATION.md`) was pre-registered to try to
falsify that inference **before** any GPU is spent.

## 5. Experiments completed this cycle

| run | question | verdict |
|---|---|---|
| `p14`/`p15` | oracle ceiling at the tie-break fix | superseded by p17 |
| `p17` | oracle ceiling, canonical ordering | PREREG SUCCESS 27/27 (vacuous); REALIZABLE **EXHAUSTED** 27/27 |
| `p18` | learned candidate scorer, plain CE | **EXHAUSTED** 4/4 on ΔR |
| `p19` | same, fully class-balanced | R collapses to 25–31; mR reaches 42.2 |
| `p20` | beta frontier | superseded by p21 (×100 Pareto bug) |
| `p21` | beta frontier, corrected, with null | full off-frontier at beta=0.20 |
| `p22` | nested beta selection | operating-point-free, but ONE partition |
| `p23` | throughput pilot, workers 7 | 0.95x -- GPU-bound, more workers cannot help |
| `p25` | nested selection resampled over 5 partitions | magnitude weakened, separation confirmed |
| `p26` | pair-matched null (image destroyed, identity kept) | **E2 SUPPORTED -- the effect is pair identity** |
| `p24` | full-validation cache extension (GPU, frozen forward pass) | COMPLETE, cache **VALID 12/12** |
| `p27` | pair-prior distillation | **WITHDRAWN** -- pair arms blindfolded on 45% of rows |
| `p28` | oracle ceiling on the full split | REALIZABLE **EXHAUSTED** 9/9; tau-restoration falsified again |
| `p29` | nested scorer on the full split | **PREREG CONFIRMED 3/3** -- and the image-destroying null passes too |
| `p30` | audit of pair-arm fold coverage | 45.4% (3k) / 33.2% (full) of rows get NO pair information |
| `p31` | learned R/mR frontier on the full split | `full` has a usable region beta in [0.10,0.20], peak **+3.334**; no-model arms have none |
| `p32` | corrected pair-prior distillation, estimable subset | registered NOT EXPLAINED — but model is **worse** than every pair statistic |
| `p33` | WPRD, a prior-free grounding metric | signal EXISTS (0.5542) and is WEAK; prior control exactly 0.5000 |
| `p34` | ensemble_alpha sweep | WPRD rises monotonically with alpha; discarded head is better |
| `p35` | WPRD stratified | classifier head wins 13/13; evaluated head has NO tail grounding |
| `p36` | rel_feat + pred_emb GPU cache | **RUNNING** |
| `p38`/`p39` | box-geometry control (cross-fit / train-fit) | geometry **0.5961** > both heads |
| `p40` | where geometry wins | margin grows toward the tail, +0.123 at tail–tail |
| `p41` | geometry on the FIELD's metric | **corrective**: model wins at tau<=0.05 |
| cross-model (IMP+) | first cross-model WPRD point + decomposition | geometry ≥ model replicates 2/2; pair identity proven invisible to WPRD; inversion untestable cross-codebase, not refuted |
| `p64` | `p41` replicated on held-out TEST (`p54` cache) + corrected nested-CV control | **REPLICATED**, 4/4 tau winners match; corrected control shows MODEL wins 7/8 cells once eval-split leakage in `w`-selection is removed |

## 6. Experiments pending

- **Corrected pair-prior distillation (CPU, no GPU).** `runs/p27` was withdrawn;
  the estimable-subset design is specified in
  `docs/PAIR_PRIOR_DISTILLATION_RESULT.md` §6. ~88.6k rows of the full `p24`
  cache — larger than the entire 3k screening set. Must recompute the prior-only
  baseline and tau frontier **on the subset**, and must report that the ~33%
  singleton-pair rows remain unaddressable by any pair-conditioned estimator.
  Open question: can a vision-free pair-conditioned model reproduce C′?

(The full-validation confirmation that was listed here is complete — see §4b/§6b.)

## 6b. Full-validation confirmation — COMPLETE

`runs/p24` finished (exit 0, 12,423 s, 10,401 images, 132,556 rows, cache
validated 12/12). All three registered analyses have now run: `p28` (oracle
ceiling), `p29` (nested scorer — carries the criterion), `p31` (frontier).

**Verdict: CONFIRMED**, 3/3 registered conditions on the registered partition.
See section 4b and `docs/FULL_VALIDATION_RESULT.md`.

Its reduced weight after `p26` stands, and the full-split data reinforces it
rather than softening it: the image-destroying `pair_matched_null` passes every
registered condition too, at `full − null = +0.031 ± 0.188`. A CONFIRMED verdict
here does not resurrect H4.

Note on execution: the first `p29` attempt was killed by a VM shutdown at
12:06 UTC with no `result.json` written. It produced no number; it was archived
as `runs/p29_full_scorer_nested_KILLED_BY_VM_REBOOT/` with a `STOPPED.md` and
relaunched with byte-identical argv. Nothing was re-run that had completed.

## 7. Experiments explicitly abandoned

- **Candidate-restricted learned reranker (GPU).** Pre-registered criterion
  returned EXHAUSTED 4/4 on ΔR and the raw score is negative as a ranker. Not
  built.
- **The visual-architecture branch as currently posed** (`runs/p26`). The
  quantity being converted is a pair prior in text-embedding space; the visual
  head is at weight zero and is untrained in this checkpoint. Building an
  architecture to exploit it would be building a second frequency prior.
- **Architecture scaling / added visual capacity.** No measured failure mode
  demands it. `docs/PHASE4_SCIENTIFIC_REASSESSMENT.md`.
- **Additive appearance scoring on frozen CLIP.** Falsified.

## 8. Historical claims that remain unverified or superseded

- **The +2.29 mR headline is not "the model adds +2.29 mR".** tau contributes
  most of that movement. Any table quoting it must use the operating-point
  separation required by the directive's section 13.
- **README's historical performance claims** predate the leak-free prior and the
  GT-extraction fix (`220c5c2e`). They have not been re-derived under the current
  evaluator and should not be cited without one.
- **`docs/CPRIME_MECHANISM_REPORT.md` §3 margin-decile table** was computed with
  the `topk` tie-break and is affected; its load-bearing claim (0.00% of rows
  above the 5th decile change) cannot move, because tied rows have margin 0 and
  live entirely in decile 1.

## 9. Protocol corrections made this cycle

1. The oracle's pre-registered R floor was applied to an arm that cannot lose
   recall, making the gate non-falsifiable. Floor now also binds on the
   realizable arm; both verdicts reported. (`a75aa03`)
2. The candidate-scorer null condition was written against the wrong baseline
   (`null > achieved` rather than `null > prior_only`). Corrected; both
   computed and stored. (`docs/CANDIDATE_SCORER_RESULT.md` §2)

Neither correction changed a pre-registered threshold.
