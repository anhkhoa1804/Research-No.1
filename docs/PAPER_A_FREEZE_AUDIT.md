# Paper A — freeze audit and manuscript blueprint

Mode: **PAPER FREEZE**, not experiment discovery. No GPU work in this session
(`nvidia-smi` verified idle before and after: 0% util, 0 MiB, no processes).
No new scientific hypothesis is introduced. This document is the frozen
specification a manuscript gets drafted from in the next session — not the
manuscript itself.

Scope reminder from `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §8: **Paper A** =
the PURE-specific mechanism findings, `docs/RESEARCH_STATE.md` §1–4c plus the
`p64` held-out replication. Paper A is explicitly a single-checkpoint case
study. Paper B (the mR@K↔WPRD inversion as a general claim) and Paper C (WPRD
as a reusable protocol) are separate documents with their own, weaker or
stronger, evidence bases and must not be conflated with Paper A's claims.

## 0. State verification (done before the audit)

- `git status -sb`: clean, `research/architecture-breakthrough` up to date
  with `origin/research/architecture-breakthrough` (fetched, no divergence).
- `nvidia-smi`: 0% util, 0 MiB used, no processes — GPU idle.
- `ps aux`: no training/experiment process running.
- Re-read in full this session: `docs/RESEARCH_STATE.md`, `docs/DIAGNOSIS.md`,
  `docs/HYPOTHESIS_MATRIX_LIVE.md`, `docs/GEOMETRY_SGG_TEST_REPLICATION_RESULT.md`,
  `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` (full, including §10), plus
  `docs/WPRD_LITERATURE_AUDIT.md`, `docs/BENCHMARK_LITERATURE_GAP.md`, and
  `docs/known_issues.md` for the evaluator-semantics caveat below.
- No separate "Paper-A blueprint" or "claim ledger" file exists on disk; the
  claim ledger for Paper A **is** `docs/RESEARCH_STATE.md` §1–4c +
  `docs/HYPOTHESIS_MATRIX_LIVE.md`, as the prior session already established.

**Confirmed frozen inputs, with the exact numbers each claim below cites:**

| source | what it fixes |
|---|---|
| `p24`/`p28` (val, full split) | prior-only R@50 66.593, mR 22.304; model adds R@50 67.168 (+0.575 over the historical read; +0.575–0.9 depending on tau) |
| `p29` (val, full split, 5-fold) | learned per-class rule Pareto −1.205±0.793, floor 0/5; `full` (prior+model) +2.947±0.190, floor 5/5 |
| `p33`/`p35` (val) | WPRD text 0.5542 [0.5495,0.5592]; classifier 0.5728 [0.5681,0.5779]; prior control exactly 0.5000; body-tail/tail-tail contain chance |
| `p38`/`p39` (val) | geometry linear (train-fitted) WPRD 0.5961 [0.5921,0.6014] |
| `p41` (val) | composed-metric Pareto: MODEL beats GEOMETRY at tau∈{0,0.05} by +1.948/+1.716; GEOMETRY beats MODEL at tau∈{0.1,0.2} by +1.826/+2.281 |
| `p54`/`p59`/`p61`/`p62` (test, pre-registered) | WPRD ordering, prior control, stratification skew, `p37` verdict all REPLICATE on test |
| `p64` (test, this session, **not** pre-registered) | `p41`'s composed-metric result REPLICATES on test (4/4 tau winners match); corrected leakage-free nested control (new, unregistered) shows MODEL wins 7/8 val+test cells |
| `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §10 | a **second, independent model** (IMP+) also shows geometry ≥ learned head on WPRD (0.6229 vs 0.6205) — **this is Paper B/C evidence, not Paper A's**, per that document's own §8 |

---

## A. Claim-by-claim audit

| # | Claim | Strongest evidence | Class |
|---|---|---|---|
| 1 | The frequency prior P(p\|s,o) dominates PredCls; the trained model adds a small increment | `p24`/`p28`, val+test | **FROZEN** (not novel — Zellers/Neural-Motifs; must cite, not claim) |
| 2 | A per-class calibration knob (tau) moves mR@50 with zero image access | `p24`/`p28`/`p29` | **FROZEN** (not novel — the debiasing literature's core observation) |
| 3 | A *learned* calibration rule cannot even match hand-tuned tau | `p25`/`p29`, 5-fold resampled, 0/5 floor | **FROZEN** — this specific negative result is this project's own measurement |
| 4 | WPRD's prior control reads exactly 0.5000 in every stratum, an algebraic identity | `p33` val, `p59` test, `p42`'s invariance proof, and independently re-derived for IMP+ (§10.2 of the cross-model doc) | **FROZEN** — holds by construction, not empirically, so it is as strong a claim as this project can make |
| 5 | The evaluated checkpoint's image-conditioned relational discrimination is weak (WPRD ≈0.55, CI excludes chance) | `p33` val 0.5542, `p59` test 0.5446 | **FROZEN WITH SCOPE LIMIT** — one checkpoint, PredCls+GT-pairs, VG150 only |
| 6 | That discrimination is statistically absent in the tail (body-tail, tail-tail contain chance) | `p35` val, `p59` test replication table | **FROZEN WITH SCOPE LIMIT** — tail cells are low-power (34–37 cells) on *both* splits; must always carry that caveat, never quoted as a bare point estimate |
| 7 | A 19-feature, train-fitted linear box-geometry probe exceeds the checkpoint's WPRD | `p38`/`p39` val 0.5961, `p59` test 0.5916 | **FROZEN WITH SCOPE LIMIT** — one checkpoint; "geometry" here means this specific linear probe, not a claim about geometry in general |
| 8 | Under the field's own composed R@50/mR@50/Pareto metric, the checkpoint beats geometry at its actual operating region (tau≤0.05) | `p41` val, `p64` test — 4/4 tau winners match | **FROZEN WITH SCOPE LIMIT**, and see item B below on how to state this |
| 9 | At tau≥0.1 geometry's apparent Pareto advantage is largely an artifact of choosing the mixing weight on the evaluation split | `p64`'s corrected, leakage-free nested control (this session) | **FROZEN WITH SCOPE LIMIT, EXPLICITLY SECONDARY** — new, not pre-registered, must never be presented as the headline; see item B |
| 10 | The above all replicate, unchanged in sign, on a held-out TEST split | `p54`/`p59`/`p61`/`p62` (pre-registered) + `p64` (descriptive) | **FROZEN**, but with the pre-registered/descriptive distinction preserved in text (item B) |
| 11 | The Pareto-gap vs. WPRD rank inversion (best-discrimination arm has the worst Pareto gap) | `p47`, n=4 arms | **FROZEN WITH SCOPE LIMIT** — n=4, explicitly low power, Spearman not distinguishable from zero on the mR/Pareto axis; must carry the n=4 caveat on every mention, never stated as "the inversion" bare |
| 12 | Reported R@50/mR@50 numbers are comparable to published VG150 PredCls literature | none — contradicted | **REMOVE.** `docs/known_issues.md` (`VERIFIED FROM CODE`): under `eval_sgg_use_gt_pairs=True` (used throughout this project — confirmed in the `p54` cache: `use_gt_pairs=True`), the evaluator's `predcls.R@K` field takes one argmax predicate per GT pair (not a multi-hypothesis ranking) and pools hits/GT globally across the dataset (not per-image-averaged). Neither matches the literature's standard R@K definition. **Every "R@50"/"mR@50" number in this project (`p24` through `p64`) is this statistic**, verified directly in `p54`'s own `latest_metrics.json` (`val_sgg.predcls.R@20/50/100 = 0.6526/0.6813/0.6814` — not identical across K, contra one line of `known_issues.md`, but also not the literature statistic). This does not invalidate any *within-evaluator* comparison this project makes (prior vs MODEL vs GEOMETRY all use the identical composition), but it means **no number in this project may be quoted next to a published R@50/mR@50 figure as if directly comparable.** |
| 13 | WPRD is a novel metric | `docs/WPRD_LITERATURE_AUDIT.md` | **NEEDS SOURCE/CITATION** — narrowed, not removed. The audit's own verdict: the *combination* (exact analytic cancellation via (s,o)-conditioning, operating-point-free, computable on existing annotations) was not found in a targeted search; each *component* (prior-only baseline, geometry-only baseline, discrimination/calibration split) is prior art (SpatialSense, Plesse et al., the debiasing literature) and must be credited, not claimed. The audit explicitly states this is a working map, not a systematic review — a formal related-work pass is a precondition for the novelty paragraph, not yet done (§D below). |
| 14 | Geometry ≥ learned model on WPRD replicates across independent models (2/2) | `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §10.2, IMP+ | **NOT PAPER A'S CLAIM.** True and frozen as a fact, but it belongs to Paper B/C's evidence base per that document's own §8 ("Paper A: unaffected... this run is about a different model entirely"). Paper A may *mention* it as a forward pointer in the discussion, explicitly labeled as a separate, ongoing generalization check — never folded into Paper A's own evidence table or abstract as if it strengthens a single-checkpoint mechanism claim. |
| 15 | The mR@K↔WPRD inversion is a property of SGG models generally | `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §10.5(b) | **REMOVE from Paper A entirely** (it was never Paper A's claim) and **note for Paper B**: explicitly "untested, not refuted" at n=1 additional model with no valid cross-codebase R@K axis. Any Paper A text that could be read as implying this must be reworded. |

---

## B. How `p64` must be presented (binding on the manuscript)

Three rules, non-negotiable for the draft:

1. **`p64`'s descriptive replication of `p41` was NOT pre-registered.** State
   this explicitly wherever the replication is reported — a table caption or
   a footnote is not enough; the running text must say so once, plainly, the
   first time `p64` is introduced. It sits in a different evidence class from
   `p54`/`p59`/`p61`/`p62`, which were pre-registered with committed
   thresholds before the GPU pass launched.
2. **The original `p41`/`p64` nested-selection arm had a known, disclosed
   leakage issue** (mixing weight `w` chosen by looking at the split being
   scored) **and, separately, its nested-CV control had a selection-key bug**
   (mR-only, always picked `w=0`). The manuscript must never present the
   tau≥0.1 "geometry wins by +1.8 to +3.0 Pareto points" number as a clean,
   causal, leakage-free result. It is the number `p41` actually reported, and
   it must be reported as such (with its own disclosed limitation attached),
   but the corrected, leakage-free control (which shows MODEL winning 7 of 8
   val+test cells once the leakage is removed) must appear alongside it,
   clearly labeled **"corrected / leakage-free control, secondary analysis,
   not pre-registered"** — never as a silent replacement, never merged into
   one number.
3. **The one headline this evidence base actually supports without caveats
   piling up is narrow and should be treated as such:** at the checkpoint's
   real operating region (tau ≤ 0.05), the model beats the geometry probe on
   the composed metric, on both validation and held-out test, under either
   selection procedure. That is the sentence the abstract gets to lean on.
   Everything about tau ≥ 0.1 belongs in the results section with its
   leakage caveat attached, not in the abstract.

---

## C. Abstract overclaiming — specific checks

- **"field"**: Paper A must not use "the field" to describe anything beyond
  the SGG debiasing/metrics literature it is positioned against (established
  prior art, correctly citable that way). It must never use "field" to mean
  "SGG models in general," because n=1 checkpoint (2 counting the IMP+ WPRD
  point, which is Paper B/C's evidence, not Paper A's) cannot support that.
  Concretely: "the field's own composed metric" (meaning: R@50/mR@50/Pareto,
  the standard SGG evaluation protocol) is fine and already used this way in
  `docs/GEOMETRY_SGG_BASELINE_RESULT.md`; "a field-wide finding" or "SGG
  models generally exhibit..." is not.
- **"visual grounding"**: WPRD measures *discrimination* — whether the score
  can tell two GT predicates apart within a fixed pair — not calibration,
  not ranking, not whether the representation is causally using pixels
  versus some other correlate. `docs/DIAGNOSIS.md`'s own limitations section
  already states this ("WPRD measures discrimination, not calibration or
  ranking. A model could be useful in ways WPRD does not see."). The
  manuscript must use "discrimination" as the primary term and introduce
  "grounding" only with that qualifier attached, every time — never as a
  bare, stronger-sounding synonym.
- **"mR@K measures calibration"**: this must be framed as a demonstrated
  mechanism *for this checkpoint and this composition formula*
  (`score = alpha*(prior - tau*logP) + model_term`), not a universal claim
  about the metric. The correct scope: "for this checkpoint, moving tau
  changes mR@50 by more than the entire measured contribution of the trained
  model, with no image access" — a case-study finding, stated as such.
- **Geometry as a "replacement model"**: every existing doc (`p39`, `p41`,
  `docs/GEOMETRY_SGG_TEST_REPLICATION_RESULT.md`) already states this
  correctly ("a lower bound on triviality, not a proposal"). The manuscript
  must preserve that framing exactly — geometry never gets called a model,
  a baseline to deploy, or an alternative architecture; it is a diagnostic
  probe establishing what a checkpoint should be expected to exceed trivially.

---

## D. Citation / related-work gap audit

Direct pull from `docs/WPRD_LITERATURE_AUDIT.md` and
`docs/BENCHMARK_LITERATURE_GAP.md`, both already structured for this purpose.

| statement | status | what must be checked before submission |
|---|---|---|
| Frequency-prior dominance in VG150 | **established prior art** | Cite Zellers et al. (Neural Motifs, CVPR 2018); nothing further to check |
| Per-pair majority-relation confound, 50–75% of examples | **established prior art** | Cite Plesse et al. (WACV 2020) explicitly; this project's 69.23%/48.7% figures are a rediscovery, must be labeled as independent confirmation, not a new finding |
| mR@K, TDE, the SGG diagnosis toolkit | **established prior art** | Cite Tang et al. (CVPR 2020); frame WPRD as a complement, not a replacement |
| Prior-only and geometry/2D-only baselines | **established prior art** | Cite SpatialSense (Tobia et al./Yang, ICCV 2019) explicitly and prominently — the audit is unambiguous that these baselines are SpatialSense's construction, independently rediscovered here |
| Pair Recall / Predicate Rank (nearest-miss metrics) | **established prior art, must be distinguished** | Cite Lorenz et al. (CVPRW 2024); the manuscript must explain precisely why neither conditions on (s,o) identity (Pair Recall pools across pairs; Predicate Rank conditions on the pair being *correct*, not on its *identity*) |
| Winoground / ARO / SugarCrepe | **established prior art, analogous motivation, different mechanism** | Cite as the compositional-VLM analogue (adversarial curation vs. analytic conditioning); SugarCrepe's "drive blind models to chance" framing is the closest rhetorical parallel and should be credited by name |
| Haystack (rare-predicate PSG data) | **established prior art, orthogonal** | Cite as addressing tail *measurability* (a different axis: annotation density, not evaluation conditioning) |
| VG-OOD | **established prior art, orthogonal** | Cite as a re-split approach; contrast with WPRD's non-re-splitting conditioning |
| Exact analytic (s,o)-conditioning that cancels the prior identically, verified by an exact-0.5 control | **plausible novelty** | Requires a **systematic**, not targeted, related-work pass before this sentence appears in a submission. `docs/WPRD_LITERATURE_AUDIT.md` itself states its search was targeted, not exhaustive, and lists this as "novelty status: NOT YET CLAIMABLE" pending that pass |
| The discrimination/calibration decomposition as a general analytical move (not SGG-specific) | **unknown** | Not checked against causal-inference or fairness-metric literature at all; flag as an open related-work item, do not claim originality without a dedicated search |

**Bottom line for §D:** the manuscript may state the observations it credits
(prior dominance, per-pair confound, existing baselines) as attributed prior
art immediately. It may **not** state the exact-cancellation-by-conditioning
combination as novel until a systematic literature pass — beyond the
targeted searches already on record — is run. This is a documentation/search
task, not an experiment, and does not block the freeze of the *scientific*
claims above.

---

## E. Final evidence table

Every abstract sentence below maps to exactly one row.

| Claim | Evidence | Split | N | Scope | Confidence | Paper section |
|---|---|---|---|---|---|---|
| Frequency prior dominates PredCls; model adds a small increment | `p24`/`p28` | val, test | 132,556 / 132,334 GT rows | PURE, VG150, PredCls+GT-pairs | FROZEN (cite Zellers) | §5 |
| Learned calibration cannot match tau | `p25`/`p29` | val (full) | 132,556, 5-fold | PURE, VG150 | FROZEN | §5 |
| WPRD prior control is exactly 0.5 in every stratum | `p33` (val), `p59` (test), `p42` (invariance proof) | val, test | all strata | metric property (holds for any model by construction) | FROZEN | §4 |
| Checkpoint's discrimination is weak (WPRD ≈0.55) | `p33`, `p59` | val, test | 132,556 / 132,334 | PURE only | FROZEN WITH SCOPE LIMIT | §6 |
| No measurable tail discrimination | `p35`, `p59` | val, test | 34 / 37 tail-tail cells | PURE only, tail underpowered both splits | FROZEN WITH SCOPE LIMIT | §6 |
| Geometry probe exceeds checkpoint on discrimination | `p38`/`p39`, `p59` | val, test | full rows | PURE only (this probe) | FROZEN WITH SCOPE LIMIT | §7 |
| Composed metric: MODEL beats GEOMETRY at tau≤0.05 | `p41` (val), `p64` (test) | val, test | full rows | PURE only | FROZEN WITH SCOPE LIMIT | §8 |
| tau≥0.1 GEOMETRY advantage is mostly eval-split-leakage artifact | `p64` corrected control (this session) | val, test | full rows | PURE only, secondary/non-pre-registered | FROZEN WITH SCOPE LIMIT, EXPLICITLY SECONDARY | §8 |
| All of the above replicate on held-out TEST | `p54`/`p59`/`p61`/`p62` (pre-registered) + `p64` (descriptive) | test | 132,334 GT rows, 10,403 images | PURE only | FROZEN (registration status stated per-item) | §9 |
| Pareto-vs-WPRD rank inversion | `p47` | val | n=4 arms | PURE only, n=4 explicitly low-power | FROZEN WITH SCOPE LIMIT | §8/discussion |
| R@50/mR@50 as reported ≠ literature R@K (top-1-per-pair, pooled) | `docs/known_issues.md`, verified against `p54`'s own `latest_metrics.json` this session | n/a | n/a | applies to every number in this paper | disclosed limitation, not a result | §3, Limitations |
| Prior-only / geometry-only baselines are SpatialSense's construct | `docs/WPRD_LITERATURE_AUDIT.md` | n/a | n/a | attribution | must-cite | Related work |
| WPRD's specific combination is a plausible, not yet established, novelty | `docs/WPRD_LITERATURE_AUDIT.md`, `docs/BENCHMARK_LITERATURE_GAP.md` | n/a | n/a | pending systematic search | NEEDS CITATION PASS | Related work |

---

## F. Additional-experiment decision

**NO EXPERIMENT REQUIRED.**

Every link in Paper A's causal chain — prior dominance, calibration can move
the conventional metric without vision, WPRD provably cancels pair identity,
weak-but-real image-conditioned discrimination exists, geometry
matches/exceeds that discrimination, the composed metric can still rank the
model above geometry at the operating region — now has a held-out-test
measurement (some pre-registered, `p64` descriptive but faithful). Nothing
in the chain currently rests on an untested link, and no reviewer objection
identified in §I below requires new data to answer (each is answered by
scoping the claim correctly, not by running something).

Two **SMALL ANALYSIS REQUIRED** items, both documentation/verification work,
not experiments, and both should happen before the manuscript's Limitations
section is finalized:

1. Reconcile `docs/known_issues.md`'s specific "`R@20 == R@50 == R@100`
   identically" line against this session's direct read of `p54`'s
   `latest_metrics.json` (`R@20=0.6526, R@50=0.6813, R@100=0.6814` — close
   but **not** identical at the full 10,403-image scale, though the
   qualitative claim — one argmax predicate per GT pair, globally pooled,
   not the literature statistic — is confirmed `VERIFIED FROM CODE` this
   session at `openvocab_rel/evals.py:2091`). The Limitations section needs
   the precise, current wording, not the stale specific number from an
   earlier, smaller run.
2. Run the **systematic** related-work pass §D calls for, beyond the
   targeted searches already in `docs/WPRD_LITERATURE_AUDIT.md` and
   `docs/BENCHMARK_LITERATURE_GAP.md`, before the novelty paragraph in the
   Introduction/Related Work sections is finalized.

Neither blocks drafting the manuscript's results sections, which rest
entirely on already-frozen numbers.

---

## Manuscript blueprint

### 1. Final title

**"Calibrated, Not Grounded: A Held-Out Diagnosis of a Frequency-Prior Scene
Graph Model"**

### 2. Abstract (draft, not final prose for the paper — a specification of what it must say)

> Standard scene-graph-generation (SGG) evaluation on Visual Genome composes
> an object-pair frequency prior, a calibration term, and a trained model's
> score into a single R@K/mR@K number, which conflates how much of that score
> is relational discrimination versus prior-driven calibration. We introduce
> Within-Pair Relational Discrimination (WPRD), a metric that conditions on
> the (subject, object) category group so the frequency prior cancels
> exactly — verified by a control that reads precisely 0.5 AUC in every
> stratum, on both a validation and a held-out test split, for one widely
> used PredCls checkpoint architecture (a frozen-CLIP, frequency-prior
> ensembled model). Under WPRD, this checkpoint's measured relational
> discrimination is weak (0.554 validation / 0.545 test AUC, confidence
> intervals excluding chance) and statistically indistinguishable from
> chance in the tail, while a linear probe on 19 scale-invariant
> bounding-box features — no pixels, no learned visual representation —
> exceeds it (0.596 / 0.592). Despite this, under the field's own composed
> R@50/mR@50 metric the checkpoint outperforms the geometry probe at its
> actual operating region (tau ≤ 0.05) on both splits; a secondary,
> leakage-controlled analysis indicates that the reverse pattern observed at
> higher tau in the primary analysis is largely an artifact of selecting a
> mixing weight on the evaluation split itself, rather than a robust property
> of the checkpoint. Every core measurement replicates, unchanged in sign, on
> held-out test. These findings are a case study of one checkpoint archetype,
> not a claim about SGG models in general: they demonstrate that a model's
> apparent relational discrimination and its effect on the field's composed
> metric can be decoupled and even inverted, and they supply a metric and
> protocol — already exercised on a second, independently trained model in
> concurrent work — for checking whether this holds elsewhere.

### 3. Three contributions

1. **WPRD**: an operating-point-free relational-discrimination metric whose
   prior-control is an exact analytic zero (0.5000 in every stratum, not an
   empirical reduction), pre-registered and replicated on held-out test.
2. **A full discrimination/calibration decomposition of one checkpoint**,
   held-out-test replicated at every step: prior dominance, a calibration
   term that moves the conventional metric with zero image access, weak and
   tail-absent discrimination, and a trivial geometry baseline that exceeds
   that discrimination.
3. **A demonstrated dissociation between discrimination and the field's
   composed-metric ranking**: the same checkpoint that loses to a geometry
   probe on discrimination beats it on R@50/mR@50 at its operating region —
   replicated on held-out test — with an explicit, separately reported
   robustness check showing how much of the opposite (higher-tau) pattern is
   a selection artifact rather than a stable effect.

### 4. Section outline

1. Introduction — the confound; three contributions; explicit scope
   statement (one checkpoint, PredCls+GT-pairs, VG150)
2. Related work — debiasing/frequency-prior line; SGG metrics (mR@K, Pair
   Recall, Predicate Rank); compositional-VLM benchmarks; SpatialSense
   (baselines credited here, not claimed); positioning WPRD narrowly
3. Setup — VG150 PredCls with GT pairs; the checkpoint; the composed score
   formula; **the evaluator-semantics caveat** (R@50/mR@50 here = top-1
   accuracy over GT pairs, globally pooled — not the literature statistic;
   valid only for same-evaluator arm comparisons)
4. WPRD — definition, the exact-cancellation derivation, the prior-control
   validation (val + test)
5. Prior dominance and calibration (brief; not novel; establishes the
   composed score's two known components before isolating the third)
6. Weak, tail-absent discrimination (WPRD measurements, CIs, stratified
   head/body/tail table, val + test)
7. The geometry baseline exceeds the checkpoint on discrimination (val + test)
8. The composed-metric reversal — model wins at the operating region despite
   losing on WPRD; the tau≥0.1 pattern and its leakage-controlled robustness
   check, clearly separated
9. Held-out test replication — one table, every claim above, side by side,
   with pre-registered vs. descriptive items marked
10. Discussion — what this checkpoint actually learned; explicit statement of
    what this does and does not say about SGG generally; forward pointer to
    the (separate) cross-model program
11. Limitations
12. Conclusion

### 5. Figure list

1. Schematic of the confound and the WPRD double-difference construction
2. WPRD by arm (prior control / random null / evaluated head / discarded
   head / geometry), grouped bars with 95% CIs, validation and test panels
   side by side
3. Stratified head/body/tail WPRD, validation vs. test, with the tail-tail
   low-power caveat annotated directly on the figure
4. Pareto-gap-vs-tau curves for MODEL, GEOMETRY, and MODEL+GEOMETRY,
   validation and test panels, tau ∈ {0, 0.05, 0.1, 0.2} marked
5. The primary vs. corrected nested-geometry comparison at each tau
   (explicitly labeled "secondary, leakage-controlled robustness check")
6. The n=4-arm Pareto-vs-WPRD rank-inversion plot, with the n=4/low-power
   caveat in the caption, not just the text

### 6. Table list

1. Dataset/checkpoint/protocol summary (VG150 PredCls+GT-pairs, split sizes,
   checkpoint identity, composed-score formula)
2. WPRD by arm, validation vs. test, with CIs (the `p33`/`p59` table)
3. Stratified head/body/tail WPRD, validation vs. test
4. Composed-metric Pareto gap by arm and tau, validation vs. test (`p41`/`p64`)
5. Corrected leakage-free nested control, validation vs. test — captioned as
   secondary and not pre-registered
6. The n=4-arm rank-inversion table, with its Spearman coefficients and the
   explicit low-power caveat in the same table, not a separate footnote

### 7. Related-work structure

1. Frequency-prior dominance and debiasing (Zellers; Tang et al. TDE;
   Plesse et al.) — credit the observation, position WPRD as a complementary
   diagnostic, not a debiasing method
2. SGG evaluation metrics (mR@K/toolkit; Pair Recall; Predicate Rank;
   Haystack; VG-OOD) — enumerate why none conditions on (s,o) identity,
   per `docs/BENCHMARK_LITERATURE_GAP.md`'s table
3. Compositional VLM benchmarks (Winoground, ARO, SugarCrepe) — the closest
   rhetorical analogue ("drive a blind baseline to chance"), different
   domain and mechanism
4. SpatialSense — closest prior art for the specific baselines used here;
   credited prominently, not treated as a related-but-distinct entry
5. Positioning statement — what combination is being claimed as new (pending
   the systematic pass in §D), stated narrowly

### 8. Limitations (for the manuscript, not exhaustive of this audit)

- Single checkpoint (PURE). Every mechanism claim is scoped to it; the
  cross-model geometry finding (2/2 models) is mentioned only as a forward
  pointer to separate, ongoing work, never as Paper A's own evidence.
- PredCls with GT pairs only; no SGDet, no SGCls.
- VG150 only; no other SGG dataset.
- **R@50/mR@50 as computed here is top-1 predicate accuracy over GT pairs,
  pooled across the dataset — not the literature's multi-hypothesis,
  per-image-averaged R@K.** Valid for internal, same-evaluator arm
  comparisons only; never juxtaposed with a published number.
- Tail-tail WPRD cells are underpowered on both splits (34–37 cells); never
  read as a point estimate.
- The tau≥0.1 "geometry wins" result carries a disclosed evaluation-split
  leakage mode in how the mixing weight is chosen; the corrected control is
  itself new and not pre-registered.
- `p64`'s composed-metric replication was not pre-registered before running,
  unlike the WPRD-side test-split items.
- The Pareto-vs-WPRD rank inversion rests on n=4 arms; not statistically
  established.
- The geometry probe is a diagnostic lower bound, not a proposed model.
- The novelty claim for WPRD's specific combination awaits a systematic
  (not merely targeted) related-work pass.

### 9. Reviewer objections and answers

1. *"Your R@50/mR@50 don't match published numbers for this architecture
   family — is the evaluator even standard?"* — No, and the paper says so
   explicitly (§3, Limitations): under GT-pairs PredCls this evaluator
   computes top-1 predicate accuracy per pair, pooled, not literature R@K.
   Every claim is an internal, same-evaluator comparison between arms, never
   a literature comparison.
2. *"The prior dominates VG150 — that's known."* — Cited as such (Zellers,
   Plesse); it is scaffolding for the discrimination/calibration
   decomposition, not a claimed contribution.
3. *"Why trust WPRD isn't gameable?"* — Its prior control is an algebraic
   identity (exactly 0.5 in every stratum, verified on two independent
   models' cache formats), not an empirical approximation; it has no free
   parameters to tune.
4. *"n=1 checkpoint — how do you know this generalizes?"* — The paper does
   not claim it does. It is stated as a case study throughout, with the
   cross-model program named explicitly as a separate, ongoing check, not
   folded into this paper's evidence.
5. *"Isn't 'geometry beats the model' just saying your model is bad?"* — The
   finding is the dissociation between discrimination (WPRD) and the
   composed metric's ranking, not a verdict on model quality; geometry is
   explicitly not proposed as a model.
6. *"Is the tau≥0.1 geometry-wins result robust?"* — No, and the paper says
   so: a leakage-controlled secondary analysis shows it mostly disappears;
   both numbers are reported, with the corrected one given interpretive
   priority.
7. *"Was the test-split replication pre-registered?"* — Partially: the
   WPRD-side items were (and passed); the composed-metric replication
   (`p64`) was descriptive and is labeled as such throughout.
8. *"What's actually novel about WPRD?"* — Narrowly scoped per the
   literature audit: the specific combination of exact analytic cancellation
   via (s,o)-conditioning with an operating-point-free construction; every
   component baseline is credited to prior work (SpatialSense, Plesse,
   the debiasing/metrics literature).

### 10. Exact claims to avoid

- "SGG models don't understand images" or any field-wide grounding claim.
- "The field's evaluation protocol is broken" (field-wide, unscoped).
- "Geometry beats the model" without the tau/operating-region qualifier.
- Quoting this project's R@50/mR@50 next to a published number without the
  GT-pairs/top-1/pooled disclosure.
- "The mR@K↔WPRD inversion is a general SGG phenomenon" — not Paper A's
  claim, and even for Paper B it is explicitly "untested, not refuted."
- "This is the first metric to control for the prior" without crediting
  SpatialSense/Plesse/the debiasing literature in the same breath.
- Presenting the tau≥0.1 "geometry wins" number as clean or uncontested.
- "IMP+ confirms/extends Paper A's finding" — that result belongs to
  Paper B/C's evidence base, not Paper A's, per
  `docs/CROSS_MODEL_IMP_PLUS_RESULT.md` §8.
- Describing `p64`'s replication as pre-registered.
- Calling the geometry probe a model, a baseline to deploy, or a proposal.

---

## Binary decision

**PAPER A = READY TO DRAFT**

Every central claim in the causal chain is FROZEN or FROZEN WITH SCOPE
LIMIT, with the scope limits fully enumerated above and none of them
requiring new data to state correctly. The one REMOVE item (literal
R@K-literature-comparability) is a claim Paper A was never actually making
in its own docs, only a disclosure gap to close in the setup/limitations
text. The two NEEDS SOURCE/CITATION items are literature-search chores, not
scientific gaps, and do not block drafting the results sections.

**Next session's exact manuscript files to create** (no LaTeX yet — this
session produced the specification, not the prose):

- `paper_a/00_outline.md` — the section outline above, expanded to
  bullet-level detail per section
- `paper_a/abstract.md` — the abstract drafted from §2 above, iterated to
  final prose
- `paper_a/related_work_notes.md` — the systematic literature pass (§D,
  item 2) written up before the Related Work section is drafted in LaTeX
- `paper_a/figures/` — data-extraction scripts (CPU-only, reading existing
  `runs/p*` JSON, no new computation) for figures 2–6 above
- `paper_a/tables/` — the six tables above, generated directly from
  `runs/p33`, `p35`, `p38`/`p39`, `p41`, `p47`, `p54`/`p59`, `p64` JSON —
  no new numbers, only formatting existing ones

**Results frozen and citable as of this audit:** every row in §E's evidence
table, with the scope and confidence columns exactly as stated — most
directly, `p24`, `p28`, `p29`, `p33`, `p35`, `p38`, `p39`, `p41`, `p47`,
`p54`, `p59`, `p61`, `p62`, and `p64` (the last split into its pre-registered
absence and its descriptive-but-faithful replication, per item B).
