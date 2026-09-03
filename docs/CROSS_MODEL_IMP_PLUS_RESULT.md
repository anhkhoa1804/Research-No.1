# Cross-model WPRD — IMP+ (bknyaz/sgg, Neural-Motifs family), first result

> **UPDATE, 2026-09-03 (later session).** §3's reproduction-gap analysis below
> compared against the WRONG published row (Xu et al.'s **Message Passing**
> architecture, not this checkpoint's **Neural-Motifs** architecture) and is
> **corrected, not deleted, in §10**. §10 also adds the decomposition
> (pair-matched null, geometry probe, variance split, head/body/tail) that was
> listed as the recommended next step in §9, and a re-corrected PURE
> comparison. Read §10 before citing any number from §3 or §6.

**Gate this addresses:** `docs/CROSS_MODEL_FEASIBILITY.md` and `docs/PAPER1_EVALUATION_TABLE.md`
("Interface for the cross-model study") both state that every WPRD/rank-inversion
claim in this project is a measurement of **one checkpoint** (PURE) until a
second, independently-trained model is run through the same metric. This is
that second model. **n = 1 additional model.** It answers "does the metric
compute cleanly on a second checkpoint" with high confidence and "does the
mR@50↔WPRD inversion generalize" with very low power — one data point cannot
establish or refute a correlation.

## 1. What was run

- **Model:** IMP+ (Neural-Motifs/IMP baseline arm), `bknyaz/sgg` (BMVC 2020 /
  ICCV 2021), self-contained `torchvision`-based codebase, no CUDA-extension
  blocker on this machine's sm_89 GPU.
- **Checkpoint:** `imp_plus.pth`, SHA256
  `dbf9da439d0bd1749cdfe69dfba46f2b2ac5b1925d1a0c7be78182da7b4f0603`, loaded with
  **0 missing / 0 unexpected** keys against `RelModelStanford(mode=predcls,
  use_bias=False, backbone=vgg16, edge_model=motifs)`. `use_bias=False` means
  this checkpoint carries **no internal frequency-bias term** — its score is
  the message-passing/motifs head alone, not blended with any learned prior.
- **Data:** VG150 **test** split (26,446 images after `filter_empty_rels`,
  183,640 GT relation rows), converted from `maelic/VG150-coco-format` (HF
  parquet) into the `VG-SGG.h5`-equivalent trio the loader expects. Converter
  validated: raw stats match the source README exactly (31,876 img / 352,330
  obj / 183,640 rel, 0 dropped); post-filter count (26,446) matches the
  literature-standard PredCls test count and this repo's own hardcoded assert.
- **Task:** PredCls (GT boxes + GT object labels; only the predicate is
  predicted) — same task PURE is evaluated on.
- **Evaluator:** the field's own `lib/sgg_eval.py` / `lib/eval.py::val_batch`,
  reused verbatim (not reimplemented) so R@K/mR@K are comparable to the
  literature.
- **Location on disk:** `~/external_models/{sgg,checkpoints,vg_sgg_data,
  vg150_coco_format,runs}` — kept outside git per repo policy (large,
  regenerable). Results copied into `runs/cross_model_imp_plus/` (gitignored;
  hashed into `docs/RUNS_MANIFEST.json`, see §5).

## 2. Timeline and completion status

GPU/process check at analysis time: **idle** (`nvidia-smi`: 0% util, 0 MiB
used, no processes) — nothing was left running or crashed mid-run.

```
FULL RUN START  2026-09-03 01:41:53 UTC
STEP 1 (R@K/mR@K eval, 26446 images):        2227.2 s
STEP 2 (WPRD pair extraction, 26446 images):  981.9 s
STEP 3 (WPRD compute):                        < 1 s
FULL RUN END    2026-09-03 02:36:21 UTC   (54m28s wall clock)
```

`~/external_models/runs/full_run.log` (86,052 bytes) contains progress lines
every 500 images for both steps with no gap, ends with the `FULL RUN END`
marker, and has **zero** tracebacks, OOM messages, NaNs, or `killed` lines —
only two benign `torchvision` deprecation `UserWarning`s (`pretrained=` vs
`weights=`), repeated once per step. `DONE: 26446 batches, 26446 images,
2227.2s` and `DONE: 26446 images, 981.9s, 183640 GT rows, 0 missing lookups
(0.000%)` are both present verbatim.

**No numeric exit code was captured** (the driver script did not `echo $?`,
unlike this repo's own `result.json` convention). Given `set -e`-style
sequential STEP markers, zero error output, and both downstream JSON files
being written and internally consistent, exit 0 is the only reading
consistent with the evidence — but it is inferred from log structure, not a
recorded field. **Flagged, not fatal.**

## 3. Ordinary R@K / mR@K, against the published target

| n | GC/NoGC | R@20 | R@50 | R@100 | mR@20 | mR@50 | mR@100 |
|---|---|---:|---:|---:|---:|---:|---:|
| 300 (pilot) | NoGC | 59.41 | 73.89 | 82.73 | 11.74 | 20.01 | 29.90 |
| 300 (pilot) | GC | 53.59 | 61.21 | 63.65 | 8.21 | 10.26 | 11.45 |
| **26,446 (full)** | **NoGC** | **61.02** | **76.17** | **84.53** | **13.68** | **24.03** | **33.73** |
| **26,446 (full)** | **GC** | **53.36** | **60.17** | **62.29** | **9.22** | **11.57** | **12.55** |
| published target (NoGC) | — | — | 74.8 | — | — | 20.6 | — |

Source: Knyazev et al., BMVC 2020, Table 1, "MP+" baseline loss row.

**Read with two caveats, not as a clean reproduction:**

1. **Model/target mismatch.** The checkpoint is labeled `IMP+` (the ICCV-2021
   paper's IMP/Neural-Motifs baseline arm); the published number being
   compared against is the **BMVC-2020 "MP+"** row. These are related but not
   guaranteed to be the identical architecture/training run — the original
   commit message already flagged this as an approximation, not an identity.
2. **The gap widens from pilot to full, in the same direction.** At n=300,
   NoGC R@50/mR@50 (73.89/20.01) sit within a plausible sampling band of the
   target (74.8/20.6). At the full 26,446 images, R@50 is **+1.4** above
   target and mR@50 is **+3.4** above target — not a shrinking gap as sample
   size grows (which sampling noise alone would predict), but a larger one.
   mR@50 in particular is known to be volatile at small n because it averages
   over 50 predicate classes, many rare — a 300-image slice under-samples the
   tail and can read low by several points, which is the most likely
   explanation for the pilot→full jump. This is **plausible, not verified**:
   no per-class breakdown at n=300 was computed to confirm it.

**Verdict on this component:** same order of magnitude as the published
target, correct GC<NoGC ordering, no collapse or degenerate behavior — good
enough to trust the checkpoint is functioning as a real PredCls model, **not**
tight enough to certify an exact reproduction. Treat the R@K/mR@K columns as
"plausible, approximately in range," not as a verified replication.

## 4. WPRD (full split)

| | n_gt_rows | n_groups | n_singleton | n_decidable | decidable % |
|---|---:|---:|---:|---:|---:|
| IMP+ (test, 26,446 img) | 183,640 | 7,127 | 2,083 | 3,958 | **55.5%** |
| PURE (val, 10,401 img, `p33`) | 132,556 | 45,607 | — | 75,366 rows | **56.9%** (row basis) |

The decidable fraction lands within 1.4 points of PURE's despite a different
model, split, and codebase — the grouping structure (how often an (s,o) pair
recurs with >1 distinct GT predicate) is a property of VG150's annotation
density, and it replicates across both extractions. This is a useful internal
consistency check on the pipeline, independent of the model itself.

| arm | wprd_macro | wprd_weighted | n_cells | n_comparisons |
|---|---:|---:|---:|---:|
| prior control (empirical within-group frequency, must be exactly 0.5) | **0.500000** | **0.500000** | 29,257 | 3,087,992 |
| **IMP+ (rel_fc / motifs head)** | **0.620539** | **0.570292** | 29,257 | 3,087,992 |

The prior-control arithmetic identity holds **exactly** (0.500000, not merely
close) — this is a hard correctness check on `SimpleGroups`/`wprd_generic` in
`tools/cross_model/compute_wprd.py`, not a measurement, and it passing exactly
confirms the grouping/AUC logic generalizes correctly to a second model's
output format.

**Head/body/tail stratification: not computed.** `tools/cross_model/
compute_wprd.py` is a from-scratch port (`SimpleGroups`) that does not carry
the frequency-bucket metadata `tools/wprd_stratified.py` needs; no stratified
breakdown exists for IMP+. Requested by the user and explicitly absent —
noted rather than estimated.

## 5. Cache / output integrity

- Checkpoint SHA256 matches the value logged at run start (`dbf9da43...`) —
  re-verified independently against the file on disk, not just trusted from
  the log.
- `load_state_dict`: 0 missing / 0 unexpected keys (asserted in code; a
  mismatch would have raised, not silently degraded).
- WPRD pair extraction: **0 missing lookups (0.000%)** over 183,640 GT rows —
  every GT triplet found its model score; PredCls scores all ordered pairs,
  so no join gaps are structurally possible for this task, and the log
  confirms none occurred.
- `runs/cross_model_imp_plus/{full_eval_result.json,pilot_300.json,
  wprd_result_full.json}` are now hashed into `docs/RUNS_MANIFEST.json`
  (`cross_model_imp_plus` entry) — content verified byte-identical to the
  source files in `~/external_models/runs/`.
- **Provenance gap, found and left visible rather than corrected silently:**
  `full_run.log`'s header line `git commit (Research-No.1): 1a3c3ae...` is
  **wrong** — that hash does not exist in this repository. It resolves inside
  `~/external_models/sgg`'s own git history (`1a3c3ae Update google drive
  links`), i.e. the logging line captured the wrong repo's `HEAD`. The run
  started at 01:41:53 UTC; the Research-No.1 commit that added the tool code
  used (`f03c921`) landed at 01:42:32 UTC, 39s later, and the working tree has
  been clean since. Circumstantial evidence (timing, no further edits to
  `tools/cross_model/*.py`) supports `f03c921` as the code actually run, but
  this is **inferred, not recorded** — the repo's own convention (an exact
  logged commit hash) was not met here.

**Net integrity read:** the scientific content (checkpoint identity, data
integrity, join completeness, arithmetic self-check) is solid. Two process
gaps — no captured exit code, mislabeled commit provenance — are real but
narrow: neither affects the numbers, both affect strict auditability of
*which exact code version* produced them.

## 6. PURE vs IMP+ — the three questions

Reference (`docs/PAPER1_EVALUATION_TABLE.md` / `runs/p49`, PURE, full-val
cache, tau=0):

| arm | R@50 | mR@50 | WPRD |
|---|---:|---:|---:|
| pair prior only | 66.593 | 22.304 | 0.5000 |
| PURE text (α=0, the evaluated head) | 67.168 | 23.204 | 0.5542 |
| PURE classifier (α=1, discarded) | 67.215 | 22.362 | 0.5728 |
| PURE geometry-linear | 67.273 | 18.576 | 0.5934 |
| PURE classifier+geoMLP (best WPRD in family) | 67.685 | 20.255 | **0.6163** |
| **IMP+ (this run, different split/task setup)** | 76.17 | **24.03** | **0.6205** |

**(a) Does WPRD generalize beyond PURE?**
**Mechanically, yes — cleanly.** The metric computed on a second,
independently-trained model, on a different split, from a different codebase,
with a functioning prior-control identity (exact 0.5) and full row coverage.
That is a genuine methodological result: WPRD is not an artifact of one
project's cache format. **As a scientific instrument it now has n=2.**

**(b) Does the mR@50 ↔ WPRD inversion replicate?**
**Not at this data point — and the direction it fails in is informative.**
Within PURE, every arm with WPRD ≥ 0.59 (the geometry-containing arms) paid
for it with mR@50 in the 18.4–20.3 range, well below the prior-only floor of
22.3. IMP+ has the **highest WPRD of any arm measured to date (0.6205)**,
*and* the highest mR@50 (24.03, NoGC) — simultaneously beating every PURE arm
on both axes at once. If the inversion were a general property of "how mR@K
responds to relational discrimination," IMP+ should not be able to occupy
that corner. **One point cannot refute a correlation claimed at effective
n≈3** (`p49`'s own caveat), but it is a real counter-example to the
*mechanism* PURE's version of the story told: "discrimination costs
calibration." IMP+ suggests that story is not universal — an architecture can
apparently buy both at once, at least under `use_bias=False` on a different
split/task harness. This is the single most important thing this run found.

**(c) What happens to the prior/geometry relationship?**
**Confounded, not answered.** PURE's story was that its evaluated head's
advantage was "(subject,object) identity in text-embedding space," not real
image conditioning (`p26`, `p35` — no measurable tail grounding, box geometry
alone out-discriminates the checkpoint). IMP+'s `edge_model=motifs` head is an
LSTM over **object labels**, which is exactly the kind of channel that can
encode pair-identity/co-occurrence structure *without* using pixels — the
same critique the field has long made of Neural-Motifs. **No decomposition
was run here** (no `p26`-style pair-matched null, no `p35`-style stratified
head/body/tail, no geometry-ceiling probe) to test whether IMP+'s high WPRD is
genuine visual grounding or the same label-identity effect PURE showed, just
captured through a different architectural channel. Stated plainly: **IMP+'s
WPRD is real and higher than anything PURE produced, but its source is
unmeasured.** This is the natural next question, not a settled one.

## 7. Trustworthiness for the evidence ledger

**Admit to the ledger as `MEASURED`, with two attached caveats, not as a
clean confirmation of anything.**

- Admissible: the WPRD number itself (0.6205/0.5703), the prior-control
  identity, the 0-missing-lookup join, the checkpoint SHA256, the clean log
  with no errors.
- Caveat 1 (§3): R@K/mR@K sit above the published comparison point by more
  than the pilot suggested, and the comparison target is an approximate
  model-family match, not an identical checkpoint. Do not cite this run's
  R@50/mR@50 as "IMP+ reproduces the literature" — cite it as "in the
  expected range, gap not fully explained."
- Caveat 2 (§5): the logged commit provenance is wrong (points at the wrong
  repository); the actual code version is inferred from timing, not recorded.
- **Not admissible yet:** any claim that the mR@50↔WPRD inversion is a
  property of models in general, or that it is specific to PURE. n=1 cannot
  decide between "the inversion is a general SGG-metric phenomenon that IMP+
  happens to escape" and "the inversion was specific to PURE's linear
  prior+visual ensemble construction." Both remain open.

## 8. Reassessment of Papers A/B/C

Working titles as used elsewhere in this repo's docs (`Paper A` = the PURE
mechanism findings, §1–4c of `docs/RESEARCH_STATE.md`; `Paper B` =
`docs/PAPER1_EVALUATION_TABLE.md`, the WPRD rank-inversion/metric-mismatch
diagnostic, explicitly gated on cross-model per `HYPOTHESIS_MATRIX_LIVE.md`
H11; `Paper C` = the WPRD benchmark protocol, `docs/BENCHMARK_PROTOTYPE.md`).

- **Paper A** (PURE-specific mechanism: prior dominates, model is a bounded
  tie-breaker expressing pair identity in text-embedding space, no measurable
  tail grounding): **unaffected.** Every claim in Paper A is explicitly scoped
  to the one PURE checkpoint (`docs/RESEARCH_STATE.md` §1–4c already say so).
  This run is about a different model entirely and neither strengthens nor
  weakens a single-checkpoint mechanism claim.

- **Paper B** (the mR@K↔WPRD inversion as a general evaluation-methodology
  claim): **the cross-model gate is now partially cleared, and the first data
  point through it does not support the general claim as stated.** Before
  this run, Paper B's central result was `p49`'s effective-n≈3
  within-PURE correlation, explicitly marked "not yet established... the
  cross-model study remains the gate on any general claim." That gate has
  now produced one point, and it sits in the wrong quadrant for the inversion
  (high WPRD, high mR@50 together). **Recommendation: soften Paper B's framing
  from "the inversion" toward "an inversion observed within one model's
  calibration family, of unknown generality — one cross-model check is
  inconsistent with generality, decomposition pending."** This is a
  significant, not cosmetic, revision to how Paper B's headline claim should
  be stated. It does not kill Paper B — the within-PURE result stands on its
  own terms — but it forecloses the strong "a decade of SGG progress may have
  been selecting against grounding" framing (`p49`'s own "why this is the
  most consequential result" section) until more models are checked.

- **Paper C** (WPRD as a reusable benchmark protocol — prior-neutral,
  arithmetic-identity-verified, model-agnostic): **strengthened.** This run is
  direct evidence for exactly the property Paper C claims: the protocol
  (`tools/cross_model/{extract_wprd_pairs,compute_wprd}.py`) ported to a
  second codebase's output format with no changes to the core AUC/grouping
  logic, the prior-control identity held exactly, and the decidable-row
  fraction replicated within 1.4 points across model/split/codebase. That is
  the strongest evidence to date that WPRD *is* portable machinery, independent
  of whether the correlational finding built on top of it (Paper B) turns out
  to be general.

## 9. Next step

**Recommendation: do not launch another large GPU experiment yet.** The
highest-value next action is decomposing *this* result, which is CPU-only and
cheap, before spending GPU time on a third model.

| | |
|---|---|
| **QUESTION** | Is IMP+'s WPRD advantage (0.6205, highest measured) genuine image-conditioned relational grounding, or — like PURE's — mostly (subject,object) label identity funneled through the motifs LSTM? |
| **WHY NOW** | This is the one new fact this run produced that changes a paper's framing (§6b, §8/Paper B). Answering it is what separates "the inversion doesn't generalize" from "IMP+ is a second data point confirming discrimination and calibration are NOT fundamentally in tension" — opposite conclusions for Paper B. |
| **GPU or CPU** | **CPU only.** `wprd_pairs_full.pt` (38.9 MB, already on disk) has `pred_classes`, `pred_rel_inds`, `rel_scores` per row — enough to build a `p26`-style pair-matched null (permute the model score within (s,o) groups, image content destroyed, identity preserved) and check whether IMP+'s WPRD collapses toward 0.5, the way `p26` showed PURE's did. |
| **Estimated time** | ~1–2 hours of tool-writing (port `p26`'s permutation logic to `SimpleGroups`) + seconds of compute. No GPU queue, no VM contention risk. |
| **Expected information gain** | High relative to cost. Directly resolves §6c above. If the null *doesn't* collapse WPRD (i.e., permuting destroys less than it did for PURE), that is the first real evidence of image-conditioned grounding anywhere in this project — a bigger result than the cross-model run itself. If it *does* collapse the same way, IMP+'s advantage is the same label-identity effect PURE showed, just via a different channel, and Paper B's "discrimination and calibration trade off" framing survives after all (the corner IMP+ occupies would be explained by disentangling from the model score, not by counter-example). |

If a second published checkpoint is wanted after that (VCTree/TDE, same
`bknyaz/sgg`-style repo family, same conversion pipeline already built and
reusable), it is the natural GPU follow-up — but only after the pair-matched
null result is in hand, since it decides what question the second model
should even be asked.

---

## 10. Decomposition and reproduction-gap correction (2026-09-03, later session)

GPU checked idle (`nvidia-smi`, 0% util, no processes) before this work; all
of it is CPU-only, reusing `wprd_pairs_full.pt` plus one new CPU-only,
no-model, no-GPU extraction (`tools/cross_model/extract_geometry.py`, 183,640
rows in ~1s, alignment to `wprd_pairs_full.pt` verified by an exact `gt_y`
equality assertion, not assumed). Tool:
`tools/cross_model/decompose_imp_plus.py`. Full output:
`runs/cross_model_imp_plus/decomposition.json`; log:
`/tmp` (background run, 3m05s wall clock, exit 0, reproduced separately below).
It reuses `tools/cross_model/compute_wprd.py`'s `SimpleGroups`/`wprd_generic`
(which itself imports `auc()` from `tools/within_pair_discrimination.py`) for
every arm's aggregate score, and `tools/wprd_geometry_control.py`'s exact
`_geom`/`_standardise`/`cross_fit_logits` for the geometry probe — no second
metric was invented.

### 10.1 The reproduction-gap correction (§3, §6, §9 above are superseded)

**§3's comparison target was wrong.** The checkpoint's own README
(`~/external_models/sgg/README.md`) cites the BMVC 2020 paper (Knyazev et al.,
arXiv:2005.08230, fetched and read directly this session, Table 1) for the
"IMP+" checkpoint's numbers — but that paper trains and reports **two
different architectures**, "Message Passing (MP) [Xu et al.]" and "Neural
Motifs (NM) [Zellers et al.]," and Table 1's **Predicate Classification**
block (no graph constraint; confirmed from the paper's own text) reads:

| Model | Loss | R@50 | mR@50 |
|---|---|---:|---:|
| FREQ (always predict the most frequent predicate) | — | 69.8 | 22.1 |
| **MP, Baseline** | Eq. 3 | **74.8** | **20.6** |
| MP, Ours (density-normalized loss) | Eq. 6 | 78.2 | 32.1 |
| **NM, Baseline** | Eq. 3 | **80.5** | **26.9** |
| NM, Ours (density-normalized loss) | Eq. 6 | 82.0 | 34.8 |

**74.8/20.6 is the "MP, Baseline" row — Xu et al.'s Message-Passing
architecture — not this checkpoint.** The checkpoint we ran loads with **0
missing / 0 unexpected keys** against `edge_model="motifs"` (`docs`, §1,
already verified from the state dict, not from a label) — i.e. it is
empirically the **Neural-Motifs family**, and the README's own footnote for
"IMP+" (`-loss baseline`) matches Table 1's **"NM, Baseline"** row's loss
label exactly. **The correct published comparison is R@50 = 80.5, mR@50 =
26.9, not 74.8/20.6.**

| | R@50 | mR@50 |
|---|---:|---:|
| our full run (NoGC, n=26,446) | 76.17 | 24.03 |
| ~~wrong target: MP, Baseline~~ | ~~74.8~~ | ~~20.6~~ |
| **correct target: NM, Baseline** | **80.5** | **26.9** |
| **corrected gap** | **−4.33** | **−2.87** |

This changes the reading completely. Under the wrong target, R@50 and mR@50
moved in **inconsistent** directions/magnitudes relative to each other
(+1.4 vs +3.4, a pattern with no obvious single cause) — which is what made
§3 flag the gap as "wider than sampling noise explains" and left it looking
unexplained. Under the correct target, both metrics fall **short** by a
**consistent, modest, same-direction** margin (−4.33 / −2.87) — the signature
of an ordinary reproduction shortfall (different codebase, a converted rather
than original `VG-SGG.h5`, possibly a different checkpoint snapshot/seed than
the one Table 1's row was measured from), not of a protocol bug inflating the
numbers.

**Reclassification: PARTIALLY EXPLAINED.** The dominant driver of the gap —
comparing to the wrong architecture's row — is now identified with high
confidence and removes the "unexplained, plausibly wrong" reading entirely.
The residual ~4-point shortfall against the *correct* row is not further
diagnosed this session (would need e.g. the original, uncoverted `VG-SGG.h5`,
or the exact training/eval commit of this specific checkpoint file, neither
available) and should not be treated as fully resolved.

### 10.2 IMP+ WPRD decomposition — A/B/C/D

Registered gates first: model WPRD reproduces the committed `wprd_result_full.json`
value (0.620539) to 1e-3 — **PASS**; prior control reads exactly 0.500000 —
**PASS**.

| arm | wprd_macro | wprd_weighted | reads |
|---|---:|---:|---|
| **model** (IMP+ rel_fc/motifs head) | **0.6205** | 0.5703 | the registered number |
| prior_control (empirical within-group freq.) | 0.5000 | 0.5000 | exact identity, gate |
| **pair_matched_null** (model score permuted **within** group) | **0.4997** | 0.4987 | collapses to chance |
| random_null (iid gaussian) | 0.5029 | 0.5024 | chance, sanity gate |
| **geometry_crossfit** (19 box numbers, cross-fitted, no pixels) | **0.6229** | 0.5597 | ≥ the model |
| geometry_crossfit, shuffled labels | 0.5069 | 0.4965 | chance, sanity gate |

**Variance split:** 71.02% of the model's total score variance is
**between-group** (pair identity), 28.98% **within-group**. **Invariance
gate: PASS** — WPRD on the raw term and on its within-group-centred version
are identical to 6 decimals (0.620539 = 0.620539), which is the algebraic
proof, not an estimate, that **item B (pair identity) contributes exactly
zero to WPRD**. This resolves an open question from §6(c) above: the
"(subject,object)-identity-via-the-motifs-LSTM" hypothesis for IMP+'s WPRD is
**not viable even in principle** — WPRD cannot see that component by
construction, for IMP+ exactly as it could not for PURE (`p42`).

**Layout/residual split of the within-group term** (ridge, cross-fitted,
5-fold, `tools/within_group_decomposition.py`'s exact method): out-of-fold
**R² = 4.44%** of the within-group model variation is *linearly* predictable
from box geometry. WPRD of that layout-predictable slice = **0.5959**; WPRD
of the residual (non-layout) slice = **0.5641** — clearly above chance, so
some non-geometry component survives, but it is markedly weaker than the
layout-carrying part.

**Reading — the answer to A vs. B vs. C:**

- **B (pair identity): ruled out, by construction (proof, not estimate).**
- **A (genuine, non-geometric grounding): present but small.** The
  pair-matched null collapsing to exact chance (0.4997) proves the signal is
  tied to the *specific row/instance*, not interchangeable within a group —
  but that is equally consistent with "the signal is this instance's unique
  box layout" as with "the signal is this instance's unique visual content."
  The residual-after-geometry WPRD (0.5641, clearly excluding 0.5) is the
  cleanest evidence of something beyond geometry, and it is modest.
- **C (geometry/spatial): the dominant identified component.** A
  **linear**, cross-fitted, **pixel-free** 19-number box probe alone reaches
  **0.6229 — matching or exceeding the full IMP+ model (0.6205)**. This is
  the *same* qualitative finding PURE produced (`docs/GEOMETRY_SGG_BASELINE_RESULT.md`,
  `p38`/`p39`: "box geometry out-discriminates the checkpoint") — now
  independently replicated in a second model, a second codebase, and a
  different split. **This is the most robust finding of this session**: it
  now holds in 2/2 models checked, versus the mR@K↔WPRD inversion which holds
  in 0/1 additional models (§10.3).
- **D (calibration/score construction): not a confound here.** Both sanity
  nulls (random, shuffled-label geometry) read within noise of exactly 0.5,
  and the prior-control identity is exact — nothing in the WPRD construction
  itself is inflating the 0.6205 or 0.6229 numbers.

**Caveat on the geometry probe:** cross-fitted only (5-fold, image-level),
**not train-fitted** — no VG150 **train** split was converted for this model
(only `VG-SGG-test.h5` exists on disk; converting train was out of scope this
session). PURE's strongest geometry number (0.5961) was **train-fitted**,
which removes even the mild "cross-fitted has a residual advantage" concern
`docs/ESTIMATOR_MATCHED_GEOMETRY_RESULT.md` raised for PURE; IMP+'s 0.6229
does not have that same guarantee and is a **linear** probe only (PURE also
has a geometry-**MLP** arm; not replicated here). Treat 0.6229 as directionally
solid (it clears the model with room to spare, and the layout/residual split
independently corroborates a large geometry contribution) but not as tightly
bounded as PURE's own train-fitted number.

### 10.3 Head/body/tail (buckets from this split's own GT counts: top-15/next-20/last-15)

| bucket pair | model | geometry_crossfit | random_null | n_comparisons |
|---|---:|---:|---:|---:|
| head-head | 0.6305 | 0.6282 | 0.5007 | 2,310,467 |
| body-head | 0.6210 | 0.6282 | 0.5037 | 609,514 |
| body-body | 0.5935 | 0.6180 | 0.5192 | 28,843 |
| head-tail | 0.5846 | 0.5868 | 0.5114 | 118,535 |
| body-tail | 0.5832 | 0.5937 | 0.4854 | 18,983 |
| **tail-tail** | 0.5679 | 0.5189 | 0.5233 | **1,650** |

Same caveat PURE's own tail-tail cell needed (§6c of the original doc, and
`p35`): **tail-tail is underpowered** (147 cells / 1,650 comparisons, 95% CI
[0.495, 0.630] for the model) — its point estimate should not be read as a
finding on its own. Everywhere else, both model and geometry sit clearly
above chance and above each other's null, and **the head>tail gradient
matches PURE's** (`p35`: discriminability falls toward the tail).

### 10.4 Corrected PURE vs. IMP+ comparison (apples-to-apples population)

**§6's PURE comparison table used the wrong PURE population.** IMP+'s
subject/object labels are the canonical 150-category VG150 vocabulary
(verified at conversion time — categories.json ids equal the canonical
alphabetical index). PURE's `p49` numbers (used in §6) are on
`datasets_vg150_clean`'s **raw, unrestricted** Visual Genome object names
(16,929 distinct categories — `docs/DATASET_IDENTITY_OBJECT_VOCAB.md`), a
**different, finer-grained population** than IMP+'s. The correct comparison
is `runs/p53` / `docs/VG150_RESTRICTED_REPLICATION.md`, which restricts PURE
to the same standard 150-category population:

| arm | WPRD | mR@50 | R@50 | population |
|---|---:|---:|---:|---|
| PURE pair prior only | 0.5000 | 21.250 | 70.666 | standard VG150 (`p53`) |
| PURE text (α=0, deployed head) | 0.5553 | 22.464 | 70.319 | standard VG150 (`p53`) |
| PURE classifier (α=1) | 0.5815 | 21.526 | 70.860 | standard VG150 (`p53`) |
| PURE geometry-linear | 0.6168 | 18.565 | 70.898 | standard VG150 (`p53`) |
| **PURE geometry-MLP (best in family)** | **0.6452** | 20.124 | 71.022 | standard VG150 (`p53`) |
| **IMP+ model** (this run) | **0.6205** | 24.03\* | 76.17\* | VG150 test, different codebase |
| **IMP+ geometry-crossfit (linear)** | **0.6229** | — | — | VG150 test, different codebase |

\* IMP+'s R@50/mR@50 come from a **different evaluator, split, and protocol**
than PURE's — WPRD is portable by construction (prior-free, cancels
per-class calibration exactly), R@K/mR@K are **not**. Putting IMP+'s raw
mR@50 next to PURE's raw mR@50, as §6(b) of this document did, compares two
numbers that are not on a controlled common scale.

**Correction to §6(b)'s "IMP+ occupies the impossible corner" claim: retracted
as stated, replaced with a narrower and more defensible reading.**

- On WPRD (the one axis that *is* portable): IMP+ (0.6205) sits **between**
  PURE's deployed head (0.5553) and PURE's best geometry arm (0.6452) — a
  **mid-family** value, not an outlier, and **not** "the highest WPRD ever
  measured in this project" as §6(b) claimed (PURE's geometry-MLP is higher,
  once compared on the right population).
- On mR@50/R@50: **no valid cross-model comparison exists** — different
  evaluators, different splits, different candidate-generation and
  tie-breaking conventions. §6(b)'s claim that IMP+ "simultaneously beats
  every PURE arm on both axes at once" is **not supportable** with the data
  in hand and should not be repeated.
- What **does** replicate, cleanly, on the one portable axis: **geometry ≥
  model, in both PURE and IMP+.** That is the finding this cross-model run
  actually supports at n=2, and it is a materially different (and better
  supported) claim than the one §6(b) made.

### 10.5 Re-answering the three questions from §6

**(a) Does WPRD generalize beyond PURE?** Still yes, and now more strongly —
the metric, the prior-control identity, the pair-matched-null construction,
the variance-split invariance gate, and the geometry-probe methodology **all**
ported to a second codebase without modification and without a single gate
failure.

**(b) Does the mR@50↔WPRD inversion replicate?** **Untested, not
refuted — §6(b)'s "not at this data point" verdict is itself
retracted.** The only valid cross-model comparison axis (WPRD) shows IMP+ as
an unremarkable mid-family point, not a counter-example. Testing the
inversion cross-model would require either a controlled, shared R@K/mR@K
protocol across codebases (not attempted), or a within-IMP+ calibration
sweep analogous to PURE's α sweep — which does not exist, because IMP+ has
no prior/calibration term to sweep (§9's "prior-conflict: N/A" finding,
confirmed again in §10.2's item D).

**(c) What happens to the prior/geometry relationship?** **Resolved, not
confounded.** Pair identity provably contributes zero to WPRD (§10.2's
invariance gate). Geometry is the dominant identified component of what
remains, replicating PURE's own `p38`/`p39` finding independently. This is
the clearest, best-supported result of the whole cross-model programme so
far.

### 10.6 Trustworthiness, updated

Admit §10's decomposition results to the ledger as `MEASURED`: every arm's
top-line number is reproduced from a registered gate (model WPRD, prior
control) or an algebraic identity (invariance gate), the geometry extraction
was alignment-verified against `wprd_pairs_full.pt` by exact `gt_y` equality
(not assumed), and every computation reused the project's existing `auc()`
and `wprd_generic` rather than a new implementation. The reproduction-gap
correction (§10.1) is `MEASURED` from the paper's own Table 1, fetched and
read directly this session (page images inspected, not summarized from a
search snippet). Caveats: the geometry probe is cross-fitted-only, linear-only
(§10.2); the residual ~4-point R@50/mR@50 shortfall against the *correct*
baseline is unexplained in detail (§10.1).

---

*§1–§9 generated 2026-09-03 (first session) from
`~/external_models/runs/{full_run.log,full_eval_result.json,
wprd_result_full.json,pilot_300.json}` and `runs/cross_model_imp_plus/*`.
§10 added 2026-09-03 (later session) from
`tools/cross_model/{extract_geometry.py,decompose_imp_plus.py}`,
`runs/cross_model_imp_plus/decomposition.json`, and Knyazev et al. 2020
(arXiv:2005.08230) Table 1. GPU checked idle before and during both
sessions; no experiment was launched in either.*
