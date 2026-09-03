# Cross-model WPRD — IMP+ (bknyaz/sgg, Neural-Motifs family), first result

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

*Generated from `~/external_models/runs/{full_run.log,full_eval_result.json,
wprd_result_full.json,pilot_300.json}` and `runs/cross_model_imp_plus/*` on
2026-09-03. GPU checked idle (`nvidia-smi`, 0% util, no processes) before and
during this analysis; no experiment was launched.*
