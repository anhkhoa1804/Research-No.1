# Forensic diagnostic: why Experiment A shows ~0.94% mR@50 vs. the checkpoint's self-reported ~22.64%

Status: diagnostic only. No production code changed. No new long-running
evaluation was run to produce this document — everything below is either
(a) already-computed output from the completed Experiment A run, reread
and cross-tabulated, or (b) a cheap, static, no-model-loading computation
(file diffs, vocabulary comparisons). Evidence preserved in
`runs/historical_checkpoint_diagnostic/`.

## Phase A — evidence preserved

`runs/historical_checkpoint_diagnostic/` now contains: `exp_a_smoke_command.py`
(exact script), `exp_a_smoke_metrics.json` (full raw `eval_sgg_standard`
output, all six tasks + all diagnostics), `checkpoint_metadata.json`,
`checkpoint_sha256.txt`, `git_commit.txt` (`220c5c2e...`), `environment.txt`.
Full effective runtime config is tabulated in that directory's `README.md`
rather than repeated here.

Checkpoint SHA256 re-verified unchanged immediately before writing this
report: `8845c3af...ad442` (`VERIFIED FROM EXPERIMENT` — recomputed twice
this engagement, identical both times). Not modified, not moved, not
converted.

## Phase B — is 0.94%/1.41% internally plausible for this sample?

From `exp_a_smoke_metrics.json` directly (no recomputation):

| Quantity | Value |
|---|---|
| Images evaluated | 16 |
| Objects (from `object_diag`) | 439 |
| GT relations (`predcls.num_gt`) | 213 |
| GT pairs (unique subj/obj, `pair_proposal_diag.n_gt_pairs`) | 204 |
| Candidate pairs (`pair_proposal_diag.candidate_pairs`) | 213 (== `n_gt_triplets` exactly, because `eval_sgg_use_gt_pairs=True` makes the candidate set *equal* the GT pair set) |
| Distinct predicate classes present in GT (`object_diag`/`predicate_diag.exposure.num_predicates`) | 29 of 50 |
| R@50 hits | 3 of 213 (0.0141 × 213 ≈ 3) |
| mR@50 | 0.0094 (per-class average recall, `_compute_global_mr`) |

**Why `R@20 == R@50 == R@100` exactly**: under GT-pairs mode the candidate
pool per image is tiny (avg 13.3 pairs/image, confirmed by
`pair_proposal_diag.avg_candidate_pairs_per_image`). K=20 already exceeds
essentially every image's entire candidate pool, so the ranked-list top-20,
top-50, and top-100 are identical sets for nearly all images — this is
structural, expected behavior of GT-pairs mode with VG150's typical
per-image relation count, **not a bug**.

**Isolating the cause, using `pair_proposal_diag` — the single most
important fact in this diagnostic**:

```
pair_proposal_diag.per_predicate["on"]:  n=83, recall@32=1.0, recall@64=1.0, ... recall@256=1.0
```

Pair-level GT recovery is **100% across every K, every predicate bucket
(head/body/tail), and every individual predicate shown** — including
`"on"`, the single largest class in this sample (83/213 = 39%). This means:
the (subject, object) identity of every GT relation was correctly located
among the candidates. **The collapse is not in pair identification. It is
entirely in predicate classification** — for a pair correctly identified
as GT-annotated "on", the model's chosen predicate label is (almost
always) something other than "on".

Ruled out by this evidence alone: malformed GT, candidate-pair filtering/
pruning (none occurred — `prune_score_mode` block never triggers, `prune_k=0`
in this config), relationness gating (explicitly disabled and irrelevant
under GT-pairs mode regardless), and the GT-extraction fix itself (see
Phase G — if anything, this data is a direct, positive demonstration that
the fix works exactly as intended).

**`predicate_diag`** confirms the qualitative signature: GT's top class is
`"on"` (83); the model's top-predicted class is `"holding"` (47), followed
by **`"flying in"` (39)** — a rare, low-frequency VG150 predicate —
while `"on"` **does not appear in the model's top-8 predictions at all**.
A weak-but-correctly-wired raw classifier over-predicts the majority class
on imbalanced data; it does not invert to prefer a rare tail class over a
head class it saw 39% of the time in the very same sample. This pattern is
not explainable by small sample size alone (Phase H) — it requires a
structural explanation.

## Phase C — historical vs. current configuration, field by field

| Field | Historical (checkpoint's embedded cfg / inferred) | Current (Experiment A) | Classification |
|---|---|---|---|
| `explicit_spoa_enabled` | Field absent — architecture didn't exist yet (effectively off) | `False` (explicit override) | **SAME** (intent preserved) |
| `asymmetric_pair_fusion_enabled` | Absent (off) | `False` (current default) | **SAME** |
| `relationness_enabled` / `eval_sgg_use_relationness` | Absent / `False` | `False` (explicit) | **SAME** |
| `text_conditioned_projection_enabled` | Absent (off) | `False` (current default) | **SAME** |
| CLIP model (`clip_name`) | `openai/clip-vit-large-patch14-336` | same (carried via cfg overlay) | **SAME** |
| CLIP preprocessing (`clip_input_res`) | `336` | same (carried) | **SAME** |
| **Predicate vocabulary (index mapping)** | `vg150_root='datasets'` — file no longer exists on this machine; true index order **unrecoverable** | `datasets_vg150_clean/vocabulary/predicates.json` — **confirmed to contain a wrong 50-predicate set** (see Phase E) | **MISSING/UNKNOWN vs. current — POTENTIALLY OUTPUT-CHANGING** — the single highest-priority unresolved variable in this entire diagnostic |
| Frequency prior (`freq_bias_enabled`/`alpha`) | `False` / `1.0` (checkpoint's own training-time eval) | `False` / n/a (explicit, "raw" by design) | **SAME as checkpoint's own training eval** (but **DIFFERENT, by design**, from `demo_config.env`'s separately-reported best-eval config, which used `alpha=3.75` + implied `enabled=True`) |
| `adaptive_calibration_enabled` | `True` (checkpoint's training-time value) | `False` (explicit — Experiment A is defined as "raw") | **INTENTIONALLY DIFFERENT** (by design, not a bug) |
| `use_rfs` | `True` | `True` (carried) | **SAME**, and also `SAFE DEFAULT` — confirmed no-op for the `local-jsonl` backend either way (`use_rfs_is_inert`, pre-existing documented behavior) |
| `use_all_pairs` | `True` | `True` (carried) | **SAME** |
| `eval_sgg_use_gt_pairs` | `True` | `True` (explicit) | **SAME** |
| `source_mode` (loader) | `local-jsonl` (inferred from `vg150_source`) | `local-jsonl` (explicit) | **SAME format**, but pointed at a **DIFFERENT root** — `datasets` (historical, gone) vs. `datasets_vg150_clean` (this session's prep) |
| Predicate aliases (data-prep-time) | Whatever `VG150_PREDICATE_ALIASES` contained at the historical training commit (bounded to before `2026-05-27`ish) | Current `VG150_PREDICATE_ALIASES`, which received *additional* alias entries in commits dated `2026-05-28` (one day after the checkpoint's bounded upper commit estimate) | **DIFFERENT / POTENTIALLY OUTPUT-CHANGING** — affects which raw predicate strings get canonicalized during data prep, not the index-mapping bug directly, but a second, independent source of dataset-content drift |
| Background predicate handling | 51-way classifier (50 + synthetic "relation") | Same (`predicate_classifier_classes=51`, confirmed via loaded weight shape) | **SAME** |
| Pair proposal / relationness | Inactive (architecture didn't exist) | Inactive (explicit) | **SAME** |
| Top-K settings | `ks=[20,50,100]`, hardcoded in `evals.py`, unchanged across the checkpoint's bounded commit range | same | **SAME** |
| Temperature/logit scaling | `eval_sgg_classifier_temperature=1.0`, `eval_sgg_text_temperature=1.0` | same (carried) | **SAME** |
| Model dimensions | `emb_dim=768`, CLIP vision/text dims 1024/768 | same (verified via successful weight load, shape-matched) | **SAME** |
| Checkpoint loading (missing/unexpected keys) | n/a | 59 missing / 0 unexpected, all traced (Phase F) | **VERIFIED SAFE** |

**The one field that dominates every other consideration in this table**
is predicate vocabulary index mapping — everything else is either
confirmed identical, a safe default, or an intentional, small, by-design
difference (raw vs. calibrated). Only the vocabulary field is both
*unverifiable against the true historical value* and *independently
confirmed wrong* in its current form.

## Phase D — provenance of the historical 22.64%

Traced as far as the available artifacts allow (repeating and tightening
the findings from `docs/HISTORICAL_CHECKPOINT_PROVENANCE.md`):

- **Source**: `checkpoints/demo_best/demo_config.env`, field
  `BEST_FULL_PREDCLS_MR50=0.2264`. No training log, `metrics.jsonl`, or
  other artifact survived alongside the checkpoint — this env file is the
  *only* record of this number anywhere.
- **Protocol**: `PROTOCOL=PredCls_GT_pair` (human-written label, not
  read by any script — grep-confirmed) → `INFERENCE`: PredCls + GT pairs,
  consistent with what I ran.
- **Score mode**: `EVAL_SCORE_MODE=ensemble`, `EVAL_ENSEMBLE_ALPHA=0.0` →
  **DIFFERENT** from my "classifier"-only raw config, by design.
- **Frequency bias**: `FREQ_BIAS_ALPHA=3.75`, `FREQ_BIAS_PATH` set — but
  **`FREQ_BIAS_ENABLED` is never set in this file**, and
  `_load_frequency_bias` requires it `True` for `alpha` to have any effect.
  `INFERENCE`: almost certainly passed as a bare `--freq_bias_enabled true`
  CLI flag not captured in this env snapshot.
- **Aggregation** (pooled `mR@50` vs. `image_mean_R@50`): **UNKNOWN**.
  `train.py`'s own checkpoint-selection code (`predcls_mr50 =
  predcls_metrics.get("mR@50", 0.0)`) reads the *pooled* key specifically
  — if this number came from that exact code path, it's pooled. But the
  filename (`pure_best_adapt_light_mR50.pt`) doesn't match what that same
  code would have produced (`core_l3_balanced_adapt_light_best_mR50.pt`,
  from the checkpoint's own embedded `save_path`) — `INFERENCE`:
  the file was manually renamed/copied into `demo_best/`, which also means
  it may not have come from the automated checkpoint-selection path at all,
  and could instead be from a manual, standalone `eval_l4_phase34.sh`-style
  invocation. **Marked `UNKNOWN`, not assumed.**
- **Split**: not stated anywhere in the recovered artifacts. `INFERENCE`
  (weak): "validation" is this repo's overwhelmingly standard default for
  a "best" checkpoint-selection metric, but this is not verified.
  **Marked `UNKNOWN`.**
- **Dataset preparation version**: not recorded — no hash, no manifest, no
  diagnostics.json survived alongside the checkpoint. **`UNKNOWN`.**
- **Predicate vocabulary used for that specific eval**: **`UNKNOWN`** —
  same root problem as Phase C's central finding. If that eval's
  `vg150_root` had a *correct* vocabulary file, its classifier-mode
  contribution to the `"ensemble"` score would have been meaningful; if it
  shared the same defect found in this session's data, the ensemble's
  text-scoring component may have been doing most of the real work, with
  the classifier component contributing noise that the ensemble blend
  partially absorbed. **Cannot be resolved without the original
  `datasets/` directory.**

**Conclusion**: 22.64% cannot be independently reproduced or fully
provenance-traced from what survived. It is a **self-reported, single-source
number** from a differently-configured (calibrated, ensemble, freq-bias-boosted)
evaluation, run in an environment whose exact dataset/vocabulary state is
unrecoverable. This is not evidence the number is false — but it is not
independently verified either.

## Phase E — dataset compatibility

Reusing this session's earlier independent validation
(`data/manifests/vg150_clean_validation_report.md`: 0 duplicate IDs, 0 bad
boxes, 0 invalid predicate strings *within the current, possibly-wrong
vocab*, 0 invalid relationship indices, 0 cross-split contamination) plus
new findings this phase:

| Aspect | Finding |
|---|---|
| Image IDs / split membership | Independently re-validated this session (Phase 4-5 of the prior turn) — clean, no contamination. Unaffected by anything in this diagnostic. |
| **Predicate vocabulary (index mapping)** | **CONFIRMED WRONG.** `datasets_vg150_clean/vocabulary/predicates.json` (byte-identical to the pre-existing, not-created-this-session `datasets/vg_raw/vocabulary/predicates.json`) contains `"growing on"` (idx 18) and `"says"` (idx 39) — **zero occurrences** in the actual cleaned data — and is **missing** `"next to"` (**19,410 occurrences** in `train.jsonl`) and `"wrapped around"` (**4,478 occurrences**). Comparing this file's order against `STANDARD_VG150_PREDICATES` sorted alphabetically (the set actually used to *filter* the data, one function away in the same tool): **24 of 50 indices (48%) point to a different predicate name.** See the full index table in `docs/PREDICATE_VOCAB_INDEX_BUG_TRIAGE.md`. |
| Object class mapping | `datasets_vg150_clean/vocabulary/objects.json` has the correct **count** (150). **Could not cross-validate order** the way predicates were checked — this codebase has no independent hardcoded canonical *object* list to diff against (unlike `STANDARD_VG150_PREDICATES` for predicates). **Marked UNKNOWN, not cleared.** |
| Background relation representation | Synthetic `"relation"` class, 51st slot — consistent between historical (`predicate_classifier_classes=51` in checkpoint cfg) and current. **SAME.** |
| Relation/pair ordering | This is exactly what commit `220c5c2e` fixed for the *GT-extraction* path; unrelated to the vocabulary index question. **Verified separately (Phase G).** |
| Predicate aliases | Current `VG150_PREDICATE_ALIASES` (`tools/prepare_vg150_subset.py`) received additions in commits dated `2026-05-28`, one day after this checkpoint's estimated upper training-commit bound. **Historical dataset prep almost certainly used fewer aliases.** Affects which raw strings get canonicalized during prep — a second, independent, smaller source of dataset-content drift, not investigated further in this diagnostic (out of scope: doesn't change the vocabulary *index* bug's severity). |
| 50/50 predicate coverage | `diagnostics.json` reports `predicate_coverage: 50` for all splits — but this counts *distinct predicate strings observed*, which is silent to whether the *index assignment* for those 50 strings is correct. **Coverage count passing does not clear this bug** — worth noting as a real gap in `tools/check_vg150_diagnostics.py`'s current checks. |
| JSONL schema | Unaffected — confirmed via this session's earlier independent field-level validation. |
| Image resolution/preprocessing | `clip_input_res=336`, same both historical and current (Phase C). |

## Phase F — checkpoint-loading semantic verification

| Missing key group (of 59) | Feature | Used in `forward()`? | Active under this diagnostic's config? | Risk |
|---|---|---|---|---|
| `decoder.subject_role_embedding`, `object_role_embedding`, `predicate_query`, `{subject,object}_branch.*`, `{subject,object}_attr_branch.*`, `predicate_branch.*`, `spoa_fusion.*` (33 keys) | Explicit SPOA branches | Only if `self.explicit_spoa_enabled` (`model.py:469,515`, clean `if/else`, `rel_feat = self.rel_seed(fused_feat)` in the `else` branch — no reference to these modules at all when off) | **No** — explicitly forced `False` this run | **NONE** (correctly neutralized) |
| `decoder.asymmetric_pair_proj.*` (8 keys) | Asymmetric pair fusion | Only if `self.asymmetric_pair_fusion_enabled` (`model.py:421`) | **No** — defaults `False`, absent from checkpoint cfg so never set `True` | **NONE** |
| `text_space_projection.*` (6 keys) | Text-conditioned projection | Computed unconditionally but **output discarded** unless `self.text_conditioned_projection_enabled` (`model.py:838-844`) | Computed (wasted CPU) but **never blended into output** — flag `False` | **NONE** (dead computation only, zero numerical effect) |
| `relationness_head.*` (6 keys) | Pair-proposal relationness scoring | Only via the separate `relationness_scores()` method, gated by `eval_sgg_use_relationness` in `evals.py` (not `forward()` itself) | **No** — explicitly forced `False` | **NONE** for this run's R@K/mR@K/predcls path (would be **HIGH** if anyone runs relationness-mode prop@K against this specific checkpoint — those weights are random) |

**All 59 missing keys accounted for. None are accidentally active in this
diagnostic's configuration.** This audit **rules out** "checkpoint loading
is silently activating newly-introduced architecture components" (decision
tree cause C) as a contributor to the observed collapse.

## Phase G — the GT-extraction fix, isolated

Per Phase B, `pair_proposal_diag` from *this actual run* already
demonstrates the fix operating correctly on real data: 100% GT-pair
recovery at every K, including for the dominant class "on". This alone is
strong evidence the fix is not the cause.

A theoretical argument for *direction*, using the exact counterexample from
`docs/GT_EXTRACTION_BUG_TRIAGE.md` §2 (3 objects, 2 relations: real GT is
`(0,1,'wearing')`, `(1,2,'near')`; the pre-fix bug reassigns the second
relation to `(0,2,'near')`, a pair with no real annotation):

- **Pre-fix**, a model that correctly predicts `'near'` for the true pair
  `(1,2)` gets **no credit** — the GT entry it needs to match is (wrongly)
  attached to `(0,2)`, a different pair with (presumably) a different or no
  confident prediction. This is a **false miss**.
- **Post-fix**, that same correct prediction for `(1,2)` **matches
  correctly**.

So the fix, when it changes anything, should make a genuinely accurate
model's *measured* recall go **up or stay flat, not down** — it removes
false misses caused by mis-attributed ground truth; it does not introduce
new ones (the fix doesn't touch prediction generation, ranking, or
candidate construction, only which GT entry a correct prediction is scored
against). **This directly argues against decision-tree cause E** (`GT
extraction was historically wrong and historical numbers were themselves
invalid` — implying the fix caused a regression): the arithmetic points the
opposite direction. It does **not**, however, prove the historical 22.64%
was measured under the pre-fix bug at all — Phase D already established
that provenance is unrecoverable — so this argument is conditional
("if the historical eval also had this bug, fixing it should have helped,
not hurt") rather than a direct explanation of the historical number
itself.

**No full-dataset comparison was run**, per instructions — this is a
closed-form argument plus the real, observed 100% pair-recovery evidence
from Phase B, not a new experiment.

## Phase H — statistical meaningfulness of this smoke test

213 GT relations, 3 hits at R@50, 29/50 predicate classes represented
across 16 images. The **overall low rate** is influenced by small sample
size in the ordinary statistical sense (wide confidence interval on a
handful of hits) — but the **qualitative pattern** (an 83-instance majority
class receiving zero top-8 predictions, a rare tail class topping the
prediction list) is **not** a small-sample artifact; that specific
inversion would be a remarkable coincidence under any noisy-but-unbiased
classifier and requires a structural explanation. **Label this run exactly
as instructed: "historical checkpoint smoke-test diagnostic — NOT A
RESEARCH BASELINE."**

## Phase I — decision tree, ranked by confidence

| # | Cause | Evidence | Confidence | Severity | Fixable w/o retraining? | Retraining required? |
|---|---|---|---|---|---|---|
| **B** | **Dataset/vocabulary incompatibility** — `vocabulary/predicates.json` contains a wrong 50-predicate set (extra: growing on/says, zero occurrences; missing: next to/wrapped around, 23,888 occurrences); 48% of indices point to a different name than the one alternative ordering available for comparison | Direct file diff + real-data occurrence counts + `predicate_diag`'s inversion pattern (rare class dominant, head class absent) | **HIGH** | Critical | Yes — fix the vocab file, re-evaluate | Not for evaluation; **unknown** whether the original training run itself used a correct or defective vocab (open question, see Phase D/caveat below) |
| **D** | **Historical number used a different (calibrated/ensemble/freq-bias) configuration** than this "raw" run, by design | `demo_config.env` fields vs. this run's explicit raw overrides (Phase C table) | **MEDIUM**, as a *contributing*, expected, non-bug factor — does not by itself explain a ~24x gap | Not applicable (expected) | N/A | N/A |
| **F** | **Checkpoint genuinely weak under the corrected protocol** | Not ruled out, but the specific inversion pattern (tail-over-head) is hard to explain as genuine model weakness alone — a weak-but-correctly-wired model should still correlate with true class frequency | **LOW-MEDIUM**, unresolved | Unknown | Unknown | Unknown — needs a clean re-test with a corrected vocabulary to isolate |
| **A** | **Current eval config incompatible with historical checkpoint** (architecture drift) | Fully audited in Phase C/F; the one required override (`explicit_spoa_enabled=False`) was correctly applied and verified | **LOW** — already neutralized, not a remaining cause | N/A (handled) | Already handled | No |
| **C** | **Checkpoint loading silently activates new architecture components** | Phase F table: all 59 missing keys traced, none active under this config | **RULED OUT** for this run | N/A | N/A | N/A |
| **E** | **GT extraction was historically wrong, invalidating historical numbers / the new fix caused a regression** | Phase G's closed-form argument and this run's own 100% pair-recovery both point the opposite direction (fix should help, not hurt, real recall) | **LOW** | N/A | N/A | N/A |
| **G** | **Multiple causes** | Most defensible single framing: **B is primary and dominant**, **D is a real but secondary, by-design contributor**, **F is unresolved and cannot be ruled out without a corrected-vocab re-test**, **A/C are neutralized**, **E argues the wrong direction** | — | — | — | — |

**Important caveat on cause B's severity ceiling**: the index-shift table
in `docs/PREDICATE_VOCAB_INDEX_BUG_TRIAGE.md` compares the *current*
(wrong) vocab file against *one* plausible reference ordering
(alphabetical `STANDARD_VG150_PREDICATES`) — and under that specific
comparison, `"on"` happens to land at the **same** index (30) in both
orderings. This means the vocabulary-defect hypothesis, exactly as
quantified, does not by itself fully explain why `"on"` specifically was
never predicted; the checkpoint's *true* historical training-time index
order is unrecoverable (Phase C/D), so it may differ from *both* orderings
compared here. **B remains the highest-confidence cause because of the
qualitative match (rare-class inversion) and the confirmed, independent
existence of a real vocabulary bug — not because the exact index-shift
arithmetic has been proven to reproduce every observed symptom.** This is
flagged honestly rather than overstated.

## Phase J — options (not chosen)

Per instructions, no option below is selected — this is the menu for your
decision:

1. **Reproduce historical evaluation cheaply using the correct historical
   configuration** — blocked until the true historical vocabulary order is
   either recovered or ruled irrelevant; not currently executable.
2. **Run a very small corrected smoke test** — feasible once (and only
   once) `vocabulary/predicates.json` is regenerated from
   `STANDARD_VG150_PREDICATES` (a documentation/data-fix, not a code
   change to `openvocab_rel/`); would directly test cause B in isolation
   at low cost (same 16-image sample, ~same runtime).
3. **Fix another compatibility issue** — none currently known beyond the
   vocabulary bug and the already-fixed GT-extraction bug.
4. **Proceed to full GPU evaluation once compute is available** — premature
   until cause B is resolved or ruled out; would otherwise waste GPU-hours
   measuring a confounded pipeline.
5. **Retrain/fine-tune because the historical checkpoint is incompatible**
   — premature; nothing found so far shows the checkpoint *itself* is
   unusable, only that the current evaluation environment's vocabulary
   file is wrong.
6. **Abandon the historical checkpoint as a baseline candidate** —
   premature for the same reason.

---

## Concise final report

**1. What is definitely known**
- Experiment A's raw numbers (R@50=1.41%, mR@50=0.94%, n=16 images,
  213 GT relations) are real, reproducible outputs of the current pipeline,
  `VERIFIED FROM EXPERIMENT`.
- Pair-level GT recovery in this same run is **100%** at every K, for
  every predicate including the dominant class — the GT-extraction fix
  (`220c5c2e`) is working correctly; the collapse is isolated to predicate
  classification, not pair identification.
- `datasets_vg150_clean/vocabulary/predicates.json` (and its pre-existing,
  not-created-this-session source `datasets/vg_raw/vocabulary/predicates.json`)
  contains a **confirmed wrong** 50-predicate set: 2 entries with zero
  real occurrences included, 2 real, common predicates (23,888 combined
  occurrences in `train.jsonl`) excluded.
- The checkpoint loads with 59 missing/0 unexpected keys; every missing
  key is verified inactive under this diagnostic's configuration — the
  `explicit_spoa_enabled=False` override is correctly applied and
  sufficient.
- The checkpoint file is unmodified (SHA256 re-verified).

**2. What is probably wrong**
- The predicate-vocabulary index defect is the most likely dominant cause
  of the observed collapse (qualitative pattern strongly matches: rare
  class dominant, head class absent from predictions) — `HIGH` confidence
  as *a* major cause, not fully proven as the *complete* explanation.
- The historical 22.64% number's exact protocol (aggregation, split,
  whether its own environment had a correct or defective vocabulary) is
  under-documented and cannot be independently verified from what
  survived.

**3. What is still unknown**
- The checkpoint's true training-time predicate-vocabulary index order
  (the original `datasets/` root no longer exists).
- Whether the historical 22.64% eval used a correct or defective
  vocabulary itself.
- Whether object-vocabulary ordering has an analogous defect (count is
  right, order unverifiable — no independent reference list exists in
  this codebase for objects the way `STANDARD_VG150_PREDICATES` exists
  for predicates).
- Whether, once the vocabulary bug is corrected, the checkpoint's
  predicate classification will land near the historical 22.64% or
  somewhere else — genuinely unresolved, not assumed.

**4. Is the checkpoint usable?**
Not yet determined either way. Nothing found in this diagnostic shows the
*checkpoint itself* (weights, architecture compatibility) is broken — the
compatibility audit (Phase F, plus the prior turn's full audit) is clean.
The blocking issue is environmental (vocabulary file), not the checkpoint.
**Do not discard it and do not certify it as usable — re-test after the
vocabulary fix before deciding.**

**5. Is the current evaluator trustworthy after commit `220c5c2e`?**
For **GT extraction specifically**: yes — directly demonstrated by this
run's 100% pair-level recovery on real data, in addition to the 7 passing
regression tests. For **end-to-end R@K/mR@K under classifier-based
scoring**: **no, not yet** — it is currently confounded by the separate,
newly-found vocabulary bug, which predates and is unrelated to the
GT-extraction fix.

**6. Exact next experiment once compute/fix status allows**
Not a GPU experiment yet. The cheapest next step is **Option 2** above: fix
`vocabulary/predicates.json` (or bypass it with an explicit, correct vocab
override) and re-run *this exact same 16-image, 4-batch smoke test*
(same script, same checkpoint, same seed/config otherwise) to see whether
predicate-classification recall becomes plausible. This requires no new
compute tier — it's the same CPU-only smoke shape already validated to run
end-to-end.

**7. Minimum GPU/RAM/storage requirements**
Unchanged from `runs/pre_fix_code_reference/COMPUTE_REQUIREMENTS.md`
(updated this phase with real measurements: CLIP ViT-L/14-336 = 1.6GB
measured, a full checkpoint = ~888MB measured). No new requirement
discovered by this diagnostic. The one new, practical constraint: **a
16-image smoke test took ~23 min/image on CPU** — any GPU-time estimate for
a real evaluation pass should be sized from that per-image cost (likely
dominated by `SGCLS`/`SGDET`'s per-object CLIP classification calls, not
yet isolated) rather than assumed from first principles.
