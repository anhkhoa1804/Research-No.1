# Project status

**Canonical current-state document.** When this document and any other
document disagree about current state, trust this one and fix the other —
but never silently rewrite a historical claim to make it agree; mark the
older document superseded instead, with a pointer here.

Evidence labels used throughout: `VERIFIED FROM CODE`, `VERIFIED FROM
EXPERIMENT`, `HISTORICAL EVIDENCE` (from a recovered artifact, not
independently reproduced), `INFERENCE`, `RECOMMENDATION`, `UNKNOWN`.

Last updated: **pre-GCP stabilization phase**, at HEAD
`140e163faa7c82ac4ea5ee060fdd255235148ec9` (`chore: add GCP evaluation
preflight`). Working tree clean at the time of writing.

State at this HEAD, in one line each — all `VERIFIED FROM TEST` or
`VERIFIED FROM CODE` unless marked otherwise:

| Item | State |
|---|---|
| GT-extraction fix (`220c5c2e`) | **COMPLETE**, 7 regression tests |
| Predicate-vocabulary fix (`9dc8f45d`, `7d91af49`) | **COMPLETE**, 11 regression tests |
| Predicate-metadata fix (`fa8c0c3b`) | **COMPLETE**, 3 regression tests |
| GCP preflight tool (`tools/gcp_preflight.py`) | **EXISTS**, hardened + tested |
| Canary workflow | **EXISTS**, tested, **not yet executed on GPU** |
| Test suite | **184 passing** (97 at `140e163f` + 87 infrastructure) |
| Historical checkpoint | recovered, SHA256-tracked (§9) |
| Frequency prior | recovered, SHA256-tracked (§9) |
| Scientific baseline | **none exists** (§10) |

> **Retraction notice for readers of earlier revisions of this file.** Three
> claims previously stated here have been falsified by later evidence and are
> corrected in place below, with the original claim preserved as struck text:
> the test count (§7), the runtime figure (§10), and "the text path has not
> been measured" (§11). None of the *fixes* recorded here were retracted —
> only the status claims about them.

## 1. Project purpose

PURE (Predicate-aware Uncropped Relation Embedding): a Scene Graph
Generation (SGG) model for VG150 — dense, never-cropped CLIP ViT-L/14-336
tokens, deformable box-conditioned object routing, Fourier-geometry pair
fusion, optional explicit SPOA (subject/predicate/object/attribute)
branches, edge-conditioned relation decoder layers. `VERIFIED FROM CODE`.

## 2. Current research hypothesis

Not yet re-established as a *tested* hypothesis this phase — the prior
research-bottleneck analysis (an earlier phase, pre-checkpoint-recovery)
proposed a ranked set of candidate bottlenecks (pair proposal recall,
predicate classification, long-tail calibration) as *untested hypotheses
only*. **Nothing in this phase changes or tests that ranking** — this
phase is entirely validity/infrastructure work, per explicit instruction.
~~`RECOMMENDATION`: the vocabulary fix (§11) must land and be verified
before that bottleneck analysis can be meaningfully re-attempted.~~
**That precondition is now met** — the vocabulary fix landed
(`9dc8f45d`, `7d91af49`) and is test-verified. The bottleneck analysis is
nonetheless still **not** unblocked, for a different reason: no
trustworthy baseline number exists yet (§10, §11). The gate is now the
historical reproduction, not the vocabulary.

## 3. Current architecture

`openvocab_rel/models/relational_model.py:RelationalModel` — 79.9M
trainable parameters in the one recovered checkpoint inspected this
engagement (`checkpoints/demo_best/pure_best_adapt_light_mR50.pt`,
`VERIFIED FROM CHECKPOINT`). Frozen CLIP ViT-L/14-336 backbone (default;
`freeze_clip` configurable), deformable object router, Fourier geometry
fusion, optional explicit-SPOA branches (`explicit_spoa_enabled`, default
`True` in current `TrainConfig` — **the recovered checkpoint predates this
architecture entirely and requires it forced `False` for a faithful
load**, `VERIFIED FROM EXPERIMENT`), optional asymmetric pair fusion
(default `False`), optional relationness head for pair-proposal scoring
(default `False`; untrained/random-weight for the recovered checkpoint).
Full architecture detail: `docs/architecture/overview.md`.

## 4. Current repository structure

```
openvocab_rel/{models,datasets}/   core package (untouched this phase)
configs/                            predicate_metadata_vg150.json (drift FIXED, fa8c0c3b), presets.yaml (documentation only, not loaded)
scripts/{train,eval,notebooks}/     entrypoints; eval_historical_checkpoint.sh is the ONLY safe one for the recovered checkpoint (§15)
tools/                               dataset prep, validation, diagnostics, gcp_preflight.py, verify_canary.py
tests/                               184 passing (§7)
docs/
    architecture/                    overview/training/evaluation/data_flow
    reproducibility.md, known_issues.md   living registers
    PROJECT_STATUS.md                this file
    GT_EXTRACTION_BUG_TRIAGE.md, PREDICATE_VOCAB_INDEX_BUG_TRIAGE.md,
    PREDICATE_VOCAB_HISTORICAL_FORENSICS.md, HISTORICAL_CHECKPOINT_DIAGNOSTIC.md,
    HISTORICAL_CHECKPOINT_PROVENANCE.md    audit/diagnostic reports (see §15 note)
    HISTORICAL_CHECKPOINT_MANIFEST.md      canonical artifact freeze (§9)
    GCP_EXPERIMENT_PROTOCOL.md             the exact GCP workflow
notes/                               historical .tex notes + INDEX.md
data/{README.md,manifests/}          dataset policy + VG150 manifest + validation report
runs/                                gitignored — see §5
checkpoints/                         gitignored — see §6
datasets/, datasets_vg150_clean/     gitignored — see §5
```

**§15 note on `docs/` layout**: a subdirectory split (`docs/audits/`,
`docs/experiments/`) was considered this phase to group the growing set of
diagnostic reports. **Not done** — `docs/GT_EXTRACTION_BUG_TRIAGE.md` is
already referenced by an in-code comment in the *committed* fix
(`openvocab_rel/evals.py`, commit `220c5c2e`), and moving only the newer
files would leave an inconsistent half-flat, half-nested layout, which is
worse than fully flat. Revisit if `docs/` grows past roughly a dozen files
— `RECOMMENDATION`, not decided.

## 5. Dataset policy

Nothing dataset-sized is tracked in git — `VERIFIED FROM CODE`
(`git check-ignore -v` confirms `/datasets/`, `/datasets_vg150_clean/`,
`/runs/`, `/checkpoints/` are all root-anchored `.gitignore` patterns).
Only `data/README.md`, `data/manifests/*.yaml`, and validation *reports*
(small, human-readable) are tracked. `datasets_vg150_clean/` (this
session's canonical prep) is independently validated: 0 duplicate IDs, 0
bad boxes, 0 invalid predicate strings *within the current vocabulary*, 0
invalid relationship indices, 0 cross-split contamination
(`data/manifests/vg150_clean_validation_report.md`, `VERIFIED FROM DATA`).
Images are an NTFS junction to the pre-existing raw corpus, not a copy —
no unnecessary 14.6GB duplication.

## 6. Checkpoint policy

Never committed, never modified in place, SHA256-tracked. The one
recovered historical checkpoint
(`checkpoints/demo_best/pure_best_adapt_light_mR50.pt`, SHA256
`8845c3af...ad442`) has been re-verified byte-identical at three separate
points this engagement — `VERIFIED FROM EXPERIMENT`. If a compatibility
conversion is ever needed, the original must be preserved alongside any
derived artifact, never overwritten (not yet needed — the current
`strict=False` load + one config override is sufficient, §9).

## 7. Current test status

~~83 tests passing~~ → **97 tests passing** (`py -3 -m pytest -q`),
`VERIFIED FROM TEST` at HEAD `140e163f`. The 83 figure was correct when
first written and was **not updated** as the vocabulary/metadata fixes
landed; it is corrected here.

Per-file breakdown at this HEAD (`pytest --collect-only -q`):

| File | Tests |
|---|---:|
| `test_imports.py` | 17 |
| `test_config.py` | 14 |
| `test_pair_construction.py` | 12 |
| `test_predicate_vocab.py` | 11 |
| `test_calibration_prior.py` | 7 |
| `test_eval_gt_extraction.py` | 7 |
| `test_geometry.py` | 7 |
| `test_losses.py` | 6 |
| `test_dataset_loader.py` | 5 |
| `test_model_forward.py` | 4 |
| `test_predicate_metadata_coverage.py` | 3 |
| `test_checkpoint_roundtrip.py` | 2 |
| `test_report_card_metric_semantics.py` | 2 |
| **Subtotal at `140e163f`** | **97** |

The pre-GCP stabilization pass added 87 infrastructure regression tests
(no research behavior asserted anywhere in them):

| File | Tests | Guards |
|---|---:|---|
| `test_canary_verifier.py` | 39 | every frozen protocol setting is individually enforced; the verifier never judges metric quality |
| `test_gcp_preflight.py` | 25 | artifact/dataset/vocabulary/prior/image failure modes; no hardcoded hashes; the historical claim is never labelled a baseline |
| `test_historical_eval_protocol.py` | 23 | every script flag exists in the argparser and resolves correctly through all presets |
| **Total** | **184** | |

None of the 184 requires the checkpoint, the dataset, a GPU, or network
access — all of which are absent from a fresh clone.

The coverage gap noted in the earlier revision ("no test exercises the
predicate-vocabulary index-mapping bug") is **closed**:
`tests/test_predicate_vocab.py` (11 tests) accompanied the `9dc8f45d` /
`7d91af49` fix, and `tests/test_predicate_metadata_coverage.py` (3)
accompanied `fa8c0c3b`.

**Remaining coverage gap** (`VERIFIED FROM CODE`): no test exercises the
full `eval_sgg_standard` pipeline end to end — it needs real CLIP weights
and network/HF-cache access. See `docs/known_issues.md`.

## 8. Validity fixes already applied

| Fix | Commit | Status |
|---|---|---|
| Frequency-prior eval-split fallback (silent leak → fail loud) | `65686b5f` | `VERIFIED FROM TEST` (`tests/test_calibration_prior.py`) |
| `negative_pair_ratio` silent inertness → warns loudly | `e043d21c` | `VERIFIED FROM TEST` |
| `use_rfs` silent inertness → warns loudly | `9c6d495b` | `VERIFIED FROM TEST` |
| `.gitignore` gaps (`datasets_vg150_clean/` was unignored) | `45eece98` | `VERIFIED FROM CODE` (`git check-ignore -v`) |
| Dead code removal (4 call-site-verified deletions) | `60754cb0` | `VERIFIED FROM TEST` (full suite unchanged) |
| **GT-triplet extraction index misalignment** | `220c5c2e` | `VERIFIED FROM TEST` + `VERIFIED FROM EXPERIMENT` (§9) |

## 9. Historical checkpoint status

Recovered, inventoried, integrity-verified, load-compatibility audited.
**Loads successfully** with 59 missing / 0 unexpected keys, all 59 traced
to architecture added *after* this checkpoint's training window and
confirmed inactive under the one required override
(`explicit_spoa_enabled=False`) — `VERIFIED FROM EXPERIMENT`
(`docs/HISTORICAL_CHECKPOINT_DIAGNOSTIC.md` Phase F). Training commit
bounded to `[5ed4e429, 215fd6d0)` (2026-05-23 to 2026-06-13), most
plausibly near `2b7ba82f` (2026-05-27) — `VERIFIED FROM GIT HISTORY` for
the bound, `INFERENCE` for the point estimate. Self-reported historical
result (mR@50≈22.64%, `demo_config.env`) is `HISTORICAL EVIDENCE` only —
single-source, unreproduced, protocol partially unrecoverable (§10).
**Not yet certified usable or discarded** — blocked on §11.

### Tracked artifact hashes (`VERIFIED FROM DATA`, re-verified at this HEAD)

| Artifact | SHA256 | Size |
|---|---|---:|
| `checkpoints/demo_best/pure_best_adapt_light_mR50.pt` | `8845c3af7dc39ad7c4c3aa0ba6dfd064a95d182db30be47cccc5f90f7f0ad442` | 931,057,422 B |
| `checkpoints/demo_best/frequency_prior.json` | `144d9f928ed9cc213acbe081b1a8791488a92ddab0ed885da7fd6bb6058c6e6a` | 101,944,045 B |
| `checkpoints/demo_best/demo_config.env` | `c73180698b5b5f6d3a58e5e6b78a39fcb8a81b6250478145d0538a5a107226a6` | 355 B |
| `datasets_vg150_clean/train.jsonl` | `36bc2923b1a3ddb56e331e9b645607606003bd1c8298d1c950b2cdcca7e31ae5` | — |
| `datasets_vg150_clean/validation.jsonl` | `4348ddbb3ce85160d0ebc7522634c68f197b0c99906794e0e4731156740f3412` | 10,401 rows |
| `datasets_vg150_clean/vocabulary/predicates.json` | `e4e88e87c3c26bf65426957ae4028b45f03d33643d5520b6ff9b42b7b60d4dc5` | 50 predicates |

**Frequency-prior structural validation** (`VERIFIED FROM DATA`, this
phase): `predicate_vocab` has exactly 50 entries and equals
`sorted(STANDARD_VG150_PREDICATES)`; `global_log_probs` has 50 entries;
all sampled `pair_log_probs` (74,884), `subject_log_probs` (12,030) and
`object_log_probs` (9,689) rows are length-50. It is therefore loadable by
`_load_frequency_bias` without silent degradation — a check that matters
precisely because that function fails **silently** (see
`docs/known_issues.md`, P1 register).

**Canonical manifest**: `docs/HISTORICAL_CHECKPOINT_MANIFEST.md` (human)
and `data/manifests/historical_checkpoint_v1.yaml` (machine-readable) are
the single source of truth for these artifacts.
**Preflight enforcement**: `tools/gcp_preflight.py` verifies every hash
above and refuses to pass on any mismatch.

## 10. Current evaluation status

The GT-extraction fix is `VERIFIED FROM EXPERIMENT` on real data: a raw
16-image run against the recovered checkpoint showed **100% pair-level GT
recovery** at every K, for every predicate including the dominant class
(`pair_proposal_diag`, `docs/HISTORICAL_CHECKPOINT_DIAGNOSTIC.md` Phase
B). ~~**End-to-end classifier-based R@K/mR@K is not yet trustworthy** —
confounded by the predicate-vocabulary bug (§11).~~ **Corrected**: the
classifier path is untrustworthy *for this checkpoint* because its
`predicate_classifier` head is untrained — **not** because of the
vocabulary bug, which is fixed and was never the cause (§11).

**No number produced to date is a baseline.** The complete list of
measurements against this checkpoint, so nothing gets promoted by
accident:

| Run | n | score path | calibration | R@50 | mR@50 | Status |
|---|---:|---|---|---:|---:|---|
| `runs/historical_checkpoint_diagnostic/` | 16 | `classifier` (untrained head) | none | 1.41 % | 0.94 % | diagnostic, misconfigured |
| `runs/text_path_gate/gate_10.json` | 10 | `ensemble` α=0.0 (pure text) | **none** | 19.71 % | 7.94 % | diagnostic, sample far too small |
| `runs/text_path_gate/gate_50.json` | 50 | `ensemble` α=0.0 (pure text) | **none** | 25.26 % | 5.84 % | diagnostic, sample far too small |
| `demo_config.env` self-report | ? | `ensemble` α=0.0 | freq prior α=3.75 | 67.09 % | 22.64 % | `HISTORICAL EVIDENCE`, unreproduced |

Every row above is explicitly labeled NOT A RESEARCH BASELINE in its own
run directory's `README.md`. The historical row is **single-source and
unreproduced**; its split and its pooled-vs-image-mean aggregation are both
`UNKNOWN`.

**Runtime** — ~~real, actively-measured cost is ~11s/image marginal + ~85s
fixed setup (`runs/runtime_benchmark/`)~~ **RETRACTED**. That figure came
from a 2-point (1-vs-2-image) linear fit, which is unsound: per-image cost
scales with *object count* (per-object CLIP classification with prompt
ensembling), and the 2-image sample was object-light.

**Current measured figure** (`VERIFIED FROM EXPERIMENT`,
`runs/text_path_gate/`, staged 10→50 images, actively monitored, no host
sleep): **~36 s/image marginal, CPU-only, this machine.**

| Stage | images | `eval_sgg_standard` | s/image |
|---|---:|---:|---:|
| 1 | 10 | 687 s | 68.7 (cold CLIP-object cache) |
| 2 | 50 | 2,121 s | 42.4 |
| marginal 10→50 | 40 | 1,434 s | **35.8** |

Extrapolated CPU cost: 200 images ≈ 2 h, 1,000 ≈ 10 h, the full
10,401-image validation split ≈ **104 h (~4.3 days)**. **A GPU is required
for any full-split run** — this is the single hardest constraint driving
the GCP decision. The separately-retracted 22,021.6s/16-image figure (host
slept mid-run) remains retracted.

## 11. Known blockers

**RESOLVED since this section was first written:**

- ~~Predicate-vocabulary index mismatch~~ — **fixed** (`9dc8f45d`,
  `7d91af49`). Also **retracted**: the claim that it caused Experiment A's
  0.94% collapse. It did not — `on`/`of`/`behind`/`flying in` were at
  identical indices in both orderings. See the retraction banners in
  `docs/PREDICATE_VOCAB_INDEX_BUG_TRIAGE.md` and
  `docs/HISTORICAL_CHECKPOINT_DIAGNOSTIC.md`.
- ~~`configs/predicate_metadata_vg150.json` drift~~ — **fixed**
  (`fa8c0c3b`), diagnostic-only impact.

**The real explanation of the 0.94% result** (`VERIFIED`): a **scoring-path
mismatch**. The historical run used `EVAL_SCORE_MODE=ensemble` with
`EVAL_ENSEMBLE_ALPHA=0.0` = **100% CLIP text-cosine, 0% classifier**.
Experiment A forced `score_mode="classifier"`, and this checkpoint's
`predicate_classifier` head is at/near random initialization (row-norms
mean 0.5752 vs 0.5796 for fresh `nn.Linear(768,51)`; biases collapsed to
absmean 0.0012 vs 0.0180, under `lr=2e-06`). Neither the model, the
dataset, nor the evaluator was at fault.

**Open blockers:**

1. ~~**The text path has not yet been measured** — no run has used the
   historical scoring configuration.~~ **RETRACTED — the raw text path
   HAS been measured.** `runs/text_path_gate/` (commit `439d1319`,
   CPU, `datasets_vg150_clean` validation split) ran
   `score_mode=ensemble` / `ensemble_alpha=0.0` / `use_gt_pairs=true` at
   10 and 50 images, and *proved* the text path was actually taken rather
   than assuming it: `_relation_predicate_logits` was wrapped and compared
   against a pure text-only reference, giving
   `max_abs_diff_vs_text_only = 0.0` and `classifier_used = false` at both
   stages. `VERIFIED FROM EXPERIMENT`.

   | Metric | 10 img | 50 img |
   |---|---:|---:|
   | R@50 | 19.71 % | **25.26 %** |
   | mR@50 | 7.94 % | **5.84 %** |
   | `image_mean_R@50` | 23.30 % | 22.55 % |
   | head / body / tail mR@50 | 37.73 / 0 / 0 | 29.22 / 0 / 0 |
   | `gt_pair_recall@32` | 1.0 | 1.0 |

   Versus the classifier path: R@50 **1.41 % → 25.26 %**, mR@50
   **0.94 % → 5.84 %**. This corroborates the scoring-path diagnosis above
   from a second direction.

2. **What remains genuinely unmeasured — this is the real open blocker.**
   The text-path gate deliberately ran **raw and uncalibrated**
   (`freq_bias_enabled=false`), but `demo_config.env` records
   `FREQ_BIAS_ALPHA=3.75` and `MODEL_NOTE=... calibrated with frequency
   prior alpha 3.75`. **No run has yet used the full historical
   configuration** (text path **+** pair-conditioned frequency prior at
   α=3.75), on any sample size. The gate's mR@50 = 5.84 % is therefore
   **not comparable** to the historical mR@50 = 22.64 %, and neither
   number is a baseline. Closing this gap is exactly what the GCP run
   exists to do — see `docs/GCP_EXPERIMENT_PROTOCOL.md`.

   `INFERENCE`, not established: the gate's predicate distribution
   collapsed onto two classes (`in` 334 + `has` 201 = 93 % of predictions
   at 50 images) while GT is dominated by `on`/`has`/`in`/`wearing` —
   consistent with an uncalibrated CLIP text-cosine scorer, and precisely
   the failure mode a pair-conditioned frequency prior exists to correct.
   Untested; do **not** treat as established.

3. **Do not use the classifier path with this checkpoint** for any research
   claim; it measures untrained weights. `RECOMMENDATION`.
4. **Sample size.** Both measured points (10, 50 images) are far too small
   for a research claim. The full split is 10,401 images.
5. Everything in `docs/known_issues.md`'s P1/P2 register (role-swap
   fabricated split, SGDet box-preprocessing silent fallback, the
   frequency-prior silent-fallback gap, `use_all_pairs` script-vs-default
   drift, etc.) — unchanged, still open.

## 12. Known research issues

Deferred entirely this phase per explicit instruction (no architecture,
loss, or metric changes). The prior research-bottleneck ranking (pair
proposal / predicate classification / calibration) remains an untested
hypothesis set — see §2.

## 13. Reproducibility status

- **Code**: fully reproducible from git history; every fix/finding this
  engagement has an exact commit hash or is explicitly marked uncommitted.
- **Dataset**: `datasets_vg150_clean/` is reproducible from
  `datasets/vg_raw/` via `tools/prepare_vg150_drive_clean.py` — but that
  raw source itself is **not** reproducible from this repo alone (no
  download URL verified working this engagement; see `data/README.md`).
- **Checkpoint**: **not** reproducible — the original training run's exact
  code commit is bounded, not pinned; its dataset preparation, alias-map
  version, and (per §11) predicate vocabulary are partially or fully
  unrecoverable. Treat the recovered `.pt` file as irreplaceable historical
  evidence, not a reproducible artifact.
- **Historical numbers** (mR@50≈22.64%, and everything in
  `notes/current_status.tex`): **not independently reproduced this
  engagement.** Marked `HISTORICAL EVIDENCE` everywhere they're cited.

## 14. Next recommended experiments (ranked, none executed this phase)

Items 1 and 2 of the earlier revision's list are **DONE** and are struck
below rather than deleted, so the reasoning chain stays legible:

1. ~~**Corrected-vocabulary micro-verification**~~ — **DONE**. The
   vocabulary fix landed (`9dc8f45d`, `7d91af49`) and
   `runs/text_path_gate/` re-ran against it.
2. ~~Scale to a larger CPU-tractable smoke sample~~ — **DONE** at n=50
   (`runs/text_path_gate/gate_50.json`). Outcome: the raw *text* path is
   healthy (R@50 25.26 %) but uncalibrated mR@50 is poor (5.84 %,
   body/tail both 0).

**Remaining, ranked:**

1. **Historical-configuration canary** — the full historical protocol
   (text path **+** frequency prior α=3.75) on **2 batches**, on GPU.
   Cheap, fast, and the only outstanding *configuration* question. Gated by
   `scripts/eval/eval_historical_checkpoint.sh` +
   `tools/verify_canary.py`. **This is the immediate next action.**
2. **Full-split historical reproduction** — same protocol, all 10,401
   validation images, on GPU. Produces the first number that could
   legitimately be compared against the historical 67.09 / 22.64. Must not
   start until the canary passes.
3. Only after #2: decide whether the historical result **reproduces**,
   **partially reproduces**, or **fails to reproduce** — and record that
   verdict here. Each outcome implies a different research path; do not
   pre-commit to one.
4. Only after #3: resume the research-bottleneck analysis with real,
   current-code measurements instead of the prior phase's untested
   hypothesis ranking.

## 15. What must NOT be changed casually

- `checkpoints/demo_best/*` — never modify, convert, or delete. Historical
  evidence, not a working artifact.
- ~~`datasets_vg150_clean/vocabulary/predicates.json` — confirmed wrong
  (§11), but **do not patch it outside a dedicated, tested fix phase**~~
  **SUPERSEDED — the fix landed.** The dedicated, tested fix phase this
  entry asked for happened (`9dc8f45d`, `7d91af49`, 11 regression tests in
  `tests/test_predicate_vocab.py`). The on-disk file is now canonical:
  re-verified this phase that its `idx_to_predicate[1..50]` equals
  `sorted(STANDARD_VG150_PREDICATES)` exactly, and equals the recovered
  `frequency_prior.json`'s own `predicate_vocab` exactly
  (`VERIFIED FROM DATA`). It is now hash-pinned
  (`e4e88e87…d4dc5`) and checked by `tools/gcp_preflight.py`. **What must
  not change now is the opposite of the original warning**: do not
  regenerate this file with a different ordering, or the recovered
  checkpoint's index mapping and the frequency prior both break.
- **The historical checkpoint's compatibility overrides** — every flag
  listed in `docs/HISTORICAL_CHECKPOINT_MANIFEST.md` must be passed
  explicitly on every run against that checkpoint. `--stage 3` alone
  silently forces `relationness_enabled`,
  `text_conditioned_projection_enabled`, `eval_sgg_use_relationness` and
  `eval_sgg_use_object_uncertainty` to `True`, none of which this
  checkpoint has trained weights for (`VERIFIED FROM CODE`,
  `config.py:741-747`). Use `scripts/eval/eval_historical_checkpoint.sh`,
  which sets all of them explicitly — never `eval_l4_phase34.sh`.
- The GT-extraction fix (`220c5c2e`) and its 7 regression tests — do not
  revert or weaken; directly verified against real data this phase.
- `openvocab_rel/**` model/loss/eval-metric logic — frozen for this entire
  validity/infrastructure line of work; any change here is a Category-C
  research decision requiring its own explicit phase.

## Status table

| Area | Status | Confidence | Notes |
|---|---|---|---|
| Code imports | Clean | `VERIFIED FROM TEST` | `import openvocab_rel` + full suite pass |
| Tests | **184 passing** | `VERIFIED FROM TEST` | 97 at `140e163f` (was documented as 83) + 87 infrastructure tests this pass |
| Dataset preparation | Clean, validated | `VERIFIED FROM DATA` | 0 integrity issues; predicate vocab file **now canonical and hash-pinned** |
| Checkpoint | Loads, compatibility understood | `VERIFIED FROM EXPERIMENT` | 59/0 missing/unexpected, all traced safe |
| Checkpoint integrity | SHA256-pinned | `VERIFIED FROM DATA` | `8845c3af…ad442`, re-verified at this HEAD |
| Frequency prior | SHA256-pinned + structurally validated | `VERIFIED FROM DATA` | `144d9f92…c6e6a`; 50-predicate vocab == canonical; all arrays length-50 |
| GT extraction | Fixed and verified on real data | `VERIFIED FROM EXPERIMENT` | 100% pair recovery observed post-fix |
| Predicate vocabulary | **Fixed** (`9dc8f45d`, `7d91af49`) | `VERIFIED FROM TEST` | canonical vocab always regenerated; was *not* the cause of the 0.94% collapse |
| Predicate metadata | **Fixed** (`fa8c0c3b`) | `VERIFIED FROM TEST` | `wrapped around` entry added; diagnostic-only impact |
| Classifier scoring head | **Untrained in the recovered checkpoint** | `VERIFIED FROM CHECKPOINT` | weight norms match fresh init; use the **text path** for this checkpoint |
| Raw text scoring path | **Measured** (n=10, n=50) | `VERIFIED FROM EXPERIMENT` | R@50 25.26 % @ n=50, text path *proven* taken; sample far too small for a claim |
| Full historical protocol (text + α=3.75 prior) | **NOT measured at any n** | `UNKNOWN` | the single outstanding configuration question; what the GCP run exists to answer |
| Evaluation baseline | **None exists yet** | — | do not cite any number produced so far as a baseline |
| GCP preflight | Implemented + tested | `VERIFIED FROM TEST` | `tools/gcp_preflight.py` |
| Canary workflow | Implemented + tested | `VERIFIED FROM TEST` | `scripts/eval/eval_historical_checkpoint.sh` + `tools/verify_canary.py`; **not yet executed on GPU** |
| Training reproducibility | Partial | `INFERENCE`/`UNKNOWN` | code yes, checkpoint/dataset/vocab provenance no |

### What is verified vs. inferred vs. unknown — explicit split

**`VERIFIED`** (reproducible from this checkout or from a hash check):
the three fixes and their 97 tests; all six artifact hashes; the
frequency prior's structure; the checkpoint's 59/0 key delta; the
classifier head being at/near random init; the text path being the one
actually taken in `runs/text_path_gate/`; ~36 s/image CPU cost; that
`--stage 3` forces four architecture flags on.

**`INFERENCE`** (reasoned, not proved): the training commit point estimate
`2b7ba82f` (the *bounded range* `[5ed4e429, 215fd6d0)` is verified); that
`frequency_prior.json` is train-derived; that the checkpoint file was
manually renamed into `demo_best/`; that the gate's two-class predicate
collapse is a calibration artifact.

**`UNKNOWN`**: whether the historical 22.64 % used pooled or image-mean
aggregation; which split it was measured on; how many images it covered;
whether it reproduces at all. **The GCP run resolves the last of these
and only the last** — the first three may remain permanently unknown, and
a mismatch on any of them is a legitimate explanation for a failed
reproduction that must not be mistaken for a model defect.
