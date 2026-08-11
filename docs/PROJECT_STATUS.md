# Project status

**Canonical current-state document.** When this document and any other
document disagree about current state, trust this one and fix the other —
but never silently rewrite a historical claim to make it agree; mark the
older document superseded instead, with a pointer here.

Evidence labels used throughout: `VERIFIED FROM CODE`, `VERIFIED FROM
EXPERIMENT`, `HISTORICAL EVIDENCE` (from a recovered artifact, not
independently reproduced), `INFERENCE`, `RECOMMENDATION`, `UNKNOWN`.

Last updated: this phase (post GT-extraction fix `220c5c2e`, post
historical-checkpoint recovery and forensic diagnosis, pre any
vocabulary fix).

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
`RECOMMENDATION`, not yet `VERIFIED FROM EXPERIMENT`: the vocabulary fix
(§11) must land and be verified before that bottleneck analysis can be
meaningfully re-attempted.

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
configs/                            predicate_metadata_vg150.json (has a known drift, §11), presets.yaml (documentation only, not loaded)
scripts/{train,eval,notebooks}/     entrypoints
tools/                               dataset prep, validation, diagnostics
tests/                               83 passing (§7)
docs/
    architecture/                    overview/training/evaluation/data_flow
    reproducibility.md, known_issues.md   living registers
    PROJECT_STATUS.md                this file
    GT_EXTRACTION_BUG_TRIAGE.md, PREDICATE_VOCAB_INDEX_BUG_TRIAGE.md,
    PREDICATE_VOCAB_HISTORICAL_FORENSICS.md, HISTORICAL_CHECKPOINT_DIAGNOSTIC.md,
    HISTORICAL_CHECKPOINT_PROVENANCE.md    audit/diagnostic reports (see §15 note)
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

**83 tests passing** (`py -3 -m pytest -q`), `VERIFIED FROM TEST`, most
recently confirmed this engagement. 7 of the 83 are the GT-extraction
regression suite (`tests/test_eval_gt_extraction.py`, commit `220c5c2e`).
No test currently exercises the predicate-vocabulary index-mapping bug
(§11) — a real coverage gap; a regression test for this should accompany
whatever fix is eventually chosen.

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

## 10. Current evaluation status

The GT-extraction fix is `VERIFIED FROM EXPERIMENT` on real data: a raw
16-image run against the recovered checkpoint showed **100% pair-level GT
recovery** at every K, for every predicate including the dominant class
(`pair_proposal_diag`, `docs/HISTORICAL_CHECKPOINT_DIAGNOSTIC.md` Phase
B). **End-to-end classifier-based R@K/mR@K is not yet trustworthy** —
confounded by the predicate-vocabulary bug (§11). No number produced this
engagement (R@50=1.41%, mR@50=0.94%, n=16) should be cited as a baseline —
explicitly labeled `historical checkpoint smoke-test diagnostic — NOT A
RESEARCH BASELINE` in `runs/historical_checkpoint_diagnostic/README.md`.

**Runtime**: real, actively-measured cost is ~11s/image marginal + ~85s
fixed setup per `eval_sgg_standard` call, CPU-only, this machine
(`runs/runtime_benchmark/`, `VERIFIED FROM EXPERIMENT`). The earlier
22,021.6s/16-image figure was an artifact of the host sleeping mid-run and
has been retracted as a throughput estimate.

## 11. Known blockers

1. **Predicate-vocabulary index mismatch** (P0, `docs/PREDICATE_VOCAB_INDEX_BUG_TRIAGE.md`,
   `docs/PREDICATE_VOCAB_HISTORICAL_FORENSICS.md`) — blocks any trustworthy
   classifier-based R@K/mR@K measurement. Not fixed. Two candidate fixes
   identified, neither chosen.
2. **`configs/predicate_metadata_vg150.json` drift** (P2, non-crashing,
   `docs/known_issues.md`) — related, smaller, likely worth fixing
   alongside #1.
3. Everything in `docs/known_issues.md`'s P1/P2 register (role-swap
   fabricated split, SGDet box-preprocessing silent fallback, `use_all_pairs`
   script-vs-default drift, etc.) — unchanged this phase, still open.

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

1. **Corrected-vocabulary micro-verification** — regenerate (or override)
   the predicate vocabulary to match `sorted(STANDARD_VG150_PREDICATES)`
   (directly corroborated by `frequency_prior.json`, §11) and re-run the
   *exact same* 1-2 image benchmark shape already proven fast and reliable
   (`runs/runtime_benchmark/`). Cheapest possible test of the dominant
   open hypothesis. Not yet authorized/run.
2. Scale to a larger (but still CPU-tractable, actively-monitored — not
   unattended-overnight) PredCls+GT-pairs smoke sample once #1 resolves,
   to get a real, trustworthy raw baseline number.
3. Only after a trustworthy raw baseline exists: attempt to reproduce
   (not assume) the historical ensemble+calibrated 22.64% configuration.
4. Only after both of the above: resume the research-bottleneck analysis
   with real, current-code measurements instead of the prior phase's
   untested hypothesis ranking.

## 15. What must NOT be changed casually

- `checkpoints/demo_best/*` — never modify, convert, or delete. Historical
  evidence, not a working artifact.
- `datasets_vg150_clean/vocabulary/predicates.json` — confirmed wrong
  (§11), but **do not patch it outside a dedicated, tested fix phase**;
  it's shared infrastructure and any fix needs its own regression test
  proving old-broken/new-correct, matching this project's established
  fix discipline.
- The GT-extraction fix (`220c5c2e`) and its 7 regression tests — do not
  revert or weaken; directly verified against real data this phase.
- `openvocab_rel/**` model/loss/eval-metric logic — frozen for this entire
  validity/infrastructure line of work; any change here is a Category-C
  research decision requiring its own explicit phase.

## Status table

| Area | Status | Confidence | Notes |
|---|---|---|---|
| Code imports | Clean | `VERIFIED FROM TEST` | `import openvocab_rel` + full suite pass |
| Tests | 83 passing | `VERIFIED FROM TEST` | includes 7 GT-extraction regression tests |
| Dataset preparation | Clean, validated | `VERIFIED FROM DATA` | 0 integrity issues found independently; predicate vocab *content* filtering is separately correct, only the *index* file is wrong (§11) |
| Checkpoint | Loads, compatibility understood | `VERIFIED FROM EXPERIMENT` | 59/0 missing/unexpected, all traced safe |
| GT extraction | Fixed and verified on real data | `VERIFIED FROM EXPERIMENT` | 100% pair recovery observed post-fix |
| Predicate vocabulary | **Confirmed wrong, not fixed** | `HIGH` (not `VERIFIED` — one open caveat, §11) | blocks trustworthy R@K/mR@K |
| Evaluation baseline | **None exists yet** | — | do not cite any number produced so far as a baseline |
| Training reproducibility | Partial | `INFERENCE`/`UNKNOWN` | code yes, checkpoint/dataset/vocab provenance no |
