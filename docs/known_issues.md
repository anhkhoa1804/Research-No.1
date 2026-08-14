# Known issues

This is the durable register for Phase 0/10 of the infrastructure cleanup:
issues identified while cleaning up the repository that were **deliberately
left unfixed**, so cleanup, bug fixes, and research changes stay in separate,
traceable commits. Nothing on this page has been changed by the cleanup
pass. Fix these in a dedicated, controlled bug-fix phase, one at a time,
each with its own test.

Severity: **P0** = could invalidate reported results, **P1** = serious
correctness/behavior gap, **P2** = moderate (drift, missing coverage, dead
weight), **P3** = minor/cosmetic.

## P0 — Corrects evaluation ground truth or classifier semantics

### GT-triplet extraction index misalignment in the default SGG eval path — FIXED
- **File/function:** `openvocab_rel/evals.py`, `eval_sgg_standard`'s
  per-image prep loop + `_collect_gt_triplets`.
- **What happened:** under the default `eval_sgg_use_gt_pairs=False`, the
  eval loop overwrote `ex_eval["pairs"]` with a freshly built,
  differently-ordered candidate list, while leaving `rel_preds`/
  `rel_pos_mask` in their original ordering. `_collect_gt_triplets` zipped
  the two positionally, silently misattributing GT predicates to the wrong
  (subject, object) pair for any image with more than one relationship —
  corrupting R@K/mR@K for all six SGG tasks plus prop@K and role-swap
  diagnostics. **Not specific to any one checkpoint or dataset** — present
  since the repository's first commit (`31e89601`).
- **Fix:** commit `220c5c2e3ee8eb40f5cc8fcd0d46f376ca4ae4a8`. Snapshots the
  original `pairs`/`rel_preds`/`rel_pos_mask` under new `_gt_pairs`/
  `_gt_preds`/`_gt_pos_mask` keys before the overwrite;
  `_collect_gt_triplets` reads those instead. Verified: 7 new regression
  tests (`tests/test_eval_gt_extraction.py`), 4/7 failing pre-fix, 7/7
  passing post-fix; re-verified live on real data during the historical
  checkpoint evaluation (`docs/HISTORICAL_CHECKPOINT_DIAGNOSTIC.md` Phase B:
  100% pair-level GT recovery observed post-fix).
- **Full root-cause writeup:** `docs/GT_EXTRACTION_BUG_TRIAGE.md`.

### Predicate-vocabulary index mismatch in prepared VG150-clean datasets — FIXED (`9dc8f45d`, `7d91af49`); causal claim below RETRACTED

> **Correction.** The entry below states this bug was "observed directly" as
> the cause of Experiment A's collapse. **That is retracted** (`VERIFIED`):
> `on` (idx 30), `of` (29), `behind` (7) and `flying in` (14) were at
> *identical* indices in both the broken and canonical orderings, so the bug
> cannot explain the observed symptom. The real cause was a **scoring-path
> mismatch** — the run used the untrained `classifier` head instead of the
> historical `ensemble`/`alpha=0.0` (pure CLIP text-cosine) path. The
> vocabulary bug was nonetheless real (24/50 indices shifted) and is now
> fixed: `_copy_vocab` always regenerates the canonical vocabulary, with an
> opt-in `--allow_source_vocab_mismatch` for raw archives. Regression tests:
> `tests/test_predicate_vocab.py`. See `docs/PREDICATE_VOCAB_INDEX_BUG_TRIAGE.md`.

### (original entry, retained)
- **File:** `datasets_vg150_clean/vocabulary/predicates.json` (and its
  source, `datasets/vg_raw/vocabulary/predicates.json` — not created by
  any tool in this repo; a pre-existing raw data file), consumed by
  `openvocab_rel/datasets/vg150_loader.py:_load_vg150_vocab`/
  `scan_vg150_predicate_vocab`.
- **What happens:** this vocabulary file's 50-predicate set diverges from
  `STANDARD_VG150_PREDICATES` (`tools/prepare_vg150_subset.py`) — the set
  actually used one function away, in the same prep tool, to *filter*
  which relationships survive into the cleaned data. The vocab file
  includes `"growing on"`/`"says"` (**zero** occurrences in the actual
  cleaned data) and excludes `"next to"`/`"wrapped around"` (**23,888**
  combined occurrences in `train.jsonl`). Because index assignment is
  positional, this shifts the classifier-index-to-label mapping for a
  large contiguous range of the vocabulary. Directly corroborated by a
  second, independently recovered historical artifact
  (`checkpoints/demo_best/frequency_prior.json`'s `predicate_vocab` field
  matches `sorted(STANDARD_VG150_PREDICATES)` exactly, not the current
  vocab file) — see `docs/PREDICATE_VOCAB_HISTORICAL_FORENSICS.md`.
- **Why it matters:** corrupts classifier-based predicate scoring
  (`eval_sgg_predicate_score_mode in {"classifier","ensemble"}`, this
  repo's default combination) for any evaluation run against a
  `vg150_root` carrying this vocabulary file — observed directly:
  Experiment A's raw-classifier run against the recovered historical
  checkpoint showed the model's most-predicted class as a rare tail
  predicate while the actual majority class ("on", 39% of the sample)
  never appeared in its top-8 predictions at all
  (`docs/HISTORICAL_CHECKPOINT_DIAGNOSTIC.md`). **Not proven to be the
  complete explanation** — an honest, unresolved caveat remains (see that
  document's Phase I) that could equally reflect the checkpoint's raw,
  uncalibrated, long-tail-reweighted training objective rather than pure
  index scrambling; only a corrected-vocab re-test can fully separate the
  two.
- **Status:** ~~confirmed, **not fixed** (explicitly out of scope for the
  diagnostic phase that found it — do not modify `predicates.json` without
  a dedicated, tested fix phase). Two candidate fixes identified but not
  chosen: regenerate the vocab file from `STANDARD_VG150_PREDICATES` inside
  `tools/prepare_vg150_drive_clean.py`'s `_copy_vocab` step, or validate
  and fail loudly on mismatch at prepare-time. `configs/predicate_metadata_vg150.json`
  has a related but distinct, non-crashing drift (52 keys, 3 extra/1
  missing relative to the canonical 50) — flagged in an earlier phase,
  still unfixed, likely worth resolving in the same pass as this one.~~

  **SUPERSEDED — both are fixed.** The line above was accurate when
  written and is struck rather than deleted so the decision trail stays
  readable. The dedicated, tested fix phase it asked for happened, and it
  chose the *first* of the two candidate fixes:
  - `9dc8f45d` — `_copy_vocab` now always regenerates the canonical
    vocabulary from `STANDARD_VG150_PREDICATES`.
  - `7d91af49` — adds an opt-in `--allow_source_vocab_mismatch` escape
    hatch for raw archives, so the regeneration fails loudly by default
    rather than silently accepting a divergent source.
  - `fa8c0c3b` — `configs/predicate_metadata_vg150.json`'s drift fixed
    (`wrapped around` entry added).
  - Regression tests: `tests/test_predicate_vocab.py` (11),
    `tests/test_predicate_metadata_coverage.py` (3).

  **Re-verified at HEAD `140e163f`** (`VERIFIED FROM DATA`): the on-disk
  `datasets_vg150_clean/vocabulary/predicates.json` `idx_to_predicate[1..50]`
  equals `sorted(STANDARD_VG150_PREDICATES)` exactly, **and** equals the
  independently recovered `frequency_prior.json`'s `predicate_vocab`
  exactly. It is now hash-pinned at `e4e88e87…d4dc5` and enforced by
  `tools/gcp_preflight.py`.
- **Full writeup:** `docs/PREDICATE_VOCAB_INDEX_BUG_TRIAGE.md`,
  `docs/PREDICATE_VOCAB_HISTORICAL_FORENSICS.md`.

## P1 — Dangerous silent fallbacks

### `eval_swap_consistency` reports a fabricated symmetric/asymmetric split — NOT FIXED (out of scope this pass)
- **File/function:** `openvocab_rel/evals.py`, `eval_swap_consistency`
- **What happens:** the function returns `sym_cos`, `asym_cos`, `n_sym`,
  `n_asym` as if it filtered role-swap cosine similarity by predicate
  symmetry, but no such filter exists anywhere in the function body — all
  four fields are populated with the same aggregate value/count computed
  over every predicate, symmetric or not.
- **Why it matters:** any report or downstream tool that reads this
  function's output believing it separates symmetric- from
  asymmetric-predicate consistency is reading a plausible-looking but wrong
  number. It is a standalone diagnostic, not in the R@K/mR@K path.
- **Status:** CONFIRMED (unchanged since the validity investigation), left
  unfixed — outside the explicit confirmed-fix list for the current
  validity-fix pass. The correct pattern (`is_symmetric(pred_metadata,
  pred_name)`) already exists and is used correctly elsewhere in the same
  file, inside `eval_sgg_standard`'s own role-swap diagnostic
  (`role_swap_diag`) — use that mechanism for any real directionality
  claim in the meantime.
- **How to test later:** unit test asserting `n_sym + n_asym == n` on a
  synthetic input with a known mix of symmetric/asymmetric predicates, and
  that `sym_cos != asym_cos` when the input is constructed so they should
  differ.

### Conditional eval-label leak in the frequency-prior helper — FIXED
- **File/function:** `openvocab_rel/evals.py`, `_predicate_log_prior_for_eval`
- **What happened:** the function correctly tried `train.jsonl` under
  `cfg.vg150_root` first. If that file was missing or failed to parse, it
  **silently** fell back to counting predicate frequencies from the loader
  object actually passed into `eval_sgg_standard` — i.e., potentially the
  validation/test split currently being scored — with no warning printed.
  Empirically confirmed via `git show`-extracted pre-fix function + a
  synthetic reproduction (see the fix commit message).
- **Fix:** the eval-loader fallback is removed entirely; missing/
  unreadable/empty train-split statistics now raise
  `MissingTrainStatisticsError` instead. A new `_calibration_prior_is_needed`
  gate means eval runs with calibration genuinely disabled never require
  `train.jsonl` to exist at all. See `tests/test_calibration_prior.py` for
  the 4-invariant regression suite (A/B/C/D per the validity-fix phase).

### Frequency-prior loading fails **silently**, producing an uncalibrated run that looks calibrated — NOT FIXED (newly found, pre-GCP stabilization pass)
- **File/function:** `openvocab_rel/evals.py`, `_load_frequency_bias`
  (lines ~1008-1063).
- **What happens:** the function has **six** separate `return None` exits
  and **none of them warns**:
  1. `freq_bias_path` is `""` or the file does not exist (line ~1012)
  2. the file exists but `json.load` raises — bare `except Exception`
     (line ~1017)
  3. `predicate_vocab` is empty (line ~1021)
  4. `global_log_probs` is not a list, or its length ≠ `len(source_vocab)`
     (via `remap`, line ~1027)

  A `None` return means "no frequency bias applied". The caller does not
  distinguish "calibration deliberately off" from "calibration requested
  but the artifact was unusable".
- **Why it matters — this is the highest-risk item for the GCP run.** The
  historical reproduction is defined by
  `--freq_bias_enabled true --freq_bias_path checkpoints/demo_best/frequency_prior.json
  --freq_bias_alpha 3.75`. If that ~97 MiB file fails to transfer to the
  GCP instance, or is truncated mid-copy, or lands at a different path,
  the evaluation **still runs to completion and still emits a full
  `metrics.jsonl`** — just silently uncalibrated. Given that the raw
  uncalibrated text path is already known to score mR@50 ≈ 5.84 %
  (`runs/text_path_gate/`) versus the historical 22.64 %, this failure
  mode would manifest as an apparently-clean "failed to reproduce"
  result. **That is a P1 scientific-validity hazard, not a convenience
  issue.**
- **Status: NOT FIXED — deliberately.** Making this fail loud means
  changing `openvocab_rel/evals.py`, which is frozen for this
  infrastructure phase (`docs/PROJECT_STATUS.md` §15). Changing a silent
  `return None` into a raise *is* a behavior change in the failure path
  and belongs in its own tested fix phase.
- **Mitigated instead, outside the evaluator** (`VERIFIED FROM TEST`):
  1. `tools/gcp_preflight.py` verifies the prior's SHA256, that it parses,
     that `predicate_vocab` has exactly 50 canonical entries, and that
     `global_log_probs` and a sample of `pair`/`subject`/`object` rows are
     all length-50 — i.e. it independently checks every condition that
     would make `_load_frequency_bias` return `None`.
  2. `tests/test_gcp_preflight.py` pins those conditions against the real
     function's behavior so the two cannot drift apart.
  3. `scripts/eval/eval_historical_checkpoint.sh` refuses to launch if the
     prior file is absent.
- **How to fix properly later:** raise a dedicated
  `MissingFrequencyPriorError` (mirroring the existing
  `MissingTrainStatisticsError` pattern already used in the same file for
  the analogous train-statistics leak) whenever `freq_bias_enabled=True`
  and `freq_bias_alpha>0` but the prior cannot be loaded. Add a test
  asserting each of the four failure conditions raises rather than
  returning `None`.

### Unknown/misspelled CLI flags are silently swallowed — NOT FIXED (newly found, pre-GCP stabilization pass)
- **File/function:** `openvocab_rel/train.py`, `main()` line ~1252:
  `args, _unknown = parser.parse_known_args(argv)`.
- **What happens:** `parse_known_args` (as opposed to `parse_args`) routes
  any unrecognized `--flag` into `_unknown`, which is discarded with no
  warning. **Verified by execution this phase**: passing
  `--totally_bogus_flag true` produces no error and no output.
  Downstream, `called_args` (built by scanning `argv` for `--`-prefixed
  tokens) will contain the bogus name, but both the merge loop and
  `_reapply_explicit_cli_args` guard on `hasattr(args, key)`, so it has no
  effect at all.
- **Why it matters:** the safety of every run against the historical
  checkpoint rests entirely on explicitly passing ~10 compatibility flags
  to override `--stage 3`'s defaults. A single typo — `--explicit_spoa_enable`
  for `--explicit_spoa_enabled` — leaves stage 3's `True` in place, loads
  the checkpoint into an architecture it was never trained for, and
  **reports nothing**. There is no runtime signal distinguishing this from
  a correct run.
- **Status: NOT FIXED — deliberately.** Switching to `parse_args` would
  make every currently-working script that passes an extra flag start
  failing, a broad behavior change well outside this phase's scope.
- **Mitigated instead** (`VERIFIED FROM TEST`):
  `tests/test_historical_eval_protocol.py` parses
  `scripts/eval/eval_historical_checkpoint.sh`, extracts every `--flag` it
  passes, and asserts each one exists in `build_argparser()`'s namespace
  **and** is a real `TrainConfig` field. A typo in that script becomes a
  test failure at commit time instead of a silently-wrong GPU run.
- **How to fix properly later:** keep `parse_known_args` but log a loud
  warning listing `_unknown` whenever it is non-empty, or add a
  `--strict_args` flag that promotes it to an error.

### SGDet box-preprocessing failure silently uses un-preprocessed boxes — NOT FIXED (newly found this re-audit pass)
- **File/function:** `openvocab_rel/evals.py`, `_make_detected_example`
  (builds the pseudo-example SGDet scores against, from Grounding-DINO
  detections)
- **What happens:** `preprocess_boxes_to_clip224(image, det_boxes, ...)` is
  wrapped in a bare `try/except Exception: boxes_224 = det_boxes.clone()`
  — if resizing/cropping the detected boxes into CLIP's 224-input
  coordinate space fails for any reason, the function silently substitutes
  the **raw, un-preprocessed, wrong-coordinate-space** boxes instead, with
  no warning.
- **Why it matters:** unlike the diagnostic-only fallback below, this feeds
  directly into the SGDet forward pass — a triggered failure would silently
  corrupt real SGDet geometry/routing with no error surfaced, potentially
  producing wrong (not just missing) SGDet numbers.
- **Status:** newly confirmed during this phase's post-fix static re-audit
  (Phase 9). Not fixed here — out of the explicit confirmed-fix list for
  this validity-fix pass; flagged for a future controlled fix with its own
  test (construct an image that reliably fails `preprocess_boxes_to_clip224`
  and assert the SGDet path either raises or excludes that example, rather
  than silently scoring it in the wrong coordinate space).

### Routing-diagnostic box-preprocessing failure falls back to raw boxes — lower severity, NOT FIXED
- **File/function:** `openvocab_rel/evals.py`, `_routing_diag_update`
- **What happens:** same `preprocess_boxes_to_clip224` failure pattern as
  above, but this instance only feeds the deformable-routing attention
  diagnostic (`routing_diag`, printed/logged, not part of R@K/mR@K or the
  SGDet forward pass itself) — a triggered failure would produce a
  misleading diagnostic printout, not a corrupted evaluation number.
- **Status:** newly confirmed during this phase's post-fix static
  re-audit. Not fixed here (diagnostic-only, lower priority than the
  SGDet-path instance above).

## P1 — Architecture vs. stated claim

### Default pair fusion is symmetric, not directional
- **File:** `openvocab_rel/models/relational_model.py`,
  `ProgressiveRelationalDecoder._semantic_pair_feature`
- **What happens:** the active branch under the dataclass default
  (`asymmetric_pair_fusion_enabled=False`) is
  `F.normalize(sub_norm + obj_norm, ...)` — a symmetric, order-invariant
  combination. The README's architecture claim ("ordered relation
  embedding... explicit direction") is only fully true when
  `asymmetric_pair_fusion_enabled=True`, which is off by default.
- **Why it matters:** directionality under the default config depends
  entirely on the role-embedding/SPOA branches and geometry asymmetry, not
  this term — a real (if partial) gap between the stated design and the
  shipped default.
- **How to test later:** role-swap consistency metric (already implemented,
  `eval_sgg_role_swap_metric_enabled`) compared with the flag on vs. off, on
  a matched checkpoint.

## P2 — Config / script drift

### `--stage 3` unconditionally forces four architecture flags on, after any preset — NOT FIXED (documented; this is a research default)
- **File:** `openvocab_rel/config.py`, `apply_stage_config`, the
  unconditional `if int(cfg.stage) == 3:` block at lines ~741-747 (note:
  this is a *second*, separate block that runs after the main
  `elif stage == 3` branch).
- **What happens** (`VERIFIED FROM CODE` + verified by execution):
  constructing a stage-3 config forces
  `text_conditioned_projection_enabled=True`, `relationness_enabled=True`,
  `eval_sgg_use_relationness=True`, `eval_sgg_use_object_uncertainty=True`.
  Combined with dataclass/branch defaults, a bare `--stage 3` resolves to:

  | Flag | stage-3 resolved value | Safe for the historical checkpoint? |
  |---|---|---|
  | `explicit_spoa_enabled` | `True` | **NO** — checkpoint predates SPOA |
  | `text_conditioned_projection_enabled` | `True` | **NO** — untrained |
  | `relationness_enabled` | `True` | **NO** — untrained/random |
  | `eval_sgg_use_relationness` | `True` | **NO** — would prune on random scores |
  | `eval_sgg_use_object_uncertainty` | `True` | questionable |
  | `clip_input_res` | `448` | **NO** — checkpoint trained at 336 |
  | `eval_sgg_predicate_ensemble_alpha` | `0.45` | **NO** — historical is `0.0` |
  | `eval_sgg_use_gt_pairs` | `False` | **NO** — historical is `True` |
  | `eval_sgg_grounding_dino_enabled` | `True` | **NO** — pulls a detector needlessly |

- **Why it matters:** these are correct, deliberate defaults *for current
  Phase 3/4 work* and must not be changed. They are simply wrong for a
  checkpoint that predates the architecture. Explicit CLI flags do
  override them — `_reapply_explicit_cli_args` (`train.py:1326`) runs
  after both `apply_stage_config` and `apply_gpu_preset`, so the override
  order is sound (`VERIFIED FROM CODE`) — but **only if every single flag
  is passed**, and see the `parse_known_args` entry above for why a typo
  is undetectable.
- **Status: NOT FIXED and must not be "fixed".** Changing any of these
  defaults is a research change. The mitigation is a dedicated
  entrypoint, not a default change:
  `scripts/eval/eval_historical_checkpoint.sh` sets every row of the table
  explicitly and prints the resolved protocol before launching.
  `scripts/eval/eval_l4_phase34.sh` is deliberately left **untouched** so
  the current Phase 3/4 workflow is unaffected.

### Re-running a script silently overwrites the previous run's outputs — NOT FIXED for existing scripts
- **File:** `scripts/eval/eval_l4_phase34.sh:17` (`OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"`),
  and the same pattern in `scripts/train/train_l4_phase34.sh`.
- **What happens:** `RUN_NAME` defaults to a fixed string
  (`eval_l4_phase34`). Re-running without setting it overwrites
  `runs/eval_l4_phase34/metrics.jsonl` and `logs/eval_l4_phase34.log` in
  place. Prior results are lost with no prompt or backup.
- **Status:** not fixed for the existing scripts (changing their output
  paths would break the current workflow's expectations and any tooling
  pointed at those fixed paths). **The new
  `scripts/eval/eval_historical_checkpoint.sh` does not have this
  problem** — it refuses to start if its target run directory already
  exists, unless `ALLOW_OVERWRITE=1` is set explicitly.

### `eval_calibration_sweep_l4.sh` hardcodes the frequency-prior path to the wrong root
- **File:** `scripts/eval/eval_calibration_sweep_l4.sh:34` —
  `FREQ_BIAS_PATH="${FREQ_BIAS_PATH:-datasets/frequency_prior.json}"`.
- **What happens:** every other script resolves `DATA_ROOT` dynamically and
  prefers `datasets_vg150_clean` when it exists; this one hardcodes
  `datasets/`. It also does not define `DATA_ROOT` at all, so the
  `FREQ_BIAS_ENABLED=auto` resolution inside `eval_l4_phase34.sh` (which
  requires `-f "${FREQ_BIAS_PATH}"`) will resolve to `false` whenever the
  prior only exists under `datasets_vg150_clean/`.
- **Why it matters:** a calibration *sweep* that silently runs with
  calibration disabled produces a flat, meaningless sweep — every alpha
  gives the same answer — and nothing says so. Related to, but distinct
  from, the `_load_frequency_bias` P1 entry above.
- **Status:** not fixed (out of scope; touching it would alter an existing
  workflow). Pass `FREQ_BIAS_PATH=` explicitly when using this script.

### `use_all_pairs` dataclass default vs. shipped-script default
- `openvocab_rel/config.py`: dataclass default `True`.
- `scripts/train/train_l4_phase34.sh`, `scripts/train/run_pure_next.sh`:
  both explicitly pass `--use_all_pairs false`.
- Under the actually-shipped setting, only GT-positive pairs are used for
  CE supervision at train time (no explicit negative-pair supervision),
  while eval time (`eval_sgg_use_gt_pairs=False` default) scores all
  N·(N−1) pairs including true negatives — a train/eval protocol asymmetry.
- **Not fixed** — this is a training-protocol question (which candidate
  pair space to train on), not a code-correctness bug; changing it would
  change training data distribution, which is a research decision, not a
  validity fix. Left exactly as the shipped scripts configure it.

### `negative_pair_ratio` is a fully inert config flag under the shipped
### training script's own defaults — investigated, classified, now WARNS
- **Intent** (from `config.py:184`'s own field comment, present since the
  repo's first commit, `31e89601`, and never edited since): "0 keeps
  positives only; positive values sample negatives per positive" —
  `negative_pair_ratio` was designed to be *the* control for how many
  negative (non-relation) pairs get sampled per positive pair at train
  time, including expressing "positives only" via `0`.
- **Actual behavior**: `_build_relation_entries`
  (`openvocab_rel/datasets/vg150_loader.py`) uses a *different* field,
  `use_all_pairs`, as the real gate — `if not bool(use_all_pairs):` returns
  positive-only pairs immediately, before `negative_pair_ratio` is ever
  read. `negative_pair_ratio` only has any effect when `use_all_pairs=True`.
- **Classification: C — accidentally unreachable**, not obsolete, not an
  intentional positives-only design. Evidence from `git log
  -S"use_all_pairs" -- scripts/`: `--use_all_pairs "${USE_ALL_PAIRS:-false}"`
  and `--negative_pair_ratio "${NEGATIVE_PAIR_RATIO:-2.0}"` were added to
  `scripts/train/train_l4_phase34.sh` in the *same diff hunk* of the *same
  commit* (`96160957`, "Modify the codebase to make the research protocol
  clearer") — i.e. both flags were added together, with no comment or
  commit message acknowledging that one silently overrides the other. If
  positives-only training were actually intended, `config.py`'s own
  documented convention (`negative_pair_ratio=0`) would have been used
  instead of leaving a non-zero value that looks like it's doing something.
  The validation-split loader construction (`train.py:1483-1484`) has the
  same issue doubled: it hardcodes `negative_pair_ratio=-1.0` regardless of
  `cfg.negative_pair_ratio`, also inert under `use_all_pairs=False`.
- **Action taken (infrastructure, not a protocol change)**:
  `negative_pair_ratio_is_inert(use_all_pairs, negative_pair_ratio)`
  (`vg150_loader.py`) makes the condition explicit and testable
  (`tests/test_pair_construction.py`), and `VG150DataLoader.__init__` now
  emits a `logging.warning` once per loader construction whenever the
  configured `negative_pair_ratio` would silently do nothing — pointing at
  this entry. **Which pairs actually get constructed is unchanged** — this
  is a category-B (infrastructure/diagnostics) fix, not a category-C
  (research) change; per instruction, the shipped script's default was
  deliberately **not** flipped to activate negative sampling automatically.
- **If/when this should actually be activated**: the minimal code change
  would be to `_build_relation_entries`'s branch condition — gate
  positives-only behavior on `negative_pair_ratio == 0` (matching its own
  documented semantics) instead of on the separate `use_all_pairs` flag, or
  simply flip `scripts/train/train_l4_phase34.sh`'s `USE_ALL_PAIRS` default
  to `true`. Either is a training-data-distribution change and belongs in
  a research-improvement phase with its own matched-compute ablation (see
  the research bottleneck analysis), not bundled into this validity pass.

### `use_rfs` / `rfs_t` are a silent no-op under the maintained data path
- Repeat-factor sampling is implemented only in
  `openvocab_rel/datasets/vg150_loader.py:VG150LocalDataset._build_valid_indices`
  (the raw VG-SGG-format loader). `VG150JSONLDataset` — the loader every
  current script actually uses via `--vg150_source local-jsonl` — has no
  RFS logic at all. The config fields read as "on by default" but do
  nothing under the maintained path.
- **Action taken (infrastructure, same pattern as `negative_pair_ratio`
  above)**: `use_rfs_is_inert(source_mode, split, use_rfs)`
  (`vg150_loader.py`) makes the condition explicit and testable
  (`tests/test_pair_construction.py`), and `VG150DataLoader.__init__` now
  emits a `logging.warning` once per loader construction whenever
  `use_rfs=True` is configured under a backend where it does nothing.
  Sampling behavior itself is unchanged — implementing RFS for the JSONL
  backend, or removing the dead flag, is a follow-up decision (research/
  data-pipeline scope), not made here.

### `adaptive_calibration_enabled`'s true source of truth is the shipped
### scripts, not `config.py`'s stage-3 preset
- `TrainConfig`'s dataclass default is `False`. `apply_stage_config`'s
  stage-1 and stage-2 branches explicitly set it `True`; the **stage-3**
  branch does not touch it at all — it silently stays at the dataclass
  default (`False`) unless something else sets it. That "something else"
  is `scripts/train/train_l4_phase34.sh:57` and
  `scripts/eval/eval_l4_phase34.sh:65`, both of which pass
  `--adaptive_calibration_enabled true` explicitly.
- Not a bug in the sense of producing wrong behavior under the documented
  workflow (both scripts agree, and CLI flags always win over stage
  presets) — but a real discoverability gap: reading `config.py` in
  isolation, a stage-3 run looks uncalibrated by default. Anyone
  constructing a stage-3 `TrainConfig` a different way (a notebook, a new
  script) would silently get `adaptive_calibration_enabled=False` and not
  notice.
- **Action taken (documentation only)**: a code comment was added at the
  stage-3 branch in `apply_stage_config` (`config.py`) pointing at exactly
  this. No default or behavior changed.

### `configs/presets.yaml` is documentation only, never loaded
- See the header comment added to that file in this cleanup pass. Real
  `--gpu_preset`/`--stage`/`--a100_ddp_preset` behavior comes from hardcoded
  Python in `openvocab_rel/config.py`; nothing enforces the YAML stays in
  sync with it. Deciding whether to wire it up for real or delete it is a
  follow-up decision, not made in this cleanup pass.

### Config / script drift — summary table

| Setting | Config default | Script override | Effective behavior | Intended behavior | Classification | Action |
|---|---|---|---|---|---|---|
| `use_all_pairs` | `True` | `--use_all_pairs false` (train scripts) | `False` under shipped training | Uncertain (no doc states either way) | **SCRIPT BUG or INTENTIONAL — uncertain** | Not fixed — a training-protocol question, not a code bug; see negative_pair_ratio entry and the research bottleneck analysis |
| `negative_pair_ratio` | `2.0` | `--negative_pair_ratio 2.0` (train scripts) | Inert whenever `use_all_pairs=False` | Reads as "2:1 negatives" per its own doc-comment | **SCRIPT BUG** (added alongside `use_all_pairs=false` in the same commit, evidence of an unintended interaction) | **Fixed**: loud warning added, behavior unchanged |
| `use_rfs` / `rfs_t` | `True` / `0.001` | not passed by any script | No-op under the `local-jsonl` backend every script uses | Reads as "on by default" | **LEGACY CONFIG** (correct for a backend, `VG150LocalDataset`, that the maintained scripts don't use) | **Fixed**: loud warning added, behavior unchanged |
| `adaptive_calibration_enabled` | `False` | `--adaptive_calibration_enabled true` (train + eval scripts) | `True` under shipped scripts | `True` (both scripts agree) | **INTENTIONAL OVERRIDE**, documentation gap only | **Fixed**: code comment added, no behavior change |
| `configs/presets.yaml` | n/a | n/a | Never loaded by any code | Documentation | **DOCUMENTATION ONLY** | Already fixed (header notice) in the cleanup pass |

### `tools/build_vg150_clean_vocab.py` default scans train+validation together
- `--splits` defaults to `"train,validation"`. The resulting object
  vocabulary (used, when present, as the CLIP zero-shot SGCls classifier's
  target vocabulary) is influenced by validation-split label frequencies —
  a mild vocabulary-construction leak. `evals.py`'s own internal fallback
  vocab loader (`_load_vg150_object_vocab`) does the same when no frozen
  vocabulary file is present.
- Likely low real-world impact (VG150's 150-object vocabulary is a
  well-known fixed list), but a real methodological objection until fixed.

## P2 — Evaluation metric naming

### Headline `R@K` is dataset-global-pooled, not per-image-averaged
- `openvocab_rel/evals.py`'s accumulation logic computes the headline
  `R@K`/`mR@K` fields as `sum(hits across all images) / sum(GT across all
  images)`. Most VG150 literature reports "R@K" as the per-image recall
  averaged across images. This repo *also* computes and reports that
  variant, under a different field name (`image_mean_R@K`) — but anything
  citing the headline `R@K` field next to a literature number (as the
  README's own LaTeX table snippet does) is comparing two different
  statistics without saying so.

## P2 — Coverage gaps

### No CI-friendly end-to-end eval smoke test
- The new `tests/` suite (added in this cleanup pass) deliberately does not
  exercise the full `eval_sgg_standard` pipeline, since that requires real
  CLIP weights (and optionally Grounding-DINO) and network/HF-cache access.
  There is currently no automated check that the full PredCls/SGCls/SGDet
  eval path runs end to end without a human manually launching
  `scripts/eval/eval_l4_phase34.sh` against real data.

### No checksum/version-pinning mechanism for the VG150 data itself
- `data/README.md` documents why (no single canonical checksum exists
  across VG150 redistribution mirrors this repo has observed) and what a
  user can do manually (hash `train.jsonl`/`validation.jsonl` themselves).
  There is no automated mechanism recording or checking this today.

### `scripts/notebooks/kaggle-pure-full-train.ipynb` prints a reference to a script that doesn't exist
- The notebook's final summary cell prints a suggestion:
  `"3. Scale training on H200 (Stage 3): bash scripts/run_h200.sh ..."`.
  No `scripts/run_h200.sh` exists anywhere in this repo (tracked or
  otherwise) — an informational print statement referencing a script that
  was apparently never added. Cosmetic (it's just printed text, not an
  executed command), left as-is per the "don't fabricate missing files"
  instruction — flagged here rather than silently invented.

## P3 — Dead code (evals.py / train.py) — FIXED

Call-site-verified (repo-wide grep, not just same-file) before removal;
all four zero-risk deletions confirmed with `ast.parse` afterward and the
full test suite passing unchanged:

- `openvocab_rel/evals.py`: `_mean_recall_from_matches` (a per-image
  mean-recall-across-predicate-classes helper, a third variant distinct
  from both the headline pooled `R@K` and the global `mR@K`) had zero call
  sites anywhere in the file. **Removed.**
- `openvocab_rel/evals.py`: the `if __name__ == "__main__":` block's body
  (standalone CLI: parse args, load checkpoint, load CLIP, build model,
  run `eval_query_grounding`) was duplicated verbatim in full, back to
  back, under the same single `if` guard — running the script would parse
  argv, run the whole evaluation, print metrics, then silently do the
  entire thing a second time. **Removed the second copy** (confirmed
  byte-identical logic; the tiny cosmetic difference — the second copy's
  redundant direct `CLIPModel.from_pretrained` call that gets immediately
  overwritten by `configure_clip` two lines later — was itself dead
  within the dead copy).
- `openvocab_rel/evals.py`: an `_update_object_diag`-style closure
  (referencing the free variable `object_diag`, which only exists in the
  one function that actually defines *and calls* it,
  `eval_sgg_standard`) was copy-pasted into 10 other standalone eval
  functions and never invoked in any of them. Verified programmatically
  that all 10 dead copies were byte-for-byte identical to each other
  before deleting them in one pass; the one real, live copy (defined and
  called inside `eval_sgg_standard`) is untouched. **Removed the 10 dead
  copies.**
- `openvocab_rel/train.py`: `_cfg_from_args` (a config-construction
  helper) had zero call sites anywhere in the repo; `main()` reimplements
  equivalent but behaviorally different config-merge logic inline instead
  (see the CLI-merge logic documented in `docs/architecture/training.md`).
  **Removed.**

### Not fixed here — out of scope for a dead-code pass

- `openvocab_rel/train.py`: `torch.nn.utils.clip_grad_norm_` is applied to
  `model.parameters()` only — CLIP's gradients are never norm-clipped once
  CLIP is unfrozen in stage 2/3. This is *not* dead code (the clipping call
  is live and does something), and changing its scope would change
  training/optimization behavior — a category-A/C judgment call, not a
  zero-risk deletion. Left untouched.

## P2 — README references two files that don't exist in this checkout

`README.md` references `notes/breakthrough_branch_plan.md` and
`notes/pure_conference_upgrade_roadmap.md`. Neither file exists in this
repository (confirmed via `ls notes/`, which lists only 6 tracked `.tex`
files). Fixed in this cleanup pass by removing the dead links from
`README.md` (see the README changelog note at the top of that file) — the
underlying content those files were meant to hold was never fabricated or
recreated, since that would misrepresent what actually exists.
