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

## P1 — Dangerous silent fallbacks

### `eval_swap_consistency` reports a fabricated symmetric/asymmetric split
- **File/function:** `openvocab_rel/evals.py`, `eval_swap_consistency`
- **What happens:** the function returns `sym_cos`, `asym_cos`, `n_sym`,
  `n_asym` as if it filtered role-swap cosine similarity by predicate
  symmetry, but no such filter exists anywhere in the function body — all
  four fields are populated with the same aggregate value/count computed
  over every predicate, symmetric or not.
- **Why it matters:** any report or downstream tool that reads this
  function's output believing it separates symmetric- from
  asymmetric-predicate consistency is reading a plausible-looking but wrong
  number.
- **How to test later:** unit test asserting `n_sym + n_asym == n` on a
  synthetic input with a known mix of symmetric/asymmetric predicates, and
  that `sym_cos != asym_cos` when the input is constructed so they should
  differ.

### Conditional eval-label leak in the frequency-prior helper
- **File/function:** `openvocab_rel/evals.py`, `_predicate_log_prior_for_eval`
- **What happens:** the function correctly tries `train.jsonl` under
  `cfg.vg150_root` first. If that file is missing or fails to parse, it
  **silently** falls back to counting predicate frequencies from the loader
  object actually passed into `eval_sgg_standard` — i.e., potentially the
  validation/test split currently being scored — with no warning printed.
- **Why it matters:** if ever triggered (misconfigured `vg150_root`, or a
  swallowed JSON parse error), the resulting prior — fed into
  `logit_adj_tau` adjustment and/or `RelationalModel.calibrated_predicate_logits`
  — would be fit on the same split's ground-truth labels being evaluated,
  silently inflating reported numbers.
- **Not currently active** under the documented, `train.jsonl`-present
  workflow; this is a latent footgun, not a demonstrated live leak.
- **How to test later:** point `eval_sgg_standard` at a `vg150_root` that
  has no `train.jsonl` and assert the function either raises or logs loudly
  instead of silently falling back to the eval loader.

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

### `configs/presets.yaml` is documentation only, never loaded
- See the header comment added to that file in this cleanup pass. Real
  `--gpu_preset`/`--stage`/`--a100_ddp_preset` behavior comes from hardcoded
  Python in `openvocab_rel/config.py`; nothing enforces the YAML stays in
  sync with it. Deciding whether to wire it up for real or delete it is a
  follow-up decision, not made in this cleanup pass.

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

## P3 — Dead code (evals.py / train.py)

- `openvocab_rel/evals.py`: a duplicated `if __name__ == "__main__":` CLI
  block appears twice, verbatim, at the end of the file.
- `openvocab_rel/evals.py`: an `_update_object_diag`-style closure
  (referencing an undefined free variable `object_diag`) is copy-pasted
  into roughly 10 standalone eval functions, never invoked in any of them
  (the one real, live copy is inside `eval_sgg_standard`).
- `openvocab_rel/train.py`: `_cfg_from_args` (a config-construction helper)
  has zero call sites; `main()` reimplements equivalent but behaviorally
  different config-merge logic inline instead.
- `openvocab_rel/train.py`: `torch.nn.utils.clip_grad_norm_` is applied to
  `model.parameters()` only — CLIP's gradients are never norm-clipped once
  CLIP is unfrozen in stage 2/3.

## P2 — README references two files that don't exist in this checkout

`README.md` references `notes/breakthrough_branch_plan.md` and
`notes/pure_conference_upgrade_roadmap.md`. Neither file exists in this
repository (confirmed via `ls notes/`, which lists only 6 tracked `.tex`
files). Fixed in this cleanup pass by removing the dead links from
`README.md` (see the README changelog note at the top of that file) — the
underlying content those files were meant to hold was never fabricated or
recreated, since that would misrepresent what actually exists.
