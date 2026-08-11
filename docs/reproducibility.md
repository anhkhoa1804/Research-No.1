# Reproducibility

What you need to obtain externally before running anything in this repo,
what this repo tracks vs. doesn't, and how to check your setup is ready —
before spending GPU time finding out the hard way.

## 1. Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
# optional: diffusion rendering / VLM filtering tooling
pip install -r requirements-optional.txt
# optional: to run the smoke-test suite in tests/
pip install -r requirements-dev.txt
```

`requirements.txt`/`requirements-optional.txt` pin **lower-bound** versions
anchored to a known-working local dev snapshot (see the comment header in
each file). They are not yet re-verified against the L4/A100 cloud VMs
actually used for the training runs referenced in `README.md` — if you hit
an incompatibility there, that gap is real and worth reporting, not a sign
your setup is wrong.

## 2. Dataset setup

See `data/README.md` for the full picture. Short version: only VG150 is
consumed by any code path in this repo. Get it via
`tools/prepare_vg150_drive_clean.py` (preferred) or
`tools/prepare_vg150_subset.py` (small smoke subset from Hugging Face), both
already documented with exact commands in the root `README.md`. No dataset
files are committed to this repository.

## 3. External model setup

CLIP (`openai/clip-vit-large-patch14-336` by default, `--clip_name` to
override) and, only when SGDet evaluation is enabled, Grounding-DINO
(`IDEA-Research/grounding-dino-tiny` by default,
`--eval_sgg_grounding_dino_model_id` to override) are both resolved
automatically from the Hugging Face Hub at runtime via `transformers`. There
is no separate manual download step today — the first run that needs them
will pull them into the local HF cache, which requires network access (or a
pre-warmed cache) at that point. This repo does not track or pin exact
model revisions beyond the HF model-id string; that string is captured into
every checkpoint's `experiment.train_config` snapshot (see §10), so you can
always tell *which* model id a given checkpoint was trained/evaluated
against, even though there's no hash/revision pin today (see
`docs/known_issues.md`).

## 4. Validate your local setup

```bash
python3 tools/validate_dataset.py --dataset vg150 --vg150_root <your DATA_ROOT>
```
Pre-flight structural check (new in this cleanup pass): confirms the
expected files/dirs exist and the JSONL splits parse, and fails loudly with
a specific message instead of letting a bad path surface confusingly deep
into a training run.

```bash
python3 tools/check_vg150_diagnostics.py \
  --diagnostics <DATA_ROOT>/diagnostics.json \
  --min_train_rows 50000 --min_val_rows 5000 \
  --min_predicate_coverage 50 --require_no_validation_issues
```
Full coverage/quality check against the output of
`tools/prepare_vg150_drive_clean.py`.

## 5. Training

```bash
bash scripts/train/train_l4_phase34.sh
```
is the maintained default entrypoint. `bash scripts/train/run_pure_next.sh`
is an older, lower-level, still-functional entrypoint with a different flag
surface. `bash scripts/train/train_phasec_pair_proposal.sh` is the Phase-C
pair-proposal pilot (wraps `train_l4_phase34.sh` with a different loss
configuration, freezing most of the model). `scripts/notebooks/
kaggle-pure-full-train.ipynb` is a Kaggle-specific alternate entrypoint.

Smoke test before a long run:
```bash
EPOCHS=1 SAMPLES_PER_EPOCH=1000 EVAL_BATCHES=20 MAX_IMAGES=1000 bash scripts/train/train_l4_phase34.sh
```

## 6. Evaluation

```bash
CKPT=checkpoints/pure_l4_phase34.pt EVAL_BATCHES=500 bash scripts/eval/eval_l4_phase34.sh
```
`scripts/eval/eval_calibration_sweep_l4.sh` sweeps the fixed-prior
calibration alpha; `scripts/eval/eval_phasec_pairgate_smoke.sh` runs the
Phase-C pair-proposal gate; `scripts/eval/report_breakthrough_phase_ab.sh`
summarizes existing `metrics.jsonl` files without running anything new.

## 7. Diagnostics

`python3 tools/sgg_gate_report.py runs/<run_name>/metrics.jsonl` and
`python3 tools/model_report_card.py runs/*/metrics.jsonl --train_jsonl
<DATA_ROOT>/train.jsonl` summarize `metrics.jsonl` output into head/body/tail
mR@50 breakdowns, best-checkpoint tables, and pair-proposal (`prop@K`)
diagnostics. `python3 tools/predicate_delta_report.py` compares per-predicate
recall between two metric rows (e.g. before/after a change).

## 8. Where outputs are stored

None of this is committed to Git (see `.gitignore`):

```text
runs/<run_name>/
  metrics.jsonl            one JSON line appended per epoch
  epoch_metrics/epoch_NNN.json
  latest_metrics.json
checkpoints/<run_name>*.pt   4 variants per run: latest, _best_R50,
                              _best_mR50, _best_tail_mR50, _best_selection
logs/<run_name>.log          from the shell wrappers' `tee`
runs/clip_obj_cache/, runs/grounding_dino_cache/
                              content-hash-keyed eval-time caches
```

## 9. How to identify an experiment

Every checkpoint and every `metrics.jsonl` row embeds an `experiment`
snapshot: git commit hash (`git rev-parse --short HEAD` at train time),
a sha256 hash of the full training config dict, and a hash of the
predicate-vocabulary list (so you can tell, after the fact, whether two
checkpoints were trained against the same predicate index ordering — see
`docs/known_issues.md` for why that ordering isn't otherwise guaranteed
stable across dataset backends). This is a genuine existing strength of the
codebase, not something added by this cleanup pass — it's documented here,
not reimplemented.

## 10. What is intentionally not stored in Git, and why

Datasets, pretrained/fine-tuned weights, checkpoints, caches, run logs, and
generated embeddings are all excluded (`.gitignore`) because they're large,
regenerable, and would make the repository unusable to clone. Their
*absence* is made explicit instead, via `data/README.md` +
`data/manifests/vg150.yaml` + `tools/validate_dataset.py`, rather than
silently assumed.

## Reproducibility status: **NOT YET REPRODUCIBLE end-to-end**

This checkout ships **zero** committed checkpoints, run logs, or prepared
datasets. Every number quoted in `README.md` or `notes/current_status.tex`
is a claim from an external run this repository cannot currently
independently reproduce or verify — that is stated plainly, not glossed
over. What *is* reproducible today: the code path itself (imports, config
construction, forward pass on synthetic data — see `tests/`), and, given the
external VG150 data and CLIP/Grounding-DINO access described above, the
training/eval commands themselves run as documented. Getting to a state
where a specific reported number can be independently regenerated requires,
at minimum: the exact VG150 data (no checksum pinned yet, see
`docs/known_issues.md`), the exact training config (available: embedded in
each checkpoint), and the exact random seed and hardware/library versions
(seed is embedded; library versions are pinned as lower bounds, not exact).
