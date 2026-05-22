# PURE Relation Modeling

PURE (Predicate-aware Uncropped Relation Embedding) is a compact VG150-style visual relation learning codebase. The maintained path is now deliberately narrow:

1. train a core L1 -> L3 relation model;
2. evaluate it with transparent text/frequency calibration;
3. only add new mechanisms if they beat the calibrated L3 baseline.

Legacy branch-ramp wrappers, high-curriculum probes, Kaggle notebooks, visual hard-negative defaults, bilinear probes, object-language anchor scaling, and relation-context scaling are removed from the maintained workflow because recent ablations showed they hurt mean recall or created protocol confusion.

## Codebase Map

```text
openvocab_rel/train.py                    main train/eval loop
openvocab_rel/config.py                   TrainConfig and GPU/runtime knobs
openvocab_rel/models/relational_model.py  core relation model and decoder
openvocab_rel/datasets/vg150_loader.py    VG150 local/HF loaders and pair construction
openvocab_rel/evals.py                    PredCls/SGCls/SGDet and score-mode evals
openvocab_rel/losses.py                   PredCE/SPOA/grounding-related losses
openvocab_rel/clip_utils.py               CLIP setup and text/image helpers
scripts/run_pure_next.sh                  low-level configurable train/eval entrypoint
scripts/run_core_recovery_l4.sh           maintained L4 core L1->L3 training runner
scripts/eval_l3_calibrated.sh             maintained calibrated L3 eval runner
scripts/run_l3_eval_calibration_sweep.sh  calibration probe around known-good L3
scripts/run_l3_eval_calibration_refine.sh calibration refinement around FA≈2.25
tools/prepare_vg150_subset.py             HF -> local JSONL/images with validation
tools/check_vg150_diagnostics.py          local dataset diagnostics guard
tools/build_vg150_frequency_prior.py      subject-object predicate prior builder
tools/summarize_metrics.py                compact metrics summary across runs
notes/current_status.tex                  concise implementation/status note
```

## Maintained Baseline

The current reliable baseline is:

- checkpoint: `checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt`
- score: `EVAL_SCORE_MODE=ensemble`
- ensemble alpha: `EVAL_ENSEMBLE_ALPHA=0.0` (text-only normalized score before prior)
- frequency prior: `FREQ_BIAS_ENABLED=true`, `FREQ_BIAS_ALPHA=2.25`
- observed fast PredCls metric on the current L4 subset: `R@50≈0.6215`, `mR@50≈0.1880`

Report no-prior/classifier/text/calibrated metrics separately. Do not mix calibrated eval numbers with no-prior model capacity claims.

## Train Core L1 -> L3

```bash
bash scripts/run_core_recovery_l4.sh
```

Resume only L3 from an existing L1 checkpoint:

```bash
RUN_L1=false \
RUN_L3=true \
L3_RESUME_FROM=checkpoints/l1_spoa_ground_recovery_l4_best_mR50.pt \
bash scripts/run_core_recovery_l4.sh
```

## Calibrated Evaluation

```bash
bash scripts/eval_l3_calibrated.sh
```

Override checkpoint or prior strength:

```bash
CKPT=checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt \
FREQ_BIAS_ALPHA=2.25 \
EVAL_BATCHES=200 \
bash scripts/eval_l3_calibrated.sh
```

## Calibration Sweeps

Use these only after a stable L3 checkpoint exists:

```bash
bash scripts/run_l3_eval_calibration_sweep.sh
bash scripts/run_l3_eval_calibration_refine.sh
```

Summarize metrics:

```bash
python3 tools/summarize_metrics.py runs/eval_l3*/metrics.jsonl
```

## Dataset Preparation

Prepare a local VG150 JSONL subset when needed:

```bash
python3 tools/prepare_vg150_subset.py \
  --dataset_id anhkhoa1804/VG150-SGG-Standard \
  --out_dir datasets \
  --train_images 5000 \
  --val_images 500 \
  --max_objects 32 \
  --min_relationships 1 \
  --min_predicate_coverage 50
```

Check diagnostics:

```bash
python3 tools/check_vg150_diagnostics.py \
  --diagnostics datasets/diagnostics.json \
  --min_train_rows 5000 \
  --min_val_rows 500 \
  --min_predicate_coverage 50 \
  --require_no_validation_issues
```

Build the frequency prior:

```bash
python3 tools/build_vg150_frequency_prior.py \
  --train_jsonl datasets/train.jsonl \
  --out_path datasets/frequency_prior.json \
  --vg150_root datasets \
  --smoothing 1.0
```

## Next Research Direction

The next top-tier direction is not more post-hoc branches. It is bias-aware calibrated relation learning:

1. adaptive calibration head instead of fixed `FREQ_BIAS_ALPHA`;
2. sample-level bias residual for subject-object-predicate priors;
3. tail-aware predicate prototypes based on CLIP text and visual positives;
4. one-to-many relation assignment / ambiguous-negative suppression;
5. full VG150/A100 scaling only after the above improves the 10k subset baseline.

# PURE Refactor and Ablation Plan

## Maintained Baseline

- Core checkpoint: `checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt`
- Calibrated eval: `EVAL_SCORE_MODE=ensemble`, `EVAL_ENSEMBLE_ALPHA=0.0`, `FREQ_BIAS_ALPHA=2.25`
- Current fast PredCls result: `R@50≈0.6215`, `mR@50≈0.1880`

## Removed From Maintained Workflow

- Object-language anchor scaling: negative mR ablation.
- Relation-context scaling: negative/unstable mR ablation.
- Bilinear mixing probes: legacy branch-ramp idea, not maintained.
- Visual hard negatives: legacy branch-ramp idea, not maintained.
- High-curriculum and Kaggle scripts: removed to avoid protocol drift.

## Next Strong Ablation Sequence

1. Confirm baseline with `scripts/eval_l3_calibrated.sh`.
2. Retrain core L3 with controlled seeds and loss weights:
   - `lambda_counterfactual`: 0.02, 0.04, 0.06
   - `lambda_spoa_alignment`: 0.35, 0.50, 0.65
   - `lambda_dense_grounding`: 0.12, 0.18, 0.25
3. Evaluate every checkpoint with the fixed calibrated protocol.
4. Implement adaptive calibration only after fixed-prior baseline is stable.
5. Scale to full VG150/A100 after subset gains are reproducible.

## Top-Tier Upgrade Targets

- Adaptive calibration head replacing fixed frequency-prior alpha.
- Sample-level bias residual using subject/object/geometry/entropy features.
- Tail-aware predicate prototypes built from text and visual positives.
- One-to-many relation assignment and ambiguous-negative suppression.