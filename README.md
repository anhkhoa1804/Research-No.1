# PURE Relation Modeling

PURE (Predicate-aware Uncropped Relation Embedding) is a compact VG150-style visual relation learning codebase. The maintained path now follows a balanced-debias roadmap:

1. keep the compact L3 core as the backbone;
2. measure raw classifier-only capacity before any prior boost;
3. add regularized debias heads only if raw mR improves without collapsing R@50;
4. run checkpoint-specific alpha sweeps only after the raw model passes acceptance criteria;
5. add ambiguous-negative suppression and tail prototypes only after the balanced head is stable;
6. scale to full VG150/A100 after subset gains are reproducible.

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
scripts/run_core_l3_l4_sweep.sh           optional CF/SPOA/grounding core-loss sweep
scripts/run_core_l3_balanced_debias.sh    balanced adaptive-prior/bias-residual A/B/C ablation
scripts/reeval_balanced_debias_matrix.sh  raw classifier + checkpoint-specific alpha sweep matrix
scripts/eval_l3_calibrated.sh             single-checkpoint fixed-prior calibrated eval runner
tools/prepare_vg150_subset.py             HF -> local JSONL/images with validation
tools/check_vg150_diagnostics.py          local dataset diagnostics guard
tools/build_vg150_frequency_prior.py      subject-object predicate prior builder
tools/summarize_metrics.py                compact metrics summary across runs
notes/current_status.tex                  concise implementation/status note
```

## Maintained Baseline

The current best local mean-recall baseline is:

- checkpoint: `checkpoints/eval_l3_final_a0_fa225_eb200_best_mR50.pt`
- calibrated PredCls metric: `R@50≈0.6215`, `mR@50≈0.1880`
- latest eval-only summary provided by the running workspace: `R@50≈0.6204`, `mR@50≈0.1853`
- best local R@50 observed after the older L3 sweep: `R@50≈0.6341`, with lower `mR@50≈0.1720`
- best mR@50 from that sweep: `mR@50≈0.1726`, still below the L3 final baseline

Report no-prior/classifier/text/fixed-prior/adaptive-calibration metrics separately. Do not mix calibrated eval numbers with no-prior model capacity claims.

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
CKPT=checkpoints/eval_l3_final_a0_fa225_eb200_best_mR50.pt \
FREQ_BIAS_ALPHA=2.25 \
EVAL_BATCHES=200 \
bash scripts/eval_l3_calibrated.sh
```

## Balanced Debias Ablation

Train the three maintained balanced-debias candidates after a stable core checkpoint exists:

```bash
BASE_CKPT=checkpoints/core_l3_cf004_spoa050_g018_lr2e6_best_mR50.pt \
EPOCHS=3 \
SAMPLES_PER_EPOCH=12000 \
EVAL_BATCHES=200 \
bash scripts/run_core_l3_balanced_debias.sh
```

Then evaluate raw capacity and checkpoint-specific calibration curves:

```bash
ALPHAS="0 0.75 1.25 1.50 1.75 2.25 2.75" \
EVAL_BATCHES=200 \
bash scripts/reeval_balanced_debias_matrix.sh
```

Acceptance criteria for a new default checkpoint:

- raw classifier-only: `R@50 >= 40.0` and `mR@50 >= 13.5`;
- calibrated: `R@50 >= 62.0` and `mR@50 > 18.6`;
- prediction histogram: no collapse into only head predicates or only tail predicates.

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


## CORE Download, Inspect, and VG150 Merge

CORE should be used in two separate roles: first as a held-out diagnostic set for VG150-only checkpoints, then as an optional robustness-training resource in ablations with a held-out CORE split. Do not train on all CORE rows and report that same CORE score as a diagnostic result.

Download the Google Drive folder on a GCP VM:

```bash
CORE_MODE=zip \
CORE_CLEAN=true \
CORE_OUT_DIR=datasets/core \
CORE_ZIP_PATH=datasets/core.zip \
CORE_ZIP_URL="https://drive.google.com/file/d/1eWdgbrQo_XTO4Ubfy2ygYtmojATlx6jJ/view?usp=drive_link" \
bash scripts/download_core_gdrive.sh
```

If `gdown` is missing, install it in the user environment, not a new virtualenv:

```bash
python3 -m pip install --user -U gdown
```

Inspect the downloaded folder and write a schema/box/image report:

```bash
python3 tools/inspect_core.py \
  --core-root datasets/core \
  --report runs/core_inspect/report.json
```

Convert CORE into the JSONL schema consumed by `VG150JSONLDataset`:

```bash
python3 tools/convert_core_to_vg150_jsonl.py \
  --core-root datasets/core \
  --out-root datasets/core_vg150_jsonl \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --holdout-v2
```

Merge converted CORE train rows into VG150 for robustness ablations. By default validation remains VG150-only so calibrated VG150 reporting is not contaminated:

```bash
python3 tools/merge_vg150_core_jsonl.py \
  --vg-root datasets \
  --core-root datasets/core_vg150_jsonl \
  --out-root datasets/vg150_core_merged \
  --core-train-repeat 1
```

Recommended run order:

1. Train/evaluate VG150-only as the primary SGG benchmark.
2. Evaluate the VG150-only checkpoint qualitatively or with a separate CORE diagnostic script/report.
3. Train VG150+CORE only as a robustness ablation using `DATA_ROOT=datasets/vg150_core_merged`.
4. Keep CORE v2 and `Extreme_Compositional_OOD` as preferred held-out stress tests when possible.

Run a CORE-integrated ablation after merge:

```bash
DATA_ROOT=datasets/vg150_core_merged \
RUN_NAME=vg150_core_l3_ablation \
OUT_ROOT=runs/vg150_core/l3_ablation \
SAVE_PATH=checkpoints/vg150_core_l3_ablation.pt \
bash scripts/run_pure_next.sh
```

## Calibration Policy

Adaptive prior gates and sample-level bias residuals are trained only in balanced-debias ablations. Fixed frequency priors are treated as post-hoc reporting boosters, not as evidence of raw model capacity.

Report every serious checkpoint in three tiers:

1. raw classifier-only, no frequency prior, no text ensemble;
2. fixed-prior calibrated with a checkpoint-specific alpha sweep;
3. adaptive-calibrated only for checkpoints trained with adaptive calibration enabled.

## Report-Ready LaTeX Snippets

### Metric Table

```latex
\begin{table}[h]
\centering
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrl}
\toprule
Method & Setting & PredCls R@50 & PredCls R@100 & PredCls mR@50 & PredCls mR@100 & Notes \\
\midrule
PURE L3 sweep cf004/spoa050 & local, calibrated eval & \textbf{63.41} & 63.41 & 17.20 & 17.20 & best local R@50, lower mR \\
PURE L3 sweep cf004/spoa035 & local, calibrated eval & 63.21 & 63.21 & 17.26 & 17.26 & best sweep mR, below baseline \\
PURE L3 final & local, calibrated checkpoint & 62.15 & -- & \textbf{18.80} & -- & best local mR@50 so far \\
PURE eval recovery & local, eval-only summary & 62.04 & 62.04 & 18.53 & 18.53 & latest provided eval summary \\
ResCAGCN + PUM & VG150, published & -- & -- & 20.20 & 22.00 & semantic ambiguity debiasing \\
Motifs + CFA & VG150, published & 54.10 & 56.60 & 35.70 & 38.20 & compositional feature augmentation \\
SBG & VG150, published & 55.80 & 57.60 & 33.30 & 35.70 & sample-level bias guidance \\
Hydra-SGG & VG150/GQA, published & -- & -- & 16.00 & -- & one-stage SGG setting; not directly PredCls-comparable \\
\bottomrule
\end{tabular}%
}
\caption{Current PURE metrics and representative external references. Values are percentages. External rows are included for orientation, not as a direct claim of protocol-matched comparison.}
\end{table}
```

### Architecture Diagram

See `notes/current_status.tex` for the full TikZ figure. The diagram describes the current PURE pipeline: dense CLIP feature grid, deformable object grounding, ordered subject--object relation construction, Fourier geometry and vector fusion, edge-conditioned relation decoding, predicate scoring, and optional adaptive calibration.

## Next Research Direction

The next top-tier direction is balanced debiasing, not more post-hoc boosting:

1. lock the compact L3 core and keep legacy branches disabled;
2. train `adapt_light`, `adapt_mid`, and `prior_only` with calibration-head gradient clipping;
3. select by raw classifier-only acceptance criteria before looking at calibrated scores;
4. run checkpoint-specific alpha sweeps only for accepted raw candidates;
5. add one-to-many relation assignment / ambiguous-negative suppression if over-debiasing persists;
6. scale the winning configuration to full VG150/A100 only after subset gains are reproducible.

## Locked SGG Architecture and Full Ablation Cycle

The final thesis framing is **SGG-first**: retrieval is a downstream demo, while the scientific target is predicate-aware scene graph generation.

### Locked PURE-Core architecture

The maintained architecture for the next training cycle is:

1. **Dense CLIP visual field** from the uncropped image.
2. **Deformable box-conditioned routing** for object evidence.
3. **Ordered subject--object relation embedding** with geometry-aware pair features.
4. **Predicate scoring head** trained with the three maintained losses:
   - `PredCE-LA`,
   - `Counterfactual-SPOA`,
   - `Dense Grounding`.
5. **Evaluation-time calibration** reported separately from raw classifier capacity.

The following branches stay disabled in the default architecture and are only ablation probes: object-language anchors, relation-context transformer, bilinear mixing, and visual hard negatives.

### Full train / ablation runner

Use the new runner for the upcoming full SGG cycle:

```bash
DATA_ROOT=datasets/vg150_core_merged \
CHECKPOINT_DIR=checkpoints/sgg_full_cycle \
RUN_ROOT=runs/sgg_full_cycle \
GPU_PRESET=l4_24gb \
EPOCHS=8 \
EVAL_BATCHES=200 \
bash scripts/run_sgg_full_ablation_cycle.sh
```

`DATA_ROOT` must contain `train.jsonl`, `validation.jsonl`, and, for calibrated evaluation, `frequency_prior.json`. To include CORE, convert the current ~1k CORE images into the same VG150 JSONL schema and merge them into `train.jsonl` before launching the cycle.

Useful subsets:

```bash
# Print all commands without running training.
DRY_RUN=true RUN_GROUPS=all bash scripts/run_sgg_full_ablation_cycle.sh

# Train only the locked core architecture and calibration sweep.
RUN_GROUPS=core bash scripts/run_sgg_full_ablation_cycle.sh

# Run only loss ablations.
RUN_GROUPS=loss bash scripts/run_sgg_full_ablation_cycle.sh

# Run only branch ablations.
RUN_GROUPS=architecture bash scripts/run_sgg_full_ablation_cycle.sh

# Run only calibration / bias residual ablations.
RUN_GROUPS=calibration bash scripts/run_sgg_full_ablation_cycle.sh
```

Final reporting should select models in this order: raw classifier mR@50, calibrated R@50/mR@50 Pareto point, then CORE robustness plots from the merged CORE subset.
