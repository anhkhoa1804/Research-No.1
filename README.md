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

# PURE Balanced-Debias Ablation Plan

## Maintained Baseline

- Raw classifier-only must be reported separately from calibrated results.
- Current calibrated reference: `checkpoints/eval_l3_final_a0_fa225_eb200_best_mR50.pt` with `R@50≈0.6215`, `mR@50≈0.1880` under tuned fixed-prior evaluation.
- Current raw evidence shows newer CF/SPOA/adaptive checkpoints improve long-tail mR but may trade off head recall.

## Removed From Maintained Workflow

- Object-language anchor scaling: negative mR ablation.
- Relation-context scaling: negative/unstable mR ablation.
- Bilinear mixing probes: legacy branch-ramp idea, not maintained.
- Visual hard negatives: legacy branch-ramp idea, not maintained.
- High-curriculum, Kaggle, and standalone calibration-refine scripts: removed to avoid protocol drift.

## Roadmap-Aligned Sequence

1. Keep compact L3 core as the backbone and disable legacy branches by default.
2. Train balanced debias candidates with `scripts/run_core_l3_balanced_debias.sh`:
   - `core_l3_balanced_adapt_light`: light adaptive prior + small residual.
   - `core_l3_balanced_adapt_mid`: stronger adaptive prior + residual with higher regularization.
   - `core_l3_balanced_prior_only`: adaptive prior without residual.
3. Evaluate with `scripts/reeval_balanced_debias_matrix.sh`:
   - alpha `0` means raw classifier-only capacity.
   - positive alphas measure post-hoc fixed-prior compatibility.
4. Accept a new default only if raw classifier-only reaches `R@50 >= 40.0` and `mR@50 >= 13.5`.
5. Add one-to-many relation assignment / ambiguous-negative suppression only if the best raw candidate still over-debiases.
6. Scale to full VG150/A100 after subset gains are reproducible across raw and calibrated tiers.

## Reporting Rule

Every result table must separate:

- raw classifier-only model capacity;
- fixed-prior calibrated system score;
- adaptive-calibrated score for adaptive-trained checkpoints only.