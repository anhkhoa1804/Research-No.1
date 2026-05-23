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
scripts/run_core_l3_calibration_ablate.sh adaptive calibration and bias-residual ablation
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

## Adaptive Calibration Ablation

The code now includes an optional trainable calibration path: classifier logits can be augmented by an adaptive prior gate and a bounded sample-level bias residual. This is disabled by default for backward compatibility and enabled in the dedicated ablation runner:

```bash
BASE_CKPT=checkpoints/eval_l3_final_a0_fa225_eb200_best_mR50.pt \
  bash scripts/run_core_l3_calibration_ablate.sh
```

Evaluate old checkpoints with `ADAPTIVE_CALIBRATION_ENABLED=false`; only checkpoints trained by `run_core_l3_calibration_ablate.sh` should be evaluated with `ADAPTIVE_CALIBRATION_ENABLED=true`.

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

The next top-tier direction is not more post-hoc branches. It is bias-aware calibrated relation learning:

1. finish adaptive calibration head ablations instead of relying only on fixed `FREQ_BIAS_ALPHA`;
2. validate the sample-level bias residual against the current `mR@50≈0.1880` baseline;
3. add tail-aware predicate prototypes based on CLIP text and visual positives if calibration improves;
4. add one-to-many relation assignment / ambiguous-negative suppression after the calibration path is stable;
5. full VG150/A100 scaling only after the above improves the 10k subset baseline.
