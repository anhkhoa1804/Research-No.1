# PURE: Scene Graph Generation for Relation-Aware Retrieval

PURE (**Predicate-aware Uncropped Relation Embedding**) is a compact Scene Graph Generation (SGG) codebase for learning visual relations on VG150-style data. The thesis framing is **SGG-first**: the primary task is predicate-aware scene graph prediction, while text-based image retrieval is a downstream application built from predicted triplets.

The project currently has two main research components:

1. **PURE model**: a compact ordered-pair predicate model using dense uncropped visual evidence, box-conditioned routing, geometry-aware relation embeddings, and transparent calibrated evaluation.
2. **CORE benchmark**: a compositional relation evaluation benchmark for diagnosing role swaps, object swaps, spatial contradictions, predicate confusability, and other relation-level failures.

## Current Status

The maintained thesis direction is:

- Keep the SGG problem as the core task.
- Treat retrieval as an application layer over cached scene graph triplets.
- Report raw classifier capacity separately from calibrated evaluation.
- Use VG150 for main PredCls / GT-pair reporting.
- Use CORE as a diagnostic benchmark and optional robustness-training ablation, not as a replacement for VG150.

Current local headline PredCls results used in the presentation:

| Setting | R@50 | mR@50 | Notes |
| --- | ---: | ---: | --- |
| Adapt-light + fixed prior `fa45` | **67.20** | 22.38 | Best calibrated R@50 |
| Adapt-light + fixed prior `fa375` | 67.09 | **22.64** | Best calibrated mR@50 |
| Ultra-light + fixed prior `fa35` | 66.92 | 22.58 | Strong calibrated point |
| Ultra-light raw e2 | 35.22 | 20.35 | Best raw training mR@50 |
| Adapt-light raw e5 | 33.71 | 20.03 | Tail improves across epochs |
| Prior-only raw e5 | 33.50 | 19.50 | Prior-only reference |

Important reporting rule: calibrated scores are system-level evaluation results. They should not be used as evidence that the raw classifier alone learned all tail predicates.

## Repository Map

```text
openvocab_rel/train.py                    main train/eval loop
openvocab_rel/config.py                   TrainConfig and runtime knobs
openvocab_rel/models/relational_model.py  PURE relation model and decoder
openvocab_rel/datasets/vg150_loader.py    VG150 JSONL/HF loaders and pair construction
openvocab_rel/evals.py                    PredCls/SGCls/SGDet-style evaluation utilities
openvocab_rel/losses.py                   predicate, counterfactual, and grounding losses
openvocab_rel/clip_utils.py               CLIP setup and image/text helpers
openvocab_rel/geometry.py                 geometry utilities
openvocab_rel/prompts.py                  text prompt helpers
openvocab_rel/text_cache.py               text embedding cache utilities
configs/presets.yaml                      GPU/runtime presets
scripts/run_pure_next.sh                  low-level configurable train/eval entrypoint
scripts/run_core_recovery_l4.sh           L4-oriented L1 -> L3 recovery runner
scripts/run_core_l3_l4_sweep.sh           optional CF/SPOA/grounding sweep
scripts/run_core_l3_balanced_debias.sh    balanced adaptive-prior/bias-residual ablation
scripts/reeval_balanced_debias_matrix.sh  raw + fixed-prior alpha evaluation matrix
scripts/eval_l3_calibrated.sh             single-checkpoint calibrated evaluation
scripts/run_core_eval.sh                  CORE evaluation helper
scripts/run_core_finetune_ablation.sh     CORE fine-tune / robustness ablation runner
scripts/diagnose_sgcls_sgdet.sh           SGCls/SGDet diagnostic runner
scripts/cleanup_balanced_artifacts.sh     cleanup helper for balanced-debias artifacts
tools/prepare_vg150_subset.py             HF -> local VG150 JSONL/images
tools/check_vg150_diagnostics.py          dataset diagnostics guard
tools/build_vg150_frequency_prior.py      subject-object predicate prior builder
tools/build_vg150_clean_vocab.py          clean VG150 object/predicate vocab builder
tools/build_core_eval_vocab.py            CORE evaluation vocab builder
tools/inspect_core.py                     CORE schema/image/box inspection
tools/convert_core_to_vg150_jsonl.py      CORE -> VG150 JSONL conversion
tools/merge_vg150_core_jsonl.py           VG150 + CORE training merge
tools/eval_core_text_image_retrieval.py   CORE text-image retrieval evaluation
tools/summarize_metrics.py                compact metric summary across runs
tools/collect_balanced_results.py         balanced-debias result collection
notes/current_status.tex                  compact technical status note
notes/paper1_pure.tex                     PURE paper draft
notes/paper2_core.tex                     CORE paper draft
notes/presentation/main.tex               Beamer defense slides
```

## Problem Definition

The core task is **Scene Graph Generation**.

Given an image and, in the main reported setting, ground-truth object boxes and object labels:

```text
Input:  image I, boxes B = {b_i}, object labels O = {o_i}
Output: ranked scene graph G(I) = {(subject, predicate, object, score)}
```

The main evaluation setting is currently **PredCls / GT-pair**: boxes and object labels are provided, and the model focuses on predicate ranking. SGCls and SGDet diagnostics exist, but they are not the current headline claims.

## PURE Architecture

The maintained PURE architecture is intentionally compact:

1. **Dense visual field**: encode the uncropped image once with CLIP patch tokens.
2. **Box-conditioned object routing**: initialize object queries from boxes and sample evidence from the dense visual field.
3. **Ordered relation embedding**: build subject-object pair features with explicit direction and geometry.
4. **Predicate scoring head**: predict relation logits for each ordered pair.
5. **Evaluation-time calibration**: optionally add a fixed frequency-prior term during evaluation.

Default maintained branches:

| Component | Status | Reason |
| --- | --- | --- |
| Dense CLIP tokens | Kept | Stable visual evidence for relations |
| Deformable box routing | Kept | Grounds object features in the uncropped image |
| Geometry-aware pair decoder | Kept | Preserves subject-object direction |
| Fixed frequency / Bayesian calibration | Eval-time only | Report raw and calibrated metrics separately |
| Object-language anchors | Ablation-only | Unstable calibrated mR in local ablations |
| Relation context / bilinear mixing | Ablation-only | Added complexity without reliable mR gain |
| Visual hard negatives | Ablation-only | Useful diagnostic, not headline path |

## Training Objective

The maintained PURE objective currently uses **three training losses**:

```text
L_PURE = λ_ce L_PredCE-LA + λ_spoa L_CF-SPOA + λ_g L_Ground
```

- **PredCE-LA**: predicate cross-entropy with logit adjustment for long-tail predicate frequency.
- **Counterfactual-SPOA**: semantic alignment with role/predicate counterfactual negatives.
- **Dense Grounding**: keeps subject, object, and pair features visually anchored to dense image evidence.

Calibration is not counted as a fourth loss because it is applied at evaluation time.

## Metrics and Reporting Policy

Report every serious checkpoint in separate tiers:

1. **Raw classifier-only**: no frequency prior and no text ensemble.
2. **Fixed-prior calibrated**: checkpoint-specific alpha sweep.
3. **Adaptive-calibrated**: only for checkpoints trained with adaptive calibration enabled.
4. **CORE diagnostic**: relation-composition stress test, not a protocol-matched VG150 SOTA comparison.

Core metrics:

- **R@K**: aggregate recall of ground-truth triplets in the top-K predictions.
- **mR@K**: mean recall across predicate classes; this exposes long-tail behavior.

Do not mix raw and calibrated numbers in the same claim.

## Dataset Preparation

Prepare a local VG150 subset when needed:

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

Build the frequency prior used for calibrated evaluation:

```bash
python3 tools/build_vg150_frequency_prior.py \
  --train_jsonl datasets/train.jsonl \
  --out_path datasets/frequency_prior.json \
  --vg150_root datasets \
  --smoothing 1.0
```

## Training and Evaluation

### Core L1 -> L3 recovery

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

### Single calibrated evaluation

```bash
bash scripts/eval_l3_calibrated.sh
```

Override checkpoint or fixed-prior strength:

```bash
CKPT=checkpoints/eval_l3_final_a0_fa225_eb200_best_mR50.pt \
FREQ_BIAS_ALPHA=2.25 \
EVAL_BATCHES=200 \
bash scripts/eval_l3_calibrated.sh
```

### Balanced debias ablation

Train maintained balanced-debias candidates after a stable core checkpoint exists:

```bash
BASE_CKPT=checkpoints/core_l3_cf004_spoa050_g018_lr2e6_best_mR50.pt \
EPOCHS=3 \
SAMPLES_PER_EPOCH=12000 \
EVAL_BATCHES=200 \
bash scripts/run_core_l3_balanced_debias.sh
```

Evaluate raw capacity and checkpoint-specific calibration curves:

```bash
ALPHAS="0 0.75 1.25 1.50 1.75 2.25 2.75" \
EVAL_BATCHES=200 \
bash scripts/reeval_balanced_debias_matrix.sh
```

Suggested acceptance criteria for a new default checkpoint:

- raw classifier-only: strong mR@50 without collapsing R@50;
- calibrated: improves the R@50/mR@50 Pareto point;
- histogram: no collapse into only head predicates or only tail predicates;
- CORE: improved robustness or clearer diagnostic behavior without contaminating held-out evaluation.

## CORE Benchmark Workflow

CORE should be used in two separate roles:

1. **Held-out diagnostic benchmark** for VG150-only checkpoints.
2. **Optional robustness-training resource** in ablations with a held-out CORE split.

Do not train on all CORE rows and report that same CORE score as a held-out diagnostic result.

### Download CORE

Download the Google Drive zip on a GCP VM or local machine. If `gdown` is missing, install it in the user environment, not a new virtual environment:

```bash
python3 -m pip install --user -U gdown
mkdir -p datasets/core
gdown --fuzzy "https://drive.google.com/file/d/1eWdgbrQo_XTO4Ubfy2ygYtmojATlx6jJ/view?usp=drive_link" -O datasets/core.zip
unzip -q datasets/core.zip -d datasets/core
```

### Inspect CORE

```bash
python3 tools/inspect_core.py \
  --core-root datasets/core \
  --report runs/core_inspect/report.json
```

### Convert CORE to VG150 JSONL

```bash
python3 tools/convert_core_to_vg150_jsonl.py \
  --core-root datasets/core \
  --out-root datasets/core_vg150_jsonl \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --holdout-v2
```

CORE metadata uses version/group-specific entity handling: groups 1--5 use root-level `shared_entities`, while `Extreme_Compositional_OOD` uses scene-level `entities`. Grounding boxes are expected as normalized `[cx, cy, w, h]` and converted to pixel `xyxy` boxes during JSONL conversion.

### Merge VG150 + CORE for robustness ablations

By default, validation remains VG150-only so calibrated VG150 reporting is not contaminated:

```bash
python3 tools/merge_vg150_core_jsonl.py \
  --vg-root datasets \
  --core-root datasets/core_vg150_jsonl \
  --out-root datasets/vg150_core_merged \
  --core-train-repeat 1
```

Run a CORE-integrated ablation after merge:

```bash
DATA_ROOT=datasets/vg150_core_merged \
RUN_NAME=vg150_core_l3_ablation \
OUT_ROOT=runs/vg150_core/l3_ablation \
SAVE_PATH=checkpoints/vg150_core_l3_ablation.pt \
bash scripts/run_pure_next.sh
```

### CORE status used in the presentation

| CORE metric | Value | Interpretation |
| --- | ---: | --- |
| Inspected scenes | 2886 | 1443 pairs, 5772 boxes, 3806 relations |
| Strict converted rows | 188 | Many rows skipped by invalid boxes/endpoints |
| Strict vocab | 266 objects / 136 predicates | No generic object/relation labels |
| CORE PredCls R@50 | 1.06% | Low due to free-form predicate mismatch |
| CORE PredCls mR@50 | 1.47% | Diagnostic only, not SOTA comparison |
| Object oracle accuracy | 100.0% | Object labels are not the failure point |
| CORE retrieval | TBD | Rerun after CLIP output patch |

## SGCls / SGDet Diagnostics

Current SGCls diagnostic indicates that predicate scoring remains healthy when labels are controlled, but object vocabulary prediction can collapse.

Presentation status:

| Setting | R@50 | mR@50 | Conclusion |
| --- | ---: | ---: | --- |
| PredCls full eval `fa375` | 67.09 | **22.64** | Main relation result |
| PredCls full eval `fa45` | **67.20** | 22.38 | Best aggregate recall |
| SGCls oracle labels | missing | missing | Rerun folder removed; rerun needed |
| SGCls CLIP top10 | 67.04 | 22.56 | Relation scores OK |
| Object classifier top1 | 0.05% | -- | Predicted `object` 12,893 times |

Policy: final SGCls should use a clean 150-class vocabulary and top-20 rerun before it is used as a defense claim.

## Presentation and Notes

The defense slides live in:

```text
notes/presentation/main.tex
```

Build from the presentation directory:

```bash
cd notes/presentation
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The current slide structure is:

1. Introduction: motivation, topic, problem definition, objectives, scope.
2. Related Work: SGG foundations, CLIP, debiasing, open-vocabulary baselines.
3. Proposed Method: PURE architecture, three losses, CORE benchmark design.
4. Experimental Results and Analysis: VG150 protocol, metrics, training results, ablations, CORE analysis.
5. Conclusion: summary, limitations, demo, references.

## Report-Ready LaTeX Table

```latex
\begin{table}[h]
\centering
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrl}
\toprule
Setting & R@50 & mR@50 & Notes \\
\midrule
PURE adapt-light fa45 & \textbf{67.20} & 22.38 & best calibrated R@50 \\
PURE adapt-light fa375 & 67.09 & \textbf{22.64} & best calibrated mR@50 \\
PURE ultra-light fa35 & 66.92 & 22.58 & strong calibrated point \\
PURE ultra-light raw e2 & 35.22 & 20.35 & best raw mR@50 \\
\bottomrule
\end{tabular}%
}
\caption{Current local PURE PredCls results. Values are percentages. Raw and calibrated rows should be interpreted separately.}
\end{table}
```

## Recommended Final Reporting Order

1. State the thesis as **SGG-first** with retrieval as an application.
2. Define the SGG input/output and PredCls / GT-pair protocol.
3. Explain PURE architecture and the three maintained losses.
4. Report raw classifier metrics before calibrated scores.
5. Present calibrated R@50/mR@50 Pareto points.
6. Use CORE for diagnostic failure analysis and compositional robustness.
7. Keep SGCls/SGDet as diagnostics unless object-vocabulary collapse is fixed.

## Practical Notes

- Do not create new Python virtual environments in this workspace.
- Keep generated datasets, checkpoints, and runs out of git unless intentionally archived.
- Use `requirements.txt` for dependency reference.
- When reporting numbers, include protocol, score mode, checkpoint, alpha/calibration setting, and evaluation batch count.
- Prefer concise notes in `notes/current_status.tex` and defense-ready slides in `notes/presentation/main.tex`.
