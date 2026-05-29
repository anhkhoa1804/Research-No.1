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
scripts/train_l4_phase34.sh               current L4 Phase 3/4 training runner
scripts/eval_l4_phase34.sh                current L4 Phase 3/4 checkpoint evaluation runner
tools/prepare_vg150_subset.py             HF -> local VG150 JSONL/images
tools/check_vg150_diagnostics.py          dataset diagnostics guard
tools/build_vg150_frequency_prior.py      subject-object predicate prior builder
tools/build_vg150_clean_vocab.py          clean VG150 object/predicate vocab builder
tools/summarize_metrics.py                compact metric summary across runs
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

The maintained L4 workflow targets the current Phase 3/4 configuration.

Train with default L4 settings:

```bash
bash scripts/train_l4_phase34.sh
```

Resume from a checkpoint:

```bash
RESUME_FROM=checkpoints/previous.pt RUN_NAME=pure_l4_phase34_resume bash scripts/train_l4_phase34.sh
```

Run a small pipeline smoke test:

```bash
EPOCHS=1 SAMPLES_PER_EPOCH=1000 EVAL_BATCHES=20 MAX_IMAGES=1000 bash scripts/train_l4_phase34.sh
```

Evaluate a checkpoint:

```bash
CKPT=checkpoints/pure_l4_phase34.pt EVAL_BATCHES=500 bash scripts/eval_l4_phase34.sh
```

Use `scripts/run_pure_next.sh` only when you need the lower-level configurable entrypoint.

## CORE Benchmark Workflow

CORE-specific conversion, merge, evaluation, and ablation helpers have been archived from the active workspace. The maintained code path is VG150 Phase 3/4 training/evaluation through the L4 scripts above.

Historical CORE numbers in older notes should be treated as diagnostic context only, not as an active reproducibility path in this checkout.

## SGCls / SGDet Diagnostics

SGCls / SGDet diagnostics are available through the maintained evaluation script. PredCls remains the main reported setting unless a clean SGCls/SGDet rerun is produced.

```bash
CKPT=checkpoints/pure_l4_phase34.pt EVAL_SGDET=true EVAL_BATCHES=100 bash scripts/eval_l4_phase34.sh
```

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
## PURE Phase 1/2 Upgrades

The maintained PURE path now includes the first two roadmap upgrades:

- **Phase 1 — constrained learnable calibration:** enable `--adaptive_calibration_enabled true` to train bounded pair-conditioned prior gates and bias residuals. Optional `--lambda_calibration_kl` preserves the raw predicate distribution, while `--lambda_calibration_rank` adds a differentiable positive-vs-hard-negative margin surrogate.
- **Phase 2 — explicit SPOA and relaxed labels:** `--explicit_spoa_enabled true` separates subject, predicate, object, and auxiliary geometry branches with role-asymmetric subject/object projections. `--predicate_label_relaxation_enabled true` applies CLIP-lattice soft targets for semantically ambiguous predicate negatives instead of hard-masking them.

Stage presets in `openvocab_rel/config.py` now turn on Phase 1 for Stage 1 and Phase 1+2 for Stage 2. YAML presets `pure_phase1_calibrated_spoa` and `pure_phase2_spoa_relaxed` document the standalone knobs.
Stage presets in `openvocab_rel/config.py` now turn on Phase 1 for Stage 1 and Phase 1+2 for Stage 2. Stage 3 adds Phase 3/4 hooks: text-conditioned predicate scoring, relationness supervision, relationness-pruned SGDet evaluation, and object-uncertainty-aware triplet scoring. YAML presets `pure_phase1_calibrated_spoa`, `pure_phase2_spoa_relaxed`, `l4_24gb_phase34`, and `a100_40gb_phase34` document the standalone knobs for the two target GPUs.