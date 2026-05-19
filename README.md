# PURE Relation Modeling

PURE (Predicate-aware Uncropped Relation Embedding) is a compact research codebase for VG150-style visual relation learning. The maintained path is a single PURE curriculum controlled by `PURE_PHASE=core|scaling|eval`: first learn the visual router, then enable entity language anchors and relation-context message passing, then sweep Bayesian/frequency calibration. Branch-ramp and Stage 1/2 scripts are retained only for debugging, ablations, and data bring-up.

## Codebase Map

```text
openvocab_rel/train.py                 main train/eval loop
openvocab_rel/config.py                TrainConfig, stage presets, GPU presets
openvocab_rel/models/relational_model.py
                                      PURE relation model and decoder
openvocab_rel/datasets/vg150_loader.py VG150 local/HF loaders and pair construction
openvocab_rel/evals.py                 PredCls/SGCls/SGDet and diagnostic evals
openvocab_rel/losses.py                InfoNCE, queue, hard-negative losses
openvocab_rel/clip_utils.py            CLIP setup and text/image helpers
tools/prepare_vg150_subset.py          HF -> local JSONL/images with strict validation
tools/build_vg150_frequency_prior.py   subject-object predicate prior for eval-time reranking
tools/summarize_metrics.py             compact metrics summary across runs
scripts/run_branch_ramp.sh             Level 0-4 branch ablation runner
scripts/run_pure_main_l3.sh            convenience wrapper for the Level-3 path
scripts/run_pure_next.sh               clean L3/PURE path for retraining
scripts/run_debug_stage1.sh            legacy one-GPU Stage 1 debug script
scripts/run_debug_stage2.sh            legacy one-GPU Stage 2 resume debug script
configs/presets.yaml                   readable hardware/stage preset notes
notes/current_status.tex               concise implementation/status note for paper tracking
notes/paper1_pure.tex                  PURE architecture draft aligned to current code
notes/paper2_core.tex                  CORE diagnostic dataset/evaluation draft
```

## PURE Model Overview

PURE keeps dense CLIP visual context active while building relation embeddings for ordered subject-object pairs.

```text
image + object boxes
-> CLIP ViT-L/14 patch tokens
-> dense 2D feature map
-> box-conditioned object routing
-> grounded subject/object node features
-> ordered subject-object pair features
-> geometry-aware relation decoder
-> predicate classifier + CLIP-text relation scoring
```

### Dense Vision Field

`openvocab_rel/train.py` runs CLIP vision, removes the class token, and reshapes patch tokens into a dense feature map. The relation model consumes this feature map directly, so predicates can use visual context outside tight object boxes.

### Node Grounding

`RelationalModel.forward_from_featmap()` in `openvocab_rel/models/relational_model.py` receives CLIP features, boxes, and candidate pairs. The decoder initializes object queries from normalized boxes and routes them into dense visual tokens. With deformable routing enabled, each object samples learned offsets near its box instead of doing expensive full-grid attention.

### Relation Reasoning

For each ordered pair, the decoder combines:

- subject node feature
- object node feature
- Fourier geometry features from the two boxes
- edge-conditioned subject/object/relation updates
- semantic/geometry vector fusion gate
- ablation-only bilinear-safe subject-object interaction, disabled in PURE

The output relation embedding is used for text-space alignment and predicate scoring. A lightweight predicate classifier head is also available and is the main Stage 1 debug target.

### Data and Pair Construction

`openvocab_rel/datasets/vg150_loader.py` builds examples with:

- `obj_boxes`
- `obj_labels`
- ordered `pairs`
- `rel_preds`
- `rel_pred_ids`
- `rel_pos_mask`
- prompt text variants for relation alignment

Positive pairs come from VG150 relationships. Negative pairs are sampled non-relation ordered pairs controlled by `negative_pair_ratio` and `max_pairs`.

## Current Training Strategy

### Debug Branch-Ramp Path

The branch-ramp path is a diagnostic/debug workflow using `scripts/run_branch_ramp.sh` through the `scripts/run_pure_main_l3.sh` wrapper. It always enters the Stage-3 training/eval code path, then overrides branch-specific loss weights and modules. The best current base checkpoint is expected to be:

```text
checkpoints/branch_level3_cf005.pt
```

Recommended local alias:

```bash
mkdir -p checkpoints
cp checkpoints/branch_level3_cf005.pt checkpoints/best_current.pt
```

Branch meanings:

- `BRANCH_LEVEL=0`: CE + geometry baseline.
- `BRANCH_LEVEL=1`: SPOA alignment + dense grounding.
- `BRANCH_LEVEL=2`: adds visual hard negatives; telemetry is written as `visual_hard_*` fields in `metrics.jsonl`.
- `BRANCH_LEVEL=3`: counterfactual branch, no bilinear, current strongest maintained architecture.
- `BRANCH_LEVEL=4`: bilinear-safe retry using low-rank bilinear and `bilinear_residual_scale`.

### Maintained Runbook

Use the phase-controlled PURE path for report-facing experiments:

1. Build `datasets/frequency_prior.json` with fixed/scanned predicate vocab.
2. Run `PURE_PHASE=core` to warm up the visual router and relation decoder.
3. Resume with `PURE_PHASE=scaling` to enable object language anchors and relation-context message passing.
4. Run `PURE_PHASE=eval` sweeps for `BAYES_CALIBRATION_WEIGHT`/`FREQ_BIAS_ALPHA`.
5. Keep no-prior, calibrated, PredCls, and clean-SGCls metrics separated in tables.

`run_branch_ramp.sh` remains available for debugging and ablations, but it is not the maintained architecture story for the report.


### PURE Clean Path

The latest diagnostic runs reduce the maintained model to three pillars: `PredCE-LA`, `Counterfactual-SPOA`, and `Dense Grounding`. Visual hard negatives and bilinear mixing remain ablation-only code paths, but they are disabled in the clean path because they hurt mean recall in the observed runs. Clean SGCls must use CLIP object classification rather than GT object labels.

`Frequency prior` is currently the strongest confirmed boost: on the reported validation protocol, `best_current.pt + FREQ_BIAS_ALPHA=1.0` reached `PredCls R@50=0.6223` and `PredCls mR@50=0.1803`. Treat this as eval-time reranking unless a new train-time run with `LOGIT_ADJ_TAU` also improves the no-prior baseline.

Build the prior first:

```bash
python tools/build_vg150_frequency_prior.py \
  --train_jsonl datasets/train.jsonl \
  --out_path datasets/frequency_prior.json \
  --vg150_root datasets \
  --smoothing 1.0
```

Train with the maintained phase curriculum:

```bash
PURE_PHASE=core \
SAVE_PATH=checkpoints/pure_next_stage1_core.pt \
EPOCHS=4 LR=2e-5 \
FREEZE_CLIP=true \
PYTHON=python \
bash scripts/run_pure_next.sh
```

If you want to clean old artifacts while keeping the current best checkpoint for comparison:

```bash
mkdir -p checkpoints
cp checkpoints/best_current.pt best_current_backup.pt
rm -f checkpoints/*.pt
rm -rf runs/branch_ramp
mv best_current_backup.pt checkpoints/best_current.pt
```


### PURE Two-Stage Curriculum

Warm up the decoder and predicate head with CLIP frozen, then resume with a smaller LR for end-to-end fine-tuning:

```bash
PURE_PHASE=core \
SAVE_PATH=checkpoints/pure_next_stage1_core.pt \
EPOCHS=4 LR=2e-5 \
FREEZE_CLIP=true PROGRESSIVE_UNFREEZE=false \
PYTHON=python \
bash scripts/run_pure_next.sh
```

```bash
PURE_PHASE=scaling \
RESUME_FROM=checkpoints/pure_next_stage1_core.pt \
SAVE_PATH=checkpoints/pure_next_stage2_scaling.pt \
EPOCHS=6 LR=2e-6 \
FREEZE_CLIP=false PROGRESSIVE_UNFREEZE=false \
OBJECT_LANGUAGE_ANCHOR_ENABLED=true \
RELATION_CONTEXT_LAYERS=2 \
EVAL_SGG_USE_CLIP_OBJ_CLASSIFIER=true \
PYTHON=python \
bash scripts/run_pure_next.sh
```

After Stage 2, run `PURE_PHASE=eval` and sweep `BAYES_CALIBRATION_WEIGHT`/`FREQ_BIAS_ALPHA` separately from `LOGIT_ADJ_TAU`; report the full grid.

### Eval-Time Frequency Prior

`tools/build_vg150_frequency_prior.py` builds a subject/object-conditioned predicate prior from `datasets/train.jsonl`. The builder now preserves a fixed predicate vocabulary when available through `--predicate_vocab` or `--vg150_root`, so rare predicates that do not appear in a small subset are not silently dropped. Missing classes fall back to a uniform log prior rather than a large negative constant.

```bash
python tools/build_vg150_frequency_prior.py \
  --train_jsonl datasets/train.jsonl \
  --out_path datasets/frequency_prior.json \
  --vg150_root datasets \
  --smoothing 1.0
```

During eval, enable the prior with:

```bash
FREQ_BIAS_ENABLED=true \
FREQ_BIAS_ALPHA=0.25 \
FREQ_BIAS_PATH=datasets/frequency_prior.json \
RESUME_FROM=checkpoints/best_current.pt \
EPOCHS=0 EVAL_BATCHES=100 \
PYTHON=python \
bash scripts/run_pure_main_l3.sh
```

Sweep `FREQ_BIAS_ALPHA=0.25,0.50,1.00` before committing to a prior strength.

### Legacy Stage 1/2 Debug

The Stage 1/2 scripts are still useful when validating a fresh dataset or debugging predicate IDs:

- Stage 1 runs `train_objective=ce_only` with `predicate_ce_positive_only=false`.
- Stage 2 resumes from Stage 1 with SPOA alignment and grounding enabled.
- Both scripts use GT-pair, no-detector fast evaluation to avoid object-classification noise.

Use these only for data bring-up and ablation debugging; the report-facing architecture is the phase-controlled PURE path.

## Environment

Use the Python environment provided by the machine or container. Do not create a new virtual environment inside the Prism workspace unless you are outside this managed run. The GitHub repository is private, so clone with a token that has repository access.

```bash
git clone https://<GITHUB_TOKEN>@github.com/anhkhoa1804/Research-No.1.git
cd Research-No.1

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('gpu count', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

If the Hugging Face dataset requires authentication:

```bash
huggingface-cli login
```

## Prepare, Filter, and Diagnose VG150

The local dataset root is simply `datasets/`. The preparation tool writes:

```text
datasets/train.jsonl
datasets/validation.jsonl
datasets/diagnostics.json
datasets/images/*.jpg
```

The tool is strict by default. It normalizes local/global object IDs, supports dict-style subject/object refs, validates relationship indices, rejects empty predicates, maps common predicate aliases back into the standard VG150 predicate set, filters remaining non-standard predicates, downloads image URLs when only remote VG image metadata is present, checks image files, detects box/object count mismatches, and fails if predicate coverage is too low.

Use `--no_predicate_alias_map` only when measuring raw predicate labels. Use `--no_remote_images` only when intentionally allowing placeholder images. Use `--allow_unknown_predicates` only when intentionally inspecting non-standard predicate labels. Leaving it off is safer for training because unknown predicates would otherwise be written to JSONL and later collapse into the fallback `relation` class.

Debug subset for one L4:

```bash
python3 tools/prepare_vg150_subset.py \
  --dataset_id anhkhoa1804/VG150-SGG-Standard \
  --out_dir datasets \
  --train_images 5000 \
  --val_images 500 \
  --max_objects 32 \
  --min_relationships 1 \
  --min_predicate_coverage 10 \
  --max_source_scan 20000 \
  --image_download_timeout 10

cat datasets/diagnostics.json
```

Larger subset after the debug path is stable:

```bash
python3 tools/prepare_vg150_subset.py \
  --dataset_id anhkhoa1804/VG150-SGG-Standard \
  --out_dir datasets \
  --train_images 25000 \
  --val_images 2000 \
  --max_objects 32 \
  --min_relationships 1 \
  --min_predicate_coverage 30 \
  --max_source_scan 0
```

Use `--allow_validation_warnings` only when intentionally inspecting a broken sample.

Inspect raw Hugging Face schema before changing image extraction:

```bash
python3 tools/prepare_vg150_subset.py \
  --dataset_id anhkhoa1804/VG150-SGG-Standard \
  --inspect_examples 2
```

Check prepared diagnostics before training:

```bash
python3 tools/check_vg150_diagnostics.py \
  --diagnostics datasets/diagnostics.json \
  --min_train_rows 5000 \
  --min_val_rows 500 \
  --min_predicate_coverage 50 \
  --require_no_validation_issues
```


## Legacy Branch-Ramp Debug Commands

Baseline eval without frequency prior:

```bash
RESUME_FROM=checkpoints/best_current.pt \
SAVE_PATH=checkpoints/eval_l3_no_prior.pt \
EPOCHS=0 EVAL_BATCHES=100 \
FREQ_BIAS_ENABLED=false \
PYTHON=python \
bash scripts/run_pure_main_l3.sh
```

Frequency-prior alpha sweep:

```bash
for A in 0.25 0.50 1.00; do
  RESUME_FROM=checkpoints/best_current.pt \
  SAVE_PATH=checkpoints/eval_l3_freq_a${A}.pt \
  EPOCHS=0 EVAL_BATCHES=100 \
  FREQ_BIAS_ENABLED=true \
  FREQ_BIAS_ALPHA=$A \
  FREQ_BIAS_PATH=datasets/frequency_prior.json \
  PYTHON=python \
  bash scripts/run_pure_main_l3.sh
done
```

Continue Level-3 after the counterfactual-negative fix:

```bash
BRANCH_LEVEL=3 \
RESUME_FROM=checkpoints/best_current.pt \
SAVE_PATH=checkpoints/branch_l3_cf_fixed.pt \
EPOCHS=2 MAX_IMAGES=5000 SAMPLES_PER_EPOCH=5000 EVAL_BATCHES=100 \
LR=1e-6 LOG_EVERY=50 \
PYTHON=python \
bash scripts/run_pure_main_l3.sh
```

Retry Level-4 only after Level-3 is re-evaluated:

```bash
BRANCH_LEVEL=4 \
RESUME_FROM=checkpoints/branch_l3_cf_fixed.pt \
SAVE_PATH=checkpoints/branch_l4_bilinear_safe_fixed.pt \
EPOCHS=2 MAX_IMAGES=5000 SAMPLES_PER_EPOCH=5000 EVAL_BATCHES=100 \
LR=8e-7 LOG_EVERY=50 \
BILINEAR_LOW_RANK=true \
BILINEAR_RESIDUAL_SCALE=0.2 \
PYTHON=python \
bash scripts/run_pure_main_l3.sh
```

Summarize branch metrics:

```bash
python tools/summarize_metrics.py \
  runs/branch_ramp/branch_l3_counterfactual/metrics.jsonl \
  runs/branch_ramp/branch_l4_bilinear_safe/metrics.jsonl
```

## Full Debug Pipeline

Run data preparation, diagnostics checks, Stage 1, Stage 2, and a compact metrics summary in one command:

```bash
DATA_ROOT=datasets \
GPU_PRESET=l4_24gb \
TRAIN_IMAGES=5000 \
VAL_IMAGES=500 \
MAX_SOURCE_SCAN=20000 \
EVAL_BATCHES=50 \
ALLOW_FALLBACK_IMAGES=false \
bash scripts/run_debug_full.sh
```

If diagnostics fail because `fallback_images` is high, inspect the HF schema first instead of training larger runs.

## One-L4 Debug Commands

Stage 1 predicate debug:

```bash
DATA_ROOT=datasets \
GPU_PRESET=l4_24gb \
MAX_IMAGES=5000 \
SAMPLES_PER_EPOCH=5000 \
EVAL_BATCHES=50 \
bash scripts/run_debug_stage1.sh
```

Stage 2 resume debug:

```bash
DATA_ROOT=datasets \
GPU_PRESET=l4_24gb \
RESUME_FROM=checkpoints/debug_stage1.pt \
SAVE_PATH=checkpoints/debug_stage2.pt \
MAX_IMAGES=5000 \
SAMPLES_PER_EPOCH=5000 \
EVAL_BATCHES=50 \
bash scripts/run_debug_stage2.sh
```

## Direct Stage Commands

Stage 1 explicit command:

```bash
python3 -m openvocab_rel.train \
  --stage 1 \
  --gpu_preset l4_24gb \
  --vg150_enabled true \
  --vg150_source local-jsonl \
  --vg150_root datasets \
  --max_images 5000 \
  --samples_per_epoch 5000 \
  --predicate_ce_positive_only false \
  --rel_queue_min_negatives 128 \
  --train_objective ce_only \
  --lambda_predicate_ce 2.0 \
  --lambda_spoa_alignment 0.0 \
  --lambda_dense_grounding 0.0 \
  --eval_fast_mode true \
  --eval_batches 50 \
  --eval_on_train_split true \
  --eval_sgg_use_gt_pairs true \
  --eval_sgg_use_clip_obj_classifier true \
  --eval_sgg_grounding_dino_enabled false \
  --eval_sgg_report_nograph false \
  --eval_research_suite false \
  --run_name debug_stage1 \
  --out_dir runs/debug_stage1 \
  --save_path checkpoints/debug_stage1.pt \
  --save_metrics_json runs/debug_stage1/metrics.jsonl
```

Stage 2 explicit command:

```bash
python3 -m openvocab_rel.train \
  --stage 2 \
  --gpu_preset l4_24gb \
  --vg150_enabled true \
  --vg150_source local-jsonl \
  --vg150_root datasets \
  --max_images 5000 \
  --samples_per_epoch 5000 \
  --predicate_ce_positive_only false \
  --rel_queue_min_negatives 128 \
  --resume true \
  --reset_epoch true \
  --resume_from checkpoints/debug_stage1.pt \
  --eval_fast_mode true \
  --eval_batches 50 \
  --eval_on_train_split true \
  --eval_sgg_use_gt_pairs true \
  --eval_sgg_use_clip_obj_classifier true \
  --eval_sgg_grounding_dino_enabled false \
  --eval_sgg_report_nograph false \
  --eval_research_suite false \
  --run_name debug_stage2 \
  --out_dir runs/debug_stage2 \
  --save_path checkpoints/debug_stage2.pt \
  --save_metrics_json runs/debug_stage2/metrics.jsonl
```

## Debug Checklist

Before spending time on longer runs, check:

- `datasets/diagnostics.json`: non-zero rows, relationship totals, and predicate coverage.
- `datasets/diagnostics.json`: all `validation_issues` counters are zero.
- Stage 1 log: `objective=ce_only`, `eval_split=train`, and `gt_pairs=True`.
- `runs/debug_stage1/metrics.jsonl`: `positive_pairs` and `candidate_pairs` are non-zero.
- Stage 1 train-split PredCls improves before increasing subset size.
- Stage 2 starts from the intended checkpoint; `--resume true` now fails if the file is missing.
