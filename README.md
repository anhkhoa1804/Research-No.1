# PURE Relation Modeling

PURE (Predicate-aware Uncropped Relation Embedding) is a compact research codebase for VG150-style visual relation learning. The maintained path is now the phase/script-controlled PURE curriculum in `scripts/run_pure_next.sh`, plus the L4-safe wrapper `scripts/run_l4_curriculum.sh`.

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
scripts/run_pure_next.sh               maintained configurable train/eval entrypoint
scripts/run_l4_curriculum.sh           low-memory L4 curriculum runner
tools/prepare_vg150_subset.py          HF -> local JSONL/images with validation
tools/check_vg150_diagnostics.py       local dataset diagnostics guard
tools/build_vg150_frequency_prior.py   subject-object predicate prior builder
tools/summarize_metrics.py             compact metrics summary across runs
configs/presets.yaml                   readable hardware/stage preset notes
notes/current_status.tex               concise implementation/status note
```

## Current Training Protocol

Use classifier scoring as the primary debug metric. CLIP-text and ensemble scores are reported as diagnostics because text scoring can improve mean recall while hurting head-class recall.

### L4-safe full curriculum

```bash
cd /home/khoa_le1804/Research-No.1
source .venv/bin/activate
export PYTHON="$(pwd)/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash scripts/run_l4_curriculum.sh
```

Run inside tmux and keep the shell open after errors:

```bash
tmux new -s pure "bash -lc 'cd /home/khoa_le1804/Research-No.1 && source .venv/bin/activate && export PYTHON=\$(pwd)/.venv/bin/python && export CUDA_VISIBLE_DEVICES=0 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && bash scripts/run_l4_curriculum.sh; echo; echo DONE_OR_FAILED_EXIT_CODE=\$?; exec bash'"
```

Attach/detach:

```bash
tmux attach -t pure
# detach: Ctrl-b then d
```

### Main configurable entrypoint

`run_pure_next.sh` supports the current knobs through environment variables:

```bash
STAGE=3 \
PURE_PHASE=core \
GPU_PRESET=l4_22gb_lowmem \
TRAIN_OBJECTIVE=full \
PREDICATE_CE_POSITIVE_ONLY=true \
LAMBDA_PREDICATE_CE=2.0 \
LAMBDA_SPOA_ALIGNMENT=0.08 \
LAMBDA_DENSE_GROUNDING=0.04 \
LAMBDA_COUNTERFACTUAL=0.0 \
PREDICATE_COUNTERFACTUAL_ENABLED=false \
BATCH_SIZE=4 \
ACCUM_STEPS=2 \
MAX_PAIRS=32 \
NUM_WORKERS=0 \
CLIP_INPUT_RES=336 \
EVAL_SCORE_MODE=classifier \
EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
RUN_NAME=stage3_full_safe_l4 \
OUT_ROOT=runs/stage3_full_safe_l4 \
SAVE_PATH=checkpoints/stage3_full_safe_l4.pt \
bash scripts/run_pure_next.sh
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

Check diagnostics before training:

```bash
python3 tools/check_vg150_diagnostics.py \
  --diagnostics datasets/diagnostics.json \
  --min_train_rows 5000 \
  --min_val_rows 500 \
  --min_predicate_coverage 50 \
  --require_no_validation_issues
```

## Frequency Prior

Build an eval-time frequency prior separately from model training:

```bash
python3 tools/build_vg150_frequency_prior.py \
  --train_jsonl datasets/train.jsonl \
  --out_path datasets/frequency_prior.json \
  --vg150_root datasets \
  --smoothing 1.0
```

Enable it only for calibration sweeps:

```bash
FREQ_BIAS_ENABLED=true \
FREQ_BIAS_ALPHA=0.25 \
FREQ_BIAS_PATH=datasets/frequency_prior.json \
EVAL_SCORE_MODE=ensemble \
bash scripts/run_pure_next.sh
```

## Metrics

Summarize run metrics:

```bash
python3 tools/summarize_metrics.py \
  runs/stage1_ce_warmup_l4/metrics.jsonl \
  runs/stage2_light_bridge_l4/metrics.jsonl \
  runs/stage3_full_safe_l4/metrics.jsonl \
  runs/stage3_stable_continue_l4/metrics.jsonl
```

Always report `STAGE`, `TRAIN_OBJECTIVE`, `RESUME_FROM`, `EVAL_SCORE_MODE`, and whether frequency calibration is enabled.

## Checkpoint Hygiene

Keep only milestone checkpoints such as:

```text
checkpoints/best_current.pt
checkpoints/branch_level3_cf005.pt
checkpoints/stage1_ce_warmup_l4_best_R50.pt
checkpoints/stage3_full_safe_l4_best_R50.pt
checkpoints/stage3_full_safe_l4_best_mR50.pt
checkpoints/stage3_stable_continue_l4_best_R50.pt
checkpoints/stage3_stable_continue_l4_best_mR50.pt
```

List checkpoint sizes:

```bash
find checkpoints archive -type f -name '*.pt' -printf '%TY-%Tm-%Td %TH:%TM %9s %p\n' 2>/dev/null | sort
```

## Notes

- The old branch-ramp, bilinear probe, Kaggle notebook, and debug Stage 1/2 scripts were removed from the maintained workspace to avoid protocol confusion.
- Visual hard negatives and bilinear mixing are ablation ideas, not maintained default contributions.
- Use `NUM_WORKERS=0`, `CLIP_INPUT_RES=336`, and `GPU_PRESET=l4_22gb_lowmem` first on L4 if there is any OOM instability.
