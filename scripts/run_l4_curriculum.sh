#!/usr/bin/env bash
set -Eeuo pipefail

on_error() {
  local code=$?
  echo
  echo "[L4 curriculum] failed with exit code ${code} at line ${BASH_LINENO[0]}" >&2
  echo "[L4 curriculum] last command: ${BASH_COMMAND}" >&2
  echo "[L4 curriculum] inspect logs/ and nvidia-smi before retrying." >&2
  exit "${code}"
}
trap on_error ERR

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "archive/checkpoints_${STAMP}" "archive/logs_${STAMP}" checkpoints logs runs

echo "[L4 curriculum] python=${PYTHON}"
echo "[L4 curriculum] cuda=${CUDA_VISIBLE_DEVICES} alloc=${PYTORCH_CUDA_ALLOC_CONF}"
nvidia-smi || true

find checkpoints -maxdepth 1 -type f \( \
  -name 'stage*.pt' -o \
  -name 'debug_*.pt' -o \
  -name 'full_safe_*.pt' -o \
  -name 'pure_next_debug*.pt' -o \
  -name 'smoke_core_best_*.pt' \
\) -print -exec mv {} "archive/checkpoints_${STAMP}/" \; 2>/dev/null || true

find logs -maxdepth 1 -type f \( \
  -name 'stage*.log' -o \
  -name 'debug_*.log' -o \
  -name 'full_safe_*.log' -o \
  -name 'pure_next_debug*.log' \
\) -print -exec mv {} "archive/logs_${STAMP}/" \; 2>/dev/null || true

find runs -maxdepth 1 -type d \( \
  -name 'stage1_*' -o \
  -name 'stage2_*' -o \
  -name 'stage3_*' -o \
  -name 'debug_*' -o \
  -name 'full_safe_*' \
\) -print -exec rm -rf {} + 2>/dev/null || true

run_stage1() {
  STAGE=1 \
  PURE_PHASE=core \
  GPU_PRESET=l4_22gb_lowmem \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=ce_only \
  PREDICATE_CE_POSITIVE_ONLY=true \
  LAMBDA_PREDICATE_CE=2.0 \
  LAMBDA_SPOA_ALIGNMENT=0.0 \
  LAMBDA_DENSE_GROUNDING=0.0 \
  LAMBDA_COUNTERFACTUAL=0.0 \
  PREDICATE_COUNTERFACTUAL_ENABLED=false \
  GATE_REGULARIZER_WEIGHT=0.0 \
  LR=1e-4 \
  EPOCHS=3 \
  MAX_IMAGES=512 \
  SAMPLES_PER_EPOCH=2048 \
  EVAL_BATCHES=50 \
  BATCH_SIZE=32 \
  MAX_PAIRS=48 \
  NUM_WORKERS=0 \
  CLIP_INPUT_RES=336 \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage1_ce_warmup_l4 \
  OUT_ROOT=runs/stage1_ce_warmup_l4 \
  SAVE_PATH=checkpoints/stage1_ce_warmup_l4.pt \
  bash scripts/run_pure_next.sh
}

run_stage2() {
  RESUME_FROM=checkpoints/stage1_ce_warmup_l4_best_R50.pt \
  STAGE=2 \
  PURE_PHASE=core \
  GPU_PRESET=l4_22gb_lowmem \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  LAMBDA_PREDICATE_CE=2.0 \
  LAMBDA_SPOA_ALIGNMENT=0.04 \
  LAMBDA_DENSE_GROUNDING=0.02 \
  LAMBDA_COUNTERFACTUAL=0.0 \
  PREDICATE_COUNTERFACTUAL_ENABLED=false \
  GATE_REGULARIZER_WEIGHT=0.0 \
  REL_QUEUE_MIN_NEGATIVES=32 \
  LR=5e-6 \
  EPOCHS=1 \
  MAX_IMAGES=768 \
  SAMPLES_PER_EPOCH=1024 \
  EVAL_BATCHES=50 \
  BATCH_SIZE=6 \
  ACCUM_STEPS=2 \
  MAX_PAIRS=32 \
  NUM_WORKERS=0 \
  CLIP_INPUT_RES=336 \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage2_light_bridge_l4 \
  OUT_ROOT=runs/stage2_light_bridge_l4 \
  SAVE_PATH=checkpoints/stage2_light_bridge_l4.pt \
  bash scripts/run_pure_next.sh
}

run_stage3_safe() {
  RESUME_FROM=checkpoints/stage2_light_bridge_l4_best_R50.pt \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET=l4_22gb_lowmem \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  LAMBDA_PREDICATE_CE=2.0 \
  LAMBDA_SPOA_ALIGNMENT=0.08 \
  LAMBDA_DENSE_GROUNDING=0.04 \
  LAMBDA_COUNTERFACTUAL=0.0 \
  PREDICATE_COUNTERFACTUAL_ENABLED=false \
  GATE_REGULARIZER_WEIGHT=0.0 \
  REL_QUEUE_MIN_NEGATIVES=32 \
  LR=3e-6 \
  EPOCHS=2 \
  MAX_IMAGES=1024 \
  SAMPLES_PER_EPOCH=1024 \
  EVAL_BATCHES=50 \
  BATCH_SIZE=4 \
  ACCUM_STEPS=2 \
  MAX_PAIRS=32 \
  NUM_WORKERS=0 \
  CLIP_INPUT_RES=336 \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage3_full_safe_l4 \
  OUT_ROOT=runs/stage3_full_safe_l4 \
  SAVE_PATH=checkpoints/stage3_full_safe_l4.pt \
  bash scripts/run_pure_next.sh
}

run_stage3_continue() {
  local resume_ckpt="checkpoints/stage3_full_safe_l4_best_mR50.pt"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/stage3_full_safe_l4_best_R50.pt"
  fi
  RESUME_FROM="${resume_ckpt}" \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET=l4_22gb_lowmem \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  LAMBDA_PREDICATE_CE=2.0 \
  LAMBDA_SPOA_ALIGNMENT=0.10 \
  LAMBDA_DENSE_GROUNDING=0.05 \
  LAMBDA_COUNTERFACTUAL=0.0 \
  PREDICATE_COUNTERFACTUAL_ENABLED=false \
  GATE_REGULARIZER_WEIGHT=0.0 \
  REL_QUEUE_MIN_NEGATIVES=32 \
  LR=2e-6 \
  EPOCHS=2 \
  MAX_IMAGES=1024 \
  SAMPLES_PER_EPOCH=1024 \
  EVAL_BATCHES=50 \
  BATCH_SIZE=4 \
  ACCUM_STEPS=2 \
  MAX_PAIRS=32 \
  NUM_WORKERS=0 \
  CLIP_INPUT_RES=336 \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage3_stable_continue_l4 \
  OUT_ROOT=runs/stage3_stable_continue_l4 \
  SAVE_PATH=checkpoints/stage3_stable_continue_l4.pt \
  bash scripts/run_pure_next.sh
}

run_stage1
run_stage2
run_stage3_safe
run_stage3_continue

echo
echo "== final best checkpoints =="
find checkpoints -maxdepth 1 -type f \( \
  -name 'stage1_ce_warmup_l4_best_*.pt' -o \
  -name 'stage2_light_bridge_l4_best_*.pt' -o \
  -name 'stage3_full_safe_l4_best_*.pt' -o \
  -name 'stage3_stable_continue_l4_best_*.pt' \
\) -printf '%TY-%Tm-%Td %TH:%TM %9s %p\n' | sort