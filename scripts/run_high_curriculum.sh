#!/usr/bin/env bash
set -Eeuo pipefail

on_error() {
  local code=$?
  echo
  echo "[High curriculum] failed with exit code ${code} at line ${BASH_LINENO[0]}" >&2
  echo "[High curriculum] last command: ${BASH_COMMAND}" >&2
  exit "${code}"
}
trap on_error ERR

export PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_PRESET="${GPU_PRESET:-high_80gb}"
MAX_IMAGES="${MAX_IMAGES:-20000}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-20000}"
EVAL_BATCHES="${EVAL_BATCHES:-300}"
BATCH_SIZE="${BATCH_SIZE:-24}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
MAX_PAIRS="${MAX_PAIRS:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CLIP_INPUT_RES="${CLIP_INPUT_RES:-448}"
REL_QUEUE_SIZE="${REL_QUEUE_SIZE:-32768}"
REL_QUEUE_MIN_NEGATIVES="${REL_QUEUE_MIN_NEGATIVES:-256}"

mkdir -p checkpoints logs runs

echo "[High curriculum] gpu_preset=${GPU_PRESET} max_images=${MAX_IMAGES} samples=${SAMPLES_PER_EPOCH} eval_batches=${EVAL_BATCHES}"
nvidia-smi || true

run_stage3_ce_warmup_high() {
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=ce_only \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_WEIGHT_POWER=0.5 \
  LAMBDA_PREDICATE_CE=2.0 \
  LAMBDA_SPOA_ALIGNMENT=0.0 \
  LAMBDA_DENSE_GROUNDING=0.0 \
  LAMBDA_COUNTERFACTUAL=0.0 \
  PREDICATE_COUNTERFACTUAL_ENABLED=false \
  GATE_REGULARIZER_WEIGHT=0.0 \
  LR=8e-5 \
  EPOCHS=2 \
  MAX_IMAGES="${MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  ACCUM_STEPS="${ACCUM_STEPS}" \
  MAX_PAIRS="${MAX_PAIRS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  CLIP_INPUT_RES="${CLIP_INPUT_RES}" \
  REL_QUEUE_SIZE="${REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${REL_QUEUE_MIN_NEGATIVES}" \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage3_ce_warmup_high \
  OUT_ROOT=runs/stage3_ce_warmup_high \
  SAVE_PATH=checkpoints/stage3_ce_warmup_high.pt \
  bash scripts/run_pure_next.sh
}

run_stage3_full_l3_high() {
  local resume_ckpt="${RESUME_FROM:-checkpoints/stage3_ce_warmup_high_best_mR50.pt}"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/stage3_ce_warmup_high_best_R50.pt"
  fi
  RESUME_FROM="${resume_ckpt}" \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_WEIGHT_POWER=0.5 \
  LAMBDA_PREDICATE_CE=1.6 \
  LAMBDA_SPOA_ALIGNMENT=0.55 \
  LAMBDA_DENSE_GROUNDING=0.18 \
  LAMBDA_COUNTERFACTUAL=0.03 \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  LR=3e-6 \
  EPOCHS=5 \
  MAX_IMAGES="${MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  ACCUM_STEPS="${ACCUM_STEPS}" \
  MAX_PAIRS="${MAX_PAIRS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  CLIP_INPUT_RES="${CLIP_INPUT_RES}" \
  REL_QUEUE_SIZE="${REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${REL_QUEUE_MIN_NEGATIVES}" \
  BILINEAR_LAYERS=0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage3_full_l3_high \
  OUT_ROOT=runs/stage3_full_l3_high \
  SAVE_PATH=checkpoints/stage3_full_l3_high.pt \
  bash scripts/run_pure_next.sh
}

run_stage3_bilinear_probe_high() {
  local resume_ckpt="checkpoints/stage3_full_l3_high_best_mR50.pt"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/stage3_full_l3_high_best_R50.pt"
  fi
  RESUME_FROM="${resume_ckpt}" \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_WEIGHT_POWER=0.5 \
  LAMBDA_PREDICATE_CE=1.4 \
  LAMBDA_SPOA_ALIGNMENT=0.45 \
  LAMBDA_DENSE_GROUNDING=0.14 \
  LAMBDA_COUNTERFACTUAL=0.02 \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.001 \
  LR=1.5e-6 \
  EPOCHS=2 \
  MAX_IMAGES="${MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  ACCUM_STEPS="${ACCUM_STEPS}" \
  MAX_PAIRS="${MAX_PAIRS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  CLIP_INPUT_RES="${CLIP_INPUT_RES}" \
  REL_QUEUE_SIZE="${REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${REL_QUEUE_MIN_NEGATIVES}" \
  BILINEAR_LAYERS=1 \
  BILINEAR_LOW_RANK=true \
  BILINEAR_RANK=32 \
  BILINEAR_RESIDUAL_SCALE=0.1 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage3_bilinear_probe_high \
  OUT_ROOT=runs/stage3_bilinear_probe_high \
  SAVE_PATH=checkpoints/stage3_bilinear_probe_high.pt \
  bash scripts/run_pure_next.sh
}

run_stage3_ce_warmup_high
run_stage3_full_l3_high
run_stage3_bilinear_probe_high

python3 tools/summarize_metrics.py \
  runs/stage3_ce_warmup_high/metrics.jsonl \
  runs/stage3_full_l3_high/metrics.jsonl \
  runs/stage3_bilinear_probe_high/metrics.jsonl || true