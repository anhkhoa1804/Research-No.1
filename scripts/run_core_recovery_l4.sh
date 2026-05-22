#!/usr/bin/env bash
set -Eeuo pipefail

on_error() {
  local code=$?
  echo
  echo "[PURE core recovery] failed with exit code ${code} at line ${BASH_LINENO[0]}" >&2
  echo "[PURE core recovery] last command: ${BASH_COMMAND}" >&2
  exit "${code}"
}
trap on_error ERR

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p checkpoints logs runs

GPU_PRESET="${GPU_PRESET:-l4_24gb}"
MAX_IMAGES="${MAX_IMAGES:-10000}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-10000}"
EVAL_BATCHES="${EVAL_BATCHES:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
MAX_PAIRS="${MAX_PAIRS:-48}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CLIP_INPUT_RES="${CLIP_INPUT_RES:-336}"
REL_QUEUE_SIZE="${REL_QUEUE_SIZE:-16384}"
REL_QUEUE_MIN_NEGATIVES="${REL_QUEUE_MIN_NEGATIVES:-128}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
RUN_L1="${RUN_L1:-true}"
RUN_L3="${RUN_L3:-true}"
L1_EPOCHS="${L1_EPOCHS:-4}"
L3_EPOCHS="${L3_EPOCHS:-4}"
L1_LR="${L1_LR:-8e-6}"
L3_LR="${L3_LR:-3e-6}"
RUN_PREFIX="${RUN_PREFIX:-}"
L1_RUN_NAME="${L1_RUN_NAME:-${RUN_PREFIX}l1_spoa_ground_recovery_l4}"
L3_RUN_NAME="${L3_RUN_NAME:-${RUN_PREFIX}l3_counterfactual_recovery_l4}"

common_env() {
  env \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_WEIGHT_POWER=0.5 \
  PREDICATE_CE_MAX_WEIGHT=5.0 \
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
  WARMUP_STEPS="${WARMUP_STEPS}" \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  EVAL_FAST_MODE=true \
  FREQ_BIAS_ENABLED=false \
  OBJECT_LANGUAGE_ANCHOR_ENABLED=false \
  RELATION_CONTEXT_LAYERS=0 \
  BILINEAR_LAYERS=0 \
  VISUAL_HARD_NEGATIVE_ENABLED=false \
  LAMBDA_VISUAL_HARD_NEGATIVE=0.0 \
  "$@"
}

run_l1() {
  common_env \
  LR="${L1_LR}" \
  EPOCHS="${L1_EPOCHS}" \
  LAMBDA_PREDICATE_CE=1.8 \
  LAMBDA_SPOA_ALIGNMENT=0.45 \
  LAMBDA_DENSE_GROUNDING=0.12 \
  LAMBDA_COUNTERFACTUAL=0.0 \
  PREDICATE_COUNTERFACTUAL_ENABLED=false \
  GATE_REGULARIZER_WEIGHT=0.002 \
  RUN_NAME="${L1_RUN_NAME}" \
  OUT_ROOT="runs/${L1_RUN_NAME}" \
  SAVE_PATH="checkpoints/${L1_RUN_NAME}.pt" \
  bash scripts/run_pure_next.sh
}

run_l3() {
  local resume_ckpt="${L3_RESUME_FROM:-checkpoints/l1_spoa_ground_recovery_l4_best_mR50.pt}"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/l1_spoa_ground_recovery_l4_best_R50.pt"
  fi
  [[ -f "${resume_ckpt}" ]] || { echo "[PURE core recovery] missing L3 resume checkpoint: ${resume_ckpt}" >&2; exit 2; }
  common_env \
  RESUME_FROM="${resume_ckpt}" \
  LR="${L3_LR}" \
  EPOCHS="${L3_EPOCHS}" \
  LAMBDA_PREDICATE_CE=1.6 \
  LAMBDA_SPOA_ALIGNMENT=0.50 \
  LAMBDA_DENSE_GROUNDING=0.18 \
  LAMBDA_COUNTERFACTUAL=0.02 \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  RUN_NAME="${L3_RUN_NAME}" \
  OUT_ROOT="runs/${L3_RUN_NAME}" \
  SAVE_PATH="checkpoints/${L3_RUN_NAME}.pt" \
  bash scripts/run_pure_next.sh
}

echo "[PURE core recovery] core-only L1->L3 protocol"
echo "  data max_images=${MAX_IMAGES} samples=${SAMPLES_PER_EPOCH} eval_batches=${EVAL_BATCHES}"
echo "  memory preset=${GPU_PRESET} batch=${BATCH_SIZE} accum=${ACCUM_STEPS} pairs=${MAX_PAIRS} res=${CLIP_INPUT_RES}"
nvidia-smi || true

if [[ "${RUN_L1}" == "true" ]]; then
  run_l1
fi
if [[ "${RUN_L3}" == "true" ]]; then
  run_l3
fi

echo
echo "== core recovery summary =="
python3 tools/summarize_metrics.py \
  runs/l1_spoa_ground_recovery_l4/metrics.jsonl \
  runs/l3_counterfactual_recovery_l4/metrics.jsonl || true