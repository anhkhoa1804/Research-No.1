#!/usr/bin/env bash
set -Eeuo pipefail

on_error() {
  local code=$?
  echo
  echo "[L4 branch curriculum] failed with exit code ${code} at line ${BASH_LINENO[0]}" >&2
  echo "[L4 branch curriculum] last command: ${BASH_COMMAND}" >&2
  echo "[L4 branch curriculum] inspect logs/ and nvidia-smi before retrying." >&2
  exit "${code}"
}
trap on_error ERR

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p checkpoints logs runs

echo "[L4 branch curriculum] python=${PYTHON}"
echo "[L4 branch curriculum] cuda=${CUDA_VISIBLE_DEVICES} alloc=${PYTORCH_CUDA_ALLOC_CONF}"
nvidia-smi || true

L4_GPU_PRESET="${L4_GPU_PRESET:-l4_24gb}"
L4_MAX_IMAGES="${L4_MAX_IMAGES:-3000}"
L4_SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH:-3000}"
L4_EVAL_BATCHES="${L4_EVAL_BATCHES:-50}"
L4_BATCH_SIZE="${L4_BATCH_SIZE:-8}"
L4_ACCUM_STEPS="${L4_ACCUM_STEPS:-2}"
L4_MAX_PAIRS="${L4_MAX_PAIRS:-40}"
L4_NUM_WORKERS="${L4_NUM_WORKERS:-0}"
L4_CLIP_INPUT_RES="${L4_CLIP_INPUT_RES:-336}"
L4_REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE:-8192}"
L4_REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES:-96}"
L4_WARMUP_STEPS="${L4_WARMUP_STEPS:-50}"
L4_L1_LR="${L4_L1_LR:-2e-5}"
L4_L3_LR="${L4_L3_LR:-8e-6}"
L4_L4_LR="${L4_L4_LR:-3e-6}"
L4_PURGE_OLD_RUNS="${L4_PURGE_OLD_RUNS:-true}"
L4_RUN_L3="${L4_RUN_L3:-true}"
L4_RUN_L4_BILINEAR="${L4_RUN_L4_BILINEAR:-false}"
L4_L1_EPOCHS="${L4_L1_EPOCHS:-2}"
L4_L3_EPOCHS="${L4_L3_EPOCHS:-2}"
L4_L4_EPOCHS="${L4_L4_EPOCHS:-1}"
L4_WAIT_FOR_FREE_VRAM_MB="${L4_WAIT_FOR_FREE_VRAM_MB:-16000}"
L4_WAIT_FOR_FREE_VRAM_SECONDS="${L4_WAIT_FOR_FREE_VRAM_SECONDS:-0}"

wait_for_vram() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  local waited=0
  local free_mb
  while true; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null | head -n 1 | tr -d ' ')"
    if [[ -z "${free_mb}" || "${free_mb}" -ge "${L4_WAIT_FOR_FREE_VRAM_MB}" ]]; then
      break
    fi
    echo "[L4 branch curriculum] waiting for free VRAM: ${free_mb}MB < ${L4_WAIT_FOR_FREE_VRAM_MB}MB"
    nvidia-smi || true
    if [[ "${L4_WAIT_FOR_FREE_VRAM_SECONDS}" -le 0 || "${waited}" -ge "${L4_WAIT_FOR_FREE_VRAM_SECONDS}" ]]; then
      echo "[L4 branch curriculum] not enough free VRAM; stop other GPU jobs or lower L4_* batch/pairs." >&2
      exit 2
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

purge_old_l4_artifacts() {
  find checkpoints -maxdepth 1 -type f \( \
    -name 'stage3_ce_warmup_l4*.pt' -o \
    -name 'stage3_full_strong_l4*.pt' -o \
    -name 'stage3_stable_continue_l4*.pt' -o \
    -name 'l1_spoa_ground_fast_l4*.pt' -o \
    -name 'l3_counterfactual_fast_l4*.pt' -o \
    -name 'l4_bilinear_probe_fast_l4*.pt' \
  \) -print -delete 2>/dev/null || true

  find logs -maxdepth 1 -type f \( \
    -name 'stage3_ce_warmup_l4*.log' -o \
    -name 'stage3_full_strong_l4*.log' -o \
    -name 'stage3_stable_continue_l4*.log' -o \
    -name 'l1_spoa_ground_fast_l4*.log' -o \
    -name 'l3_counterfactual_fast_l4*.log' -o \
    -name 'l4_bilinear_probe_fast_l4*.log' \
  \) -print -delete 2>/dev/null || true

  find runs -maxdepth 1 -type d \( \
    -name 'stage3_ce_warmup_l4*' -o \
    -name 'stage3_full_strong_l4*' -o \
    -name 'stage3_stable_continue_l4*' -o \
    -name 'l1_spoa_ground_fast_l4*' -o \
    -name 'l3_counterfactual_fast_l4*' -o \
    -name 'l4_bilinear_probe_fast_l4*' \
  \) -print -exec rm -rf {} + 2>/dev/null || true
}

echo "[L4 branch curriculum] preset=${L4_GPU_PRESET} images=${L4_MAX_IMAGES} samples=${L4_SAMPLES_PER_EPOCH} batch=${L4_BATCH_SIZE} accum=${L4_ACCUM_STEPS} pairs=${L4_MAX_PAIRS} eval=${L4_EVAL_BATCHES} warmup=${L4_WARMUP_STEPS} l1_lr=${L4_L1_LR} l3_lr=${L4_L3_LR} l1_epochs=${L4_L1_EPOCHS} l3_epochs=${L4_L3_EPOCHS} run_l3=${L4_RUN_L3} run_l4=${L4_RUN_L4_BILINEAR}"
wait_for_vram
if [[ "${L4_PURGE_OLD_RUNS}" == "true" ]]; then
  purge_old_l4_artifacts
fi

run_l1_spoa_ground() {
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${L4_GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_WEIGHT_POWER=0.5 \
  LAMBDA_PREDICATE_CE=1.4 \
  LAMBDA_SPOA_ALIGNMENT=0.55 \
  LAMBDA_DENSE_GROUNDING=0.20 \
  LAMBDA_COUNTERFACTUAL=0.0 \
  LAMBDA_VISUAL_HARD_NEGATIVE=0.0 \
  VISUAL_HARD_NEGATIVE_ENABLED=false \
  PREDICATE_COUNTERFACTUAL_ENABLED=false \
  GATE_REGULARIZER_WEIGHT=0.002 \
  REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES}" \
  LR="${L4_L1_LR}" \
  WARMUP_STEPS="${L4_WARMUP_STEPS}" \
  EPOCHS="${L4_L1_EPOCHS}" \
  MAX_IMAGES="${L4_MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${L4_EVAL_BATCHES}" \
  BATCH_SIZE="${L4_BATCH_SIZE}" \
  ACCUM_STEPS="${L4_ACCUM_STEPS}" \
  MAX_PAIRS="${L4_MAX_PAIRS}" \
  NUM_WORKERS="${L4_NUM_WORKERS}" \
  CLIP_INPUT_RES="${L4_CLIP_INPUT_RES}" \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  BILINEAR_LAYERS=0 \
  EVAL_SCORE_MODE=ensemble \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=l1_spoa_ground_fast_l4 \
  OUT_ROOT=runs/l1_spoa_ground_fast_l4 \
  SAVE_PATH=checkpoints/l1_spoa_ground_fast_l4.pt \
  bash scripts/run_pure_next.sh
}

run_l3_counterfactual() {
  local resume_ckpt="checkpoints/l1_spoa_ground_fast_l4_best_mR50.pt"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/l1_spoa_ground_fast_l4_best_R50.pt"
  fi
  RESUME_FROM="${resume_ckpt}" \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${L4_GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_WEIGHT_POWER=0.5 \
  LAMBDA_PREDICATE_CE=1.5 \
  LAMBDA_SPOA_ALIGNMENT=0.50 \
  LAMBDA_DENSE_GROUNDING=0.18 \
  LAMBDA_COUNTERFACTUAL=0.03 \
  LAMBDA_VISUAL_HARD_NEGATIVE=0.0 \
  VISUAL_HARD_NEGATIVE_ENABLED=false \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES}" \
  LR="${L4_L3_LR}" \
  WARMUP_STEPS="${L4_WARMUP_STEPS}" \
  EPOCHS="${L4_L3_EPOCHS}" \
  MAX_IMAGES="${L4_MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${L4_EVAL_BATCHES}" \
  BATCH_SIZE="${L4_BATCH_SIZE}" \
  ACCUM_STEPS="${L4_ACCUM_STEPS}" \
  MAX_PAIRS="${L4_MAX_PAIRS}" \
  NUM_WORKERS="${L4_NUM_WORKERS}" \
  CLIP_INPUT_RES="${L4_CLIP_INPUT_RES}" \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  BILINEAR_LAYERS=0 \
  EVAL_SCORE_MODE=ensemble \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=l3_counterfactual_fast_l4 \
  OUT_ROOT=runs/l3_counterfactual_fast_l4 \
  SAVE_PATH=checkpoints/l3_counterfactual_fast_l4.pt \
  bash scripts/run_pure_next.sh
}

run_l4_bilinear_probe() {
  local resume_ckpt="checkpoints/l3_counterfactual_fast_l4_best_mR50.pt"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/l3_counterfactual_fast_l4_best_R50.pt"
  fi
  RESUME_FROM="${resume_ckpt}" \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${L4_GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_WEIGHT_POWER=0.5 \
  LAMBDA_PREDICATE_CE=1.3 \
  LAMBDA_SPOA_ALIGNMENT=0.40 \
  LAMBDA_DENSE_GROUNDING=0.12 \
  LAMBDA_COUNTERFACTUAL=0.02 \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.001 \
  REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES}" \
  LR="${L4_L4_LR}" \
  WARMUP_STEPS="${L4_WARMUP_STEPS}" \
  EPOCHS="${L4_L4_EPOCHS}" \
  MAX_IMAGES="${L4_MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${L4_EVAL_BATCHES}" \
  BATCH_SIZE="${L4_BATCH_SIZE}" \
  ACCUM_STEPS="${L4_ACCUM_STEPS}" \
  MAX_PAIRS="${L4_MAX_PAIRS}" \
  NUM_WORKERS="${L4_NUM_WORKERS}" \
  CLIP_INPUT_RES="${L4_CLIP_INPUT_RES}" \
  BILINEAR_LAYERS=1 \
  BILINEAR_LOW_RANK=true \
  BILINEAR_RANK=32 \
  BILINEAR_RESIDUAL_SCALE=0.1 \
  EVAL_SCORE_MODE=ensemble \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=l4_bilinear_probe_fast_l4 \
  OUT_ROOT=runs/l4_bilinear_probe_fast_l4 \
  SAVE_PATH=checkpoints/l4_bilinear_probe_fast_l4.pt \
  bash scripts/run_pure_next.sh
}

run_l1_spoa_ground
if [[ "${L4_RUN_L3}" == "true" ]]; then
  run_l3_counterfactual
fi
if [[ "${L4_RUN_L4_BILINEAR}" == "true" ]]; then
  run_l4_bilinear_probe
fi

echo
echo "== final best checkpoints =="
find checkpoints -maxdepth 1 -type f \( \
  -name 'l1_spoa_ground_fast_l4_best_*.pt' -o \
  -name 'l3_counterfactual_fast_l4_best_*.pt' -o \
  -name 'l4_bilinear_probe_fast_l4_best_*.pt' \
\) -printf '%TY-%Tm-%Td %TH:%TM %9s %p\n' | sort