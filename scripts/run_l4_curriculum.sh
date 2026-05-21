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

L4_GPU_PRESET="${L4_GPU_PRESET:-l4_24gb}"
L4_MAX_IMAGES="${L4_MAX_IMAGES:-10000}"
L4_SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH:-10000}"
L4_EVAL_BATCHES="${L4_EVAL_BATCHES:-150}"
L4_CE_BATCH_SIZE="${L4_CE_BATCH_SIZE:-32}"
L4_FULL_BATCH_SIZE="${L4_FULL_BATCH_SIZE:-${L4_BATCH_SIZE:-10}}"
L4_CONTINUE_BATCH_SIZE="${L4_CONTINUE_BATCH_SIZE:-${L4_FULL_BATCH_SIZE}}"
L4_ACCUM_STEPS="${L4_ACCUM_STEPS:-2}"
L4_MAX_PAIRS="${L4_MAX_PAIRS:-48}"
L4_NUM_WORKERS="${L4_NUM_WORKERS:-0}"
L4_CLIP_INPUT_RES="${L4_CLIP_INPUT_RES:-336}"
L4_REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE:-32768}"
L4_REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES:-192}"

L4_WAIT_FOR_FREE_VRAM_MB="${L4_WAIT_FOR_FREE_VRAM_MB:-18000}"
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
    echo "[L4 curriculum] waiting for free VRAM: ${free_mb}MB < ${L4_WAIT_FOR_FREE_VRAM_MB}MB"
    nvidia-smi || true
    if [[ "${L4_WAIT_FOR_FREE_VRAM_SECONDS}" -le 0 || "${waited}" -ge "${L4_WAIT_FOR_FREE_VRAM_SECONDS}" ]]; then
      echo "[L4 curriculum] not enough free VRAM; stop other GPU jobs or lower L4_* batch/pairs." >&2
      exit 2
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

echo "[L4 curriculum] preset=${L4_GPU_PRESET} images=${L4_MAX_IMAGES} samples=${L4_SAMPLES_PER_EPOCH} ce_batch=${L4_CE_BATCH_SIZE} full_batch=${L4_FULL_BATCH_SIZE} continue_batch=${L4_CONTINUE_BATCH_SIZE} accum=${L4_ACCUM_STEPS} pairs=${L4_MAX_PAIRS} res=${L4_CLIP_INPUT_RES} min_free_vram=${L4_WAIT_FOR_FREE_VRAM_MB}MB"
wait_for_vram

if [[ "${ARCHIVE_OLD_RUNS:-true}" == "true" ]]; then
  find checkpoints -maxdepth 1 -type f \( \
    -name 'stage3_ce_warmup_l4*.pt' -o \
    -name 'stage3_full_strong_l4*.pt' -o \
    -name 'stage3_stable_continue_l4*.pt' \
  \) -print -exec mv {} "archive/checkpoints_${STAMP}/" \; 2>/dev/null || true

  find logs -maxdepth 1 -type f \( \
    -name 'stage3_ce_warmup_l4*.log' -o \
    -name 'stage3_full_strong_l4*.log' -o \
    -name 'stage3_stable_continue_l4*.log' \
  \) -print -exec mv {} "archive/logs_${STAMP}/" \; 2>/dev/null || true

  find runs -maxdepth 1 -type d \( \
    -name 'stage3_ce_warmup_l4*' -o \
    -name 'stage3_full_strong_l4*' -o \
    -name 'stage3_stable_continue_l4*' \
  \) -print -exec rm -rf {} + 2>/dev/null || true
fi

run_stage3_ce_warmup() {
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${L4_GPU_PRESET}" \
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
  MAX_IMAGES="${L4_MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${L4_EVAL_BATCHES}" \
  BATCH_SIZE="${L4_CE_BATCH_SIZE}" \
  ACCUM_STEPS="${L4_ACCUM_STEPS}" \
  MAX_PAIRS="${L4_MAX_PAIRS}" \
  NUM_WORKERS="${L4_NUM_WORKERS}" \
  CLIP_INPUT_RES="${L4_CLIP_INPUT_RES}" \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage3_ce_warmup_l4 \
  OUT_ROOT=runs/stage3_ce_warmup_l4 \
  SAVE_PATH=checkpoints/stage3_ce_warmup_l4.pt \
  bash scripts/run_pure_next.sh
}

run_stage3_full_strong() {
  RESUME_FROM=checkpoints/stage3_ce_warmup_l4_best_R50.pt \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${L4_GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  LAMBDA_PREDICATE_CE=1.6 \
  LAMBDA_SPOA_ALIGNMENT=0.50 \
  LAMBDA_DENSE_GROUNDING=0.18 \
  LAMBDA_COUNTERFACTUAL=0.02 \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES}" \
  LR=3e-6 \
  EPOCHS=3 \
  MAX_IMAGES="${L4_MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${L4_EVAL_BATCHES}" \
  BATCH_SIZE="${L4_FULL_BATCH_SIZE}" \
  ACCUM_STEPS="${L4_ACCUM_STEPS}" \
  MAX_PAIRS="${L4_MAX_PAIRS}" \
  NUM_WORKERS="${L4_NUM_WORKERS}" \
  CLIP_INPUT_RES="${L4_CLIP_INPUT_RES}" \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  BILINEAR_LAYERS=0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage3_full_strong_l4 \
  OUT_ROOT=runs/stage3_full_strong_l4 \
  SAVE_PATH=checkpoints/stage3_full_strong_l4.pt \
  bash scripts/run_pure_next.sh
}

run_stage3_continue() {
  local resume_ckpt="checkpoints/stage3_full_strong_l4_best_mR50.pt"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/stage3_full_strong_l4_best_R50.pt"
  fi
  RESUME_FROM="${resume_ckpt}" \
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${L4_GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  LAMBDA_PREDICATE_CE=1.6 \
  LAMBDA_SPOA_ALIGNMENT=0.50 \
  LAMBDA_DENSE_GROUNDING=0.18 \
  LAMBDA_COUNTERFACTUAL=0.02 \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES}" \
  LR=3e-6 \
  EPOCHS=2 \
  MAX_IMAGES="${L4_MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${L4_EVAL_BATCHES}" \
  BATCH_SIZE="${L4_CONTINUE_BATCH_SIZE}" \
  ACCUM_STEPS="${L4_ACCUM_STEPS}" \
  MAX_PAIRS="${L4_MAX_PAIRS}" \
  NUM_WORKERS="${L4_NUM_WORKERS}" \
  CLIP_INPUT_RES="${L4_CLIP_INPUT_RES}" \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  BILINEAR_LAYERS=0 \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  FREQ_BIAS_ENABLED=false \
  RUN_NAME=stage3_stable_continue_l4 \
  OUT_ROOT=runs/stage3_stable_continue_l4 \
  SAVE_PATH=checkpoints/stage3_stable_continue_l4.pt \
  bash scripts/run_pure_next.sh
}

run_stage3_ce_warmup
run_stage3_full_strong
run_stage3_continue

echo
echo "== final best checkpoints =="
find checkpoints -maxdepth 1 -type f \( \
  -name 'stage3_ce_warmup_l4_best_*.pt' -o \
  -name 'stage3_full_strong_l4_best_*.pt' -o \
  -name 'stage3_stable_continue_l4_best_*.pt' \
\) -printf '%TY-%Tm-%Td %TH:%TM %9s %p\n' | sort