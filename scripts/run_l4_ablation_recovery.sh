#!/usr/bin/env bash
set -Eeuo pipefail

on_error() {
  local code=$?
  echo
  echo "[L4 ablation recovery] failed with exit code ${code} at line ${BASH_LINENO[0]}" >&2
  echo "[L4 ablation recovery] last command: ${BASH_COMMAND}" >&2
  echo "[L4 ablation recovery] inspect logs/, metrics.jsonl, and nvidia-smi before retrying." >&2
  exit "${code}"
}
trap on_error ERR

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p checkpoints logs runs

echo "[L4 ablation recovery] python=${PYTHON}"
echo "[L4 ablation recovery] cuda=${CUDA_VISIBLE_DEVICES} alloc=${PYTORCH_CUDA_ALLOC_CONF}"
nvidia-smi || true

L4_GPU_PRESET="${L4_GPU_PRESET:-l4_24gb}"
L4_MAX_IMAGES="${L4_MAX_IMAGES:-10000}"
L4_SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH:-10000}"
L4_EVAL_BATCHES="${L4_EVAL_BATCHES:-200}"
L4_BATCH_SIZE="${L4_BATCH_SIZE:-8}"
L4_ACCUM_STEPS="${L4_ACCUM_STEPS:-2}"
L4_MAX_PAIRS="${L4_MAX_PAIRS:-48}"
L4_NUM_WORKERS="${L4_NUM_WORKERS:-0}"
L4_CLIP_INPUT_RES="${L4_CLIP_INPUT_RES:-336}"
L4_REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE:-16384}"
L4_REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES:-128}"
L4_WARMUP_STEPS="${L4_WARMUP_STEPS:-100}"
L4_PURGE_OLD_RUNS="${L4_PURGE_OLD_RUNS:-false}"
L4_RUN_L1="${L4_RUN_L1:-true}"
L4_RUN_L3="${L4_RUN_L3:-true}"
L4_RUN_L4_BILINEAR="${L4_RUN_L4_BILINEAR:-false}"
L4_L1_EPOCHS="${L4_L1_EPOCHS:-4}"
L4_L3_EPOCHS="${L4_L3_EPOCHS:-4}"
L4_L4_EPOCHS="${L4_L4_EPOCHS:-2}"
L4_L1_LR="${L4_L1_LR:-8e-6}"
L4_L3_LR="${L4_L3_LR:-3e-6}"
L4_L4_LR="${L4_L4_LR:-1.5e-6}"
L4_PREDICATE_CE_LOSS="${L4_PREDICATE_CE_LOSS:-ce}"
L4_PREDICATE_CE_WEIGHT_POWER="${L4_PREDICATE_CE_WEIGHT_POWER:-0.0}"
L4_EVAL_SCORE_MODE="${L4_EVAL_SCORE_MODE:-classifier}"
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
    echo "[L4 ablation recovery] waiting for free VRAM: ${free_mb}MB < ${L4_WAIT_FOR_FREE_VRAM_MB}MB"
    nvidia-smi || true
    if [[ "${L4_WAIT_FOR_FREE_VRAM_SECONDS}" -le 0 || "${waited}" -ge "${L4_WAIT_FOR_FREE_VRAM_SECONDS}" ]]; then
      echo "[L4 ablation recovery] not enough free VRAM; stop old jobs or lower L4_* batch/pairs." >&2
      exit 2
    fi
    sleep 10
    waited=$((waited + 10))
  done
}

purge_recovery_artifacts() {
  find checkpoints -maxdepth 1 -type f \( \
    -name 'l1_spoa_ground_recovery_l4*.pt' -o \
    -name 'l3_counterfactual_recovery_l4*.pt' -o \
    -name 'l4_bilinear_recovery_l4*.pt' \
  \) -print -delete 2>/dev/null || true
  find logs -maxdepth 1 -type f \( \
    -name 'l1_spoa_ground_recovery_l4*.log' -o \
    -name 'l3_counterfactual_recovery_l4*.log' -o \
    -name 'l4_bilinear_recovery_l4*.log' \
  \) -print -delete 2>/dev/null || true
  find runs -maxdepth 1 -type d \( \
    -name 'l1_spoa_ground_recovery_l4*' -o \
    -name 'l3_counterfactual_recovery_l4*' -o \
    -name 'l4_bilinear_recovery_l4*' \
  \) -print -exec rm -rf {} + 2>/dev/null || true
}

print_config() {
  cat <<CFG
[L4 ablation recovery] protocol=branch-ramp-style
  data: max_images=${L4_MAX_IMAGES} samples_per_epoch=${L4_SAMPLES_PER_EPOCH} eval_batches=${L4_EVAL_BATCHES}
  memory: gpu_preset=${L4_GPU_PRESET} batch=${L4_BATCH_SIZE} accum=${L4_ACCUM_STEPS} max_pairs=${L4_MAX_PAIRS} res=${L4_CLIP_INPUT_RES}
  objective: ce_loss=${L4_PREDICATE_CE_LOSS} ce_weight_power=${L4_PREDICATE_CE_WEIGHT_POWER} score=${L4_EVAL_SCORE_MODE}
  stages: run_l1=${L4_RUN_L1}/${L4_L1_EPOCHS}ep lr=${L4_L1_LR}; run_l3=${L4_RUN_L3}/${L4_L3_EPOCHS}ep lr=${L4_L3_LR}; run_l4=${L4_RUN_L4_BILINEAR}/${L4_L4_EPOCHS}ep lr=${L4_L4_LR}
CFG
}

common_env() {
  STAGE=3 \
  PURE_PHASE=core \
  GPU_PRESET="${L4_GPU_PRESET}" \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS="${L4_PREDICATE_CE_LOSS}" \
  PREDICATE_CE_WEIGHT_POWER="${L4_PREDICATE_CE_WEIGHT_POWER}" \
  MAX_IMAGES="${L4_MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${L4_SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${L4_EVAL_BATCHES}" \
  BATCH_SIZE="${L4_BATCH_SIZE}" \
  ACCUM_STEPS="${L4_ACCUM_STEPS}" \
  MAX_PAIRS="${L4_MAX_PAIRS}" \
  NUM_WORKERS="${L4_NUM_WORKERS}" \
  CLIP_INPUT_RES="${L4_CLIP_INPUT_RES}" \
  REL_QUEUE_SIZE="${L4_REL_QUEUE_SIZE}" \
  REL_QUEUE_MIN_NEGATIVES="${L4_REL_QUEUE_MIN_NEGATIVES}" \
  WARMUP_STEPS="${L4_WARMUP_STEPS}" \
  LOGIT_ADJ_TAU=0.0 \
  EVAL_LOGIT_ADJ_TAU=0.0 \
  EVAL_SCORE_MODE="${L4_EVAL_SCORE_MODE}" \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  EVAL_FAST_MODE=true \
  FREQ_BIAS_ENABLED=false \
  "$@"
}

run_l1_spoa_ground() {
  common_env \
  LR="${L4_L1_LR}" \
  EPOCHS="${L4_L1_EPOCHS}" \
  LAMBDA_PREDICATE_CE=1.8 \
  LAMBDA_SPOA_ALIGNMENT=0.45 \
  LAMBDA_DENSE_GROUNDING=0.12 \
  LAMBDA_COUNTERFACTUAL=0.0 \
  LAMBDA_VISUAL_HARD_NEGATIVE=0.0 \
  VISUAL_HARD_NEGATIVE_ENABLED=false \
  PREDICATE_COUNTERFACTUAL_ENABLED=false \
  GATE_REGULARIZER_WEIGHT=0.002 \
  BILINEAR_LAYERS=0 \
  RUN_NAME=l1_spoa_ground_recovery_l4 \
  OUT_ROOT=runs/l1_spoa_ground_recovery_l4 \
  SAVE_PATH=checkpoints/l1_spoa_ground_recovery_l4.pt \
  bash scripts/run_pure_next.sh
}

run_l3_counterfactual() {
  local resume_ckpt="${L4_L3_RESUME_FROM:-checkpoints/l1_spoa_ground_recovery_l4_best_mR50.pt}"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/l1_spoa_ground_recovery_l4_best_R50.pt"
  fi
  if [[ ! -f "${resume_ckpt}" ]]; then
    echo "[L4 ablation recovery] missing L3 resume checkpoint: ${resume_ckpt}" >&2
    exit 2
  fi
  common_env \
  RESUME_FROM="${resume_ckpt}" \
  LR="${L4_L3_LR}" \
  EPOCHS="${L4_L3_EPOCHS}" \
  LAMBDA_PREDICATE_CE=1.6 \
  LAMBDA_SPOA_ALIGNMENT=0.50 \
  LAMBDA_DENSE_GROUNDING=0.18 \
  LAMBDA_COUNTERFACTUAL=0.02 \
  LAMBDA_VISUAL_HARD_NEGATIVE=0.0 \
  VISUAL_HARD_NEGATIVE_ENABLED=false \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  BILINEAR_LAYERS=0 \
  RUN_NAME=l3_counterfactual_recovery_l4 \
  OUT_ROOT=runs/l3_counterfactual_recovery_l4 \
  SAVE_PATH=checkpoints/l3_counterfactual_recovery_l4.pt \
  bash scripts/run_pure_next.sh
}

run_l4_bilinear_probe() {
  local resume_ckpt="${L4_BILINEAR_RESUME_FROM:-checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt}"
  if [[ ! -f "${resume_ckpt}" ]]; then
    resume_ckpt="checkpoints/l3_counterfactual_recovery_l4_best_R50.pt"
  fi
  if [[ ! -f "${resume_ckpt}" ]]; then
    echo "[L4 ablation recovery] missing L4 resume checkpoint: ${resume_ckpt}" >&2
    exit 2
  fi
  common_env \
  RESUME_FROM="${resume_ckpt}" \
  LR="${L4_L4_LR}" \
  EPOCHS="${L4_L4_EPOCHS}" \
  LAMBDA_PREDICATE_CE=1.3 \
  LAMBDA_SPOA_ALIGNMENT=0.40 \
  LAMBDA_DENSE_GROUNDING=0.12 \
  LAMBDA_COUNTERFACTUAL=0.02 \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.001 \
  BILINEAR_LAYERS=1 \
  BILINEAR_LOW_RANK=true \
  BILINEAR_RANK=32 \
  BILINEAR_RESIDUAL_SCALE=0.1 \
  RUN_NAME=l4_bilinear_recovery_l4 \
  OUT_ROOT=runs/l4_bilinear_recovery_l4 \
  SAVE_PATH=checkpoints/l4_bilinear_recovery_l4.pt \
  bash scripts/run_pure_next.sh
}

print_config
wait_for_vram
if [[ "${L4_PURGE_OLD_RUNS}" == "true" ]]; then
  purge_recovery_artifacts
fi

if [[ "${L4_RUN_L1}" == "true" ]]; then
  run_l1_spoa_ground
fi
if [[ "${L4_RUN_L3}" == "true" ]]; then
  run_l3_counterfactual
fi
if [[ "${L4_RUN_L4_BILINEAR}" == "true" ]]; then
  run_l4_bilinear_probe
fi

echo
echo "== recovery summary =="
python3 tools/summarize_metrics.py \
  runs/l1_spoa_ground_recovery_l4/metrics.jsonl \
  runs/l3_counterfactual_recovery_l4/metrics.jsonl \
  runs/l4_bilinear_recovery_l4/metrics.jsonl || true

echo
echo "== final best checkpoints =="
find checkpoints -maxdepth 1 -type f \( \
  -name 'l1_spoa_ground_recovery_l4_best_*.pt' -o \
  -name 'l3_counterfactual_recovery_l4_best_*.pt' -o \
  -name 'l4_bilinear_recovery_l4_best_*.pt' \
\) -printf '%TY-%Tm-%Td %TH:%TM %9s %p\n' | sort