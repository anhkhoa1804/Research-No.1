#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_CKPT="${BASE_CKPT:-checkpoints/l1_spoa_ground_recovery_l4_best_mR50.pt}"
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "[core L3 sweep] base checkpoint not found: ${BASE_CKPT}" >&2
  echo "[core L3 sweep] set BASE_CKPT to an existing checkpoint. Available candidates:" >&2
  find checkpoints -maxdepth 1 -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) \
    -printf '  %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort >&2 || true
  exit 2
fi

MAX_IMAGES="${MAX_IMAGES:-10000}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-12000}"
EVAL_BATCHES="${EVAL_BATCHES:-120}"
BATCH_SIZE="${BATCH_SIZE:-10}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
MAX_PAIRS="${MAX_PAIRS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLIP_INPUT_RES="${CLIP_INPUT_RES:-336}"

run_l3() {
  local name="$1"
  local lr="$2"
  local cf="$3"
  local spoa="$4"
  local ground="$5"
  echo
  echo "== ${name}: lr=${lr} cf=${cf} spoa=${spoa} ground=${ground} =="
  PURE_PHASE=core \
  RESUME_FROM="${BASE_CKPT}" \
  STAGE=3 \
  GPU_PRESET=l4_24gb \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_WEIGHT_POWER=0.5 \
  PREDICATE_CE_MAX_WEIGHT=5.0 \
  LAMBDA_PREDICATE_CE=1.6 \
  LAMBDA_SPOA_ALIGNMENT="${spoa}" \
  LAMBDA_DENSE_GROUNDING="${ground}" \
  LAMBDA_COUNTERFACTUAL="${cf}" \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  OBJECT_LANGUAGE_ANCHOR_ENABLED=false \
  RELATION_CONTEXT_LAYERS=0 \
  BILINEAR_LAYERS=0 \
  FREQ_BIAS_ENABLED=false \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  MAX_IMAGES="${MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  ACCUM_STEPS="${ACCUM_STEPS}" \
  MAX_PAIRS="${MAX_PAIRS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  CLIP_INPUT_RES="${CLIP_INPUT_RES}" \
  EPOCHS=3 \
  LR="${lr}" \
  RUN_NAME="${name}" \
  OUT_ROOT="runs/${name}" \
  SAVE_PATH="checkpoints/${name}.pt" \
  bash scripts/run_pure_next.sh
}

run_l3 core_l3_cf004_spoa050_g018_lr2e6 2e-6 0.04 0.50 0.18
run_l3 core_l3_cf006_spoa050_g018_lr2e6 2e-6 0.06 0.50 0.18
run_l3 core_l3_cf004_spoa035_g018_lr2e6 2e-6 0.04 0.35 0.18

echo
echo "== core L3 sweep summary =="
python3 tools/summarize_metrics.py runs/core_l3_*/metrics.jsonl || true