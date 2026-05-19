#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:-datasets}
DATASET_ID=${DATASET_ID:-anhkhoa1804/VG150-SGG-Standard}
GPU_PRESET=${GPU_PRESET:-l4_24gb}
TRAIN_IMAGES=${TRAIN_IMAGES:-5000}
VAL_IMAGES=${VAL_IMAGES:-500}
MAX_SOURCE_SCAN=${MAX_SOURCE_SCAN:-20000}
IMAGE_DOWNLOAD_TIMEOUT=${IMAGE_DOWNLOAD_TIMEOUT:-10}
MAX_IMAGES=${MAX_IMAGES:-${TRAIN_IMAGES}}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-${TRAIN_IMAGES}}
EVAL_BATCHES=${EVAL_BATCHES:-50}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
ALLOW_FALLBACK_IMAGES=${ALLOW_FALLBACK_IMAGES:-false}

mkdir -p "${DATA_ROOT}" "${CHECKPOINT_DIR}" runs

"${PYTHON}" tools/prepare_vg150_subset.py \
  --dataset_id "${DATASET_ID}" \
  --out_dir "${DATA_ROOT}" \
  --train_images "${TRAIN_IMAGES}" \
  --val_images "${VAL_IMAGES}" \
  --max_objects 32 \
  --min_relationships 1 \
  --min_predicate_coverage 10 \
  --max_source_scan "${MAX_SOURCE_SCAN}" \
  --image_download_timeout "${IMAGE_DOWNLOAD_TIMEOUT}"

CHECK_ARGS=(
  --diagnostics "${DATA_ROOT}/diagnostics.json"
  --min_train_rows "${TRAIN_IMAGES}"
  --min_val_rows "${VAL_IMAGES}"
  --min_predicate_coverage 50
  --require_no_validation_issues
)
if [[ "${ALLOW_FALLBACK_IMAGES}" == "true" ]]; then
  CHECK_ARGS+=(--allow_fallback_images)
fi
"${PYTHON}" tools/check_vg150_diagnostics.py "${CHECK_ARGS[@]}"

DATA_ROOT="${DATA_ROOT}" \
GPU_PRESET="${GPU_PRESET}" \
MAX_IMAGES="${MAX_IMAGES}" \
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH}" \
EVAL_BATCHES="${EVAL_BATCHES}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
bash scripts/run_debug_stage1.sh 2>&1 | tee runs/debug_stage1_full.log

DATA_ROOT="${DATA_ROOT}" \
GPU_PRESET="${GPU_PRESET}" \
RESUME_FROM="${CHECKPOINT_DIR}/debug_stage1.pt" \
SAVE_PATH="${CHECKPOINT_DIR}/debug_stage2.pt" \
MAX_IMAGES="${MAX_IMAGES}" \
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH}" \
EVAL_BATCHES="${EVAL_BATCHES}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
bash scripts/run_debug_stage2.sh 2>&1 | tee runs/debug_stage2_full.log

"${PYTHON}" tools/summarize_metrics.py \
  runs/debug_stage1/metrics.jsonl \
  runs/debug_stage2/metrics.jsonl