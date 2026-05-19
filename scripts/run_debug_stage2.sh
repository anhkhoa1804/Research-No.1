#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:-datasets}
OUT_DIR=${OUT_DIR:-runs/debug_stage2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
GPU_PRESET=${GPU_PRESET:-l4_24gb}
RESUME_FROM=${RESUME_FROM:-checkpoints/debug_stage1.pt}
SAVE_PATH=${SAVE_PATH:-checkpoints/debug_stage2.pt}
MAX_IMAGES=${MAX_IMAGES:-5000}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-5000}
EVAL_BATCHES=${EVAL_BATCHES:-50}
SEED=${SEED:-0}

mkdir -p "${OUT_DIR}" "${CHECKPOINT_DIR}"

"${PYTHON}" -m openvocab_rel.train \
  --stage 2 \
  --gpu_preset "${GPU_PRESET}" \
  --vg150_enabled true \
  --vg150_source local-jsonl \
  --vg150_root "${DATA_ROOT}" \
  --max_images "${MAX_IMAGES}" \
  --samples_per_epoch "${SAMPLES_PER_EPOCH}" \
  --seed "${SEED}" \
  --amp true \
  --amp_dtype bf16 \
  --predicate_ce_positive_only false \
  --rel_queue_min_negatives 128 \
  --resume true \
  --reset_epoch true \
  --resume_from "${RESUME_FROM}" \
  --eval_fast_mode true \
  --eval_batches "${EVAL_BATCHES}" \
  --eval_on_train_split true \
  --eval_sgg_use_gt_pairs true \
  --eval_sgg_use_clip_obj_classifier false \
  --eval_sgg_grounding_dino_enabled false \
  --eval_sgg_report_nograph false \
  --eval_research_suite false \
  --log_every 25 \
  --run_name debug_stage2 \
  --out_dir "${OUT_DIR}" \
  --save_path "${SAVE_PATH}" \
  --save_metrics_json "${OUT_DIR}/metrics.jsonl"