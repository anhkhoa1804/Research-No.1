#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-datasets}"
CKPT="${CKPT:-checkpoints/pure_l4_phase34.pt}"
RUN_NAME="${RUN_NAME:-eval_l4_phase34}"
OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"
EVAL_SGG_USE_GT_PAIRS="${EVAL_SGG_USE_GT_PAIRS:-false}"
EVAL_SGG_USE_RELATIONNESS="${EVAL_SGG_USE_RELATIONNESS:-true}"
EVAL_SGG_REPORT_NOGRAPH="${EVAL_SGG_REPORT_NOGRAPH:-true}"
EVAL_COMPARE_SCORE_MODES="${EVAL_COMPARE_SCORE_MODES:-classifier,text,ensemble}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[eval_l4_phase34] checkpoint not found: ${CKPT}" >&2
  exit 2
fi

mkdir -p runs logs "${OUT_DIR}"

PYTHONUNBUFFERED=1 "${PYTHON}" -u -m openvocab_rel.train \
  --stage "${STAGE:-3}" \
  --gpu_preset "${GPU_PRESET:-l4_24gb}" \
  --eval_only true \
  --epochs 0 \
  --run_name "${RUN_NAME}" \
  --out_dir "${OUT_DIR}" \
  --save_metrics_json "${OUT_DIR}/metrics.jsonl" \
  --resume true \
  --resume_from "${CKPT}" \
  --vg150_root "${DATA_ROOT}" \
  --vg150_enabled true \
  --vg150_source "${VG150_SOURCE:-local-jsonl}" \
  --device cuda \
  --batch_size "${BATCH_SIZE:-12}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --clip_input_res "${CLIP_INPUT_RES:-448}" \
  --max_images "${MAX_IMAGES:-0}" \
  --eval_batches "${EVAL_BATCHES:-500}" \
  --explicit_spoa_enabled true \
  --adaptive_calibration_enabled true \
  --predicate_label_relaxation_enabled true \
  --text_conditioned_projection_enabled true \
  --relationness_enabled true \
  --eval_sgg_predicate_score_mode "${EVAL_SCORE_MODE:-ensemble}" \
  --eval_sgg_compare_score_modes "${EVAL_COMPARE_SCORE_MODES}" \
  --eval_sgg_predicate_ensemble_alpha "${EVAL_ENSEMBLE_ALPHA:-0.45}" \
  --eval_sgg_use_gt_pairs "${EVAL_SGG_USE_GT_PAIRS}" \
  --eval_sgg_use_relationness "${EVAL_SGG_USE_RELATIONNESS}" \
  --eval_sgg_relationness_weight "${RELATIONNESS_WEIGHT:-0.75}" \
  --eval_sgg_relationness_prune_k "${RELATIONNESS_PRUNE_K:-96}" \
  --eval_sgg_detector_prune_k "${DETECTOR_PRUNE_K:-128}" \
  --eval_sgg_use_object_uncertainty true \
  --eval_sgg_object_score_power "${OBJECT_SCORE_POWER:-1.0}" \
  --eval_sgg_report_nograph "${EVAL_SGG_REPORT_NOGRAPH}" \
  --eval_sgg_grounding_dino_enabled "${EVAL_SGDET:-false}" \
  2>&1 | tee "logs/${RUN_NAME}.log"