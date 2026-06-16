#!/usr/bin/env bash
set -Eeuo pipefail

CKPT="${CKPT:-checkpoints/pure_l4_p3_asym_sampler_best_selection.pt}"
ALPHAS="${ALPHAS:-0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50}"
PREFIX="${PREFIX:-eval_p3_gt_pairs}"
EVAL_BATCHES="${EVAL_BATCHES:-500}"
EVAL_SGG_USE_GT_PAIRS="${EVAL_SGG_USE_GT_PAIRS:-true}"
EVAL_SGG_USE_RELATIONNESS="${EVAL_SGG_USE_RELATIONNESS:-true}"
EVAL_SCORE_MODE="${EVAL_SCORE_MODE:-classifier}"
EVAL_COMPARE_SCORE_MODES="${EVAL_COMPARE_SCORE_MODES:-classifier,text,ensemble}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[eval_calibration_sweep_l4] checkpoint not found: ${CKPT}" >&2
  exit 2
fi

mkdir -p logs runs

for alpha in ${ALPHAS}; do
  tag=$(awk -v a="${alpha}" 'BEGIN { printf "fa%03d", int(a * 1000 + 0.5) }')
  run_name="${PREFIX}_${tag}"
  echo "[eval_calibration_sweep_l4] alpha=${alpha} run=${run_name}"
  CKPT="${CKPT}" \
  RUN_NAME="${run_name}" \
  OUT_DIR="runs/${run_name}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  EVAL_SGG_USE_GT_PAIRS="${EVAL_SGG_USE_GT_PAIRS}" \
  EVAL_SGG_USE_RELATIONNESS="${EVAL_SGG_USE_RELATIONNESS}" \
  EVAL_SCORE_MODE="${EVAL_SCORE_MODE}" \
  EVAL_COMPARE_SCORE_MODES="${EVAL_COMPARE_SCORE_MODES}" \
  BAYES_CALIBRATION_WEIGHT="${alpha}" \
  FREQ_BIAS_ENABLED="${FREQ_BIAS_ENABLED:-auto}" \
  FREQ_BIAS_PATH="${FREQ_BIAS_PATH:-datasets/frequency_prior.json}" \
  bash scripts/eval_l4_phase34.sh
done