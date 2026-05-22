#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT="${CKPT:-checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt}"
FREQ_BIAS_PATH="${FREQ_BIAS_PATH:-datasets/frequency_prior.json}"
EVAL_BATCHES_SWEEP="${EVAL_BATCHES_SWEEP:-100}"
EVAL_BATCHES_CONFIRM="${EVAL_BATCHES_CONFIRM:-200}"
MAX_IMAGES="${MAX_IMAGES:-10000}"
BASE_RUN_PREFIX="${BASE_RUN_PREFIX:-eval_l3_refine}"

[[ -f "${CKPT}" ]] || { echo "[L3 eval refine] checkpoint not found: ${CKPT}" >&2; exit 2; }
[[ -f "${FREQ_BIAS_PATH}" ]] || { echo "[L3 eval refine] frequency prior not found: ${FREQ_BIAS_PATH}" >&2; exit 2; }

run_eval() {
  local name="$1"
  local eval_batches="$2"
  shift 2
  echo
  echo "== ${name} =="
  PURE_PHASE=eval \
  RESUME_FROM="${CKPT}" \
  OBJECT_LANGUAGE_ANCHOR_ENABLED=false \
  RELATION_CONTEXT_LAYERS=0 \
  BILINEAR_LAYERS=0 \
  FREQ_BIAS_ENABLED=true \
  FREQ_BIAS_PATH="${FREQ_BIAS_PATH}" \
  EVAL_SCORE_MODE=ensemble \
  EVAL_ENSEMBLE_ALPHA=0.0 \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  MAX_IMAGES="${MAX_IMAGES}" \
  EVAL_BATCHES="${eval_batches}" \
  EPOCHS=1 \
  LR=1e-8 \
  FREEZE_CLIP=true \
  RUN_NAME="${name}" \
  OUT_ROOT="runs/${name}" \
  SAVE_PATH="checkpoints/${name}.pt" \
  env "$@" bash scripts/run_pure_next.sh
}

# Confirm the current best at the larger eval budget.
run_eval "${BASE_RUN_PREFIX}_confirm_a0_fa20_eb${EVAL_BATCHES_CONFIRM}" "${EVAL_BATCHES_CONFIRM}" \
  FREQ_BIAS_ALPHA=2.0

# Probe whether the prior is still underweighted or starts over-biasing.
run_eval "${BASE_RUN_PREFIX}_a0_fa225_eb${EVAL_BATCHES_SWEEP}" "${EVAL_BATCHES_SWEEP}" \
  FREQ_BIAS_ALPHA=2.25

run_eval "${BASE_RUN_PREFIX}_a0_fa25_eb${EVAL_BATCHES_SWEEP}" "${EVAL_BATCHES_SWEEP}" \
  FREQ_BIAS_ALPHA=2.5

# Check whether a tiny classifier contribution can recover R without sacrificing too much mR.
run_eval "${BASE_RUN_PREFIX}_a01_ct20_fa20_eb${EVAL_BATCHES_SWEEP}" "${EVAL_BATCHES_SWEEP}" \
  EVAL_ENSEMBLE_ALPHA=0.1 EVAL_CLASSIFIER_TEMPERATURE=2.0 FREQ_BIAS_ALPHA=2.0

echo
echo "== calibration refine summary =="
python3 tools/summarize_metrics.py runs/${BASE_RUN_PREFIX}_*/metrics.jsonl || true