#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT="${CKPT:-checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt}"
if [[ ! -f "${CKPT}" ]]; then
  echo "[L3 eval calibration] checkpoint not found: ${CKPT}" >&2
  exit 2
fi
if [[ ! -f "${FREQ_BIAS_PATH:-datasets/frequency_prior.json}" ]]; then
  echo "[L3 eval calibration] frequency prior not found: ${FREQ_BIAS_PATH:-datasets/frequency_prior.json}" >&2
  exit 2
fi

EVAL_BATCHES_SWEEP="${EVAL_BATCHES_SWEEP:-100}"
EVAL_BATCHES_CONFIRM="${EVAL_BATCHES_CONFIRM:-200}"
MAX_IMAGES="${MAX_IMAGES:-10000}"
BASE_RUN_PREFIX="${BASE_RUN_PREFIX:-eval_l3_calib}"

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
  FREQ_BIAS_PATH="${FREQ_BIAS_PATH:-datasets/frequency_prior.json}" \
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

# Current best mR direction: text-heavy + stronger frequency prior.
run_eval "${BASE_RUN_PREFIX}_a0_fa175_eb${EVAL_BATCHES_SWEEP}" "${EVAL_BATCHES_SWEEP}" \
  EVAL_SCORE_MODE=ensemble EVAL_ENSEMBLE_ALPHA=0.0 FREQ_BIAS_ALPHA=1.75

run_eval "${BASE_RUN_PREFIX}_a0_fa20_eb${EVAL_BATCHES_SWEEP}" "${EVAL_BATCHES_SWEEP}" \
  EVAL_SCORE_MODE=ensemble EVAL_ENSEMBLE_ALPHA=0.0 FREQ_BIAS_ALPHA=2.0

# Balanced candidate: keeps R@50 near 0.59 while preserving mR with softened classifier.
run_eval "${BASE_RUN_PREFIX}_a02_ct15_fa15_eb${EVAL_BATCHES_SWEEP}" "${EVAL_BATCHES_SWEEP}" \
  EVAL_SCORE_MODE=ensemble EVAL_ENSEMBLE_ALPHA=0.2 EVAL_CLASSIFIER_TEMPERATURE=1.5 FREQ_BIAS_ALPHA=1.5

# Confirm the current best known point at larger eval budget.
run_eval "${BASE_RUN_PREFIX}_confirm_a0_fa15_eb${EVAL_BATCHES_CONFIRM}" "${EVAL_BATCHES_CONFIRM}" \
  EVAL_SCORE_MODE=ensemble EVAL_ENSEMBLE_ALPHA=0.0 FREQ_BIAS_ALPHA=1.5

echo
echo "== calibration sweep summary =="
python3 tools/summarize_metrics.py runs/${BASE_RUN_PREFIX}_*/metrics.jsonl || true