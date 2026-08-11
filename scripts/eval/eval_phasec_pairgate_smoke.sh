#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -f datasets_vg150_clean/train.jsonl && -f datasets_vg150_clean/validation.jsonl ]]; then
    DATA_ROOT="datasets_vg150_clean"
  else
    DATA_ROOT="datasets"
  fi
fi

if [[ -z "${CKPT:-}" ]]; then
  candidates=(
    checkpoints/*v13*best_mR50.pt
    checkpoints/*best_mR50.pt
    checkpoints/*best_selection.pt
    checkpoints/*.pt
  )
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "[eval_phasec_pairgate_smoke] no checkpoint found under checkpoints/*.pt" >&2
    exit 2
  fi
  CKPT="${candidates[0]}"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[eval_phasec_pairgate_smoke] checkpoint not found: ${CKPT}" >&2
  exit 2
fi

echo "[eval_phasec_pairgate_smoke] DATA_ROOT=${DATA_ROOT} CKPT=${CKPT}"

DATA_ROOT="${DATA_ROOT}" \
MAX_IMAGES="${MAX_IMAGES:-64}" \
EVAL_BATCHES="${EVAL_BATCHES:-1}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
EVAL_SGG_USE_GT_PAIRS=false \
EVAL_COMPARE_SCORE_MODES="${EVAL_COMPARE_SCORE_MODES:-classifier}" \
EVAL_SCORE_MODE="${EVAL_SCORE_MODE:-classifier}" \
EVAL_SGG_REPORT_NOGRAPH="${EVAL_SGG_REPORT_NOGRAPH:-false}" \
EVAL_SGG_USE_RELATIONNESS=true \
RELATIONNESS_PRUNE_K="${RELATIONNESS_PRUNE_K:-96}" \
CKPT="${CKPT}" \
RUN_NAME="${RUN_NAME:-phasec_pairgate_smoke}" \
bash scripts/eval/eval_l4_phase34.sh

python3 tools/sgg_gate_report.py "runs/${RUN_NAME:-phasec_pairgate_smoke}/metrics.jsonl"