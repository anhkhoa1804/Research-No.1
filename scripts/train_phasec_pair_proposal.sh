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

if [[ -z "${RESUME_FROM:-}" ]]; then
  candidates=(
    checkpoints/*v13*best_mR50.pt
    checkpoints/*best_mR50.pt
    checkpoints/*best_selection.pt
    checkpoints/*.pt
  )
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "[train_phasec_pair_proposal] no checkpoint found under checkpoints/*.pt" >&2
    exit 2
  fi
  RESUME_FROM="${candidates[0]}"
fi

if [[ ! -f "${RESUME_FROM}" ]]; then
  echo "[train_phasec_pair_proposal] resume checkpoint not found: ${RESUME_FROM}" >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-phasec_pair_proposal_l4_smoke}"
SAVE_PATH="${SAVE_PATH:-checkpoints/${RUN_NAME}.pt}"

echo "[train_phasec_pair_proposal] DATA_ROOT=${DATA_ROOT} RESUME_FROM=${RESUME_FROM} SAVE_PATH=${SAVE_PATH}"

DATA_ROOT="${DATA_ROOT}" \
RUN_NAME="${RUN_NAME}" \
SAVE_PATH="${SAVE_PATH}" \
RESUME_FROM="${RESUME_FROM}" \
RESET_EPOCH="${RESET_EPOCH:-true}" \
EPOCHS="${EPOCHS:-1}" \
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-1500}" \
MAX_IMAGES="${MAX_IMAGES:-0}" \
EVAL_BATCHES="${EVAL_BATCHES:-1}" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
ACCUM_STEPS="${ACCUM_STEPS:-2}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
FREEZE_CLIP=true \
EVAL_SGG_USE_GT_PAIRS=false \
EVAL_COMPARE_SCORE_MODES="${EVAL_COMPARE_SCORE_MODES:-classifier}" \
EVAL_SCORE_MODE="${EVAL_SCORE_MODE:-classifier}" \
EVAL_SGG_REPORT_NOGRAPH="${EVAL_SGG_REPORT_NOGRAPH:-false}" \
RELATIONNESS_PRUNE_K="${RELATIONNESS_PRUNE_K:-96}" \
LAMBDA_PREDICATE_CE=0.0 \
LAMBDA_SPOA_ALIGNMENT=0.0 \
LAMBDA_DENSE_GROUNDING=0.0 \
LAMBDA_COUNTERFACTUAL=0.0 \
LAMBDA_TEXT_PREDICATE_CE=0.0 \
LAMBDA_CALIBRATION_KL=0.0 \
LAMBDA_CALIBRATION_RANK=0.0 \
LAMBDA_CALIBRATION_REG=0.0 \
LAMBDA_RELATIONNESS="${LAMBDA_RELATIONNESS:-1.0}" \
LAMBDA_RELATIONNESS_RANK="${LAMBDA_RELATIONNESS_RANK:-0.5}" \
LAMBDA_PAIR_TOPK_SURROGATE="${LAMBDA_PAIR_TOPK_SURROGATE:-1.0}" \
PAIR_TOPK_SURROGATE_K="${PAIR_TOPK_SURROGATE_K:-96}" \
PAIR_TOPK_SURROGATE_MARGIN="${PAIR_TOPK_SURROGATE_MARGIN:-0.10}" \
LAMBDA_OBJECT_BRIDGE=0.0 \
LAMBDA_TRIPLET_RANK=0.0 \
LAMBDA_ROLE_SWAP_RANK=0.0 \
bash scripts/train_l4_phase34.sh \
  --freeze_predicate_head true \
  --freeze_non_relationness true \
  --lambda_pair_topk_surrogate "${LAMBDA_PAIR_TOPK_SURROGATE:-1.0}" \
  --pair_topk_surrogate_k "${PAIR_TOPK_SURROGATE_K:-96}" \
  --pair_topk_surrogate_margin "${PAIR_TOPK_SURROGATE_MARGIN:-0.10}"

python3 tools/sgg_gate_report.py "runs/${RUN_NAME}/metrics.jsonl"