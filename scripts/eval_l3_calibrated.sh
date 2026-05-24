#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT="${CKPT:-checkpoints/l3_counterfactual_recovery_l4_best_mR50.pt}"
FREQ_BIAS_PATH="${FREQ_BIAS_PATH:-datasets/frequency_prior.json}"
FREQ_BIAS_ALPHA="${FREQ_BIAS_ALPHA:-2.25}"
EVAL_BATCHES="${EVAL_BATCHES:-200}"
MAX_IMAGES="${MAX_IMAGES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-6}"
RUN_NAME="${RUN_NAME:-eval_l3_final_a0_fa225_eb${EVAL_BATCHES}}"
CLEAN_EVAL_RUN="${CLEAN_EVAL_RUN:-false}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[PURE calibrated eval] checkpoint not found: ${CKPT}" >&2
  echo "[PURE calibrated eval] set CKPT to an existing checkpoint. Available candidates:" >&2
  find checkpoints -maxdepth 1 -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) \
    -printf '  %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort >&2 || true
  exit 2
fi
[[ -f "${FREQ_BIAS_PATH}" ]] || { echo "[PURE calibrated eval] frequency prior not found: ${FREQ_BIAS_PATH}" >&2; exit 2; }

if [[ "${CLEAN_EVAL_RUN}" == "true" ]]; then
  rm -rf "runs/${RUN_NAME}"
  rm -f "checkpoints/${RUN_NAME}.pt"
fi

PURE_PHASE=eval \
RESUME_FROM="${CKPT}" \
OBJECT_LANGUAGE_ANCHOR_ENABLED=false \
RELATION_CONTEXT_LAYERS=0 \
BILINEAR_LAYERS=0 \
FREQ_BIAS_ENABLED=true \
FREQ_BIAS_ALPHA="${FREQ_BIAS_ALPHA}" \
FREQ_BIAS_PATH="${FREQ_BIAS_PATH}" \
EVAL_SCORE_MODE=ensemble \
EVAL_ENSEMBLE_ALPHA=0.0 \
EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
MAX_IMAGES="${MAX_IMAGES}" \
EVAL_BATCHES="${EVAL_BATCHES}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
EVAL_ONLY=true \
EPOCHS=0 \
LR=0 \
FREEZE_CLIP=true \
RUN_NAME="${RUN_NAME}" \
OUT_ROOT="runs/${RUN_NAME}" \
SAVE_PATH="checkpoints/${RUN_NAME}.pt" \
bash scripts/run_pure_next.sh

python3 tools/summarize_metrics.py "runs/${RUN_NAME}/metrics.jsonl" || true