#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EVAL_BATCHES="${EVAL_BATCHES:-200}"
MAX_IMAGES="${MAX_IMAGES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-6}"
ALPHAS="${ALPHAS:-0 0.75 1.25 1.50 1.75 2.25 2.75}"

ckpts=()
if [[ "$#" -gt 0 ]]; then
  ckpts=("$@")
else
  for stem in \
    core_l3_balanced_adapt_light \
    core_l3_balanced_adapt_mid \
    core_l3_balanced_prior_only \
    core_l3_cf004_spoa050_g018_lr2e6 \
    eval_l3_final_a0_fa225_eb200; do
    for suffix in _best_mR50 _best_R50 ""; do
      path="checkpoints/${stem}${suffix}.pt"
      [[ -f "${path}" ]] && ckpts+=("${path}") && break
    done
  done
fi

if [[ "${#ckpts[@]}" -eq 0 ]]; then
  echo "[reeval matrix] no checkpoints found. Pass checkpoint paths explicitly." >&2
  exit 2
fi

run_eval() {
  local ckpt="$1"
  local alpha="$2"
  local base
  local alpha_tag
  base="$(basename "${ckpt}" .pt)"
  alpha_tag="${alpha//./}"
  if [[ "${alpha}" == "0" || "${alpha}" == "0.0" ]]; then
    PURE_PHASE=eval \
    RESUME_FROM="${ckpt}" \
    OBJECT_LANGUAGE_ANCHOR_ENABLED=false \
    RELATION_CONTEXT_LAYERS=0 \
    BILINEAR_LAYERS=0 \
    ADAPTIVE_CALIBRATION_ENABLED=false \
    BAYES_CALIBRATION_WEIGHT=0.0 \
    FREQ_BIAS_ENABLED=false \
    FREQ_BIAS_ALPHA=0.0 \
    EVAL_SCORE_MODE=classifier \
    EVAL_ENSEMBLE_ALPHA=0.0 \
    EVAL_COMPARE_SCORE_MODES=classifier \
    MAX_IMAGES="${MAX_IMAGES}" \
    EVAL_BATCHES="${EVAL_BATCHES}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    EVAL_ONLY=true \
    EPOCHS=0 \
    LR=0 \
    FREEZE_CLIP=true \
    RUN_NAME="reeval_${base}_raw_classifier_eb${EVAL_BATCHES}" \
    OUT_ROOT="runs/reeval_${base}_raw_classifier_eb${EVAL_BATCHES}" \
    SAVE_PATH="checkpoints/reeval_${base}_raw_classifier_eb${EVAL_BATCHES}.pt" \
    bash scripts/run_pure_next.sh
  else
    CKPT="${ckpt}" \
    FREQ_BIAS_ALPHA="${alpha}" \
    EVAL_BATCHES="${EVAL_BATCHES}" \
    MAX_IMAGES="${MAX_IMAGES}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    RUN_NAME="reeval_${base}_fa${alpha_tag}_eb${EVAL_BATCHES}" \
    bash scripts/eval_l3_calibrated.sh
  fi
}

for ckpt in "${ckpts[@]}"; do
  [[ -f "${ckpt}" ]] || { echo "[reeval matrix] missing ${ckpt}" >&2; continue; }
  for alpha in ${ALPHAS}; do
    echo
    echo "== reeval ${ckpt} alpha=${alpha} =="
    run_eval "${ckpt}" "${alpha}"
  done
done

echo
echo "== reeval matrix summary =="
python3 tools/summarize_metrics.py runs/reeval_*_eb${EVAL_BATCHES}/metrics.jsonl || true