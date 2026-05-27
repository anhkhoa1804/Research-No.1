#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT="${CKPT:-checkpoints/core_l3_balanced_adapt_light_best_mR50.pt}"
ALPHA="${FREQ_BIAS_ALPHA:-3.75}"
EVAL_BATCHES="${EVAL_BATCHES:-40}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_IMAGES="${MAX_IMAGES:-0}"
TOPK="${EVAL_SGG_CLIP_OBJ_TOPK:-10}"

run_eval() {
  local run_name="$1"
  local oracle="$2"
  local use_clip="$3"
  local use_scores="$4"
  local use_gt_pairs="$5"
  local use_dino="$6"
  local report_nograph="$7"

  echo
  echo "== diagnose ${run_name} oracle=${oracle} clip=${use_clip} obj_scores=${use_scores} gt_pairs=${use_gt_pairs} dino=${use_dino} =="
  PURE_PHASE=eval \
  RESUME_FROM="${CKPT}" \
  STAGE=3 \
  GPU_PRESET=l4_24gb \
  FREEZE_CLIP=true \
  FREQ_BIAS_ENABLED=true \
  FREQ_BIAS_ALPHA="${ALPHA}" \
  EVAL_SCORE_MODE=ensemble \
  EVAL_ENSEMBLE_ALPHA=0.0 \
  EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble \
  EVAL_ONLY=true \
  EPOCHS=0 \
  LR=0 \
  MAX_IMAGES="${MAX_IMAGES}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  EVAL_FAST_MODE=false \
  EVAL_SGG_USE_GT_PAIRS="${use_gt_pairs}" \
  EVAL_SGG_USE_CLIP_OBJ_CLASSIFIER="${use_clip}" \
  EVAL_SGG_CLIP_OBJ_TOPK="${TOPK}" \
  EVAL_SGG_SGCLS_USE_OBJ_SCORES="${use_scores}" \
  EVAL_SGG_SGCLS_ORACLE_LABELS="${oracle}" \
  EVAL_SGG_GROUNDING_DINO_ENABLED="${use_dino}" \
  EVAL_SGG_REPORT_NOGRAPH="${report_nograph}" \
  RUN_NAME="${run_name}" \
  OUT_ROOT="runs/${run_name}" \
  SAVE_PATH="checkpoints/${run_name}.pt" \
  bash scripts/run_pure_next.sh
}

run_eval diagnose_sgcls_oracle_gtlabels_eb${EVAL_BATCHES} true false false true false true
run_eval diagnose_sgcls_clip_top${TOPK}_noscores_eb${EVAL_BATCHES} false true false true false true
run_eval diagnose_sgcls_clip_top${TOPK}_scores_eb${EVAL_BATCHES} false true true true false true

if [[ "${RUN_SGDET:-false}" == "true" ]]; then
  run_eval diagnose_sgdet_dino_top${TOPK}_eb${EVAL_BATCHES} false true false false true true
fi

"${PYTHON}" tools/summarize_metrics.py runs/diagnose_*_eb${EVAL_BATCHES}/metrics.jsonl || true