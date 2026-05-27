#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CORE_ROOT="${CORE_ROOT:-datasets/core_benchmark}"
CORE_JSONL_ROOT="${CORE_JSONL_ROOT:-datasets/core_vg150_jsonl}"
VG_ROOT="${VG_ROOT:-datasets}"
MERGED_ROOT="${MERGED_ROOT:-datasets/vg150_core_merged}"
BASE_CKPT="${BASE_CKPT:-checkpoints/core_l3_balanced_adapt_light_best_mR50.pt}"
CORE_TRAIN_REPEAT="${CORE_TRAIN_REPEAT:-3}"
HOLDOUT_V2="${HOLDOUT_V2:-true}"
INCLUDE_CORE_VALIDATION="${INCLUDE_CORE_VALIDATION:-false}"
RUN_CORE_INSPECT="${RUN_CORE_INSPECT:-true}"
RUN_CORE_CONVERT="${RUN_CORE_CONVERT:-true}"
RUN_CORE_MERGE="${RUN_CORE_MERGE:-true}"
RUN_CORE_TRAIN="${RUN_CORE_TRAIN:-true}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[CORE finetune] python not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "[CORE finetune] base checkpoint not found: ${BASE_CKPT}" >&2
  find checkpoints -maxdepth 1 -type f -name '*.pt' -printf '  %p\n' 2>/dev/null | sort >&2 || true
  exit 2
fi

mkdir -p logs runs/core_inspect

if [[ "${RUN_CORE_INSPECT}" == "true" ]]; then
  "${PYTHON}" tools/inspect_core.py \
    --core-root "${CORE_ROOT}" \
    --report runs/core_inspect/report.json
fi

if [[ "${RUN_CORE_CONVERT}" == "true" ]]; then
  convert_args=(
    --core-root "${CORE_ROOT}"
    --out-root "${CORE_JSONL_ROOT}"
    --train-ratio "${CORE_TRAIN_RATIO:-0.80}"
    --val-ratio "${CORE_VAL_RATIO:-0.10}"
  )
  if [[ "${HOLDOUT_V2}" == "true" ]]; then
    convert_args+=(--holdout-v2)
  fi
  "${PYTHON}" tools/convert_core_to_vg150_jsonl.py "${convert_args[@]}"
fi

if [[ "${RUN_CORE_MERGE}" == "true" ]]; then
  merge_args=(
    --vg-root "${VG_ROOT}"
    --core-root "${CORE_JSONL_ROOT}"
    --out-root "${MERGED_ROOT}"
    --core-train-repeat "${CORE_TRAIN_REPEAT}"
  )
  if [[ "${INCLUDE_CORE_VALIDATION}" == "true" ]]; then
    merge_args+=(--include-core-validation)
  fi
  "${PYTHON}" tools/merge_vg150_core_jsonl.py "${merge_args[@]}"
fi

if [[ "${RUN_CORE_TRAIN}" != "true" ]]; then
  echo "[CORE finetune] prepared CORE data only. merged_root=${MERGED_ROOT}"
  exit 0
fi

MAX_IMAGES="${MAX_IMAGES:-0}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-14000}"
EVAL_BATCHES="${EVAL_BATCHES:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
MAX_PAIRS="${MAX_PAIRS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-6}"
CORE_VARIANTS="${CORE_VARIANTS:-core_ft_light core_ft_tail}" 

run_core_variant() {
  local variant="$1"
  local run_name=""
  local ce_power="0.50"
  local lambda_pred="1.2"
  local cf_weight="0.04"
  local ground_weight="0.18"
  case "${variant}" in
    core_ft_light)
      run_name="core_l3_core_ft_light"
      ce_power="0.50"; lambda_pred="1.2"; cf_weight="0.04"; ground_weight="0.18"
      ;;
    core_ft_tail)
      run_name="core_l3_core_ft_tail"
      ce_power="0.65"; lambda_pred="1.5"; cf_weight="0.06"; ground_weight="0.20"
      ;;
    core_ft_ground)
      run_name="core_l3_core_ft_ground"
      ce_power="0.50"; lambda_pred="1.2"; cf_weight="0.04"; ground_weight="0.30"
      ;;
    *)
      echo "[CORE finetune] unknown variant: ${variant}" >&2
      exit 2
      ;;
  esac

  echo
  echo "== CORE finetune ${run_name} data=${MERGED_ROOT} repeat=${CORE_TRAIN_REPEAT} =="
  DATA_ROOT="${MERGED_ROOT}" \
  PURE_PHASE=core \
  RESUME_FROM="${BASE_CKPT}" \
  STAGE=3 \
  GPU_PRESET=l4_24gb \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_GAMMA=1.5 \
  PREDICATE_CE_WEIGHT_POWER="${ce_power}" \
  PREDICATE_CE_MAX_WEIGHT=5.0 \
  LAMBDA_PREDICATE_CE="${lambda_pred}" \
  LAMBDA_SPOA_ALIGNMENT=0.50 \
  LAMBDA_DENSE_GROUNDING="${ground_weight}" \
  LAMBDA_COUNTERFACTUAL="${cf_weight}" \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  ADAPTIVE_CALIBRATION_ENABLED=true \
  ADAPTIVE_PRIOR_ENABLED=true \
  BIAS_RESIDUAL_ENABLED=true \
  ADAPTIVE_PRIOR_SCALE=0.50 \
  BIAS_RESIDUAL_SCALE=0.10 \
  LAMBDA_CALIBRATION_REG=0.003 \
  CALIBRATION_GRAD_CLIP_NORM=0.50 \
  OBJECT_LANGUAGE_ANCHOR_ENABLED=false \
  RELATION_CONTEXT_LAYERS=0 \
  BILINEAR_LAYERS=0 \
  FREQ_BIAS_ENABLED=false \
  EVAL_SCORE_MODE=classifier \
  EVAL_COMPARE_SCORE_MODES=classifier \
  MAX_IMAGES="${MAX_IMAGES}" \
  SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  ACCUM_STEPS="${ACCUM_STEPS}" \
  MAX_PAIRS="${MAX_PAIRS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  EPOCHS="${EPOCHS}" \
  LR="${LR}" \
  RUN_NAME="${run_name}" \
  OUT_ROOT="runs/${run_name}" \
  SAVE_PATH="checkpoints/${run_name}.pt" \
  bash scripts/run_pure_next.sh
}

for variant in ${CORE_VARIANTS}; do
  run_core_variant "${variant}"
done

echo
echo "== CORE finetune summary =="
"${PYTHON}" tools/summarize_metrics.py runs/core_l3_core_ft_*/metrics.jsonl || true