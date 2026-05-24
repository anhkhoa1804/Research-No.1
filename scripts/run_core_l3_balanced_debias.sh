#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_CKPT="${BASE_CKPT:-checkpoints/core_l3_cf004_spoa050_g018_lr2e6_best_mR50.pt}"
if [[ ! -f "${BASE_CKPT}" ]]; then
  ALT="checkpoints/eval_l3_final_a0_fa225_eb200_best_mR50.pt"
  if [[ -f "${ALT}" ]]; then
    echo "[balanced debias] preferred base missing; falling back to ${ALT}" >&2
    BASE_CKPT="${ALT}"
  else
    echo "[balanced debias] base checkpoint not found: ${BASE_CKPT}" >&2
    find checkpoints -maxdepth 1 -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) \
      -printf '  %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort >&2 || true
    exit 2
  fi
fi

MAX_IMAGES="${MAX_IMAGES:-10000}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-12000}"
EVAL_BATCHES="${EVAL_BATCHES:-200}"
BATCH_SIZE="${BATCH_SIZE:-10}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
MAX_PAIRS="${MAX_PAIRS:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLIP_INPUT_RES="${CLIP_INPUT_RES:-336}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-6}"
BALANCED_VARIANTS="${BALANCED_VARIANTS:-ultra_light adapt_light prior_only}"
BALANCED_CLEAN_RUN="${BALANCED_CLEAN_RUN:-false}"

run_balanced() {
  local name="$1"
  local prior_enabled="$2"
  local bias_enabled="$3"
  local prior_scale="$4"
  local residual_scale="$5"
  local reg="$6"
  local calib_clip="$7"
  local ce_weight_power="${8:-${PREDICATE_CE_WEIGHT_POWER:-0.5}}"
  local lambda_predicate_ce="${9:-${LAMBDA_PREDICATE_CE:-1.6}}"

  if [[ "${BALANCED_CLEAN_RUN}" == "true" ]]; then
    rm -rf "runs/${name}"
    rm -f "checkpoints/${name}.pt" "checkpoints/${name}_best_mR50.pt" "checkpoints/${name}_best_R50.pt"
  fi

  echo
  echo "== ${name}: prior=${prior_enabled}:${prior_scale} residual=${bias_enabled}:${residual_scale} reg=${reg} calib_clip=${calib_clip} =="
  PURE_PHASE=core \
  RESUME_FROM="${BASE_CKPT}" \
  STAGE=3 \
  GPU_PRESET=l4_24gb \
  FREEZE_CLIP=true \
  TRAIN_OBJECTIVE=full \
  PREDICATE_CE_POSITIVE_ONLY=true \
  PREDICATE_CE_LOSS=focal \
  PREDICATE_CE_GAMMA=1.5 \
  PREDICATE_CE_WEIGHT_POWER="${ce_weight_power}" \
  PREDICATE_CE_MAX_WEIGHT=5.0 \
  LAMBDA_PREDICATE_CE="${lambda_predicate_ce}" \
  LAMBDA_SPOA_ALIGNMENT=0.50 \
  LAMBDA_DENSE_GROUNDING=0.18 \
  LAMBDA_COUNTERFACTUAL=0.04 \
  PREDICATE_COUNTERFACTUAL_ENABLED=true \
  GATE_REGULARIZER_WEIGHT=0.002 \
  ADAPTIVE_CALIBRATION_ENABLED=true \
  ADAPTIVE_PRIOR_ENABLED="${prior_enabled}" \
  BIAS_RESIDUAL_ENABLED="${bias_enabled}" \
  ADAPTIVE_PRIOR_SCALE="${prior_scale}" \
  BIAS_RESIDUAL_SCALE="${residual_scale}" \
  LAMBDA_CALIBRATION_REG="${reg}" \
  CALIBRATION_GRAD_CLIP_NORM="${calib_clip}" \
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
  CLIP_INPUT_RES="${CLIP_INPUT_RES}" \
  EPOCHS="${EPOCHS}" \
  LR="${LR}" \
  RUN_NAME="${name}" \
  OUT_ROOT="runs/${name}" \
  SAVE_PATH="checkpoints/${name}.pt" \
  bash scripts/run_pure_next.sh
}

run_variant() {
  case "$1" in
    ultra_light)
      run_balanced core_l3_balanced_ultra_light true true 0.30 0.05 0.008 0.35
      ;;
    ultra_light_long)
      run_balanced core_l3_balanced_ultra_light_long true true 0.30 0.05 0.008 0.35
      ;;
    adapt_light)
      run_balanced core_l3_balanced_adapt_light true true 0.50 0.10 0.003 0.50
      ;;
    prior_only)
      run_balanced core_l3_balanced_prior_only true false 0.35 0.00 0.006 0.35
      ;;
    bias_only)
      run_balanced core_l3_balanced_bias_only false true 0.00 0.05 0.008 0.35
      ;;
    bias_only_seed_lr5e7)
      run_balanced core_l3_seed_bias_only_lr5e7 false true 0.00 0.05 0.008 0.35
      ;;
    tail_probe_p06)
      run_balanced core_l3_seed_tail_probe_p06 true true 0.30 0.05 0.008 0.35 0.60 1.6
      ;;
    *)
      echo "[balanced debias] unknown variant: $1" >&2
      exit 2
      ;;
  esac
}

for variant in ${BALANCED_VARIANTS}; do
  run_variant "${variant}"
done

echo
echo "== balanced debias summary =="
python3 tools/summarize_metrics.py runs/core_l3_balanced_*/metrics.jsonl || true