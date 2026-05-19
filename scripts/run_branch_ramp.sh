#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:-datasets}
OUT_ROOT=${OUT_ROOT:-runs/branch_ramp}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
GPU_PRESET=${GPU_PRESET:-l4_24gb}
BRANCH_LEVEL=${BRANCH_LEVEL:-1}
if [[ -z "${RESUME_FROM+x}" ]]; then
  if [[ "${BRANCH_LEVEL}" =~ ^[0-9]+$ && "${BRANCH_LEVEL}" -ge 3 && -f "${CHECKPOINT_DIR}/branch_level3_cf005.pt" ]]; then
    RESUME_FROM="${CHECKPOINT_DIR}/branch_level3_cf005.pt"
  else
    RESUME_FROM="${CHECKPOINT_DIR}/debug_stage2.pt"
  fi
fi
SAVE_PATH=${SAVE_PATH:-${CHECKPOINT_DIR}/branch_level${BRANCH_LEVEL}.pt}
MAX_IMAGES=${MAX_IMAGES:-5000}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-5000}
EVAL_BATCHES=${EVAL_BATCHES:-50}
EVAL_ON_TRAIN_SPLIT=${EVAL_ON_TRAIN_SPLIT:-false}
SEED=${SEED:-0}
LOG_EVERY=${LOG_EVERY:-10}
TIMING_BREAKDOWN=${TIMING_BREAKDOWN:-false}
EPOCHS=${EPOCHS:-4}
LR=${LR:-2e-5}

mkdir -p "${OUT_ROOT}" "${CHECKPOINT_DIR}" logs

if [[ ! -f "${RESUME_FROM}" ]]; then
  echo "[BranchRamp] resume checkpoint not found: ${RESUME_FROM}" >&2
  echo "[BranchRamp] set RESUME_FROM=/path/to/checkpoint.pt or create the default checkpoint." >&2
  exit 2
fi

case "${BRANCH_LEVEL}" in
  0)
    RUN_NAME="branch_l0_ce_geom"
    TRAIN_OBJECTIVE="ce_only"
    LAMBDA_CE="2.0"; LAMBDA_SPOA="0.0"; LAMBDA_GROUND="0.0"; LAMBDA_CF="0.0"; LAMBDA_VIS="0.0"
    VIS_HARD="false"; PRED_CF="false"; GATE_W="0.0"; BILINEAR_LAYERS="0"; DEFORMABLE="true"
    ;;
  1)
    RUN_NAME="branch_l1_spoa_ground"
    TRAIN_OBJECTIVE="full"
    LAMBDA_CE="1.5"; LAMBDA_SPOA="0.5"; LAMBDA_GROUND="0.25"; LAMBDA_CF="0.0"; LAMBDA_VIS="0.0"
    VIS_HARD="false"; PRED_CF="false"; GATE_W="0.01"; BILINEAR_LAYERS="0"; DEFORMABLE="true"
    ;;
  2)
    RUN_NAME="branch_l2_visual_hard"
    TRAIN_OBJECTIVE="full"
    LAMBDA_CE="1.5"; LAMBDA_SPOA="0.5"; LAMBDA_GROUND="0.25"; LAMBDA_CF="0.0"; LAMBDA_VIS="0.10"
    VIS_HARD="true"; PRED_CF="false"; GATE_W="0.01"; BILINEAR_LAYERS="0"; DEFORMABLE="true"
    ;;
  3)
    RUN_NAME="branch_l3_counterfactual"
    TRAIN_OBJECTIVE="full"
    LAMBDA_CE="1.2"; LAMBDA_SPOA="0.75"; LAMBDA_GROUND="0.25"; LAMBDA_CF="0.05"; LAMBDA_VIS="0.0"
    VIS_HARD="false"; PRED_CF="true"; GATE_W="0.01"; BILINEAR_LAYERS="0"; DEFORMABLE="true"
    ;;
  4)
    RUN_NAME="branch_l4_bilinear_safe"
    TRAIN_OBJECTIVE="full"
    LAMBDA_CE="1.2"; LAMBDA_SPOA="0.75"; LAMBDA_GROUND="0.25"; LAMBDA_CF="0.05"; LAMBDA_VIS="0.0"
    VIS_HARD="false"; PRED_CF="true"; GATE_W="0.01"; BILINEAR_LAYERS="1"; DEFORMABLE="true"
    ;;
  *)
    echo "Unknown BRANCH_LEVEL=${BRANCH_LEVEL}; use 0,1,2,3,4" >&2
    exit 2
    ;;
esac

LAMBDA_VIS=${LAMBDA_VIS_OVERRIDE:-${LAMBDA_VIS}}
LAMBDA_CE=${LAMBDA_CE_OVERRIDE:-${LAMBDA_CE}}
LAMBDA_SPOA=${LAMBDA_SPOA_OVERRIDE:-${LAMBDA_SPOA}}
LAMBDA_GROUND=${LAMBDA_GROUND_OVERRIDE:-${LAMBDA_GROUND}}
LAMBDA_CF=${LAMBDA_CF_OVERRIDE:-${LAMBDA_CF}}

OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "[BranchRamp] level=${BRANCH_LEVEL} run=${RUN_NAME} resume=${RESUME_FROM} save=${SAVE_PATH} lr=${LR} epochs=${EPOCHS} lambda_ce=${LAMBDA_CE} lambda_spoa=${LAMBDA_SPOA} lambda_ground=${LAMBDA_GROUND} lambda_cf=${LAMBDA_CF} lambda_vis=${LAMBDA_VIS}"

PYTHONUNBUFFERED=1 "${PYTHON}" -u -m openvocab_rel.train \
  --stage 3 \
  --gpu_preset "${GPU_PRESET}" \
  --vg150_enabled true \
  --vg150_source local-jsonl \
  --vg150_root "${DATA_ROOT}" \
  --max_images "${MAX_IMAGES}" \
  --samples_per_epoch "${SAMPLES_PER_EPOCH}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --amp true \
  --amp_dtype bf16 \
  --resume true \
  --reset_epoch true \
  --resume_from "${RESUME_FROM}" \
  --train_objective "${TRAIN_OBJECTIVE}" \
  --lambda_predicate_ce "${LAMBDA_CE}" \
  --lambda_spoa_alignment "${LAMBDA_SPOA}" \
  --lambda_dense_grounding "${LAMBDA_GROUND}" \
  --lambda_counterfactual "${LAMBDA_CF}" \
  --lambda_visual_hard_negative "${LAMBDA_VIS}" \
  --visual_hard_negative_enabled "${VIS_HARD}" \
  --predicate_counterfactual_enabled "${PRED_CF}" \
  --gate_regularizer_weight "${GATE_W}" \
  --fusion_gate_temperature "${FUSION_GATE_TEMPERATURE:-0.7}" \
  --logit_adj_tau "${LOGIT_ADJ_TAU:-0.0}" \
  --progressive_bilinear_layers "${BILINEAR_LAYERS}" \
  --bilinear_low_rank "${BILINEAR_LOW_RANK:-true}" \
  --bilinear_residual_scale "${BILINEAR_RESIDUAL_SCALE:-0.2}" \
  --deformable_routing_enabled "${DEFORMABLE}" \
  --predicate_ce_positive_only false \
  --rel_queue_min_negatives 128 \
  --eval_fast_mode true \
  --eval_batches "${EVAL_BATCHES}" \
  --eval_on_train_split "${EVAL_ON_TRAIN_SPLIT}" \
  --eval_sgg_use_gt_pairs true \
  --eval_sgg_use_clip_obj_classifier false \
  --eval_sgg_grounding_dino_enabled false \
  --eval_sgg_report_nograph false \
  --freq_bias_enabled "${FREQ_BIAS_ENABLED:-false}" \
  --freq_bias_path "${FREQ_BIAS_PATH:-}" \
  --freq_bias_alpha "${FREQ_BIAS_ALPHA:-0.5}" \
  --freq_bias_smoothing "${FREQ_BIAS_SMOOTHING:-1.0}" \
  --eval_research_suite false \
  --log_every "${LOG_EVERY}" \
  --timing_breakdown "${TIMING_BREAKDOWN}" \
  --run_name "${RUN_NAME}" \
  --out_dir "${OUT_DIR}" \
  --save_path "${SAVE_PATH}" \
  --save_metrics_json "${OUT_DIR}/metrics.jsonl" 2>&1 | tee -a "logs/${RUN_NAME}.log"