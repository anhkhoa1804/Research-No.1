#!/usr/bin/env bash
set -euo pipefail

# PURE-next clean path after branch-ramp diagnostics:
# - keeps L3 counterfactual + deformable routing
# - disables visual hard negatives and bilinear mixing by default
# - enables train-time logit adjustment for long-tail predicate CE
# - keeps eval-time frequency prior configurable for validation

PYTHON=${PYTHON:-python3}
DATA_ROOT=${DATA_ROOT:-datasets}
OUT_ROOT=${OUT_ROOT:-}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
GPU_PRESET=${GPU_PRESET:-l4_24gb}
STAGE=${STAGE:-3}
RESUME_FROM=${RESUME_FROM:-}
SAVE_PATH=${SAVE_PATH:-${CHECKPOINT_DIR}/pure_next_l3.pt}
MAX_IMAGES=${MAX_IMAGES:-0}
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-0}
EVAL_BATCHES=${EVAL_BATCHES:-100}
EVAL_ON_TRAIN_SPLIT=${EVAL_ON_TRAIN_SPLIT:-false}
EVAL_ONLY=${EVAL_ONLY:-false}
SEED=${SEED:-0}
LOG_EVERY=${LOG_EVERY:-50}
TIMING_BREAKDOWN=${TIMING_BREAKDOWN:-false}
EPOCHS=${EPOCHS:-8}
LR=${LR:-2e-6}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
WARMUP_STEPS=${WARMUP_STEPS:-500}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
PURE_PHASE=${PURE_PHASE:-core}
FREEZE_CLIP=${FREEZE_CLIP:-true}
PROGRESSIVE_UNFREEZE=${PROGRESSIVE_UNFREEZE:-false}

case "${PURE_PHASE}" in
  core)
    DEFAULT_OUT_ROOT=runs/pure_next_core
    DEFAULT_OBJECT_LANGUAGE_ANCHOR_ENABLED=false
    DEFAULT_RELATION_CONTEXT_LAYERS=0
    DEFAULT_LOGIT_ADJ_TAU=0.0
    DEFAULT_EVAL_LOGIT_ADJ_TAU=0.0
    DEFAULT_EVAL_SCORE_MODE=classifier
    DEFAULT_EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble
    DEFAULT_FREQ_BIAS_ENABLED=false
    ;;
  scaling)
    DEFAULT_OUT_ROOT=runs/pure_next_scaling
    DEFAULT_OBJECT_LANGUAGE_ANCHOR_ENABLED=true
    DEFAULT_RELATION_CONTEXT_LAYERS=0
    DEFAULT_LOGIT_ADJ_TAU=0.0
    DEFAULT_EVAL_LOGIT_ADJ_TAU=0.0
    DEFAULT_EVAL_SCORE_MODE=classifier
    DEFAULT_EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble
    DEFAULT_FREQ_BIAS_ENABLED=false
    ;;
  eval)
    DEFAULT_OUT_ROOT=runs/pure_next_eval
    DEFAULT_OBJECT_LANGUAGE_ANCHOR_ENABLED=true
    DEFAULT_RELATION_CONTEXT_LAYERS=0
    DEFAULT_LOGIT_ADJ_TAU=0.0
    DEFAULT_EVAL_LOGIT_ADJ_TAU=0.0
    DEFAULT_EVAL_SCORE_MODE=ensemble
    DEFAULT_EVAL_COMPARE_SCORE_MODES=classifier,text,ensemble
    DEFAULT_FREQ_BIAS_ENABLED=true
    ;;
  *)
    echo "[PURE-next] unknown PURE_PHASE=${PURE_PHASE}; expected core, scaling, or eval" >&2
    exit 2
    ;;
esac

OUT_ROOT=${OUT_ROOT:-${DEFAULT_OUT_ROOT}}
RUN_NAME=${RUN_NAME:-pure_next_l3}
LOG_PATH=${LOG_PATH:-logs/${RUN_NAME}.log}
LOGIT_ADJ_TAU=${LOGIT_ADJ_TAU:-${DEFAULT_LOGIT_ADJ_TAU}}
EVAL_LOGIT_ADJ_TAU=${EVAL_LOGIT_ADJ_TAU:-${DEFAULT_EVAL_LOGIT_ADJ_TAU}}
EVAL_SCORE_MODE=${EVAL_SCORE_MODE:-${DEFAULT_EVAL_SCORE_MODE}}
EVAL_ENSEMBLE_ALPHA=${EVAL_ENSEMBLE_ALPHA:-0.5}
EVAL_CLASSIFIER_TEMPERATURE=${EVAL_CLASSIFIER_TEMPERATURE:-1.0}
EVAL_TEXT_TEMPERATURE=${EVAL_TEXT_TEMPERATURE:-1.0}
EVAL_COMPARE_SCORE_MODES=${EVAL_COMPARE_SCORE_MODES:-${DEFAULT_EVAL_COMPARE_SCORE_MODES}}
FUSION_GATE_TEMPERATURE=${FUSION_GATE_TEMPERATURE:-0.7}
GATE_REGULARIZER_WEIGHT=${GATE_REGULARIZER_WEIGHT:-0.005}
OBJECT_LANGUAGE_ANCHOR_ENABLED=${OBJECT_LANGUAGE_ANCHOR_ENABLED:-${DEFAULT_OBJECT_LANGUAGE_ANCHOR_ENABLED}}
OBJECT_LANGUAGE_ANCHOR_SOURCE=${OBJECT_LANGUAGE_ANCHOR_SOURCE:-auto}
OBJECT_LANGUAGE_ANCHOR_ALPHA=${OBJECT_LANGUAGE_ANCHOR_ALPHA:-0.35}
RELATION_CONTEXT_LAYERS=${RELATION_CONTEXT_LAYERS:-${DEFAULT_RELATION_CONTEXT_LAYERS}}
RELATION_CONTEXT_HEADS=${RELATION_CONTEXT_HEADS:-8}
BAYES_CALIBRATION_WEIGHT=${BAYES_CALIBRATION_WEIGHT:-0.0}
ADAPTIVE_CALIBRATION_ENABLED=${ADAPTIVE_CALIBRATION_ENABLED:-false}
ADAPTIVE_PRIOR_ENABLED=${ADAPTIVE_PRIOR_ENABLED:-true}
BIAS_RESIDUAL_ENABLED=${BIAS_RESIDUAL_ENABLED:-true}
ADAPTIVE_PRIOR_SCALE=${ADAPTIVE_PRIOR_SCALE:-1.0}
BIAS_RESIDUAL_SCALE=${BIAS_RESIDUAL_SCALE:-0.25}
LAMBDA_CALIBRATION_REG=${LAMBDA_CALIBRATION_REG:-0.001}
CALIBRATION_GRAD_CLIP_NORM=${CALIBRATION_GRAD_CLIP_NORM:-0.0}
FREQ_BIAS_ENABLED=${FREQ_BIAS_ENABLED:-${DEFAULT_FREQ_BIAS_ENABLED}}
FREQ_BIAS_ALPHA=${FREQ_BIAS_ALPHA:-1.0}
FREQ_BIAS_PATH=${FREQ_BIAS_PATH:-${DATA_ROOT}/frequency_prior.json}
EVAL_SGG_USE_CLIP_OBJ_CLASSIFIER=${EVAL_SGG_USE_CLIP_OBJ_CLASSIFIER:-true}
EVAL_SGG_CLIP_OBJ_TOPK=${EVAL_SGG_CLIP_OBJ_TOPK:-5}
TRAIN_OBJECTIVE=${TRAIN_OBJECTIVE:-full}
PREDICATE_CE_POSITIVE_ONLY=${PREDICATE_CE_POSITIVE_ONLY:-false}
LAMBDA_PREDICATE_CE=${LAMBDA_PREDICATE_CE:-1.2}
LAMBDA_SPOA_ALIGNMENT=${LAMBDA_SPOA_ALIGNMENT:-0.75}
LAMBDA_DENSE_GROUNDING=${LAMBDA_DENSE_GROUNDING:-0.25}
LAMBDA_COUNTERFACTUAL=${LAMBDA_COUNTERFACTUAL:-0.05}
LAMBDA_VISUAL_HARD_NEGATIVE=${LAMBDA_VISUAL_HARD_NEGATIVE:-0.0}
VISUAL_HARD_NEGATIVE_ENABLED=${VISUAL_HARD_NEGATIVE_ENABLED:-false}
PREDICATE_COUNTERFACTUAL_ENABLED=${PREDICATE_COUNTERFACTUAL_ENABLED:-true}
PREDICATE_CE_LOSS=${PREDICATE_CE_LOSS:-focal}
PREDICATE_CE_GAMMA=${PREDICATE_CE_GAMMA:-1.5}
PREDICATE_CE_WEIGHT_POWER=${PREDICATE_CE_WEIGHT_POWER:-0.5}
PREDICATE_CE_MAX_WEIGHT=${PREDICATE_CE_MAX_WEIGHT:-5.0}
BILINEAR_LAYERS=${BILINEAR_LAYERS:-0}
BILINEAR_LOW_RANK=${BILINEAR_LOW_RANK:-false}
BILINEAR_RANK=${BILINEAR_RANK:-32}
BILINEAR_RESIDUAL_SCALE=${BILINEAR_RESIDUAL_SCALE:-0.2}
REL_QUEUE_SIZE=${REL_QUEUE_SIZE:-32768}
REL_QUEUE_MIN_NEGATIVES=${REL_QUEUE_MIN_NEGATIVES:-128}
EVAL_FAST_MODE=${EVAL_FAST_MODE:-true}
BATCH_SIZE=${BATCH_SIZE:-}
MAX_PAIRS=${MAX_PAIRS:-}
NUM_WORKERS=${NUM_WORKERS:-}
ACCUM_STEPS=${ACCUM_STEPS:-}
CLIP_INPUT_RES=${CLIP_INPUT_RES:-}

EXTRA_ARGS=()
if [[ -n "${BATCH_SIZE}" ]]; then
  EXTRA_ARGS+=(--batch_size "${BATCH_SIZE}")
fi
if [[ -n "${MAX_PAIRS}" ]]; then
  EXTRA_ARGS+=(--max_pairs "${MAX_PAIRS}")
fi
if [[ -n "${NUM_WORKERS}" ]]; then
  EXTRA_ARGS+=(--num_workers "${NUM_WORKERS}")
fi
if [[ -n "${ACCUM_STEPS}" ]]; then
  EXTRA_ARGS+=(--accum_steps "${ACCUM_STEPS}")
fi
if [[ -n "${CLIP_INPUT_RES}" ]]; then
  EXTRA_ARGS+=(--clip_input_res "${CLIP_INPUT_RES}")
fi

mkdir -p "${OUT_ROOT}" "${CHECKPOINT_DIR}" logs

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ ! -f "${RESUME_FROM}" ]]; then
    echo "[PURE-next] resume checkpoint not found: ${RESUME_FROM}" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume true --reset_epoch true --resume_from "${RESUME_FROM}")
else
  RESUME_ARGS=(--resume false)
fi

if [[ "${FREQ_BIAS_ENABLED}" == "true" && ! -f "${FREQ_BIAS_PATH}" ]]; then
  echo "[PURE-next] frequency prior not found: ${FREQ_BIAS_PATH}" >&2
  echo "[PURE-next] build it with tools/build_vg150_frequency_prior.py or set FREQ_BIAS_ENABLED=false." >&2
  exit 2
fi

if [[ "${STAGE}" != "1" && "${STAGE}" != "2" && "${STAGE}" != "3" ]]; then
  echo "[PURE-next] unknown STAGE=${STAGE}; expected 1, 2, or 3" >&2
  exit 2
fi

if [[ "${STAGE}" != "3" && -n "${RESUME_FROM}" ]]; then
  echo "[PURE-next] warning: resuming with STAGE=${STAGE}; make sure the checkpoint was trained for the same stage/protocol." >&2
fi

echo "[PURE-next] stage=${STAGE} phase=${PURE_PHASE} out=${OUT_ROOT} save=${SAVE_PATH} resume=${RESUME_FROM:-none} lr=${LR} epochs=${EPOCHS} objective=${TRAIN_OBJECTIVE} freeze_clip=${FREEZE_CLIP} anchor=${OBJECT_LANGUAGE_ANCHOR_ENABLED} rel_ctx=${RELATION_CONTEXT_LAYERS} logit_adj_tau=${LOGIT_ADJ_TAU} eval_tau=${EVAL_LOGIT_ADJ_TAU} score=${EVAL_SCORE_MODE} freq_alpha=${FREQ_BIAS_ALPHA}"

PYTHONUNBUFFERED=1 "${PYTHON}" -u -m openvocab_rel.train \
  --stage "${STAGE}" \
  --gpu_preset "${GPU_PRESET}" \
  --vg150_enabled true \
  --vg150_source local-jsonl \
  --vg150_root "${DATA_ROOT}" \
  --max_images "${MAX_IMAGES}" \
  --samples_per_epoch "${SAMPLES_PER_EPOCH}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --lr_schedule "${LR_SCHEDULE}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --freeze_clip "${FREEZE_CLIP}" \
  --progressive_unfreeze "${PROGRESSIVE_UNFREEZE}" \
  --pure_phase "${PURE_PHASE}" \
  --object_language_anchor_enabled "${OBJECT_LANGUAGE_ANCHOR_ENABLED}" \
  --object_language_anchor_source "${OBJECT_LANGUAGE_ANCHOR_SOURCE}" \
  --object_language_anchor_alpha "${OBJECT_LANGUAGE_ANCHOR_ALPHA}" \
  --relation_context_layers "${RELATION_CONTEXT_LAYERS}" \
  --relation_context_heads "${RELATION_CONTEXT_HEADS}" \
  --bayes_calibration_weight "${BAYES_CALIBRATION_WEIGHT}" \
  --adaptive_calibration_enabled "${ADAPTIVE_CALIBRATION_ENABLED}" \
  --adaptive_prior_enabled "${ADAPTIVE_PRIOR_ENABLED}" \
  --bias_residual_enabled "${BIAS_RESIDUAL_ENABLED}" \
  --adaptive_prior_scale "${ADAPTIVE_PRIOR_SCALE}" \
  --bias_residual_scale "${BIAS_RESIDUAL_SCALE}" \
  --lambda_calibration_reg "${LAMBDA_CALIBRATION_REG}" \
  --calibration_grad_clip_norm "${CALIBRATION_GRAD_CLIP_NORM}" \
  "${EXTRA_ARGS[@]}" \
  --amp true \
  --amp_dtype bf16 \
  "${RESUME_ARGS[@]}" \
  --train_objective "${TRAIN_OBJECTIVE}" \
  --lambda_predicate_ce "${LAMBDA_PREDICATE_CE}" \
  --lambda_spoa_alignment "${LAMBDA_SPOA_ALIGNMENT}" \
  --lambda_dense_grounding "${LAMBDA_DENSE_GROUNDING}" \
  --lambda_counterfactual "${LAMBDA_COUNTERFACTUAL}" \
  --lambda_visual_hard_negative "${LAMBDA_VISUAL_HARD_NEGATIVE}" \
  --visual_hard_negative_enabled "${VISUAL_HARD_NEGATIVE_ENABLED}" \
  --predicate_counterfactual_enabled "${PREDICATE_COUNTERFACTUAL_ENABLED}" \
  --gate_regularizer_weight "${GATE_REGULARIZER_WEIGHT}" \
  --fusion_gate_temperature "${FUSION_GATE_TEMPERATURE}" \
  --logit_adj_tau "${LOGIT_ADJ_TAU}" \
  --eval_logit_adj_tau "${EVAL_LOGIT_ADJ_TAU}" \
  --progressive_bilinear_layers "${BILINEAR_LAYERS}" \
  --bilinear_low_rank "${BILINEAR_LOW_RANK}" \
  --bilinear_rank "${BILINEAR_RANK}" \
  --bilinear_residual_scale "${BILINEAR_RESIDUAL_SCALE}" \
  --deformable_routing_enabled true \
  --predicate_ce_loss "${PREDICATE_CE_LOSS}" \
  --predicate_ce_gamma "${PREDICATE_CE_GAMMA}" \
  --predicate_ce_weight_power "${PREDICATE_CE_WEIGHT_POWER}" \
  --predicate_ce_max_weight "${PREDICATE_CE_MAX_WEIGHT}" \
  --predicate_ce_positive_only "${PREDICATE_CE_POSITIVE_ONLY}" \
  --rel_queue_size "${REL_QUEUE_SIZE}" \
  --rel_queue_min_negatives "${REL_QUEUE_MIN_NEGATIVES}" \
  --eval_fast_mode "${EVAL_FAST_MODE}" \
  --eval_only "${EVAL_ONLY}" \
  --eval_batches "${EVAL_BATCHES}" \
  --eval_on_train_split "${EVAL_ON_TRAIN_SPLIT}" \
  --eval_sgg_use_gt_pairs true \
  --eval_sgg_predicate_score_mode "${EVAL_SCORE_MODE}" \
  --eval_sgg_predicate_ensemble_alpha "${EVAL_ENSEMBLE_ALPHA}" \
  --eval_sgg_classifier_temperature "${EVAL_CLASSIFIER_TEMPERATURE}" \
  --eval_sgg_text_temperature "${EVAL_TEXT_TEMPERATURE}" \
  --eval_sgg_compare_score_modes "${EVAL_COMPARE_SCORE_MODES}" \
  --eval_sgg_use_clip_obj_classifier "${EVAL_SGG_USE_CLIP_OBJ_CLASSIFIER}" \
  --eval_sgg_clip_obj_topk "${EVAL_SGG_CLIP_OBJ_TOPK}" \
  --eval_sgg_grounding_dino_enabled false \
  --eval_sgg_report_nograph false \
  --freq_bias_enabled "${FREQ_BIAS_ENABLED}" \
  --freq_bias_path "${FREQ_BIAS_PATH}" \
  --freq_bias_alpha "${FREQ_BIAS_ALPHA}" \
  --freq_bias_smoothing "${FREQ_BIAS_SMOOTHING:-1.0}" \
  --eval_research_suite false \
  --log_every "${LOG_EVERY}" \
  --timing_breakdown "${TIMING_BREAKDOWN}" \
  --run_name "${RUN_NAME}" \
  --out_dir "${OUT_ROOT}" \
  --save_path "${SAVE_PATH}" \
  --save_metrics_json "${OUT_ROOT}/metrics.jsonl" 2>&1 | tee -a "${LOG_PATH}"