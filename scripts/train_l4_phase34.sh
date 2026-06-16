#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-datasets}"
RUN_NAME="${RUN_NAME:-pure_l4_phase34}"
OUT_DIR="${OUT_DIR:-runs/${RUN_NAME}}"
SAVE_PATH="${SAVE_PATH:-checkpoints/${RUN_NAME}.pt}"
RESUME_FROM="${RESUME_FROM:-}"

mkdir -p checkpoints runs logs "${OUT_DIR}"

ARGS=(
  -u -m openvocab_rel.train
  --stage "${STAGE:-3}"
  --gpu_preset "${GPU_PRESET:-l4_24gb}"
  --run_name "${RUN_NAME}"
  --out_dir "${OUT_DIR}"
  --save_path "${SAVE_PATH}"
  --save_metrics_json "${OUT_DIR}/metrics.jsonl"
  --vg150_root "${DATA_ROOT}"
  --vg150_enabled true
  --vg150_source "${VG150_SOURCE:-local-jsonl}"
  --device cuda
  --epochs "${EPOCHS:-6}"
  --batch_size "${BATCH_SIZE:-12}"
  --accum_steps "${ACCUM_STEPS:-2}"
  --num_workers "${NUM_WORKERS:-4}"
  --clip_input_res "${CLIP_INPUT_RES:-448}"
  --max_images "${MAX_IMAGES:-0}"
  --samples_per_epoch "${SAMPLES_PER_EPOCH:-12000}"
  --max_objects "${MAX_OBJECTS:-32}"
  --max_pairs "${MAX_PAIRS:-64}"
  --learned_prune_k "${LEARNED_PRUNE_K:-64}"
  --use_all_pairs "${USE_ALL_PAIRS:-false}"
  --negative_pair_ratio "${NEGATIVE_PAIR_RATIO:-2.0}"
  --lr "${LR:-2e-5}"
  --lr_schedule "${LR_SCHEDULE:-cosine}"
  --warmup_steps "${WARMUP_STEPS:-300}"
  --amp true
  --amp_dtype bf16
  --channels_last true
  --gradient_checkpointing true
  --freeze_clip "${FREEZE_CLIP:-false}"
  --progressive_unfreeze "${PROGRESSIVE_UNFREEZE:-false}"
  --train_objective full
  --explicit_spoa_enabled true
  --adaptive_calibration_enabled true
  --adaptive_prior_enabled true
  --bias_residual_enabled true
  --predicate_label_relaxation_enabled true
  --predicate_group_relaxation_enabled "${PREDICATE_GROUP_RELAXATION_ENABLED:-false}"
  --predicate_group_relaxation_epsilon "${PREDICATE_GROUP_RELAXATION_EPSILON:-0.04}"
  --lambda_role_swap_rank "${LAMBDA_ROLE_SWAP_RANK:-0.0}"
  --role_swap_rank_margin "${ROLE_SWAP_RANK_MARGIN:-0.20}"
  --tail_logit_adjustment_enabled "${TAIL_LOGIT_ADJUSTMENT_ENABLED:-false}"
  --tail_logit_adjustment_tau "${TAIL_LOGIT_ADJUSTMENT_TAU:-0.0}"
  --predicate_sampler_enabled "${PREDICATE_SAMPLER_ENABLED:-false}"
  --predicate_sampler_power "${PREDICATE_SAMPLER_POWER:-0.75}"
  --predicate_sampler_max_weight "${PREDICATE_SAMPLER_MAX_WEIGHT:-20.0}"
  --asymmetric_pair_fusion_enabled "${ASYMMETRIC_PAIR_FUSION_ENABLED:-false}"
  --asymmetric_pair_fusion_include_reverse_diff "${ASYMMETRIC_PAIR_FUSION_INCLUDE_REVERSE_DIFF:-true}"
  --asymmetric_pair_fusion_hidden_mult "${ASYMMETRIC_PAIR_FUSION_HIDDEN_MULT:-2.0}"
  --predicate_metadata_path "${PREDICATE_METADATA_PATH:-configs/predicate_metadata_vg150.json}"
  --text_conditioned_projection_enabled true
  --text_conditioned_projection_residual "${TEXT_PROJ_RESIDUAL:-0.35}"
  --relationness_enabled true
  --lambda_predicate_ce "${LAMBDA_PREDICATE_CE:-1.2}"
  --lambda_spoa_alignment "${LAMBDA_SPOA_ALIGNMENT:-0.75}"
  --lambda_dense_grounding "${LAMBDA_DENSE_GROUNDING:-0.25}"
  --lambda_counterfactual "${LAMBDA_COUNTERFACTUAL:-0.05}"
  --lambda_text_predicate_ce "${LAMBDA_TEXT_PREDICATE_CE:-0.4}"
  --lambda_relationness "${LAMBDA_RELATIONNESS:-0.15}"
  --lambda_relationness_rank "${LAMBDA_RELATIONNESS_RANK:-0.10}"
  --relationness_rank_margin "${RELATIONNESS_RANK_MARGIN:-0.25}"
  --relationness_rank_topk "${RELATIONNESS_RANK_TOPK:-32}"
  --lambda_triplet_rank "${LAMBDA_TRIPLET_RANK:-0.0}"
  --triplet_rank_margin "${TRIPLET_RANK_MARGIN:-0.20}"
  --triplet_rank_topk "${TRIPLET_RANK_TOPK:-64}"
  --triplet_rank_relationness_weight "${TRIPLET_RANK_RELATIONNESS_WEIGHT:-0.25}"
  --lambda_calibration_reg "${LAMBDA_CALIBRATION_REG:-0.001}"
  --lambda_calibration_kl "${LAMBDA_CALIBRATION_KL:-0.03}"
  --lambda_calibration_rank "${LAMBDA_CALIBRATION_RANK:-0.08}"
  --calibration_rank_margin "${CALIBRATION_RANK_MARGIN:-0.25}"
  --predicate_ce_loss "${PREDICATE_CE_LOSS:-focal}"
  --predicate_ce_gamma "${PREDICATE_CE_GAMMA:-1.0}"
  --predicate_ce_weight_power "${PREDICATE_CE_WEIGHT_POWER:-0.5}"
  --predicate_ce_max_weight "${PREDICATE_CE_MAX_WEIGHT:-5.0}"
  --eval_every "${EVAL_EVERY:-1}"
  --eval_batches "${EVAL_BATCHES:-200}"
  --eval_sgg_predicate_score_mode "${EVAL_SCORE_MODE:-ensemble}"
  --eval_sgg_compare_score_modes "${EVAL_COMPARE_SCORE_MODES:-}"
  --eval_sgg_predicate_ensemble_alpha "${EVAL_ENSEMBLE_ALPHA:-0.45}"
  --eval_sgg_use_gt_pairs "${EVAL_SGG_USE_GT_PAIRS:-false}"
  --eval_sgg_role_swap_metric_enabled "${EVAL_ROLE_SWAP_METRIC_ENABLED:-true}"
  --eval_sgg_routing_diag_enabled "${EVAL_ROUTING_DIAG_ENABLED:-true}"
  --eval_sgg_use_relationness "${EVAL_SGG_USE_RELATIONNESS:-true}"
  --eval_sgg_relationness_weight "${RELATIONNESS_WEIGHT:-0.75}"
  --eval_sgg_relationness_prune_k "${RELATIONNESS_PRUNE_K:-96}"
  --eval_sgg_prune_score_mode "${PRUNE_SCORE_MODE:-relationness}"
  --eval_sgg_pair_score_mode "${PAIR_SCORE_MODE:-relationness}"
  --eval_sgg_detector_prune_k "${DETECTOR_PRUNE_K:-128}"
  --eval_sgg_use_object_uncertainty true
  --eval_sgg_object_score_power "${OBJECT_SCORE_POWER:-1.0}"
  --eval_sgg_grounding_dino_enabled "${EVAL_SGDET:-false}"
  --log_every "${LOG_EVERY:-50}"
)

if [[ -n "${RESUME_FROM}" ]]; then
  if [[ ! -f "${RESUME_FROM}" ]]; then
    echo "[train_l4_phase34] resume checkpoint not found: ${RESUME_FROM}" >&2
    exit 2
  fi
  ARGS+=(--resume true --resume_from "${RESUME_FROM}" --reset_epoch "${RESET_EPOCH:-true}")
else
  ARGS+=(--resume false)
fi

PYTHONUNBUFFERED=1 "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "logs/${RUN_NAME}.log"