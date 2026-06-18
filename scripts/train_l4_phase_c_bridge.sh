#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:=pure_l4_phase_c_pair_object_bridge}"
: "${SAVE_PATH:=checkpoints/${RUN_NAME}.pt}"
: "${RESUME_FROM:=}"
: "${RESET_EPOCH:=true}"
: "${EPOCHS:=1}"
: "${SAMPLES_PER_EPOCH:=8000}"
: "${EVAL_BATCHES:=30}"
: "${BATCH_SIZE:=20}"
: "${MAX_PAIRS:=128}"
: "${NEGATIVE_PAIR_RATIO:=8.0}"
: "${LAMBDA_PAIR_PROPOSAL:=1.0}"
: "${LAMBDA_PAIR_PROPOSAL_RANK:=0.5}"
: "${LAMBDA_OBJECT_BRIDGE:=0.5}"
: "${FREEZE_PREDICATE_HEAD:=true}"
: "${PYTHON_BIN:=python3}"
: "${VG150_SOURCE:=local-jsonl}"

"${PYTHON_BIN}" -m openvocab_rel.train \
  --run_name "${RUN_NAME}" \
  --save_path "${SAVE_PATH}" \
  --resume_from "${RESUME_FROM}" \
  --reset_epoch "${RESET_EPOCH}" \
  --train_objective pair_object_bridge \
  --vg150_source "${VG150_SOURCE}" \
  --hf_streaming false \
  --epochs "${EPOCHS}" \
  --samples_per_epoch "${SAMPLES_PER_EPOCH}" \
  --eval_batches "${EVAL_BATCHES}" \
  --batch_size "${BATCH_SIZE}" \
  --max_pairs "${MAX_PAIRS}" \
  --use_all_pairs true \
  --negative_pair_ratio "${NEGATIVE_PAIR_RATIO}" \
  --freeze_clip true \
  --freeze_predicate_head "${FREEZE_PREDICATE_HEAD}" \
  --relationness_enabled true \
  --lambda_relationness "${LAMBDA_PAIR_PROPOSAL}" \
  --lambda_relationness_rank "${LAMBDA_PAIR_PROPOSAL_RANK}" \
  --relationness_rank_topk 64 \
  --lambda_object_bridge "${LAMBDA_OBJECT_BRIDGE}" \
  --object_bridge_topk 5 \
  --lambda_predicate_ce 0.0 \
  --lambda_spoa_alignment 0.0 \
  --lambda_dense_grounding 0.0 \
  "$@"