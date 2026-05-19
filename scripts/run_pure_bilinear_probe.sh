#!/usr/bin/env bash
set -euo pipefail

BRANCH_LEVEL=4 \
RESUME_FROM=${RESUME_FROM:-checkpoints/branch_level3_cf005.pt} \
SAVE_PATH=${SAVE_PATH:-checkpoints/pure_bilinear_probe.pt} \
EPOCHS=${EPOCHS:-1} \
MAX_IMAGES=${MAX_IMAGES:-2000} \
SAMPLES_PER_EPOCH=${SAMPLES_PER_EPOCH:-2000} \
EVAL_BATCHES=${EVAL_BATCHES:-30} \
LOG_EVERY=${LOG_EVERY:-50} \
LR=${LR:-2e-6} \
PYTHON=${PYTHON:-python} \
bash scripts/run_branch_ramp.sh