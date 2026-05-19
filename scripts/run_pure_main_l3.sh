#!/usr/bin/env bash
set -euo pipefail

BRANCH_LEVEL=${BRANCH_LEVEL:-3} \
RESUME_FROM=${RESUME_FROM:-checkpoints/branch_level3_cf005.pt} \
SAVE_PATH=${SAVE_PATH:-checkpoints/pure_main_l3.pt} \
EPOCHS=${EPOCHS:-2} \
EVAL_BATCHES=${EVAL_BATCHES:-50} \
LOG_EVERY=${LOG_EVERY:-50} \
LR=${LR:-3e-6} \
PYTHON=${PYTHON:-python} \
bash scripts/run_branch_ramp.sh