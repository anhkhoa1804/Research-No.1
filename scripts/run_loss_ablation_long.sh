#!/usr/bin/env bash
set -euo pipefail

# Longer loss-ablation sweep from one existing checkpoint.
# Usage:
#   PYTHON=./.venv/bin/python RESUME_FROM=checkpoints/some.pt bash scripts/run_loss_ablation_long.sh

PYTHON=${PYTHON:-python3}
BASE_RESUME_FROM=${RESUME_FROM:-${BASE_RESUME_FROM:-}}
if [[ -z "${BASE_RESUME_FROM}" ]]; then
  echo "[loss-ablation-long] set RESUME_FROM=/path/to/existing_checkpoint.pt" >&2
  exit 2
fi
if [[ ! -f "${BASE_RESUME_FROM}" ]]; then
  echo "[loss-ablation-long] checkpoint not found: ${BASE_RESUME_FROM}" >&2
  exit 2
fi

DATA_ROOT=${DATA_ROOT:-datasets}
RUN_ROOT=${RUN_ROOT:-runs/loss_ablation_long}
CKPT_ROOT=${CKPT_ROOT:-checkpoints/loss_ablation_long}
GPU_PRESET=${GPU_PRESET:-l4_24gb}
PURE_PHASE=${PURE_PHASE:-core}
EPOCHS=${EPOCHS:-4}
LR=${LR:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-100}
EVAL_BATCHES=${EVAL_BATCHES:-300}
EVAL_FAST_MODE=${EVAL_FAST_MODE:-true}
EVAL_SCORE_MODE=${EVAL_SCORE_MODE:-classifier}
EVAL_COMPARE_SCORE_MODES=${EVAL_COMPARE_SCORE_MODES:-classifier}
EVAL_SGG_USE_GT_PAIRS=${EVAL_SGG_USE_GT_PAIRS:-true}
PREDICATE_CE_POSITIVE_ONLY=${PREDICATE_CE_POSITIVE_ONLY:-false}
PREDICATE_CE_LOSS=${PREDICATE_CE_LOSS:-focal}
SEED=${SEED:-0}
LOG_EVERY=${LOG_EVERY:-100}
BATCH_SIZE=${BATCH_SIZE:-}
MAX_PAIRS=${MAX_PAIRS:-}
NUM_WORKERS=${NUM_WORKERS:-}
ACCUM_STEPS=${ACCUM_STEPS:-}

mkdir -p "${RUN_ROOT}" "${CKPT_ROOT}" logs

run_variant() {
  local name="$1"
  local lambda_pred_ce="$2"
  local lambda_spoa="$3"
  local lambda_ground="$4"
  local lambda_cf="$5"

  echo "[loss-ablation-long] ${name}: pred_ce=${lambda_pred_ce} spoa=${lambda_spoa} ground=${lambda_ground} cf=${lambda_cf}"
  PYTHON="${PYTHON}" \
  DATA_ROOT="${DATA_ROOT}" \
  OUT_ROOT="${RUN_ROOT}/${name}" \
  CHECKPOINT_DIR="${CKPT_ROOT}" \
  RESUME_FROM="${BASE_RESUME_FROM}" \
  SAVE_PATH="${CKPT_ROOT}/${name}.pt" \
  RUN_NAME="loss_ablation_long_${name}" \
  LOG_PATH="logs/loss_ablation_long_${name}.log" \
  GPU_PRESET="${GPU_PRESET}" \
  PURE_PHASE="${PURE_PHASE}" \
  EPOCHS="${EPOCHS}" \
  LR="${LR}" \
  WARMUP_STEPS="${WARMUP_STEPS}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  EVAL_FAST_MODE="${EVAL_FAST_MODE}" \
  EVAL_SCORE_MODE="${EVAL_SCORE_MODE}" \
  EVAL_COMPARE_SCORE_MODES="${EVAL_COMPARE_SCORE_MODES}" \
  EVAL_SGG_USE_GT_PAIRS="${EVAL_SGG_USE_GT_PAIRS}" \
  PREDICATE_CE_POSITIVE_ONLY="${PREDICATE_CE_POSITIVE_ONLY}" \
  PREDICATE_CE_LOSS="${PREDICATE_CE_LOSS}" \
  LAMBDA_PREDICATE_CE="${lambda_pred_ce}" \
  LAMBDA_SPOA_ALIGNMENT="${lambda_spoa}" \
  LAMBDA_DENSE_GROUNDING="${lambda_ground}" \
  LAMBDA_COUNTERFACTUAL="${lambda_cf}" \
  SEED="${SEED}" \
  LOG_EVERY="${LOG_EVERY}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_PAIRS="${MAX_PAIRS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  ACCUM_STEPS="${ACCUM_STEPS}" \
  bash scripts/run_pure_next.sh
}

# Keep CE-heavy baseline because it usually gives R@50 > mR@50 and is easier to interpret.
run_variant full 1.2 0.75 0.25 0.05
run_variant no_spoa 1.2 0.0 0.25 0.0
run_variant no_ground 1.2 0.75 0.0 0.05

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

def walk(obj, path=()):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk(value, path + (str(key),))
    elif isinstance(obj, (int, float)):
        yield path, float(obj)

def pick(metrics, suffix):
    candidates = []
    for path, value in walk(metrics):
        joined = "/".join(path).lower()
        if "predcls" in joined and joined.endswith(suffix.lower()):
            candidates.append((len(path), joined, value))
    if not candidates:
        for path, value in walk(metrics):
            joined = "/".join(path).lower()
            if "predcls" in joined and suffix.lower() in joined:
                candidates.append((len(path), joined, value))
    return sorted(candidates)[0][2] if candidates else None

root = Path("runs/loss_ablation_long")
print("variant	PredCls_R@50	PredCls_mR@50	latest")
for latest in sorted(root.glob("*/latest_metrics.json")):
    try:
        metrics = json.loads(latest.read_text())
    except Exception as exc:
        print(f"{latest.parent.name}	read_error	{exc}	{latest}")
        continue
    r50 = pick(metrics, "r@50")
    mr50 = pick(metrics, "mr@50") or pick(metrics, "mR@50")
    r50_s = "NA" if r50 is None else f"{r50:.4f}"
    mr50_s = "NA" if mr50 is None else f"{mr50:.4f}"
    print(f"{latest.parent.name}	{r50_s}	{mr50_s}	{latest}")
PY