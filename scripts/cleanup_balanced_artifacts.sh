#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN="${DRY_RUN:-true}"
KEEP_EVAL_BATCHES="${KEEP_EVAL_BATCHES:-500}"
REMOVE_LONG_RUNS="${REMOVE_LONG_RUNS:-false}"

run_rm() {
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] rm -rf'
    printf ' %q' "$@"
    printf '\n'
  else
    rm -rf "$@"
  fi
}

shopt -s nullglob

# Evaluation-only checkpoints are reproducible and usually not needed after metrics are written.
run_rm checkpoints/reeval_core_l3_balanced_*.pt checkpoints/reeval_core_l3_seed_*.pt

# Remove non-final reeval run folders while keeping the chosen EB setting.
for path in runs/reeval_core_l3_balanced_*_eb* runs/reeval_core_l3_seed_*_eb*; do
  [[ -e "${path}" ]] || continue
  if [[ "${path}" != *"_eb${KEEP_EVAL_BATCHES}" ]]; then
    run_rm "${path}"
  fi
done

# Optional: remove exploratory long/seed training outputs, never enabled by default.
if [[ "${REMOVE_LONG_RUNS}" == "true" ]]; then
  run_rm \
    runs/core_l3_balanced_ultra_light_long \
    runs/core_l3_seed_bias_only_lr5e7 \
    runs/core_l3_seed_tail_probe_p06 \
    checkpoints/core_l3_balanced_ultra_light_long.pt \
    checkpoints/core_l3_balanced_ultra_light_long_best_mR50.pt \
    checkpoints/core_l3_balanced_ultra_light_long_best_R50.pt \
    checkpoints/core_l3_seed_bias_only_lr5e7.pt \
    checkpoints/core_l3_seed_bias_only_lr5e7_best_mR50.pt \
    checkpoints/core_l3_seed_bias_only_lr5e7_best_R50.pt \
    checkpoints/core_l3_seed_tail_probe_p06.pt \
    checkpoints/core_l3_seed_tail_probe_p06_best_mR50.pt \
    checkpoints/core_l3_seed_tail_probe_p06_best_R50.pt
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[cleanup] dry run only. Re-run with DRY_RUN=false to delete."
else
  echo "[cleanup] done."
fi