#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

DATA_ROOT="${DATA_ROOT:-datasets}"
GT_CANDIDATES=(runs/eval_p3_best_gt_pairs_final/metrics.jsonl runs/eval_p3_gt_pairs_fa*/metrics.jsonl)
ALLPAIRS_CANDIDATES=(
  runs/eval_p3_same_as_p5_k256/metrics.jsonl
  runs/pure_l4_p5_pair_rank_smoke/metrics.jsonl
  runs/pure_l4_p6_triplet_rank_smoke/metrics.jsonl
  runs/pure_l4_p7_tail_triplet_rank_smoke/metrics.jsonl
  runs/pure_l4_p8_pair_rank_balanced_smoke/metrics.jsonl
)

mkdir -p runs/breakthrough_phase_ab

echo "== Phase A/B: GT-pair report card =="
gt_existing=()
for path in "${GT_CANDIDATES[@]}"; do
  [[ -f "$path" ]] && gt_existing+=("$path")
done
if [[ "${#gt_existing[@]}" -gt 0 ]]; then
  if [[ -f "${DATA_ROOT}/train.jsonl" ]]; then
    python3 tools/model_report_card.py "${gt_existing[@]}" --train_jsonl "${DATA_ROOT}/train.jsonl" \
      | tee runs/breakthrough_phase_ab/gtpair_report.txt
  else
    python3 tools/model_report_card.py "${gt_existing[@]}" \
      | tee runs/breakthrough_phase_ab/gtpair_report.txt
  fi
else
  echo "No GT-pair metrics found."
fi

echo

echo "== Phase A/B: all-pairs diagnostic gate =="
existing=()
for path in "${ALLPAIRS_CANDIDATES[@]}"; do
  [[ -f "$path" ]] && existing+=("$path")
done
if [[ "${#existing[@]}" -gt 0 ]]; then
  python3 tools/sgg_gate_report.py "${existing[@]}" \
    | tee runs/breakthrough_phase_ab/allpairs_gate_report.txt
else
  echo "No all-pairs diagnostic metrics found."
fi

echo

echo "Reports written under runs/breakthrough_phase_ab/"