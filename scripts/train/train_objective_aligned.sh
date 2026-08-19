#!/usr/bin/env bash
#
# Objective-aligned training run.
#
# PURPOSE
# -------
# Establish the first PURE checkpoint whose training objective matches what
# the evaluator actually scores, so that its result can be compared honestly
# against the frequency-prior control (tools/frequency_prior_baseline.py:
# R@50 = 64.37 % / mR@50 = 20.30 % with the model contributing nothing).
#
# This is NOT a new architecture. Not one line of model code differs from the
# frozen baseline on main. Only the objective and the data distribution are
# changed, and every change is passed explicitly here rather than by editing
# a default -- so `main` remains the untouched reproducibility baseline.
#
# WHAT IT FIXES, AND WHY
# ----------------------
# 1. The evaluator scores the predicate CLASSIFIER (score_mode=ensemble), but
#    in the historical run that head sat at random initialisation because
#    lr=2e-6. The head evaluation depends on is the head training barely
#    touched. -> raise LR and LAMBDA_PREDICATE_CE.
#
# 2. Training used use_all_pairs=false, so CE only ever saw GT-positive pairs
#    and the model never learned "these two objects are unrelated". The
#    background class exists (predicate_classifier_classes=51, negatives are
#    labelled "relation") but nothing was ever assigned to it.
#    -> USE_ALL_PAIRS=true with a real negative ratio.
#
# 3. THREE imbalance corrections were stacked at training time -- inverse
#    frequency class weights (predicate_ce_weight_power), focal loss, and
#    (optionally) logit adjustment -- while evaluation applied a FOURTH
#    correction in the opposite direction (freq_bias alpha=3.75 pushes
#    predictions back toward the prior). They fight each other.
#    -> ARM selects which correction is active. Only the "baseline" control
#       keeps the existing stacked pair; every other arm applies at most one.
#
# ARMS (set ARM=..., default "logit_adj")
# ---------------------------------------
#   baseline    current objective, but with the classifier actually trained.
#               Keeps focal + inverse-frequency weights. This is the control
#               that isolates "we simply trained it properly".
#   logit_adj   single, theoretically grounded long-tail correction (Menon
#               et al. 2021, ce_logits - tau*log_prior). Focal and class
#               weights OFF so the imbalance is corrected exactly once.
#   none        no long-tail correction at all. Lower bound; tells us how
#               much of mR@50 is the correction versus the representation.
#
# Every arm is evaluated with BOTH raw and frequency-prior-calibrated scoring,
# and against the prior-only control. Report deltas, never absolutes.
#
# USAGE
#   bash scripts/train/train_objective_aligned.sh                 # ARM=logit_adj
#   ARM=baseline bash scripts/train/train_objective_aligned.sh
#   ARM=none     bash scripts/train/train_objective_aligned.sh
#   DRY_RUN=1    bash scripts/train/train_objective_aligned.sh    # print only

set -Eeuo pipefail

ARM="${ARM:-logit_adj}"
case "${ARM}" in
  baseline)   CE_LOSS="focal"; CE_WEIGHT_POWER="0.5"; TAIL_ADJ="false"; TAIL_TAU="0.0" ;;
  logit_adj)  CE_LOSS="ce";    CE_WEIGHT_POWER="0.0"; TAIL_ADJ="true";  TAIL_TAU="${TAIL_TAU:-1.0}" ;;
  none)       CE_LOSS="ce";    CE_WEIGHT_POWER="0.0"; TAIL_ADJ="false"; TAIL_TAU="0.0" ;;
  *) echo "[objective-aligned] unknown ARM: ${ARM} (want baseline|logit_adj|none)" >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-datasets_vg150_clean}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-objaligned_${ARM}_${RUN_STAMP}}"
OUT_DIR="runs/${RUN_NAME}"

if [[ -e "${OUT_DIR}" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "[objective-aligned] ERROR: ${OUT_DIR} exists; refusing to overwrite." >&2
  echo "[objective-aligned] Set RUN_NAME=... or ALLOW_OVERWRITE=1." >&2
  exit 2
fi
for f in "${DATA_ROOT}/train.jsonl" "${DATA_ROOT}/validation.jsonl" \
         "${DATA_ROOT}/vocabulary/predicates.json"; do
  [[ -f "${f}" ]] || { echo "[objective-aligned] MISSING REQUIRED ARTIFACT: ${f}" >&2; exit 2; }
done
[[ -d "${DATA_ROOT}/images" ]] || { echo "[objective-aligned] MISSING: ${DATA_ROOT}/images" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Every research-relevant flag is explicit. Nothing is inherited from a stage
# preset, because --stage 3 silently forces four architecture flags on
# (config.py:741-747). tests/test_objective_aligned_protocol.py parses this
# array and asserts each flag exists and resolves as intended.
PROTOCOL=(
  --stage 3
  --gpu_preset "${GPU_PRESET:-l4_24gb}"
  --run_name "${RUN_NAME}"
  --out_dir "${OUT_DIR}"
  --save_metrics_json "${OUT_DIR}/metrics.jsonl"
  --vg150_root "${DATA_ROOT}"
  --vg150_enabled true
  --vg150_source local-jsonl
  --device cuda
  --seed "${SEED:-0}"
  --epochs "${EPOCHS:-6}"
  --batch_size "${BATCH_SIZE:-12}"
  --num_workers "${NUM_WORKERS:-4}"
  --clip_input_res "${CLIP_INPUT_RES:-336}"
  --max_images "${MAX_IMAGES:-0}"
  --samples_per_epoch "${SAMPLES_PER_EPOCH:-20000}"
  --lr "${LR:-2e-5}"
  --freeze_clip true
  # --- fix 2: the model must learn "no relation" -------------------------
  --use_all_pairs true
  --negative_pair_ratio "${NEGATIVE_PAIR_RATIO:-2.0}"
  # --- fix 1: train the head the evaluator actually scores ---------------
  --predicate_classifier_enabled true
  --lambda_predicate_ce "${LAMBDA_PREDICATE_CE:-2.0}"
  --predicate_ce_loss "${CE_LOSS}"
  --predicate_ce_weight_power "${CE_WEIGHT_POWER}"
  # --- fix 3: long-tail correction, selected by ARM ----------------------
  --tail_logit_adjustment_enabled "${TAIL_ADJ}"
  --tail_logit_adjustment_tau "${TAIL_TAU}"
  # --- keep the representation losses that align rel_feats to CLIP text --
  --lambda_spoa_alignment "${LAMBDA_SPOA:-0.75}"
  --lambda_dense_grounding "${LAMBDA_GROUND:-0.25}"
  # --- architecture: identical to the frozen baseline --------------------
  --explicit_spoa_enabled true
  --asymmetric_pair_fusion_enabled false
  # --- evaluation during training: raw, uncalibrated ---------------------
  --eval_sgg_use_gt_pairs true
  --eval_sgg_predicate_score_mode classifier
  --adaptive_calibration_enabled false
  --freq_bias_enabled false
  --bayes_calibration_weight 0.0
  --eval_sgg_multi_predicate_topk 10
  --eval_batches "${EVAL_BATCHES:-300}"
  --eval_sgg_grounding_dino_enabled false
)

cat <<BANNER
======================================================================
  OBJECTIVE-ALIGNED TRAINING -- RESOLVED PROTOCOL
======================================================================
  arm                    : ${ARM}
  run name               : ${RUN_NAME}
  output                 : ${OUT_DIR}
----------------------------------------------------------------------
  LONG-TAIL CORRECTION (arm=${ARM}; only "baseline" stacks more than one)
    predicate_ce_loss             = ${CE_LOSS}
    predicate_ce_weight_power     = ${CE_WEIGHT_POWER}
    tail_logit_adjustment_enabled = ${TAIL_ADJ}   tau = ${TAIL_TAU}
  OBJECTIVE
    lambda_predicate_ce           = ${LAMBDA_PREDICATE_CE:-2.0}
    use_all_pairs                 = true   (negatives: ${NEGATIVE_PAIR_RATIO:-2.0} per positive)
    lr                            = ${LR:-2e-5}
  IN-TRAINING EVAL  (raw classifier, NO calibration -- calibrated numbers
                     are produced separately so raw and system-level scores
                     are never conflated)
    score_mode=classifier  freq_bias=false  adaptive_calibration=false
    multi_predicate_topk=10
----------------------------------------------------------------------
  CONTROL TO BEAT (tools/frequency_prior_baseline.py, model = 0):
    R@50 = 64.37 %     mR@50 = 20.30 %
  Report the DELTA over that control. An absolute calibrated score is
  not evidence the model learned anything.
======================================================================
BANNER

mkdir -p "${OUT_DIR}" logs
{
  printf '%s' "${PYTHON} -u -m openvocab_rel.train"
  for a in "${PROTOCOL[@]}"; do printf ' %q' "${a}"; done
  printf '\n'
} > "${OUT_DIR}/command.txt"
git rev-parse HEAD > "${OUT_DIR}/git_commit.txt" 2>/dev/null || echo unknown > "${OUT_DIR}/git_commit.txt"
{
  echo "arm=${ARM}"
  echo "run_name=${RUN_NAME}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "seed=${SEED:-0}"
} > "${OUT_DIR}/environment.txt"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[objective-aligned] DRY RUN -- nothing executed. Captured to ${OUT_DIR}/"
  exit 0
fi

PYTHONUNBUFFERED=1 "${PYTHON}" -u -m openvocab_rel.train "${PROTOCOL[@]}" \
  2>&1 | tee "${OUT_DIR}/run.log"
