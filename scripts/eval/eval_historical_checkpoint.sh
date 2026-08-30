#!/usr/bin/env bash
#
# Historical-checkpoint reproduction entrypoint.
#
# The ONLY supported way to evaluate
# checkpoints/demo_best/pure_best_adapt_light_mR50.pt.
#
# WHY THIS EXISTS INSTEAD OF eval_l4_phase34.sh
# ---------------------------------------------
# eval_l4_phase34.sh is correct for current Phase 3/4 work and is deliberately
# left untouched. It is UNSAFE for this checkpoint: `--stage 3` resolves
# explicit_spoa_enabled, text_conditioned_projection_enabled,
# relationness_enabled and eval_sgg_use_relationness all to `true` (four of
# them forced unconditionally at config.py:741-747), clip_input_res to 448,
# ensemble alpha to 0.45, and gt_pairs to false. This checkpoint predates every
# one of those architectural pieces and was trained at 336. Running it under
# those defaults evaluates untrained/random-weight heads.
#
# Every compatibility-sensitive flag below is therefore passed EXPLICITLY.
# None is inherited. See docs/HISTORICAL_CHECKPOINT_MANIFEST.md section 4 for
# the per-flag justification.
#
# THE FLAG-NAME HAZARD
# --------------------
# openvocab_rel/train.py uses parse_known_args, which silently discards
# unrecognized flags. A typo here would leave a stage-3 default in place with
# NO runtime signal. tests/test_historical_eval_protocol.py parses this file
# and asserts every --flag it passes exists in build_argparser() and is a real
# TrainConfig field, so a typo fails at commit time rather than on a GPU.
# Keep the flags in the PROTOCOL array below -- the test reads that array.
#
# USAGE
#   bash scripts/eval/eval_historical_checkpoint.sh --canary   # 2 batches, fast
#   bash scripts/eval/eval_historical_checkpoint.sh --full     # whole split
#   bash scripts/eval/eval_historical_checkpoint.sh --dry-run  # print, run nothing
#
# The full run refuses to start unless a canary has already passed (override
# with ALLOW_UNGATED_FULL=1, which you should not need).

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Frozen protocol -- the historical configuration. Do not edit casually; each
# value is justified in docs/HISTORICAL_CHECKPOINT_MANIFEST.md section 4.
# ---------------------------------------------------------------------------
CKPT="checkpoints/demo_best/pure_best_adapt_light_mR50.pt"
FREQ_PRIOR="checkpoints/demo_best/frequency_prior.json"
DATA_ROOT="datasets_vg150_clean"

STAGE=3
GPU_PRESET="l4_24gb"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CLIP_INPUT_RES=336            # checkpoint is clip-vit-large-patch14-336; stage 3 would give 448
ENSEMBLE_ALPHA=0.0            # demo_config.env EVAL_ENSEMBLE_ALPHA; stage 3 would give 0.45
FREQ_BIAS_ALPHA=3.75          # demo_config.env FREQ_BIAS_ALPHA
FREQ_BIAS_SMOOTHING=1.0

MODE=""
DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --canary)  MODE="canary" ;;
    --full)    MODE="full" ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "[historical-eval] unknown argument: ${arg}" >&2
       echo "[historical-eval] usage: $0 (--canary|--full) [--dry-run]" >&2
       exit 2 ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  echo "[historical-eval] ERROR: specify --canary or --full." >&2
  echo "[historical-eval] Never run --full before a canary has passed." >&2
  exit 2
fi

if [[ "${MODE}" == "canary" ]]; then
  EVAL_BATCHES="${EVAL_BATCHES:-2}"
  RUN_PREFIX="historical_canary"
else
  # 0 = the whole split. evals.py treats a non-positive cap as "no limit"
  # (_batch_limit_reached). Before that helper existed, 0 broke on the first
  # batch and this "full" run evaluated ZERO images while exiting 0 and
  # reporting all-zero metrics.
  EVAL_BATCHES="${EVAL_BATCHES:-0}"
  RUN_PREFIX="historical_full"
fi

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-${RUN_PREFIX}_${RUN_STAMP}}"
OUT_DIR="runs/${RUN_NAME}"

# ---------------------------------------------------------------------------
# Guard: never silently overwrite a previous run.
# ---------------------------------------------------------------------------
if [[ -e "${OUT_DIR}" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "[historical-eval] ERROR: ${OUT_DIR} already exists." >&2
  echo "[historical-eval] Refusing to overwrite a previous run's artifacts." >&2
  echo "[historical-eval] Set RUN_NAME=... for a new run, or ALLOW_OVERWRITE=1 to force." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Guard: the full run is gated on a passed canary.
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "full" && "${ALLOW_UNGATED_FULL:-0}" != "1" ]]; then
  if ! ls runs/historical_canary_*/canary_verdict.txt >/dev/null 2>&1 \
     || ! grep -lq '^PASS' runs/historical_canary_*/canary_verdict.txt 2>/dev/null; then
    echo "[historical-eval] ERROR: no passing canary found." >&2
    echo "[historical-eval] Run the canary first:" >&2
    echo "[historical-eval]     bash $0 --canary" >&2
    echo "[historical-eval] Override with ALLOW_UNGATED_FULL=1 only if you know why." >&2
    exit 2
  fi
  echo "[historical-eval] canary gate satisfied:"
  grep -l '^PASS' runs/historical_canary_*/canary_verdict.txt 2>/dev/null | sed 's/^/[historical-eval]   /'
fi

# ---------------------------------------------------------------------------
# Guard: required artifacts must exist BEFORE we spend startup time. In
# particular the frequency prior, whose absence _load_frequency_bias swallows
# silently -- producing a complete but uncalibrated result. See
# docs/known_issues.md.
# ---------------------------------------------------------------------------
missing=0
for f in "${CKPT}" "${FREQ_PRIOR}" "${DATA_ROOT}/train.jsonl" \
         "${DATA_ROOT}/validation.jsonl" "${DATA_ROOT}/vocabulary/predicates.json"; do
  if [[ ! -f "${f}" ]]; then
    echo "[historical-eval] MISSING REQUIRED ARTIFACT: ${f}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "[historical-eval] These artifacts are gitignored and must be transferred out of band." >&2
  echo "[historical-eval] See docs/HISTORICAL_CHECKPOINT_MANIFEST.md section 6." >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}/images" ]]; then
  echo "[historical-eval] MISSING: ${DATA_ROOT}/images -- every image would silently" >&2
  echo "[historical-eval] degrade to a gray placeholder. Refusing to run." >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---------------------------------------------------------------------------
# The resolved protocol. Every entry is passed explicitly; nothing inherited.
# tests/test_historical_eval_protocol.py parses this array.
# ---------------------------------------------------------------------------
PROTOCOL=(
  --stage "${STAGE}"
  --gpu_preset "${GPU_PRESET}"
  --eval_only true
  --epochs 0
  --resume true
  --resume_from "${CKPT}"
  --vg150_root "${DATA_ROOT}"
  --vg150_enabled true
  --vg150_source local-jsonl
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --clip_input_res "${CLIP_INPUT_RES}"
  --eval_batches "${EVAL_BATCHES}"
  --eval_fast_mode false
  --explicit_spoa_enabled false
  --text_conditioned_projection_enabled false
  --relationness_enabled false
  --eval_sgg_use_relationness false
  --eval_sgg_predicate_score_mode ensemble
  --eval_sgg_predicate_ensemble_alpha "${ENSEMBLE_ALPHA}"
  --adaptive_calibration_enabled true
  --bayes_calibration_weight 0.0
  --freq_bias_enabled true
  --freq_bias_path "${FREQ_PRIOR}"
  --freq_bias_alpha "${FREQ_BIAS_ALPHA}"
  --freq_bias_smoothing "${FREQ_BIAS_SMOOTHING}"
  --eval_sgg_use_gt_pairs true
  --eval_sgg_grounding_dino_enabled false
  --run_name "${RUN_NAME}"
  --out_dir "${OUT_DIR}"
  --save_metrics_json "${OUT_DIR}/metrics.jsonl"
)

# ---------------------------------------------------------------------------
# Print the resolved protocol BEFORE doing anything.
# ---------------------------------------------------------------------------
cat <<BANNER
======================================================================
  HISTORICAL CHECKPOINT REPRODUCTION -- RESOLVED PROTOCOL
======================================================================
  mode                      : ${MODE}$( [[ "${DRY_RUN}" == "1" ]] && echo " (DRY RUN -- nothing will execute)" )
  run name                  : ${RUN_NAME}
  output directory          : ${OUT_DIR}
  eval_batches              : ${EVAL_BATCHES}$( [[ "${EVAL_BATCHES}" == "0" ]] && echo "  (0 = entire split)" )
----------------------------------------------------------------------
  checkpoint                : ${CKPT}
  frequency prior           : ${FREQ_PRIOR}
  dataset root              : ${DATA_ROOT}
  dataset source            : local-jsonl
  device / batch / workers  : ${DEVICE} / ${BATCH_SIZE} / ${NUM_WORKERS}
----------------------------------------------------------------------
  COMPATIBILITY OVERRIDES (stage-3 default -> forced value)
    explicit_spoa_enabled               true  -> false
    text_conditioned_projection_enabled true  -> false
    relationness_enabled                true  -> false
    eval_sgg_use_relationness           true  -> false
    clip_input_res                      448   -> ${CLIP_INPUT_RES}
    eval_sgg_use_gt_pairs               false -> true
    eval_sgg_grounding_dino_enabled     true  -> false
  SCORING
    eval_sgg_predicate_score_mode             = ensemble
    eval_sgg_predicate_ensemble_alpha   0.45  -> ${ENSEMBLE_ALPHA}   (=> 100% CLIP text, 0% classifier)
  CALIBRATION
    adaptive_calibration_enabled        false -> true
    bayes_calibration_weight                  = 0.0
    freq_bias_enabled                   false -> true
    freq_bias_alpha                     0.5   -> ${FREQ_BIAS_ALPHA}
======================================================================
BANNER

mkdir -p "${OUT_DIR}" logs

# ---------------------------------------------------------------------------
# Artifact capture (Phase 7). Written BEFORE the run so a crashed run is still
# identifiable.
# ---------------------------------------------------------------------------
{
  printf '%s' "${PYTHON} -u -m openvocab_rel.train"
  for a in "${PROTOCOL[@]}"; do printf " %q" "${a}"; done
  printf '\n'
} > "${OUT_DIR}/command.txt"

git rev-parse HEAD > "${OUT_DIR}/git_commit.txt" 2>/dev/null || echo "unknown" > "${OUT_DIR}/git_commit.txt"
git status --porcelain=v1 > "${OUT_DIR}/git_status.txt" 2>/dev/null || true
{
  echo "mode=${MODE}"
  echo "run_name=${RUN_NAME}"
  echo "eval_batches=${EVAL_BATCHES}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "python=$(${PYTHON} -c 'import platform;print(platform.python_version())' 2>/dev/null || echo unknown)"
  echo "platform=$(${PYTHON} -c 'import platform;print(platform.platform())' 2>/dev/null || echo unknown)"
  echo "torch=$(${PYTHON} -c 'import torch;print(torch.__version__)' 2>/dev/null || echo unavailable)"
  echo "cuda_available=$(${PYTHON} -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null || echo unknown)"
  echo "gpu=$(${PYTHON} -c 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")' 2>/dev/null || echo unknown)"
} > "${OUT_DIR}/environment.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[historical-eval] DRY RUN complete. Captured to ${OUT_DIR}/. Nothing executed."
  exit 0
fi

# ---------------------------------------------------------------------------
# Preflight gate -- hashes, dataset identity, vocabulary, prior structure.
# ---------------------------------------------------------------------------
echo "[historical-eval] running preflight..."
PREFLIGHT_STRICT=()
[[ "${DEVICE}" == "cuda" ]] && PREFLIGHT_STRICT+=(--strict)
if ! "${PYTHON}" tools/gcp_preflight.py "${PREFLIGHT_STRICT[@]}" \
      --out "${OUT_DIR}/manifest.yaml" \
      --run-name "${RUN_NAME}" \
      --command "$(cat "${OUT_DIR}/command.txt")"; then
  echo "[historical-eval] PREFLIGHT FAILED -- aborting before any GPU time is spent." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
echo "[historical-eval] launching evaluation..."
set +e
PYTHONUNBUFFERED=1 "${PYTHON}" -u -m openvocab_rel.train "${PROTOCOL[@]}" \
  2>&1 | tee "${OUT_DIR}/run.log"
status="${PIPESTATUS[0]}"
set -e
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT_DIR}/environment.txt"
echo "exit_status=${status}" >> "${OUT_DIR}/environment.txt"

if [[ "${status}" -ne 0 ]]; then
  echo "[historical-eval] evaluation exited ${status} -- see ${OUT_DIR}/run.log" >&2
  exit "${status}"
fi

# ---------------------------------------------------------------------------
# Verify the RESOLVED runtime settings, not the intended ones.
# ---------------------------------------------------------------------------
echo "[historical-eval] verifying resolved protocol..."
"${PYTHON}" tools/verify_canary.py "${OUT_DIR}/metrics.jsonl" \
  --out "${OUT_DIR}/canary_verdict.txt"
verdict="$?"

if [[ "${verdict}" -ne 0 ]]; then
  echo "[historical-eval] PROTOCOL VERIFICATION FAILED -- do not use these numbers." >&2
  exit "${verdict}"
fi

echo "[historical-eval] done. Artifacts in ${OUT_DIR}/"
if [[ "${MODE}" == "canary" ]]; then
  echo "[historical-eval] canary passed; the full run is now unblocked:"
  echo "[historical-eval]     bash $0 --full"
fi
