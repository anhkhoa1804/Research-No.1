#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHON="${PYTHON:-$(pwd)/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CORE_ROOT="${CORE_ROOT:-datasets/core}"
CORE_JSONL_ROOT="${CORE_JSONL_ROOT:-datasets/core_vg150_jsonl}"
CORE_EVAL_ROOT="${CORE_EVAL_ROOT:-datasets/core_eval_jsonl}"
CORE_EVAL_SPLIT="${CORE_EVAL_SPLIT:-validation}"
CKPT="${CKPT:-checkpoints/core_l3_balanced_adapt_light_best_mR50.pt}"
RUN_CORE_INSPECT="${RUN_CORE_INSPECT:-true}"
RUN_CORE_CONVERT="${RUN_CORE_CONVERT:-true}"
HOLDOUT_V2="${HOLDOUT_V2:-true}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[CORE eval] python not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "[CORE eval] checkpoint not found: ${CKPT}" >&2
  exit 2
fi

mkdir -p logs runs/core_inspect "${CORE_EVAL_ROOT}"

if [[ "${RUN_CORE_INSPECT}" == "true" ]]; then
  "${PYTHON}" tools/inspect_core.py --core-root "${CORE_ROOT}" --report runs/core_inspect/report.json
fi

if [[ "${RUN_CORE_CONVERT}" == "true" ]]; then
  convert_args=(--core-root "${CORE_ROOT}" --out-root "${CORE_JSONL_ROOT}" --train-ratio "${CORE_TRAIN_RATIO:-0.80}" --val-ratio "${CORE_VAL_RATIO:-0.10}")
  if [[ "${HOLDOUT_V2}" == "true" ]]; then
    convert_args+=(--holdout-v2)
  fi
  if [[ "${CORE_EVAL_STRICT:-true}" == "true" ]]; then
    convert_args+=(--drop-generic-objects --drop-generic-relations)
  fi
  "${PYTHON}" tools/convert_core_to_vg150_jsonl.py "${convert_args[@]}"
fi

actual_core_root="$(${PYTHON} - <<'PY' "${CORE_JSONL_ROOT}" "${CORE_ROOT}"
import json, sys
from pathlib import Path
jsonl = Path(sys.argv[1])
fallback = Path(sys.argv[2])
report = jsonl / "core_conversion_report.json"
if report.exists():
    try:
        print(json.loads(report.read_text()).get("core_root") or str(fallback))
    except Exception:
        print(str(fallback))
else:
    print(str(fallback))
PY
)"

rm -rf "${CORE_EVAL_ROOT}"
mkdir -p "${CORE_EVAL_ROOT}"
ln -sfn "$(pwd)/${CORE_JSONL_ROOT}/train.jsonl" "${CORE_EVAL_ROOT}/train.jsonl"
ln -sfn "$(pwd)/${CORE_JSONL_ROOT}/${CORE_EVAL_SPLIT}.jsonl" "${CORE_EVAL_ROOT}/validation.jsonl"
for version_dir in "${actual_core_root}"/*; do
  if [[ -d "${version_dir}" ]]; then
    ln -sfn "$(pwd)/${version_dir}" "${CORE_EVAL_ROOT}/$(basename "${version_dir}")"
  fi
done

"${PYTHON}" tools/build_core_eval_vocab.py --jsonl-root "${CORE_JSONL_ROOT}" --out-root "${CORE_EVAL_ROOT}" --max-objects "${CORE_MAX_OBJECTS:-300}"

"${PYTHON}" - <<'PY' "${CORE_JSONL_ROOT}" "${CORE_EVAL_ROOT}" "${CORE_EVAL_SPLIT}" "${CORE_EVAL_STRICT:-true}"
import json
import sys
from collections import Counter
from pathlib import Path

jsonl_root = Path(sys.argv[1])
eval_root = Path(sys.argv[2])
split = sys.argv[3]
strict = sys.argv[4].lower() == "true"
generic_objects = {"", "__background__", "background", "bg", "object", "objects", "thing", "entity", "unknown", "none"}
generic_preds = {"", "__background__", "background", "bg", "relation", "relationships", "no relation", "no interaction", "unknown", "none"}

def clean(value):
    return " ".join(str(value).strip().lower().replace("_", " ").split())

def iter_rows(path):
    if not path.exists():
        raise SystemExit(f"[CORE eval] missing split file: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

rows = list(iter_rows(jsonl_root / f"{split}.jsonl"))
if not rows:
    raise SystemExit(f"[CORE eval] converted split is empty: {jsonl_root / f'{split}.jsonl'}")

objects = Counter()
preds = Counter()
for row in rows:
    for obj in row.get("objects", []):
        if isinstance(obj, dict):
            names = obj.get("names", [])
            if isinstance(names, str):
                name = names
            elif names:
                name = names[0]
            else:
                name = ""
            objects[clean(name)] += 1
    for rel in row.get("relationships", []):
        if isinstance(rel, dict):
            preds[clean(rel.get("predicate", ""))] += 1

bad_objects = objects & Counter({x: 10**9 for x in generic_objects})
bad_preds = preds & Counter({x: 10**9 for x in generic_preds})
if strict and (bad_objects or bad_preds):
    raise SystemExit(f"[CORE eval] generic labels remain after strict conversion: objects={bad_objects.most_common()} predicates={bad_preds.most_common()}")

vocab_report = eval_root / "vocabulary" / "core_eval_vocab_report.json"
if vocab_report.exists():
    report = json.loads(vocab_report.read_text(encoding="utf-8"))
    if strict and int(report.get("num_predicates", 0)) <= 0:
        raise SystemExit("[CORE eval] predicate vocab is empty after strict filtering")
print(f"[CORE eval] guard ok rows={len(rows)} top_objects={objects.most_common(5)} top_predicates={preds.most_common(5)}")
PY

echo "[CORE eval] split=${CORE_EVAL_SPLIT} ckpt=${CKPT} eval_root=${CORE_EVAL_ROOT} actual_core_root=${actual_core_root}"

PYTHON="${PYTHON}" \
PURE_PHASE=eval \
STAGE=3 \
EPOCHS=0 \
LR=0 \
RESUME_FROM="${CKPT}" \
DATA_ROOT="${CORE_EVAL_ROOT}" \
RUN_NAME="${RUN_NAME:-core_eval_${CORE_EVAL_SPLIT}}" \
OUT_ROOT="${OUT_ROOT:-runs/core_eval_${CORE_EVAL_SPLIT}}" \
SAVE_PATH="${SAVE_PATH:-checkpoints/core_eval_${CORE_EVAL_SPLIT}.pt}" \
GPU_PRESET="${GPU_PRESET:-l4_24gb}" \
EVAL_ONLY=true \
EVAL_FAST_MODE="${EVAL_FAST_MODE:-false}" \
EVAL_BATCHES="${EVAL_BATCHES:-9999}" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
MAX_PAIRS="${MAX_PAIRS:-64}" \
EVAL_SCORE_MODE="${EVAL_SCORE_MODE:-text}" \
EVAL_COMPARE_SCORE_MODES="${EVAL_COMPARE_SCORE_MODES:-text}" \
FREQ_BIAS_ENABLED=false \
EVAL_SGG_USE_GT_PAIRS=true \
EVAL_SGG_USE_CLIP_OBJ_CLASSIFIER=false \
EVAL_SGG_SGCLS_ORACLE_LABELS=true \
EVAL_SGG_GROUNDING_DINO_ENABLED=false \
EVAL_SGG_REPORT_NOGRAPH=false \
bash scripts/run_pure_next.sh

"${PYTHON}" tools/summarize_metrics.py "${OUT_ROOT:-runs/core_eval_${CORE_EVAL_SPLIT}}/metrics.jsonl" || true