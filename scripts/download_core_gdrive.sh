#!/usr/bin/env bash
set -euo pipefail

CORE_ZIP_URL="${CORE_ZIP_URL:-https://drive.google.com/file/d/1eWdgbrQo_XTO4Ubfy2ygYtmojATlx6jJ/view?usp=drive_link}"
CORE_FOLDER_URL="${CORE_FOLDER_URL:-https://drive.google.com/drive/folders/11rAVJgxZ557XPf4JHyQi7rJ7Hmw7fMHR}"
CORE_OUT_DIR="${CORE_OUT_DIR:-datasets/core}"
CORE_ZIP_PATH="${CORE_ZIP_PATH:-datasets/core.zip}"
CORE_MODE="${CORE_MODE:-zip}"
CORE_CLEAN="${CORE_CLEAN:-false}"

if command -v gdown >/dev/null 2>&1; then
  GDOWN_BIN=(gdown)
elif python3 -m gdown --version >/dev/null 2>&1; then
  GDOWN_BIN=(python3 -m gdown)
else
  cat >&2 <<'EOF'
ERROR: gdown is not available on this VM.
Install it in the user environment, not a new virtualenv:
  python3 -m pip install --user -U gdown
EOF
  exit 2
fi

if [[ "${CORE_CLEAN}" == "true" ]]; then
  echo "Cleaning old CORE outputs: ${CORE_OUT_DIR} ${CORE_ZIP_PATH} datasets/core_benchmark"
  rm -rf "${CORE_OUT_DIR}" "${CORE_ZIP_PATH}" datasets/core_benchmark
fi

mkdir -p "$(dirname "${CORE_ZIP_PATH}")" "${CORE_OUT_DIR}"

if [[ "${CORE_MODE}" == "zip" ]]; then
  echo "Downloading CORE zip from: ${CORE_ZIP_URL}"
  echo "Zip path: ${CORE_ZIP_PATH}"
  "${GDOWN_BIN[@]}" --fuzzy "${CORE_ZIP_URL}" -O "${CORE_ZIP_PATH}"

  echo "Extracting to: ${CORE_OUT_DIR}"
  rm -rf "${CORE_OUT_DIR}"
  mkdir -p "${CORE_OUT_DIR}"
  python3 - <<PY
from pathlib import Path
from zipfile import ZipFile
zip_path = Path("${CORE_ZIP_PATH}")
out_dir = Path("${CORE_OUT_DIR}")
with ZipFile(zip_path) as archive:
    archive.extractall(out_dir)
print(f"Extracted {zip_path} -> {out_dir}")
PY
else
  echo "Downloading CORE folder from: ${CORE_FOLDER_URL}"
  echo "Output directory: ${CORE_OUT_DIR}"
  "${GDOWN_BIN[@]}" --folder "${CORE_FOLDER_URL}" -O "${CORE_OUT_DIR}"
fi

cat <<EOF

CORE download finished.
Next commands:
  find ${CORE_OUT_DIR} -maxdepth 4 -name metadata.json | sort
  python3 tools/inspect_core.py --core-root ${CORE_OUT_DIR} --report runs/core_inspect/report.json
EOF