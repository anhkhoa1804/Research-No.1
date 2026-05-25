#!/usr/bin/env bash
set -euo pipefail

CORE_DRIVE_URL="${CORE_DRIVE_URL:-https://drive.google.com/drive/folders/11rAVJgxZ557XPf4JHyQi7rJ7Hmw7fMHR}"
CORE_OUT_DIR="${CORE_OUT_DIR:-datasets/core_benchmark}"

mkdir -p "${CORE_OUT_DIR}"

if command -v gdown >/dev/null 2>&1; then
  GDOWN_BIN="gdown"
elif python3 -m gdown --version >/dev/null 2>&1; then
  GDOWN_BIN="python3 -m gdown"
else
  cat >&2 <<'EOF'
ERROR: gdown is not available on this VM.
Install it in the user environment, not a new virtualenv:
  python3 -m pip install --user -U gdown
Then rerun:
  bash scripts/download_core_gdrive.sh
EOF
  exit 2
fi

echo "Downloading CORE from: ${CORE_DRIVE_URL}"
echo "Output directory: ${CORE_OUT_DIR}"
${GDOWN_BIN} --folder "${CORE_DRIVE_URL}" -O "${CORE_OUT_DIR}" --remaining-ok

cat <<EOF

Download finished. Next inspect candidate roots:
  find ${CORE_OUT_DIR} -maxdepth 3 -name metadata.json | sort
  python3 tools/inspect_core.py --core-root ${CORE_OUT_DIR} --report runs/core_inspect/report.json
EOF