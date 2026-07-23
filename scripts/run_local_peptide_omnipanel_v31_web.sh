#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${V31_WEB_PYTHON:-/root/.venvs/peptide-omnipanel-v31-web311-20260723/bin/python}

if [[ ! -x "$PYTHON" ]]; then
  echo "V31 web Python is unavailable: $PYTHON" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$PYTHON" scripts/run_peptide_omnipanel_v31_web.py \
  --device "${V31_WEB_DEVICE:-cuda}" \
  --host "${V31_WEB_HOST:-0.0.0.0}" \
  --port "${V31_WEB_PORT:-7860}" \
  "$@"
