#!/usr/bin/env bash
# Self-contained Peptide OmniPanel V33 local inference engine launcher on node2 (0 GPU VRAM, 0 node23 dependency)
set -euo pipefail

ROOT="/var/www/peptide-omnipanel/v33_next"
VENV_PY="/var/www/peptide-omnipanel/app/.venv/bin/python"

export PYTHONPATH="${ROOT}/extra_site${PYTHONPATH:+:${PYTHONPATH}}"
export V33_BASE_DIR="${ROOT}/v33_runtime"
export V33_ESMC_PYTHON="${VENV_PY}"
export V33_ESMC_MODEL_DIR="${ROOT}/pretrained/esmc_600m"
export NCAA_REPO_DIR="${ROOT}/pretrained/ncaadistill"
export V33_PEPTIDECLM2_DIR="${ROOT}/pretrained/peptideclm2"
export V33_CHEMBERTA_DIR="${ROOT}/pretrained/chemberta_mtr"
export V33_EP_WORKERS="16"
export V33_ESMC_THREADS="24"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false"
export OMP_NUM_THREADS="4"
export OPENBLAS_NUM_THREADS="4"
export MKL_NUM_THREADS="4"
export CUDA_VISIBLE_DEVICES=""

exec "${VENV_PY}" -u "${ROOT}/v33_runtime/bundle/v33_inference_service.py" \
  --host 127.0.0.1 \
  --port 8902 \
  --esmc-port 18902 \
  --threads 16
