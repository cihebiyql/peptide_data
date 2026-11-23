#!/usr/bin/env bash
# Self-contained Peptide OmniPanel V33 local inference engine launcher (0 GPU VRAM, 0 external cluster dependency)
set -euo pipefail

ROOT="${OMNIPANEL_ROOT:-/var/www/peptide-omnipanel/v33_next}"
VENV_PY="${OMNIPANEL_VENV_PY:-/var/www/peptide-omnipanel/app/.venv/bin/python}"

export PYTHONPATH="${ROOT}/extra_site${PYTHONPATH:+:${PYTHONPATH}}"
export V33_BASE_DIR="${ROOT}/v33_runtime"
export V33_ESMC_PYTHON="${VENV_PY}"
export V33_ESMC_MODEL_DIR="${ROOT}/pretrained/esmc_600m"
export NCAA_REPO_DIR="${ROOT}/pretrained/ncaadistill"
export V33_PEPTIDECLM2_DIR="${ROOT}/pretrained/peptideclm2"
export V33_CHEMBERTA_DIR="${ROOT}/pretrained/chemberta_mtr"
export V33_EP_WORKERS="16"
export V33_ESMC_THREADS="16"
export V33_MAX_CONCURRENT_INFER="4"
export V33_INFER_QUEUE_TIMEOUT="30"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false"
# Keep per-estimator BLAS/OpenMP threads at 1 so 16 endpoint workers + concurrent requests never oversubscribe CPU
export OMP_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"
export CUDA_VISIBLE_DEVICES=""

exec "${VENV_PY}" -u "${ROOT}/v33_runtime/bundle/v33_inference_service.py" \
  --host 127.0.0.1 \
  --port 8902 \
  --esmc-port 18902 \
  --threads 2 \
  --max-concurrent 4 \
  --queue-timeout 30
