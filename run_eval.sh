#!/usr/bin/env bash
# Run FFA-Net PSNR eval on host (no Docker). Recommended when Docker Hub is unreachable.
#
# Usage:
#   ./run_eval.sh          # GPU 0
#   ./run_eval.sh 3        # GPU 3
#   MAX_SIZE=1024 ./run_eval.sh 2
#
set -euo pipefail

PROJECT_ROOT="/data2/hyz/FFA-Nettest"
GPU="${1:-${GPU_DEVICE:-0}}"
MAX_SIZE="${MAX_SIZE:-0}"

cd "${PROJECT_ROOT}/net"

if [[ ! -f "trained_models/ots_train_ffa_3_19.pk" ]]; then
  echo "ERROR: weight not found: trained_models/ots_train_ffa_3_19.pk"
  echo "Download from Google Drive / Baidu (see trained_models/readme.md)"
  exit 1
fi

EVAL_ARGS=(
  --custom
  --hazy_dir "${PROJECT_ROOT}/dataset/haze"
  --clear_dir "${PROJECT_ROOT}/dataset/GT"
  --task ots
  --model_dir "${PROJECT_ROOT}/net/trained_models/ots_train_ffa_3_19.pk"
  --save_dir "${PROJECT_ROOT}/dataset/pred_FFA"
)

if [[ "${MAX_SIZE}" != "0" ]]; then
  EVAL_ARGS+=(--max_size "${MAX_SIZE}")
fi

echo "==> GPU: ${GPU}"
export CUDA_VISIBLE_DEVICES="${GPU}"
python eval.py "${EVAL_ARGS[@]}"

echo "==> Done. Predictions: ${PROJECT_ROOT}/dataset/pred_FFA"
