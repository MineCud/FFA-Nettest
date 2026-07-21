#!/usr/bin/env bash
# Evaluate FFA-Net on remote sensing haze/GT pairs inside Docker.
#
# Usage:
#   ./run_eval_docker.sh          # use all GPUs (default)
#   ./run_eval_docker.sh 0        # use GPU 0
#   ./run_eval_docker.sh 3        # use GPU 3
#   ./run_eval_docker.sh 0,2      # use GPU 0 and 2
#   MAX_SIZE=1024 ./run_eval_docker.sh 1
#
set -euo pipefail

PROJECT_ROOT="/data2/hyz/FFA-Nettest"
SAVE_DIR="${PROJECT_ROOT}/dataset/pred_FFA"
IMAGE="ffa-nettest:latest"
MAX_SIZE="${MAX_SIZE:-0}"

# GPU: positional arg > env GPU_DEVICE > default "all"
GPU="${1:-${GPU_DEVICE:-all}}"

cd "${PROJECT_ROOT}"

if [[ "${GPU}" == "all" ]]; then
  GPUS_FLAG="all"
else
  GPUS_FLAG="device=${GPU}"
fi

echo "==> Building Docker image (first run only)..."
docker build -t "${IMAGE}" .

EVAL_ARGS=(
  --custom
  --hazy_dir "/workspace/FFA-Nettest/dataset/haze"
  --clear_dir "/workspace/FFA-Nettest/dataset/GT"
  --task ots
  --model_dir "/workspace/FFA-Nettest/net/trained_models/ots_train_ffa_3_19.pk"
  --save_dir "/workspace/FFA-Nettest/dataset/pred_FFA"
)

if [[ "${MAX_SIZE}" != "0" ]]; then
  EVAL_ARGS+=(--max_size "${MAX_SIZE}")
fi

echo "==> Running eval in Docker (GPU: ${GPU})..."
docker run --rm --gpus "${GPUS_FLAG}" \
  -v "${PROJECT_ROOT}:/workspace/FFA-Nettest" \
  -w /workspace/FFA-Nettest/net \
  "${IMAGE}" \
  python eval.py "${EVAL_ARGS[@]}"

echo "==> Done. Predictions saved to: ${SAVE_DIR}"
