#!/usr/bin/env bash
# Evaluate FFA-Net on remote sensing haze/GT pairs inside Docker.
set -euo pipefail

PROJECT_ROOT="/data2/hyz/FFA-Nettest"
HAZE_DIR="${PROJECT_ROOT}/dataset/haze"
GT_DIR="${PROJECT_ROOT}/dataset/GT"
SAVE_DIR="${PROJECT_ROOT}/dataset/pred_FFA"
MODEL="${PROJECT_ROOT}/net/trained_models/ots_train_ffa_3_19.pk"
MAX_SIZE="${MAX_SIZE:-0}"   # e.g. MAX_SIZE=1024 ./run_eval_docker.sh

cd "${PROJECT_ROOT}"

echo "==> Building Docker image (first run only)..."
docker compose build

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

echo "==> Running eval in Docker..."
docker compose run --rm ffa-eval \
  python eval.py "${EVAL_ARGS[@]}"

echo "==> Done. Predictions saved to: ${SAVE_DIR}"
