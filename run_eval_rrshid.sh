#!/usr/bin/env bash
# Evaluate FFA-Net on RRSHID with PSNR per density (浅/中/浓).
#
# Usage:
#   ./run_eval_rrshid.sh 3                    # all densities
#   ./run_eval_rrshid.sh 3 tn                  # thin/浅 only
#   ./run_eval_rrshid.sh 3 m                   # moderate/中
#   ./run_eval_rrshid.sh 3 tk                  # thick/浓
#
set -euo pipefail

PROJECT_ROOT="/data2/hyz/FFA-Nettest"
GPU="${1:-${GPU_DEVICE:-0}}"
DENSITY="${2:-all}"
MODEL="${MODEL:-${PROJECT_ROOT}/net/trained_models/ots_train_ffa_3_19.pk}"

cd "${PROJECT_ROOT}/net"
export CUDA_VISIBLE_DEVICES="${GPU}"

python eval.py --rrshid \
  --data_dir "${PROJECT_ROOT}/dataset" \
  --density "${DENSITY}" \
  --model_dir "${MODEL}" \
  --save_dir "${PROJECT_ROOT}/dataset/pred_rrshid"
