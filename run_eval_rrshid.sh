#!/usr/bin/env bash
# Evaluate FFA-Net on RRSHID test set, per fog level.
#
# Usage:
#   ./run_eval_rrshid.sh 3              # all levels, test split
#   ./run_eval_rrshid.sh 3 thin         # 浅雾 only
#   ./run_eval_rrshid.sh 3 moderate     # 中雾
#   ./run_eval_rrshid.sh 3 thick        # 浓雾
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:-${GPU_DEVICE:-0}}"
DENSITY="${2:-all}"
MODEL="${MODEL:-${ROOT}/net/trained_models/ots_train_ffa_3_19.pk}"

cd "${ROOT}/net"
export CUDA_VISIBLE_DEVICES="${GPU}"

python eval.py --rrshid \
  --data_dir "${ROOT}/dataset" \
  --density "${DENSITY}" \
  --split test \
  --model_dir "${MODEL}" \
  --save_dir "${ROOT}/dataset/pred_rrshid"
