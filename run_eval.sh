#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:-${GPU_DEVICE:-0}}"

cd "${ROOT}/net"
export CUDA_VISIBLE_DEVICES="${GPU}"

python eval.py --rrshid \
  --data_dir "${ROOT}/dataset" \
  --model_dir "${ROOT}/net/trained_models/ots_train_ffa_3_19.pk" \
  --save_dir "${ROOT}/dataset/pred_rrshid"
