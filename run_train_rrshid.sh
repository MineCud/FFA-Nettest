#!/usr/bin/env bash
# Fine-tune FFA-Net on RRSHID (thin/moderate/thick = 浅/中/浓).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
GPU="${1:-${GPU_DEVICE:-0}}"

cd "${ROOT}/net"
export CUDA_VISIBLE_DEVICES="${GPU}"

python main.py \
  --net ffa \
  --crop --crop_size=240 \
  --blocks=19 --gps=3 \
  --bs=16 --lr=0.00001 \
  --trainset rrshid_train --testset rrshid_val \
  --data_dir "${ROOT}/dataset" \
  --pretrain ./trained_models/ots_train_ffa_3_19.pk \
  --steps=50000 --eval_step=1000

echo "==> Checkpoint: ${ROOT}/net/trained_models/rrshid_train_ffa_3_19.pk"
