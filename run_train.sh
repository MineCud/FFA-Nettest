#!/usr/bin/env bash
# Fine-tune FFA-Net on remote sensing haze/GT pairs.
#
# Data layout (recommended):
#   dataset/train/hazy/  dataset/train/GT/
#   dataset/val/hazy/      dataset/val/GT/
#
# Flat layout (no val split):
#   dataset/haze/  dataset/GT/   -> used for both train and val (with warning)
#
# Usage:
#   ./run_train.sh          # GPU 0
#   ./run_train.sh 3        # GPU 3
#
set -euo pipefail

PROJECT_ROOT="/data2/hyz/FFA-Nettest"
GPU="${1:-${GPU_DEVICE:-0}}"

cd "${PROJECT_ROOT}/net"

export CUDA_VISIBLE_DEVICES="${GPU}"

python main.py \
  --net ffa \
  --crop --crop_size=240 \
  --blocks=19 --gps=3 \
  --bs=8 --lr=0.00001 \
  --trainset rs_train --testset rs_test \
  --data_dir "${PROJECT_ROOT}/dataset" \
  --pair_mode same_name \
  --pretrain ./trained_models/ots_train_ffa_3_19.pk \
  --steps=50000 --eval_step=1000

echo "==> Best checkpoint: ${PROJECT_ROOT}/net/trained_models/rs_train_ffa_3_19.pk"
