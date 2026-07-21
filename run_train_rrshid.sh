#!/usr/bin/env bash
# Fine-tune FFA-Net on RRSHID (TN/M/TK = 浅/中/浓).
#
# Expected layout under dataset/:
#   RRSHID-TN/  RRSHID-TN-GT/
#   RRSHID-M/   RRSHID-M-GT/
#   RRSHID-TK/  RRSHID-TK-GT/
#
# Or:
#   Restored/RRSHID-TN/  +  Ground-truth/RRSHID-TN-GT/  ...
#
# Usage:
#   ./run_train_rrshid.sh 3
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
  --bs=16 --lr=0.00001 \
  --trainset rrshid_train --testset rrshid_test \
  --data_dir "${PROJECT_ROOT}/dataset" \
  --rrshid_val_ratio 0.1 \
  --pretrain ./trained_models/ots_train_ffa_3_19.pk \
  --steps=50000 --eval_step=1000

echo "==> Checkpoint: ${PROJECT_ROOT}/net/trained_models/rrshid_train_ffa_3_19.pk"
