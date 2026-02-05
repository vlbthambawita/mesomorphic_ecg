#!/usr/bin/env bash
set -e

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# PTB-XL MI vs NORM — Interpretable Mesomorphic Neural Network (IMN) patch-wise
# Runs ptbxl_imn_patch_lightning_GPT.py
# For quick test use --epochs 1.

python ptbxl_imn_patch_lightning_GPT.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --sampling_rate 500 \
  --epochs 1 \
  --batch_size 64 \
  --num_workers 4 \
  --seed 42 \
  --token_dim 16 \
  --backbone_dim 128 \
  --hyper_hidden 256 \
  --lambda_l1 1e-3 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --dropout 0.1 \
  --n_pos 1 \
  --n_neg 1 \
  --viz_random

echo "========== Done =========="
