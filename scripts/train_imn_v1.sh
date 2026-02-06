#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

# PTB-XL NORM vs X - Interpretable Mesomorphic Neural Network (IMN) v1 (simple)
# Runs all task types: norm_vs_mi, norm_vs_sttc, norm_vs_cd, norm_vs_hyp.
# For quick test use --epochs 1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
BASE_OUT=runs/imn_ecg_simple_implementation_GM_Basic

for task in "${TASKS[@]}"; do
  echo "========== Running task: $task =========="
  python models/imn_v1_simple.py \
    --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
    --task "$task" \
    --sampling_rate 500 \
    --epochs 100 \
    --batch_size 64 \
    --lambda_l1 1e-4 \
    --scheduler step \
    --early_stop_patience 10 \
    --early_stop_min_delta 0.0 \
    --early_stop_monitor val_auc \
    --early_stop_mode max \
    --n_pos_viz 10 \
    --n_neg_viz 10 \
    --viz_random \
    --out_dir "$BASE_OUT" \
    --viz_pdf "viz_${task}_imn.pdf" \
    --wandb_project mesormorphic_ecg
done

echo "========== All tasks finished =========="
