#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

# PTB-XL NORM vs X - IMN with Transition Network (single linear output, BCE)
# Runs all tasks at 100 Hz and 500 Hz. For quick test use --epochs 1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
RATES=(100 500)
BASE_OUT=runs/imn_ecg_transition_net_GM_one_linear

for rate in "${RATES[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "========== Running task: $task @ ${rate} Hz =========="
    python models/imn_v2_transition_one_linear.py \
      --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
      --task "$task" --sampling_rate "$rate" \
      --epochs 100 --batch_size 64 --lambda_l1 1e-4 \
      --scheduler step --early_stop_patience 10 --early_stop_min_delta 0.0 \
      --early_stop_monitor val_auc --early_stop_mode max \
      --n_pos_viz 10 --n_neg_viz 10 --viz_random \
      --out_dir "${BASE_OUT}_${rate}Hz" --viz_pdf "viz_${task}_imn.pdf" \
      --wandb_project mesormorphic_ecg
  done
done

echo "========== All experiments finished =========="
