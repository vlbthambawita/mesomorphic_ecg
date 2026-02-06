#!/usr/bin/env bash
set -e

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# PTB-XL NORM vs X - Interpretable Mesomorphic Neural Network (IMN) with Transition Network
# SINGLE LINEAR OUTPUT VERSION (Binary Classification with 1 Output Node)
# Runs all task types (norm_vs_mi, norm_vs_sttc, norm_vs_cd, norm_vs_hyp) at 100 Hz and 500 Hz.
# Separate output folders per frequency: ..._100Hz and ..._500Hz.
# For quick test use --epochs 1.

TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
RATES=(100 500)
BASE_OUT=runs/imn_ecg_transition_net_GM_one_linear_test_ecg_plots

for rate in "${RATES[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "========== Running task: $task @ ${rate} Hz =========="
    python script_02022026_v7_IMN_GM_2_with_transition_net_with_one_linear_eq.py \
      --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
      --task "$task" \
        --sampling_rate "$rate" \
      --epochs 1 \
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
      --out_dir "${BASE_OUT}_${rate}Hz" \
      --viz_pdf "viz_${task}_imn.pdf" \
      --wandb_project mesormorphic_ecg \
      --viz_ecg_plot \
      --viz_ecg_plot_n 5
  done
done

echo "========== All experiments finished =========="
