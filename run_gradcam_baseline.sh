#!/usr/bin/env bash
set -e

# PTB-XL NORM vs X baseline 2D CNN + Grad-CAM (pytorch-grad-cam).
# Runs all task types: norm_vs_mi, norm_vs_sttc, norm_vs_cd, norm_vs_hyp.
# For quick test use --epochs 1.

TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
BASE_OUT=runs/baseline_2d_gradcam

for task in "${TASKS[@]}"; do
  echo "========== Running task: $task =========="
  python script_26012026_v7_gradcam_library_baseline.py \
    --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
    --task "$task" \
    --batch_size 64 \
    --sampling_rate 500 \
    --epochs 100 \
    --n_pos_viz 10 \
    --n_neg_viz 10 \
    --viz_random \
    --out_dir "$BASE_OUT" \
    --viz_pdf "viz_${task}_gradcam.pdf" \
    --wandb_project mesormorphic_ecg \
    --early_stop_patience 10 \
    --scheduler step
done

echo "========== All tasks finished =========="
