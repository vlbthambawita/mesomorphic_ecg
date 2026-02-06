#!/usr/bin/env bash
set -e

# Use GPU 2
export CUDA_VISIBLE_DEVICES=0

# PTB-XL NORM vs X IMN 1D baseline (1D CNN patch embed + 2D CNN hypernetwork).
# Lead-wise and segment-wise XAI explanations.
# Runs all task types: norm_vs_mi, norm_vs_sttc, norm_vs_cd, norm_vs_hyp.
# For quick test use --epochs 1.

TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
BASE_OUT=runs/imn1d_lead_segment

for task in "${TASKS[@]}"; do
  echo "========== Running task: $task =========="
  python script_01022026_imn1d_lead_segment_xai.py \
    --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
    --task "$task" \
    --batch_size 64 \
    --sampling_rate 500 \
    --epochs 1 \
    --n_pos_viz 10 \
    --n_neg_viz 10 \
    --viz_random \
    --out_dir "$BASE_OUT" \
    --viz_pdf "viz_${task}_imn1d.pdf" \
    --wandb_project mesormorphic_ecg \
    --early_stop_patience 10 \
    --scheduler step
done

echo "========== All tasks finished =========="
