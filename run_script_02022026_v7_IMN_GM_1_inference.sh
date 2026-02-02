#!/usr/bin/env bash
set -e

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Inference-only: load checkpoint and run visualization with different window/stride configs.
# Runs all task types: norm_vs_mi, norm_vs_sttc, norm_vs_cd, norm_vs_hyp.
# Uses the latest checkpoint per task from runs/imn_ecg/{task}/.
# Run training first (run_script_02022026_v7_IMN_GM_1.sh) to generate checkpoints.

BASE_DIR=runs/imn_ecg
TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)

for task in "${TASKS[@]}"; do
  CKPT=$(ls -t "$BASE_DIR/$task"/*/best-imn-*.ckpt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    echo "========== No checkpoint found for $task, skipping =========="
    continue
  fi
  echo "========== Running inference for task: $task (ckpt: $CKPT) =========="
  python script_02022026_v7_IMN_GM_1.py \
    --inference_only \
    --ckpt "$CKPT" \
    --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
    --task "$task" \
    --sampling_rate 500 \
    --window 125 250 500 \
    --stride 62 125 250 \
    --n_pos_viz 25 \
    --n_neg_viz 25 \
    --viz_random \
    --viz_pdf "viz_${task}_imn.pdf"
done

echo "========== All tasks finished =========="
