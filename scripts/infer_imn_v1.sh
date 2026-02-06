#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

# IMN v1 - Inference-only: load checkpoint and run visualization.
# Uses latest checkpoint per task from runs/imn_ecg_simple_implementation_GM_Basic/{task}/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_DIR=runs/imn_ecg_simple_implementation_GM_Basic
TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)

for task in "${TASKS[@]}"; do
  CKPT=$(ls -t "$BASE_DIR/$task"/*/best-imn-*.ckpt 2>/dev/null | head -1)
  [[ -z "$CKPT" ]] && { echo "========== No checkpoint found for $task, skipping =========="; continue; }
  echo "========== Running inference for task: $task (ckpt: $CKPT) =========="
  python models/imn_v1_simple.py \
    --inference_only --ckpt "$CKPT" \
    --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
    --task "$task" --sampling_rate 500 \
    --window 125 250 500 --stride 62 125 250 \
    --n_pos_viz 25 --n_neg_viz 25 --viz_random \
    --viz_pdf "viz_${task}_imn.pdf"
done
echo "========== All tasks finished =========="
