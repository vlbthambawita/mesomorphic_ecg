#!/usr/bin/env bash
set -e

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# PTB-XL NORM vs X - IMN with Transition Network (2-class CE)
# Inference-only script:
#  - Finds the latest best checkpoint per (task, sampling_rate)
#  - Runs validation + metrics + visualization using --inference_only
#  - Optional user-selected window, stride, leads, heatmap_height, ecg_height:
#      bash run_script_02022026_v7_IMN_GM_2_with_transition_net_inference.sh [window] [stride] [leads] [heatmap_height] [ecg_height]
#    e.g.:
#      bash run_script_02022026_v7_IMN_GM_2_with_transition_net_inference.sh 50 25
#      bash run_script_02022026_v7_IMN_GM_2_with_transition_net_inference.sh 50 25 "V1,V2,V3,V4,V5,V6" 0.5 0.5

TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
RATES=(100)
BASE_OUT="runs/imn_ecg_transition_net_GM"

DATA_PATH="/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"

# Default window/stride (100Hz-friendly)
WINDOW_ARG="--window 2 10 125"
STRIDE_ARG="--stride 1 5 67"
LEADS_ARG="--leads I,V2,V3"
HEATMAP_HEIGHT_ARG="--viz_heatmap_height 0.3"
ECG_HEIGHT_ARG="--viz_ecg_height 0.3"

# Optional CLI overrides
if [[ -n "${1:-}" ]]; then
  WINDOW_ARG="--window $1"
fi
if [[ -n "${2:-}" ]]; then
  STRIDE_ARG="--stride $2"
fi
if [[ -n "${3:-}" ]]; then
  LEADS_ARG="--leads ${3}"
fi
if [[ -n "${4:-}" ]]; then
  HEATMAP_HEIGHT_ARG="--viz_heatmap_height ${4}"
fi
if [[ -n "${5:-}" ]]; then
  ECG_HEIGHT_ARG="--viz_ecg_height ${5}"
fi

for rate in "${RATES[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "========== Inference for task: ${task} @ ${rate} Hz =========="

    TASK_DIR="${BASE_OUT}_${rate}Hz/${task}"

    if [[ ! -d "${TASK_DIR}" ]]; then
      echo "  [WARN] Directory not found: ${TASK_DIR} (skipping)"
      continue
    fi

    # Find the newest run directory and then its best checkpoint
    LATEST_RUN_DIR="$(ls -td "${TASK_DIR}"/*/ 2>/dev/null | head -n 1 || true)"
    if [[ -z "${LATEST_RUN_DIR}" ]]; then
      echo "  [WARN] No run subdirectories found in ${TASK_DIR} (skipping)"
      continue
    fi

    CKPT_PATH="$(ls -t "${LATEST_RUN_DIR%/}/"best-imn-*.ckpt 2>/dev/null | head -n 1 || true)"
    if [[ -z "${CKPT_PATH}" ]]; then
      echo "  [WARN] No best-imn-*.ckpt found in ${LATEST_RUN_DIR} (skipping)"
      continue
    fi

    echo "  Using checkpoint: ${CKPT_PATH}"

    python script_02022026_v7_IMN_GM_2_with_transition_net.py \
      --path "${DATA_PATH}" \
      --task "${task}" \
      --sampling_rate "${rate}" \
      --batch_size 64 \
      --lambda_l1 1e-4 \
      --scheduler none \
      --inference_only \
      --ckpt "${CKPT_PATH}" \
      --n_pos_viz 10 \
      --n_neg_viz 10 \
      ${WINDOW_ARG} \
      ${STRIDE_ARG} \
      --viz_random \
      --viz_negative_class \
      --viz_pdf "viz_${task}_imn_inference.pdf" \
      --wandb_project mesormorphic_ecg \
      ${LEADS_ARG} \
      ${HEATMAP_HEIGHT_ARG} \
      ${ECG_HEIGHT_ARG}

    echo "========== Finished inference for task: ${task} @ ${rate} Hz =========="
  done
done

echo "========== All inference runs finished =========="
