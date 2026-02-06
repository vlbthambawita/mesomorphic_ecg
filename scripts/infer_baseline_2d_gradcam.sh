#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

# PTB-XL NORM vs X - Baseline 2D CNN + Grad-CAM (pytorch-grad-cam)
# Inference-only: finds latest checkpoint, runs validation + Grad-CAM visualization.
# Usage: bash infer_baseline_2d_gradcam.sh [window] [stride] [leads] [heatmap_height] [ecg_height] [viz_ecg_plot]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
RATES=(100)
BASE_OUT="runs/baseline_2d_gradcam"
DATA_PATH="/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"

WINDOW_ARG="--window 125"
STRIDE_ARG="--stride 67"
LEADS_ARG="--leads I,V2,V3"
HEATMAP_HEIGHT_ARG="--viz_heatmap_height 0.3"
ECG_HEIGHT_ARG="--viz_ecg_height 0.3"
VIZ_ECG_PLOT_ARG=""
[[ -n "${1:-}" ]] && WINDOW_ARG="--window $1"
[[ -n "${2:-}" ]] && STRIDE_ARG="--stride $2"
[[ -n "${3:-}" ]] && LEADS_ARG="--leads ${3}"
[[ -n "${4:-}" ]] && HEATMAP_HEIGHT_ARG="--viz_heatmap_height ${4}"
[[ -n "${5:-}" ]] && ECG_HEIGHT_ARG="--viz_ecg_height ${5}"
[[ "${6:-}" == "1" || "${6:-}" == "yes" ]] && VIZ_ECG_PLOT_ARG="--viz_ecg_plot --viz_ecg_plot_n 5"

for rate in "${RATES[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "========== Inference for task: ${task} @ ${rate} Hz =========="
    TASK_DIR="${BASE_OUT}_${rate}Hz/${task}"
    [[ ! -d "${TASK_DIR}" ]] && { echo "  [WARN] Directory not found: ${TASK_DIR} (skipping)"; continue; }
    LATEST_RUN_DIR="$(ls -td "${TASK_DIR}"/*/ 2>/dev/null | head -n 1 || true)"
    [[ -z "${LATEST_RUN_DIR}" ]] && { echo "  [WARN] No run subdirectories found (skipping)"; continue; }
    CKPT_PATH="$(ls -t "${LATEST_RUN_DIR%/}"/best-epoch*.ckpt 2>/dev/null | head -n 1 || true)"
    [[ -z "${CKPT_PATH}" ]] && { echo "  [WARN] No best-epoch*.ckpt found (skipping)"; continue; }
    echo "  Using checkpoint: ${CKPT_PATH}"
    python models/baseline_2d_gradcam.py \
      --path "${DATA_PATH}" --task "${task}" --sampling_rate "${rate}" --batch_size 64 \
      --scheduler none --inference_only --ckpt "${CKPT_PATH}" \
      --n_pos_viz 10 --n_neg_viz 10 ${WINDOW_ARG} ${STRIDE_ARG} ${LEADS_ARG} \
      ${HEATMAP_HEIGHT_ARG} ${ECG_HEIGHT_ARG} ${VIZ_ECG_PLOT_ARG} \
      --viz_random --viz_pdf "viz_${task}_gradcam_inference.pdf" --wandb_project mesormorphic_ecg
    echo "========== Finished inference for task: ${task} @ ${rate} Hz =========="
  done
done
echo "========== All inference runs finished =========="
