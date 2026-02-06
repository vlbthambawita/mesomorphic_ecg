#!/usr/bin/env bash
set -e

# Use GPU 0
export CUDA_VISIBLE_DEVICES=0

# PTB-XL NORM vs X - Baseline 2D CNN + Grad-CAM (pytorch-grad-cam)
# Inference-only script:
#  - Finds the latest best checkpoint per (task, sampling_rate)
#  - Runs validation + metrics + Grad-CAM visualization using --inference_only
#  - Optional CLI overrides: window, stride, leads, heatmap_height, ecg_height, viz_ecg_plot
#    e.g.: bash run_inference_gradcam_baseline.sh 125 67
#    e.g.: bash run_inference_gradcam_baseline.sh 125 67 "V1,V2,V3,V4,V5,V6" 0.5 0.65 1

TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
RATES=(100)
# Matches run_gradcam_baseline.sh: output is runs/baseline_2d_gradcam_100Hz (or _500Hz if trained)
BASE_OUT="runs/baseline_2d_gradcam"

DATA_PATH="/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"

# Optional CLI overrides (applied to all tasks/rates)
# Usage: bash run_inference_gradcam_baseline.sh [window] [stride] [leads] [heatmap_height] [ecg_height] [viz_ecg_plot]
#   window, stride: e.g. 125 67
#   leads: e.g. "V1,V2,V3,V4,V5,V6" or "0-5" or "I,II,III"
#   heatmap_height: e.g. 0.3 (shorter) or 1.0 (taller) for top heatmap panel
#   ecg_height: e.g. 0.5 (shorter) or 0.65 (taller) per lead for bottom panel
#   viz_ecg_plot: "1" or "yes" to enable ECG+Grad-CAM heatmap PDFs (ecg_plot-style layout)
WINDOW_ARG="--window 125"
STRIDE_ARG="--stride 67"
LEADS_ARG="--leads I,V2,V3"
HEATMAP_HEIGHT_ARG="--viz_heatmap_height 0.3"
ECG_HEIGHT_ARG="--viz_ecg_height 0.3"
VIZ_ECG_PLOT_ARG=""
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
if [[ "${6:-}" == "1" || "${6:-}" == "yes" ]]; then
  VIZ_ECG_PLOT_ARG="--viz_ecg_plot --viz_ecg_plot_n 5"
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

    CKPT_PATH="$(ls -t "${LATEST_RUN_DIR}"/best-epoch*.ckpt 2>/dev/null | head -n 1 || true)"
    if [[ -z "${CKPT_PATH}" ]]; then
      echo "  [WARN] No best-epoch*.ckpt found in ${LATEST_RUN_DIR} (skipping)"
      continue
    fi

    echo "  Using checkpoint: ${CKPT_PATH}"

    python script_26012026_v7_gradcam_library_baseline.py \
      --path "${DATA_PATH}" \
      --task "${task}" \
      --sampling_rate "${rate}" \
      --batch_size 64 \
      --scheduler none \
      --inference_only \
      --ckpt "${CKPT_PATH}" \
      --n_pos_viz 10 \
      --n_neg_viz 10 \
      ${WINDOW_ARG} \
      ${STRIDE_ARG} \
      ${LEADS_ARG} \
      ${HEATMAP_HEIGHT_ARG} \
      ${ECG_HEIGHT_ARG} \
      ${VIZ_ECG_PLOT_ARG} \
      --viz_random \
      --viz_pdf "viz_${task}_gradcam_inference.pdf" \
      --wandb_project mesormorphic_ecg

    echo "========== Finished inference for task: ${task} @ ${rate} Hz =========="
  done
done

echo "========== All inference runs finished =========="
