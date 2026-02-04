#!/usr/bin/env bash
# Visualize top-k leads and top-k segments for IMN final predictions.
# Uses inference-only mode: loads best checkpoint and generates PDFs showing
# the most important leads and segments per sample based on IMN feature attribution.

set -e

export CUDA_VISIBLE_DEVICES=0

# --- Config ---
SCRIPT="script_02022026_v7_IMN_GM_2_with_transition_net_with_one_linear_eq.py"
DATA_PATH="/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"
BASE_OUT="runs/imn_ecg_transition_net_GM_one_linear"

# Top-k for leads and segments (per-sample, based on IMN impact importance)
TOP_K_LEADS="${1:-2}"      # Default: top 4 leads
TOP_K_SEGMENTS="${2:-2}"  # Default: top 10 segments

# Window/stride for segment aggregation (100 Hz typical: w=10, s=5)
WINDOW="${3:-50}"
STRIDE="${4:-25}"

# Tasks and sampling rate
TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
RATE=100

echo "========== Top-K Leads & Segments Visualization =========="
echo "  TOP_K_LEADS=${TOP_K_LEADS}"
echo "  TOP_K_SEGMENTS=${TOP_K_SEGMENTS}"
echo "  WINDOW=${WINDOW}, STRIDE=${STRIDE}"
echo "=========================================================="

for task in "${TASKS[@]}"; do
  echo ""
  echo "========== Task: ${task} @ ${RATE} Hz =========="

  TASK_DIR="${BASE_OUT}_${RATE}Hz/${task}"
  if [[ ! -d "${TASK_DIR}" ]]; then
    echo "  [WARN] Directory not found: ${TASK_DIR} (skipping)"
    continue
  fi

  LATEST_RUN_DIR="$(ls -td "${TASK_DIR}"/*/ 2>/dev/null | head -n 1 || true)"
  if [[ -z "${LATEST_RUN_DIR}" ]]; then
    echo "  [WARN] No run subdirectories in ${TASK_DIR} (skipping)"
    continue
  fi

  CKPT_PATH="$(ls -t "${LATEST_RUN_DIR}"/best-imn-epoch*=*.ckpt 2>/dev/null | head -n 1 || true)"
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "  [WARN] No best-imn-epoch*.ckpt in ${LATEST_RUN_DIR} (skipping)"
    continue
  fi

  echo "  Checkpoint: ${CKPT_PATH}"
  echo "  Output: ${LATEST_RUN_DIR}"

  python "${SCRIPT}" \
    --path "${DATA_PATH}" \
    --task "${task}" \
    --sampling_rate "${RATE}" \
    --batch_size 64 \
    --lambda_l1 1e-4 \
    --scheduler none \
    --inference_only \
    --ckpt "${CKPT_PATH}" \
    --window "${WINDOW}" \
    --stride "${STRIDE}" \
    --top_k_leads "${TOP_K_LEADS}" \
    --top_k_segments "${TOP_K_SEGMENTS}" \
    --n_pos_viz 5 \
    --n_neg_viz 5 \
    --viz_random \
    --viz_pdf "viz_${task}_top${TOP_K_LEADS}leads_top${TOP_K_SEGMENTS}seg_inference.pdf" \
    --wandb_project mesormorphic_ecg \
    --viz_ecg_plot \
    --viz_ecg_plot_n 3

  echo "  Done: ${task}"
done

echo ""
echo "========== All top-k visualizations finished =========="
