#!/usr/bin/env bash
# Generate visualization outputs for IMN Single Linear Output models:
#   - script_02022026_v7_IMN_GM_2_with_transition_net_with_one_linear_eq.py (uses --top_k_leads, --top_k_segments)
#   - one_linear_v2.py (uses --top_k_pos, --top_k_neg; set PYTHON_SCRIPT to override)
#
# Produces:
#   - IMN explanation PDFs (heatmap + ECG traces; script_02022026: top leads/segments; one_linear_v2: Red/Blue pos/neg)
#   - Optionally: ECG+IMN heatmap PDFs (--viz_ecg_plot)
#
# Usage:
#   ./run_one_linear_v2_viz.sh                           # defaults, auto-discover checkpoint
#   ./run_one_linear_v2_viz.sh 2 3 5 5                   # top 2 leads, 3 segments, 5 samples each
#   ./run_one_linear_v2_viz.sh 2 3 10 10 "" /path/to/best-imn.ckpt
#
# Arguments:
#   $1: TOP_K_LEADS   - top-k leads to highlight (default: 2)
#   $2: TOP_K_SEGMENTS - top-k segments per lead (default: 3)
#   $3: N_POS_VIZ   - number of positive-class samples to visualize (default: 5)
#   $4: N_NEG_VIZ   - number of negative-class samples to visualize (default: 5)
#   $5: CKPT        - checkpoint path (optional; if empty, auto-discover from BASE_OUT)

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Config ---
# Use script_02022026_v7_IMN_GM_2_with_transition_net_with_one_linear_eq.py by default
SCRIPT="${PYTHON_SCRIPT:-$(dirname -- "$0")/script_02022026_v7_IMN_GM_2_with_transition_net_with_one_linear_eq.py}"
DATA_PATH="/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"
# Base directory containing task subfolders with trained runs.
# Layout: ${BASE_OUT}/${task}/YYYY-MM-DD_HH-MM-SS/best-imn-epoch*.ckpt
BASE_OUT="${BASE_OUT:-/work/vajira/DL2025/mesormorphic_ecg/runs/imn_ecg_transition_net_GM_one_linear_100Hz}"

# Arguments
# For script_02022026: TOP_K_LEADS, TOP_K_SEGMENTS
# For one_linear_v2: TOP_K_POS, TOP_K_NEG (Red/Blue)
TOP_K_LEADS="${1:-2}"
TOP_K_SEGMENTS="${2:-3}"
N_POS_VIZ="${3:-5}"
N_NEG_VIZ="${4:-5}"
CKPT="${5:-}"

# Window/stride for segment aggregation
# 100 Hz: w=50, s=25  |  500 Hz: w=250, s=125
RATE="${RATE:-100}"
if [[ "${RATE}" -eq 100 ]]; then
  WINDOW=50
  STRIDE=25
else
  WINDOW=250
  STRIDE=125
fi

# Tasks
TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)

# Optional: enable ECG+IMN heatmap plots (set VIZ_ECG_PLOT=1 to enable)
VIZ_ECG_PLOT="${VIZ_ECG_PLOT:-0}"
VIZ_ECG_PLOT_N="${VIZ_ECG_PLOT_N:-3}"

echo "========== IMN One-Linear Visualization =========="
echo "  Script: ${SCRIPT}"
echo "  TOP_K_LEADS=${TOP_K_LEADS}, TOP_K_SEGMENTS=${TOP_K_SEGMENTS}"
echo "  N_POS_VIZ=${N_POS_VIZ}, N_NEG_VIZ=${N_NEG_VIZ}"
echo "  WINDOW=${WINDOW}, STRIDE=${STRIDE}, RATE=${RATE} Hz"
echo "  BASE_OUT=${BASE_OUT}"
echo "  VIZ_ECG_PLOT=${VIZ_ECG_PLOT}"
echo "===================================================="

for task in "${TASKS[@]}"; do
  echo ""
  echo "========== Task: ${task} @ ${RATE} Hz =========="

  CKPT_PATH="${CKPT}"
  if [[ -z "${CKPT_PATH}" ]]; then
    TASK_DIR="${BASE_OUT}/${task}"
    if [[ ! -d "${TASK_DIR}" ]]; then
      # Try with rate suffix (e.g. imn_ecg_one_linear_v2_100Hz)
      TASK_DIR="${BASE_OUT}_${RATE}Hz/${task}"
    fi
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
  fi

  echo "  Checkpoint: ${CKPT_PATH}"

  VIZ_PDF="viz_${task}_top${TOP_K_LEADS}leads_top${TOP_K_SEGMENTS}segs_inference.pdf"

  EXTRA_ARGS=()
  if [[ "${VIZ_ECG_PLOT}" -eq 1 ]]; then
    EXTRA_ARGS+=(--viz_ecg_plot --viz_ecg_plot_n "${VIZ_ECG_PLOT_N}")
  fi

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
    --n_pos_viz "${N_POS_VIZ}" \
    --n_neg_viz "${N_NEG_VIZ}" \
    --viz_random \
    --viz_pdf "${VIZ_PDF}" \
    --wandb_project mesormorphic_ecg \
    --out_dir "${BASE_OUT}" \
    "${EXTRA_ARGS[@]}"

  echo "  Done: ${task}"
done

echo ""
echo "========== All visualizations finished =========="
