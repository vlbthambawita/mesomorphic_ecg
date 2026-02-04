#!/usr/bin/env bash
# Visualize top-k POSITIVE and NEGATIVE segments for IMN final predictions
# using the new --top_k_pos / --top_k_neg arguments in one_linear_04022026.py.
#
# For each task, this script:
#   1. Finds the latest run directory under BASE_OUT_<RATE>Hz/<task>/
#   2. Picks the latest best-imn-*.ckpt checkpoint
#   3. Runs one_linear_04022026.py in inference-only mode with:
#        - top_k_leads      : highlight most important leads
#        - top_k_pos        : per-lead segments with strongest POSITIVE (towards MI/STTC/CD/HYP) evidence
#        - top_k_neg        : per-lead segments with strongest NEGATIVE (towards NORM) evidence
#      and writes PDFs into a fresh inference_* folder next to the checkpoint.

set -e

export CUDA_VISIBLE_DEVICES=0

# --- Config ---
SCRIPT="one_linear_04022026.py"
DATA_PATH="/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"
# Base directory that contains task subfolders with trained runs.
# Example layout:
#   ${BASE_OUT}_100Hz/norm_vs_mi/2026-02-03_07-27-00/best-imn-epoch*=*.ckpt
BASE_OUT="runs/imn_ecg_transition_net_GM_one_linear"

# Arguments:
#   $1: TOP_K_LEADS (default: 2)
#   $2: TOP_K_POS   (default: 2)
#   $3: TOP_K_NEG   (default: 2)
#   $4: WINDOW      (default: 50)
#   $5: STRIDE      (default: 25)
TOP_K_LEADS="${1:-2}"
TOP_K_POS="${2:-2}"
TOP_K_NEG="${3:-2}"
WINDOW="${4:-50}"
STRIDE="${5:-25}"

# Tasks and sampling rate
TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)
RATE=100

echo "========== Top-K POS/NEG Segments Visualization =========="
echo "  TOP_K_LEADS=${TOP_K_LEADS}"
echo "  TOP_K_POS=${TOP_K_POS}"
echo "  TOP_K_NEG=${TOP_K_NEG}"
echo "  WINDOW=${WINDOW}, STRIDE=${STRIDE}"
echo "==========================================================="

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
  echo "  Output base dir: ${LATEST_RUN_DIR}"

  # Name PDFs to reflect POS/NEG visualization settings
  VIZ_PDF="viz_${task}_top${TOP_K_LEADS}leads_top${TOP_K_POS}pos_top${TOP_K_NEG}neg_inference.pdf"

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
    --top_k_pos "${TOP_K_POS}" \
    --top_k_neg "${TOP_K_NEG}" \
    --n_pos_viz 5 \
    --n_neg_viz 5 \
    --viz_random \
    --viz_pdf "${VIZ_PDF}" \
    --wandb_project mesormorphic_ecg \
    --viz_ecg_plot \
    --viz_ecg_plot_n 3

  echo "  Done: ${task}"
done

echo ""
echo "========== All POS/NEG visualizations finished =========="

