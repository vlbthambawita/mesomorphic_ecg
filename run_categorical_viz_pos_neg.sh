#!/usr/bin/env bash
# Visualize POSITIVE (Red) and NEGATIVE (Blue) segments for IMN categorical model.
# Uses categorical_04022206.py with --top_k_pos / --top_k_neg for class-specific
# intrinsic attribution: Red = supports target class, Blue = refutes target class.
#
# NOTE: Checkpoints must be trained with categorical_04022206.py (not other IMN scripts).
#       To train first: python categorical_04022206.py --path $DATA_PATH --epochs 20 ...
#
# Usage:
#   ./run_categorical_viz_pos_neg.sh [TOP_K_POS] [TOP_K_NEG] [N_POS_VIZ] [N_NEG_VIZ] [CKPT]
#
# Examples:
#   ./run_categorical_viz_pos_neg.sh                    # defaults, auto-discover checkpoint
#   ./run_categorical_viz_pos_neg.sh 3 3 10 10         # top 3 pos/neg per lead, 10 samples each
#   ./run_categorical_viz_pos_neg.sh 2 2 5 5 /path/to/best-imn.ckpt

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Config ---
SCRIPT="categorical_04022206.py"
DATA_PATH="/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"
BASE_OUT="/work/vajira/DL2025/mesormorphic_ecg/runs/imn_ecg_transition_net_GM_100Hz"

# Arguments:
#   $1: TOP_K_POS   - top-k positive (Red) segments per lead (default: 3)
#   $2: TOP_K_NEG   - top-k negative (Blue) segments per lead (default: 3)
#   $3: N_POS_VIZ   - number of positive-class samples to visualize (default: 5)
#   $4: N_NEG_VIZ   - number of negative-class samples to visualize (default: 5)
#   $5: CKPT        - checkpoint path for inference-only (optional; if empty, auto-discover)
TOP_K_POS="${1:-3}"
TOP_K_NEG="${2:-3}"
N_POS_VIZ="${3:-5}"
N_NEG_VIZ="${4:-5}"
CKPT="${5:-}"

# Window/stride for segment aggregation
# 100 Hz: w=50, s=25  |  500 Hz: w=250, s=125
RATE=100
WINDOW=50
STRIDE=25

# Tasks
TASKS=(norm_vs_mi norm_vs_sttc norm_vs_cd norm_vs_hyp)

echo "========== Categorical IMN: Positive & Negative Visualization =========="
echo "  TOP_K_POS=${TOP_K_POS} (Red: supports target class)"
echo "  TOP_K_NEG=${TOP_K_NEG} (Blue: refutes target class)"
echo "  N_POS_VIZ=${N_POS_VIZ}, N_NEG_VIZ=${N_NEG_VIZ}"
echo "  WINDOW=${WINDOW}, STRIDE=${STRIDE}, RATE=${RATE} Hz"
echo "========================================================================"

for task in "${TASKS[@]}"; do
  echo ""
  echo "========== Task: ${task} @ ${RATE} Hz =========="

  CKPT_PATH="${CKPT}"
  if [[ -z "${CKPT_PATH}" ]]; then
    TASK_DIR="${BASE_OUT}/${task}"
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

  VIZ_PDF="viz_${task}_top${TOP_K_POS}pos_top${TOP_K_NEG}neg_inference.pdf"

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
    --top_k_pos "${TOP_K_POS}" \
    --top_k_neg "${TOP_K_NEG}" \
    --n_pos_viz "${N_POS_VIZ}" \
    --n_neg_viz "${N_NEG_VIZ}" \
    --viz_random \
    --viz_pdf "${VIZ_PDF}" \
    --wandb_project mesormorphic_ecg \
    --out_dir "${BASE_OUT}"

  echo "  Done: ${task}"
done

echo ""
echo "========== All positive/negative visualizations finished =========="
