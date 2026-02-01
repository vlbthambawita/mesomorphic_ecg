#!/usr/bin/env bash
cd "$(dirname "$0")"

# Binary task: norm_vs_mi | norm_vs_sttc | norm_vs_cd | norm_vs_hyp
TASK="norm_vs_sttc"

python script_30012026_baseline_v3_related.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --task "$TASK" \
  --batch_size 64 \
  --sampling_rate 500 \
  --epochs 100 \
  --n_pos_viz 10 \
  --n_neg_viz 10 \
  --viz_random \
  --out_dir runs/baseline_hypernet_classifier \
  --wandb_project mesormorphic_ecg \
  --early_stop_patience 10 \
  --scheduler step

TASK="norm_vs_cd"

python script_30012026_baseline_v3_related.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --task "$TASK" \
  --batch_size 64 \
  --sampling_rate 500 \
  --epochs 100 \
  --n_pos_viz 10 \
  --n_neg_viz 10 \
  --viz_random \
  --out_dir runs/baseline_hypernet_classifier \
  --wandb_project mesormorphic_ecg \
  --early_stop_patience 10 \
  --scheduler step

TASK="norm_vs_hyp"

python script_30012026_baseline_v3_related.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --task "$TASK" \
  --batch_size 64 \
  --sampling_rate 500 \
  --epochs 100 \
  --n_pos_viz 10 \
  --n_neg_viz 10 \
  --viz_random \
  --out_dir runs/baseline_hypernet_classifier \
  --wandb_project mesormorphic_ecg \
  --early_stop_patience 10 \
  --scheduler step

TASK="norm_vs_mi"

python script_30012026_baseline_v3_related.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --task "$TASK" \
  --batch_size 64 \
  --sampling_rate 500 \
  --epochs 100 \
  --n_pos_viz 10 \
  --n_neg_viz 10 \
  --viz_random \
  --out_dir runs/baseline_hypernet_classifier \
  --wandb_project mesormorphic_ecg \
  --early_stop_patience 10 \
  --scheduler step