#python script_25012026_v1.py --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 --sampling_rate 100 --epochs 20 --batch_size 64

python script_25012026_v3_Selected_01_lightning_save_wgen_tokens.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --batch_size 3 \
  --sampling_rate 500 \
  --epochs 1 \
  --n_pos_viz 10 \
  --n_neg_viz 10 \
  --viz_random \
  --out_dir runs/mi_vs_norm_imn \
  --wandb_project mesormorphic_ecg \
  --early_stop_patience 10 \
  --scheduler step \
  --viz_topk \
  --viz_mc_dropout \
  --save_wgen_tokens \
  --save_wgen_tokens_format all \
  --save_wgen_tokens_split test \
  --save_wgen_tokens_max_samples 50 \
  --exp_name test_run \
  --verbose