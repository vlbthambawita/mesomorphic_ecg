#python script_25012026_v1.py --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 --sampling_rate 100 --epochs 20 --batch_size 64

python script_25012026_v3.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --sampling_rate 500 \
  --epochs 20 \
  --n_pos_viz 40 \
  --n_neg_viz 40 \
  --viz_random \
  --out_dir runs/mi_vs_norm_imn