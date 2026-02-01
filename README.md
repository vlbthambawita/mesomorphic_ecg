## ECG Patch IMN (Lightning) – `script_25012026_v3_Selected_01_lightning_save_wgen_tokens.py`

This script trains and evaluates a **binary MI vs NORM classifier** on PTB‑XL ECGs using the patch‑wise IMN model, logs runs to **Weights & Biases**, produces **visual explanation PDFs**, and can optionally **save the final generated weights (`Wgen`) and tokens** for downstream analysis.

---

### 1. Prerequisites

- **Python**: 3.9+ recommended.
- **GPU** (recommended): CUDA-capable GPU with recent NVIDIA drivers.
- **Python packages** (minimum):
  - `torch`, `pytorch-lightning`
  - `numpy`, `pandas`, `matplotlib`, `scikit-learn`
  - `wfdb` (for PTB‑XL waveform loading)
  - `pyyaml`
  - `wandb`
  - `h5py` (only if you want `--save_wgen_tokens_format h5` or `all`)

Example (conda + pip):

```bash
conda create -n ecg-imn python=3.10 -y
conda activate ecg-imn
pip install torch pytorch-lightning wandb numpy pandas matplotlib scikit-learn wfdb pyyaml h5py
```

---

### 2. PTB‑XL dataset layout

Download PTB‑XL from PhysioNet and point `--path` to the **root folder** that contains:

- `ptbxl_database.csv`
- `scp_statements.csv`
- `records100/` and/or `records500/`

Example layout:

```text
/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1/
  ├── ptbxl_database.csv
  ├── scp_statements.csv
  ├── records100/
  └── records500/
```

You will pass this directory to `--path`.

---

### 3. Basic usage

From the `mesormorphic_ecg` directory:

```bash
cd /work/vajira/DL2025/mesormorphic_ecg

python script_25012026_v3_Selected_01_lightning_save_wgen_tokens.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1
```

This will:

- Load PTB‑XL at the default sampling rate (`--sampling_rate 500` by default).
- Train the ECG Patch IMN model with default hyper‑parameters.
- Create a new run directory under `runs/mi_vs_norm_imn/<script>_<timestamp>/`.
- Log metrics and artifacts to **Weights & Biases** (online by default).
- Evaluate on the test fold and generate default visualization PDFs.

---

### 4. Common options

- **Dataset / IO**
  - `--path PATH` (required): PTB‑XL root directory (see above).
  - `--sampling_rate {100,500}`: 100 Hz or 500 Hz filtered signals (default: `500`).
  - `--out_dir DIR`: Base output directory for runs (default: `runs/mi_vs_norm_imn`).
  - `--exp_name NAME`: Optional experiment name; included in run folder name.

- **Training**
  - `--epochs INT` (default: `20`)
  - `--batch_size INT` (default: `64`)
  - `--lr FLOAT` (default: `1e-3`)
  - `--weight_decay FLOAT` (default: `1e-4`)
  - `--l1_lambda FLOAT` (default: `0.05`)
  - `--token_dim INT` (default: `64`)
  - `--hyper_hidden INT` (default: `256`)

- **Patching**
  - `--window INT`: Patch window in samples. Default is **0.5 s**, i.e. `50` at 100 Hz or `250` at 500 Hz.
  - `--stride INT`: Patch stride in samples. Default is `window // 2` (50 % overlap).

- **Hardware / reproducibility**
  - `--seed INT` (default: `42`)
  - `--num_workers INT` (default: `2`)
  - `--accelerator {"auto","gpu","cpu"}` (default: `"auto"`)
  - `--devices INT` (default: `1`)

- **Weights & Biases**
  - `--wandb_project NAME` (default: `ecg-patch-imn`)
  - `--wandb_name NAME`: Custom run name; otherwise script constructs one from filename + timestamp.
  - `--wandb_offline`: Run WandB in offline mode.

- **Scheduler / early stopping**
  - `--scheduler {"cosine","step","reduce_on_plateau","cosine_restarts","none"}` (default: `cosine`)
  - `--scheduler_params JSON_STRING` (default: `"{}"`)  
    Example: `--scheduler_params '{"T_0": 10, "T_mult": 2}'`
  - `--early_stop_patience INT` (default: `10`)
  - `--early_stop_min_delta FLOAT` (default: `0.0`)
  - `--early_stop_monitor {"val_loss","val_auc","val_acc"}` (default: `val_auc`)
  - `--early_stop_mode {"min","max"}` (default: `max`)

- **Visualization**
  - `--viz_pdf FNAME` (default: `viz_mi_vs_norm_test.pdf`)
  - `--n_pos_viz INT` / `--n_neg_viz INT` (default: `25`/`25`)
  - `--viz_random`: Randomly pick positives/negatives for visualization.
  - `--viz_seed INT` (default: `123`)
  - `--viz_topk`: Enable top‑k lead/patch visualizations.
  - `--topk_leads INT` (default: `3`)
  - `--topk_patches INT` (default: `10`)
  - `--viz_topk_pdf FNAME` (default: `viz_topk_test.pdf`)
  - `--viz_mc_dropout`: Enable Monte‑Carlo dropout uncertainty visualizations.
  - `--mc_samples INT` (default: `50`)
  - `--viz_uncertainty_pdf FNAME` (default: `viz_uncertainty_test.pdf`)

- **Saving Wgen and tokens**
  - `--save_wgen_tokens`: Enable saving final `Wgen` and `tokens`.
  - `--save_wgen_tokens_format {"npz","pt","h5","all"}` (default: `npz`).
  - `--save_wgen_tokens_split {"test","val","both"}` (default: `test`).
  - `--save_wgen_tokens_max_samples INT`: Optional cap on number of samples to save.

- **Misc**
  - `--verbose`: Extra logging, especially from the model forward pass.

For the authoritative list, see the argument parser in `script_25012026_v3_Selected_01_lightning_save_wgen_tokens.py`.

---

### 5. Example runs

#### 5.1. Full 500 Hz run with visualizations

```bash
python script_25012026_v3_Selected_01_lightning_save_wgen_tokens.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --sampling_rate 500 \
  --epochs 100 \
  --batch_size 64 \
  --n_pos_viz 10 \
  --n_neg_viz 10 \
  --viz_random \
  --viz_topk \
  --viz_mc_dropout \
  --out_dir runs/mi_vs_norm_imn \
  --wandb_project mesormorphic_ecg \
  --early_stop_patience 10 \
  --scheduler step \
  --exp_name my_experiment_name
```

#### 5.2. Same run + save Wgen/tokens (npz, test split)

```bash
python script_25012026_v3_Selected_01_lightning_save_wgen_tokens.py \
  --path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \
  --sampling_rate 500 \
  --epochs 100 \
  --batch_size 64 \
  --viz_random \
  --viz_topk \
  --viz_mc_dropout \
  --save_wgen_tokens \
  --save_wgen_tokens_format npz \
  --save_wgen_tokens_split test \
  --out_dir runs/mi_vs_norm_imn \
  --wandb_project mesormorphic_ecg \
  --exp_name my_experiment_with_wgen
```

---

### 6. Outputs

Each run creates a directory:

```text
{out_dir}/{script_name}[_{exp_name}]_{timestamp}/
```

Inside you will typically find:

- `args.yaml`: All parsed CLI arguments for reproducibility.
- `best-epoch=XX-val_auc=YYYY.ckpt` and `last.ckpt`: best and last Lightning checkpoints.
- `model.txt`: Text dump of the underlying PyTorch model architecture.
- `viz_mi_vs_norm_test.pdf`: Per‑lead explanation plots on the test set.
- `viz_topk_test.pdf`: (if `--viz_topk`) top‑k lead/patch visualizations.
- `viz_uncertainty_test.pdf`: (if `--viz_mc_dropout`) MC‑dropout uncertainty plots.
- `wgen_tokens_test.*` / `wgen_tokens_val.*`: (if `--save_wgen_tokens`) saved Wgen/tokens and metadata.

---

### 7. Interpreting saved Wgen/tokens

When `--save_wgen_tokens` is enabled, the script collects:

- `Wgen`: \([N, P, D]\) generated weights per patch/token.
- `tokens`: \([N, P, D]\) patch tokens.
- `labels`, `logits`, `probs`: scalar label, logit, and probability per sample.
- `P`, `T`, `token_dim`, `N`: metadata.

**Recommended formats:**

- **`npz` (default)** – good for NumPy‑based analysis and portability.
- **`pt`** – best if you will reload into PyTorch scripts.
- **`h5`** – good for very large runs where chunked/partial I/O matters.
- **`all`** – write all three and compare sizes / load times later.

Example (Python) for loading a `.npz` file:

```python
import numpy as np

d = np.load("wgen_tokens_test.npz")
Wgen   = d["Wgen"]      # [N, P, D]
tokens = d["tokens"]    # [N, P, D]
labels = d["labels"]    # [N]
probs  = d["probs"]     # [N]
P      = int(d["P"])
T      = int(d["T"])
D      = int(d["token_dim"])
N      = int(d["N"])
```

---

### 8. Notes

- The script **filters to MI vs NORM only** using PTB‑XL superclasses and uses folds 1–8 for train, 9 for val, 10 for test.
- Make sure `--window` and `--stride` are compatible with the signal length; defaults are chosen to be safe for typical PTB‑XL lengths.
- To re‑run an experiment with slightly different hyper‑parameters, you can copy the `args.yaml` from a previous run and modify CLI flags accordingly.

