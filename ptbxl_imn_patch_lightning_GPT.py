
"""
PTB-XL MI vs NORM — Interpretable Mesomorphic Neural Network (IMN) adaptation
(based on: "Interpretable Mesomorphic Neural Networks", where a shared hypernetwork
generates instance-wise linear model weights).

This script replaces the baseline 2D-CNN+GradCAM with a patch-wise IMN:
- Patch ECG into lead×time windows (overlapping allowed)
- Patch embedder (shared) -> patch tokens (vector features)
- Backbone pools tokens -> global embedding g
- Hypernetwork maps g -> instance-specific linear weights over patch tokens
- Logits are computed as a *linear* model over tokens (instance-wise), with an L1 penalty
  on the generated weights (as in the IMN paper).

Explainability:
- For a given sample and class c, patch importance is computed from contributions
  contrib[p] = sum_d (w[p,d,c] * token[p,d]).
- We visualize a lead×segment heatmap + 12-lead plots with red shading per segment.

Data:
- PTB-XL, MI vs NORM
- Splits: train folds 1-8, val fold 9, test fold 10
- Sampling rate: 100 or 500

Run:
python ptbxl_imn_patch_lightning.py --path /path/to/ptbxl --sampling_rate 500 --epochs 20

Notes:
- This is a faithful *ECG-domain* adaptation of the IMN idea from the paper:
  an end-to-end learned hypernetwork generates instance-wise linear models with sparsity (L1).
"""

import argparse
import ast
import json
import os
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
import wfdb
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# PTB-XL official loading
# -----------------------
def load_raw_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def build_superclasses(path: str):
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    return agg_df, aggregate_diagnostic




class PTBXLBinaryDatasetCE(Dataset):
    """
    X: [N, 12, L]
    y: [N] int64 {0,1}   0=NORM, 1=MI
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, per_lead_zscore: bool = True):
        super().__init__()
        assert X.ndim == 3 and X.shape[1] == 12
        assert y.ndim == 1 and y.shape[0] == X.shape[0]
        self.X = X.float()
        self.y = y.long()
        self.per_lead_zscore = per_lead_zscore

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [12, L]
        y = self.y[idx]  # scalar int64

        if self.per_lead_zscore:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        return x, y


# -----------------------
# 2D CNN baseline

# -----------------------
# Metrics# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def simple_auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    # y_true: [N] {0,1}, y_score: [N] prob for class 1
    y_true_np = y_true.detach().cpu().numpy()
    y_score_np = y_score.detach().cpu().numpy()
    if len(np.unique(y_true_np)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_np, y_score_np))


# -----------------------
# IMN model for ECG (patch-wise)
# -----------------------
class ECGPatchIMN(nn.Module):
    """
    ECG adaptation of IMN:

    - Extract patches (windows) per lead: x -> patches [B,12,T,window]
    - Patch embedder -> tokens [B,P,D] where P=12*T
    - Backbone pools tokens -> g [B,H]
    - Hypernet(g) -> W [B,P,D,C], b [B,C]
    - logits = sum_{p,d} W[p,d,c] * token[p,d] + b[c]

    This implements an instance-wise linear model over the learned token features,
    with shared hypernetwork parameters across all instances.
    """
    def __init__(
        self,
        signal_len: int,
        window: int,
        stride: int,
        token_dim: int = 16,
        backbone_dim: int = 128,
        hyper_hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert window <= signal_len and stride > 0
        self.signal_len = signal_len
        self.window = window
        self.stride = stride
        self.num_classes = num_classes
        self.token_dim = token_dim

        self.T = (signal_len - window) // stride + 1
        self.P = 12 * self.T  # patches across all leads

        # Patch embedder: small 1D CNN -> token_dim
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, token_dim),
            nn.GELU(),
        )

        # Backbone over tokens -> global embedding g
        self.backbone = nn.Sequential(
            nn.Linear(token_dim, backbone_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
        )

        # Hypernetwork: g -> weights and bias for linear classifier
        self.hyper = nn.Sequential(
            nn.Linear(backbone_dim, hyper_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hyper_hidden, self.P * self.token_dim * num_classes + num_classes),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,12,L] -> patches: [B,12,T,window]
        return x.unfold(dimension=2, size=self.window, step=self.stride)

    def forward(self, x: torch.Tensor, return_extras: bool = False):
        # x: [B,12,L]
        B, C, L = x.shape
        assert C == 12 and L == self.signal_len, f"Expected [B,12,{self.signal_len}], got {x.shape}"

        patches = self._patchify(x)                 # [B,12,T,window]
        B, C, T, W = patches.shape
        patches = patches.reshape(B * C * T, 1, W)  # [B*12*T,1,window]
        tokens = self.patch_embed(patches)          # [B*12*T, D]
        tokens = tokens.reshape(B, C * T, self.token_dim)  # [B,P,D]

        # pool tokens -> g, then backbone
        g0 = tokens.mean(dim=1)  # [B,D]
        g = self.backbone(g0)    # [B,H]

        out = self.hyper(g)  # [B, P*D*C + C]
        w_flat = out[:, : self.P * self.token_dim * self.num_classes]
        b = out[:, self.P * self.token_dim * self.num_classes :]

        W = w_flat.view(B, self.P, self.token_dim, self.num_classes)  # [B,P,D,C]
        logits = torch.einsum("bpd,bpdc->bc", tokens, W) + b          # [B,C]

        if return_extras:
            return logits, W, b, tokens
        return logits


class IMNLightning(pl.LightningModule):
    def __init__(
        self,
        signal_len: int,
        window: int,
        stride: int,
        token_dim: int = 16,
        backbone_dim: int = 128,
        hyper_hidden: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: list[float] | None = None,
        lambda_l1: float = 1e-3,
        scheduler_type: str | None = "cosine",
        scheduler_params: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ECGPatchIMN(
            signal_len=signal_len,
            window=window,
            stride=stride,
            token_dim=token_dim,
            backbone_dim=backbone_dim,
            hyper_hidden=hyper_hidden,
            num_classes=2,
            dropout=dropout,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self._class_weights = class_weights
        self.lambda_l1 = lambda_l1

        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}

        self.val_probs = []
        self.val_y = []

    def _ce_weight(self):
        if self._class_weights is None:
            return None
        w = torch.tensor(self._class_weights, dtype=torch.float32, device=self.device)
        return w

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.scheduler_type is None or self.scheduler_type == "none":
            return optimizer

        if self.scheduler_type == "cosine":
            max_epochs = getattr(self.trainer, "max_epochs", None) or 100
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, **self.scheduler_params
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if self.scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.scheduler_params.get("factor", 0.5),
                patience=self.scheduler_params.get("patience", 5),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_auc"}}

        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, W, _, _ = self.model(x, return_extras=True)
        ce = F.cross_entropy(logits, y, weight=self._ce_weight())

        # L1 penalty on generated weights (instance-wise), averaged over batch
        l1 = W.abs().mean()
        loss = ce + self.lambda_l1 * l1

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ce", ce, on_step=False, on_epoch=True)
        self.log("train_l1", l1, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, W, _, _ = self.model(x, return_extras=True)
        ce = F.cross_entropy(logits, y, weight=self._ce_weight())
        l1 = W.abs().mean()
        loss = ce + self.lambda_l1 * l1

        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_ce", ce, on_step=False, on_epoch=True)
        self.log("val_l1", l1, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_probs.append(prob.detach().cpu())
        self.val_y.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_y) if len(self.val_y) else None
        y_score = torch.cat(self.val_probs) if len(self.val_probs) else None
        if y_true is None:
            return
        auc = simple_auc_roc(y_true.float(), y_score.float())
        self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_probs.clear()
        self.val_y.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self._ce_weight())

        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"y": y.detach().cpu(), "p": prob.detach().cpu()}


# -----------------------
# Explainability visualization to PDF (IMN weights)
# -----------------------
@torch.no_grad()
def visualize_pos_neg_to_pdf_imn(
    lightning_module: IMNLightning,
    dataset: Dataset,
    device,
    pdf_path: str,
    sampling_rate: int,
    window: int,
    stride: int,
    n_pos: int,
    n_neg: int,
    pos_class_name: str = "MI",
    random_pick: bool = False,
    seed: int = 123,
    lead_names=None,
):
    model = lightning_module.model.to(device)
    model.eval()

    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    # indices for positives/negatives in dataset
    ys = torch.stack([dataset[i][1] for i in range(len(dataset))]).numpy()
    pos_idx = np.where(ys == 1)[0].tolist()
    neg_idx = np.where(ys == 0)[0].tolist()

    rng = np.random.RandomState(seed)
    if random_pick:
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

    pick_pos = pos_idx[:n_pos]
    pick_neg = neg_idx[:n_neg]
    picks = [("POS", i) for i in pick_pos] + [("NEG", i) for i in pick_neg]

    # Segment count for plotting
    # (This is determined by window/stride and L)
    _, y0 = dataset[0]
    x0, _ = dataset[0]
    Lsig = int(x0.shape[-1])
    Tseg = (Lsig - window) // stride + 1

    with PdfPages(pdf_path) as pdf:
        for tag, idx in picks:
            x, y = dataset[idx]
            x_b = x.unsqueeze(0).to(device)  # [1,12,L]

            logits, W, b, tokens = model(x_b, return_extras=True)
            prob = torch.softmax(logits, dim=1)[0, 1].item()

            # Compute contributions per patch for class 1 (MI)
            # tokens: [1,P,D], W: [1,P,D,C]
            contrib = (tokens * W[..., 1]).sum(dim=2)[0]  # [P]
            # reshape to [12,T]
            contrib = contrib.view(12, Tseg).detach().cpu().numpy()

            # Importance map (normalize to [0,1] per sample)
            imp = np.abs(contrib)
            imp = imp - imp.min()
            imp = imp / (imp.max() + 1e-6)

            # Plot: heatmap + lead plots with shading
            fig = plt.figure(figsize=(11, 8.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=0.25)

            ax0 = fig.add_subplot(gs[0, 0])
            im = ax0.imshow(imp, aspect="auto", vmin=0, vmax=1, cmap="Reds")
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            ax0.set_title(f"{tag} | true={int(y.item())} | P({pos_class_name})={prob:.3f} | idx={idx}")
            fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

            ax1 = fig.add_subplot(gs[1, 0])
            x_np = x.detach().cpu().numpy()  # [12,L]
            t = np.arange(Lsig) / sampling_rate

            # Plot each lead offset vertically
            offsets = np.arange(12)[::-1] * 2.5
            for li in range(12):
                ysig = x_np[li] + offsets[li]
                ax1.plot(t, ysig, linewidth=0.7)
                # segment shading
                for s in range(Tseg):
                    st = s * stride
                    en = st + window
                    if en > Lsig:
                        break
                    alpha = float(imp[li, s]) * 0.6
                    if alpha > 0:
                        ax1.axvspan(st / sampling_rate, en / sampling_rate, alpha=alpha)

            ax1.set_yticks(offsets)
            ax1.set_yticklabels(lead_names[::-1])
            ax1.set_xlabel("Time (s)")
            ax1.set_title("12-lead ECG with segment-wise IMN importance (red shading)")
            ax1.grid(True, alpha=0.2)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Root path to PTB-XL (folder containing records/ and ptbxl_database.csv)")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # IMN params
    parser.add_argument("--token_dim", type=int, default=16)
    parser.add_argument("--backbone_dim", type=int, default=128)
    parser.add_argument("--hyper_hidden", type=int, default=256)
    parser.add_argument("--lambda_l1", type=float, default=1e-3)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Visualization segmentation (also used by IMN patching!)
    parser.add_argument("--window", type=int, default=None,
                        help="Window in samples. Default: 0.5s => 50@100Hz or 250@500Hz.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride in samples. Default: window/2 (50% overlap).")
    parser.add_argument("--n_pos", type=int, default=8)
    parser.add_argument("--n_neg", type=int, default=8)
    parser.add_argument("--viz_random", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    
    path = args.path if args.path.endswith("/") else args.path + "/"

    # Load metadata
    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    _, aggregate_diagnostic = build_superclasses(path)
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

    # Binary task: NORM vs MI (exclusive)
    pos_class = "MI"
    labels = Y["diagnostic_superclass"].values
    is_pos = np.array([pos_class in l for l in labels])
    is_norm = np.array(["NORM" in l for l in labels])
    keep = (is_pos ^ is_norm)
    Yf = Y[keep].copy()

    y_bin = np.array([1 if pos_class in l else 0 for l in Yf["diagnostic_superclass"].values], dtype=np.int64)
    print(f"NORM vs {pos_class}: N={len(Yf)} | {pos_class}={int(y_bin.sum())} | NORM={int((1-y_bin).sum())}")

    # Load signals
    print(f"Loading signals at {args.sampling_rate}Hz ...")
    X = load_raw_data(Yf, args.sampling_rate, path)  # [N,L,12]
    X = np.transpose(X, (0, 2, 1))                   # [N,12,L]
    N, C, Lsig = X.shape
    print("X shape:", X.shape)

    # Folds: train 1-8, validation 9+10 (combined)
    fold = Yf["strat_fold"].values
    train_mask = (fold >= 1) & (fold <= 8)
    val_mask = (fold >= 9)

    X_train, y_train = X[train_mask], y_bin[train_mask]
    X_val, y_val     = X[val_mask], y_bin[val_mask]

    print("Split sizes: train=", len(y_train), ", val=", len(y_val))

    # Default window/stride
    Lsig = X.shape[-1]
    if args.window is None:
        window = 50 if args.sampling_rate == 100 else 250  # 0.5s
    else:
        window = args.window
    if args.stride is None:
        stride = window // 2
    else:
        stride = args.stride

    assert window <= Lsig and stride > 0
    Tseg = (Lsig - window) // stride + 1
    print(f"Patch/Visualization segments: window={window}, stride={stride}, T={Tseg}, P={12*Tseg}")

    
    # Torch tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t   = torch.from_numpy(X_val).float()
    y_val_t   = torch.from_numpy(y_val).long()

    # Datasets/loaders
    train_ds = PTBXLBinaryDatasetCE(X_train_t, y_train_t, per_lead_zscore=True)
    val_ds   = PTBXLBinaryDatasetCE(X_val_t, y_val_t, per_lead_zscore=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # We follow the original baseline protocol: use folds 9+10 as the held-out set
    # for both evaluation and visualization.
    test_loader = val_loader
    test_ds = val_ds

    # Class weights (optional) — keep same behavior as baseline: inverse frequency
    counts = np.bincount(y_train, minlength=2)
    class_weights = (counts.sum() / (2.0 * np.maximum(counts, 1))).tolist()
    print("Class weights:", class_weights)

    # Run dir
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs_imn", run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    # Logger
    logger = None
    if not args.no_wandb:
        logger = WandbLogger(project="ptbxl-imn", name=run_name, save_dir=run_dir)

    # Lightning module
    lit = IMNLightning(
        signal_len=Lsig,
        window=window,
        stride=stride,
        token_dim=args.token_dim,
        backbone_dim=args.backbone_dim,
        hyper_hidden=args.hyper_hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        lambda_l1=args.lambda_l1,
    )

    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=10),
        ModelCheckpoint(monitor="val_auc", mode="max", save_top_k=1, filename="best"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        default_root_dir=run_dir,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    trainer.fit(lit, train_loader, val_loader)

    # Test
    test_out = trainer.test(lit, test_loader, ckpt_path="best")
    print("Test metrics:", test_out)

    # Save checkpoint
    ckpt_path = os.path.join(run_dir, "best.ckpt")
    # Lightning saves it in the checkpoint callback dir, but we also note it here.

    # Metrics CSV on test set
    # (collect probs)
    lit.eval()
    ys_all, ps_all = [], []
    device = lit.device
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = lit.model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            ys_all.append(yb.numpy())
            ps_all.append(prob)
    y_true = np.concatenate(ys_all)
    y_score = np.concatenate(ps_all)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(run_dir, "test_metrics.csv"), index=False)
    print("Saved test_metrics.csv")

    # PDF explanations
    pdf_path = os.path.join(run_dir, "imn_explanations.pdf")
    visualize_pos_neg_to_pdf_imn(
        lightning_module=lit,
        dataset=test_ds,
        device=device,
        pdf_path=pdf_path,
        sampling_rate=args.sampling_rate,
        window=window,
        stride=stride,
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        random_pick=args.viz_random,
        seed=args.seed,
    )
    print("Saved PDF:", pdf_path)

    if logger is not None:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
