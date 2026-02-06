"""
PTB-XL binary baseline with IMN-style network using 2D CNN + PDF report.

- Model: IMN (Interpretable Mixture of Networks) with 2D CNN patch embed and 2D CNN hypernetwork.
  - Patches: time segments of shape (12, window) -> T patches per sample.
  - Patch embed: 2D CNN on [1, 12, window] -> token_dim.
  - Hyper net: 2D CNN on token grid [1, T, D] -> pool -> linear -> Wgen [B,T,D], b; logit = (Wgen*tokens).sum + b.
  - Output: logits [B,2] for CE (0=NORM, 1=MI).
- Loss: CrossEntropy with optional class weights.
- XAI: Native IMN heatmap (segment importance [12,T], broadcast from [T]).
- PDF: N positives and M negatives, heatmap + 12 lead plots with shading by segment importance.
- Same splits/task/sampling as script_26012026_v7_gradcam_library_baseline.py.

Run:
python script_26012026_imn2d_gradcam_baseline.py --path /path/to/ptbxl --sampling_rate 500 --epochs 20 --viz_random
"""

import argparse
import ast
import json
import os
import random
import numpy as np
import pandas as pd
import wfdb
from datetime import datetime
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
import wandb
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
    data = np.array([signal for signal, meta in data])  # [N, L, 12]
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


# -----------------------
# Dataset
# -----------------------
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
# 2D patchify: [B,12,L] -> [B,T,12,window]
# -----------------------
def patchify_ecg_2d(x: torch.Tensor, window: int, stride: int):
    """
    x: [B, 12, L]
    returns patches: [B, T, 12, window], T = (L - window) // stride + 1
    """
    B, C, L = x.shape
    assert C == 12
    assert window <= L and stride > 0
    T = (L - window) // stride + 1
    patches = torch.zeros(B, T, 12, window, device=x.device, dtype=x.dtype)
    for t in range(T):
        s = t * stride
        e = s + window
        patches[:, t, :, :] = x[:, :, s:e]
    return patches, T


# -----------------------
# IMN with 2D CNN patch embedding
# -----------------------
class ECGPatchIMN2D(nn.Module):
    """
    IMN-style model with 2D CNN patch embedding and 2D CNN hypernetwork.
    - Patches: T segments of shape (12, window).
    - Patch embed: 2D CNN on [1, 12, window] -> token_dim.
    - Hyper net: 2D CNN on token grid [1, T, D] -> pool -> linear -> Wgen [B,T,D], b.
    - logit = (Wgen*tokens).sum + b; output logits [B,2] for CrossEntropy (0=NORM, 1=MI).
    """
    def __init__(
        self,
        signal_len: int,
        window: int,
        stride: int,
        token_dim: int = 64,
        hyper_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.signal_len = signal_len
        self.window = window
        self.stride = stride
        self.token_dim = token_dim

        self.T = (signal_len - window) // stride + 1

        # 2D CNN patch embed: input [B*T, 1, 12, window] -> [B*T, token_dim]
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, token_dim),
            nn.GELU(),
        )
        # For Grad-CAM on patch embed (optional): last conv
        self._target_layer = self.patch_embed[3]  # Conv2d(16, 32, ...)

        # Hypernetwork as 2D CNN: input token grid [B, 1, T, D] -> convs -> pool -> linear -> Wgen, b
        self.hyper_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, hyper_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hyper_hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.hyper_linear = nn.Linear(hyper_hidden, self.T * token_dim + 1)

    def _forward_full(self, x):
        """Returns logits [B,2], Wgen [B,T,D], tokens [B,T,D], T."""
        B = x.shape[0]
        patches, T = patchify_ecg_2d(x, self.window, self.stride)  # [B,T,12,window]
        assert T == self.T
        # [B*T, 1, 12, window]
        p = patches.view(B * T, 1, 12, self.window)
        tok = self.patch_embed(p)  # [B*T, D]
        tokens = tok.view(B, T, self.token_dim)  # [B,T,D]

        # Hyper 2D CNN: token grid [B, T, D] -> [B, 1, T, D]
        token_grid = tokens.unsqueeze(1)  # [B, 1, T, D]
        h = self.hyper_conv(token_grid)   # [B, hyper_hidden]
        out = self.hyper_linear(h)        # [B, T*D+1]
        Wgen = out[:, :-1].view(B, T, self.token_dim)
        b = out[:, -1]
        logit_scalar = (Wgen * tokens).sum(dim=(1, 2)) + b  # [B]
        logits = torch.stack([-logit_scalar, logit_scalar], dim=1)  # [B,2]
        return logits, Wgen, tokens, T

    def forward(self, x):
        logits, _, _, _ = self._forward_full(x)
        return logits

    @torch.no_grad()
    def explain(self, x):
        """
        Returns prob [B], heatmap [B,12,T] (segment importance broadcast to 12 leads).
        """
        logits, Wgen, tokens, T = self._forward_full(x)
        prob = F.softmax(logits, dim=1)[:, 1]  # [B]
        segment_scores = (Wgen * tokens).sum(dim=2)  # [B,T]
        heatmap = segment_scores.unsqueeze(1).expand(-1, 12, -1)  # [B,12,T]
        return prob, heatmap


# -----------------------
# Lightning module (CE, class weights, val_auc)
# -----------------------
class IMN2DLightning(pl.LightningModule):
    def __init__(
        self,
        signal_len: int,
        window: int,
        stride: int,
        token_dim: int = 64,
        hyper_hidden: int = 256,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: list[float] | None = None,
        scheduler_type: str | None = "cosine",
        scheduler_params: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}

        self.model = ECGPatchIMN2D(
            signal_len=signal_len,
            window=window,
            stride=stride,
            token_dim=token_dim,
            hyper_hidden=hyper_hidden,
            dropout=dropout,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self._class_weights = class_weights

        self.val_probs = []
        self.val_y = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.scheduler_type is None or self.scheduler_type == "none":
            return optimizer

        if self.scheduler_type == "cosine":
            max_epochs = getattr(self.trainer, "max_epochs", None) or 100
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                **self.scheduler_params
            )
        elif self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_params.get("step_size", 10),
                gamma=self.scheduler_params.get("gamma", 0.1),
                **{k: v for k, v in self.scheduler_params.items() if k not in ["step_size", "gamma"]}
            )
        elif self.scheduler_type == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.scheduler_params.get("factor", 0.5),
                patience=self.scheduler_params.get("patience", 5),
                **{k: v for k, v in self.scheduler_params.items() if k not in ["factor", "patience"]}
            )
        elif self.scheduler_type == "cosine_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.scheduler_params.get("T_0", 10),
                T_mult=self.scheduler_params.get("T_mult", 2),
                **{k: v for k, v in self.scheduler_params.items() if k not in ["T_0", "T_mult"]}
            )
        else:
            return optimizer

        if self.scheduler_type == "reduce_on_plateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_auc",
                }
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }

    def _ce_weight(self):
        if self._class_weights is None:
            return None
        w = torch.tensor(self._class_weights, dtype=torch.float32, device=self.device)
        return w

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self._ce_weight())

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self._ce_weight())

        prob = F.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_probs.append(prob.detach().cpu())
        self.val_y.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_y) if len(self.val_y) else None
        y_score = torch.cat(self.val_probs) if len(self.val_probs) else None
        if y_true is not None:
            auc = simple_auc_roc(y_true.float(), y_score.float())
            self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_probs.clear()
        self.val_y.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self._ce_weight())

        prob = F.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"y": y.detach().cpu(), "p": prob.detach().cpu()}


# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def simple_auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.detach().cpu().float()
    y_score = y_score.detach().cpu().float()
    if y_true.min() == y_true.max():
        return float("nan")

    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(y_score) + 1, dtype=torch.float)

    pos = y_true == 1
    n_pos = pos.sum().item()
    n_neg = (~pos).sum().item()
    sum_ranks_pos = ranks[pos].sum().item()

    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# -----------------------
# PDF visualization using IMN native heatmap
# -----------------------
def visualize_pos_neg_to_pdf_imn(
    model,
    dataset,
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
    """
    Multi-page PDF: N positives and M negatives.
    Each page: IMN segment heatmap [12,T] + 12 lead plots with shading by segment importance.
    """
    model.eval()
    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    pos_idx, neg_idx = [], []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if int(y.item()) == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    if random_pick:
        rng = np.random.default_rng(seed)
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

    sel_pos = pos_idx[:n_pos]
    sel_neg = neg_idx[:n_neg]

    if len(sel_pos) < n_pos:
        print(f"WARNING: requested n_pos={n_pos}, but only found {len(sel_pos)} positives.")
    if len(sel_neg) < n_neg:
        print(f"WARNING: requested n_neg={n_neg}, but only found {len(sel_neg)} negatives.")

    selected = [(pos_class_name, i) for i in sel_pos] + [("NORM", i) for i in sel_neg]

    with PdfPages(pdf_path) as pdf:
        for tag, idx in selected:
            x, y = dataset[idx]
            x_b = x.unsqueeze(0).to(device)

            with torch.no_grad():
                prob, heatmap = model.explain(x_b)
                prob = float(prob.item())
                hm = heatmap.squeeze(0).cpu().numpy()  # [12,T]
                x_np = x.detach().cpu().numpy()        # [12,L]

            # Normalize heatmap for display (signed; red=positive, blue=negative)
            denom = np.max(np.abs(hm)) + 1e-6
            hm_disp = hm / denom

            Lsig = x_np.shape[1]
            Tseg = hm.shape[1]

            fig = plt.figure(figsize=(11.7, 16.5))
            gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1] * 12 + [0.5])

            ax0 = fig.add_subplot(gs[0, 0])
            im = ax0.imshow(hm_disp, aspect="auto", vmin=-1, vmax=1, cmap="bwr")
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            ax0.set_title(f"{tag} | true={int(y.item())} | P({pos_class_name})={prob:.3f} | idx={idx}")
            fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

            for lead in range(12):
                ax = fig.add_subplot(gs[lead + 1, 0])
                ax.plot(x_np[lead], linewidth=0.8)
                ax.set_xlim(0, Lsig - 1)
                ax.set_ylabel(lead_names[lead], rotation=0, labelpad=20, va="center")

                contrib = hm_disp[lead]
                for t in range(Tseg):
                    a = float(contrib[t])
                    alpha = min(0.35, abs(a) * 0.35)
                    if alpha > 0:
                        color = "red" if a > 0 else "blue"
                        start = t * stride
                        end = min(start + window, Lsig)
                        ax.axvspan(start, end, alpha=alpha, color=color, linewidth=0)

                ax.set_xticks([])

            axf = fig.add_subplot(gs[13, 0])
            axf.axis("off")
            axf.text(
                0, 0.5,
                f"IMN segment importance (2D CNN patch embed). Red=positive class, Blue=negative. Normalized per record.",
                fontsize=10
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved IMN visualization PDF: {pdf_path}")


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path/to/ptbxl/")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500])

    parser.add_argument("--task", type=str, default="norm_vs_mi",
                        choices=["norm_vs_mi", "norm_vs_sttc", "norm_vs_cd", "norm_vs_hyp"],
                        help="Binary task: norm_vs_mi, norm_vs_sttc, norm_vs_cd, norm_vs_hyp")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)

    # IMN 2D
    parser.add_argument("--window", type=int, default=None,
                        help="Patch window in samples. Default: 0.5s => 50@100Hz or 250@500Hz.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Patch stride. Default: window//2.")
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--hyper_hidden", type=int, default=256)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--wandb_project", type=str, default="mesormorphic_ecg")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_offline", action="store_true")

    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "reduce_on_plateau", "cosine_restarts", "none"])
    parser.add_argument("--scheduler_params", type=str, default="{}")

    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_monitor", type=str, default="val_auc",
                        choices=["val_loss", "val_auc", "val_acc"])
    parser.add_argument("--early_stop_mode", type=str, default="max", choices=["min", "max"])

    parser.add_argument("--out_dir", type=str, default="runs/imn2d_baseline", help="Base output directory")
    parser.add_argument("--viz_pdf", type=str, default="viz_norm_vs_mi_imn2d.pdf")
    parser.add_argument("--n_pos_viz", type=int, default=25)
    parser.add_argument("--n_neg_viz", type=int, default=25)
    parser.add_argument("--viz_random", action="store_true")
    parser.add_argument("--viz_seed", type=int, default=123)

    args = parser.parse_args()

    set_seed(args.seed)

    try:
        scheduler_params = json.loads(args.scheduler_params)
    except json.JSONDecodeError:
        scheduler_params = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.out_dir, args.task, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    path = args.path if args.path.endswith("/") else args.path + "/"

    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    _, aggregate_diagnostic = build_superclasses(path)
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

    TASK_TO_POS = {"norm_vs_mi": "MI", "norm_vs_sttc": "STTC", "norm_vs_cd": "CD", "norm_vs_hyp": "HYP"}
    pos_class = TASK_TO_POS[args.task]

    labels = Y["diagnostic_superclass"].values
    is_pos = np.array([pos_class in l for l in labels])
    is_norm = np.array(["NORM" in l for l in labels])
    keep = is_pos ^ is_norm
    Yf = Y[keep].copy()

    y_bin = np.array([1 if pos_class in l else 0 for l in Yf["diagnostic_superclass"].values], dtype=np.int64)
    print(f"Task {args.task}: N={len(Yf)} | {pos_class}={int(y_bin.sum())} | NORM={int((1 - y_bin).sum())}")

    print("Loading signals ...")
    X = load_raw_data(Yf, args.sampling_rate, path)
    X = np.transpose(X, (0, 2, 1))  # [N, 12, L]
    N, C, Lsig = X.shape
    print("X shape:", X.shape)

    fold = Yf["strat_fold"].values
    train_mask = (fold >= 1) & (fold <= 8)
    val_mask = fold >= 9

    X_train, y_train = X[train_mask], y_bin[train_mask]
    X_val, y_val = X[val_mask], y_bin[val_mask]
    print("Split sizes: train=", len(y_train), ", val=", len(y_val))

    if args.window is None:
        window = 50 if args.sampling_rate == 100 else 250
    else:
        window = args.window
    if args.stride is None:
        stride = window // 2
    else:
        stride = args.stride

    assert window <= Lsig and stride > 0
    Tseg = (Lsig - window) // stride + 1
    print(f"IMN 2D patches: window={window}, stride={stride}, T={Tseg}")

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    n_pos = float((y_train_t == 1).sum())
    n_neg = float((y_train_t == 0).sum())
    w0 = (n_pos + n_neg) / max(n_neg, 1.0)
    w1 = (n_pos + n_neg) / max(n_pos, 1.0)
    class_weights = [float(w0), float(w1)]
    print(f"Class weights: w0={w0:.3f}, w1={w1:.3f}")

    train_ds = PTBXLBinaryDatasetCE(X_train_t, y_train_t, per_lead_zscore=True)
    val_ds = PTBXLBinaryDatasetCE(X_val_t, y_val_t, per_lead_zscore=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )

    lit = IMN2DLightning(
        signal_len=Lsig,
        window=window,
        stride=stride,
        token_dim=args.token_dim,
        hyper_hidden=args.hyper_hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        scheduler_type=args.scheduler if args.scheduler != "none" else None,
        scheduler_params=scheduler_params,
    )

    wandb_name = args.wandb_name or timestamp
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=wandb_name,
        save_dir=run_dir,
        offline=args.wandb_offline,
    )
    wandb_logger.log_hyperparams(vars(args))
    wandb_logger.log_hyperparams({"window": window, "stride": stride, "signal_len": Lsig, "task": args.task, "pos_class": pos_class})

    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="best-{epoch:02d}-{val_auc:.4f}",
        monitor=args.early_stop_monitor,
        mode=args.early_stop_mode,
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    es_cb = EarlyStopping(
        monitor=args.early_stop_monitor,
        mode=args.early_stop_mode,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[ckpt_cb, es_cb, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=20,
        deterministic=True,
    )

    trainer.fit(lit, train_loader, val_loader)

    best_path = ckpt_cb.best_model_path
    if best_path and os.path.exists(best_path):
        print("Loading best checkpoint:", best_path)
        lit = IMN2DLightning.load_from_checkpoint(best_path)

    trainer.test(lit, val_loader)

    # Move model to target device before manual inference (trainer may leave it on CPU)
    device_obj = torch.device(device)
    lit = lit.to(device_obj)
    lit.model = lit.model.to(device_obj)

    lit.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device_obj)
            logits = lit.model(x)
            prob = F.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
            y_true_list.append(y.numpy())
            y_pred_list.append(pred.cpu().numpy())
            y_prob_list.append(prob.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_prob = np.concatenate(y_prob_list)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    metrics_csv_path = os.path.join(run_dir, "validation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
    print(f"Validation metrics saved to: {metrics_csv_path}")

    with open(os.path.join(run_dir, "model.txt"), "w") as f:
        f.write(str(lit.model))

    pdf_path = args.viz_pdf
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(run_dir, pdf_path)

    print("Saving IMN 2D visualization PDF to:", pdf_path)
    visualize_pos_neg_to_pdf_imn(
        model=lit.model,
        dataset=val_ds,
        device=device_obj,
        pdf_path=pdf_path,
        sampling_rate=args.sampling_rate,
        window=window,
        stride=stride,
        n_pos=args.n_pos_viz,
        n_neg=args.n_neg_viz,
        pos_class_name=pos_class,
        random_pick=args.viz_random,
        seed=args.viz_seed,
    )
    print("Done.")

    wandb.finish()


if __name__ == "__main__":
    main()
