"""
PTB-XL MI vs NORM baseline (2D CNN) + Grad-CAM using jacobgil/pytorch-grad-cam + PDF report.

- Model: simple 2D CNN on ECG "image" [B,1,12,L]
- Loss: CrossEntropy (2 classes: 0=NORM, 1=MI) with optional class weights
- XAI: GradCAM from pytorch-grad-cam (jacobgil) on the last conv layer
- PDF: pick N positives (MI) and M negatives (NORM) from TEST, visualize:
      (1) lead x segments heatmap (GradCAM averaged into segments)
      (2) 12 lead plots with red shading by segment importance
- Splits: train folds 1-8, val fold 9, test fold 10
- Sampling rate: 100 or 500
- Run dir: runs/.../<timestamp> with args.yaml, metrics.csv, last.pt, best.pt, model.txt, pdf

Run:
python baseline_ptbxl_2d_gradcam_jacobgil.py --path /path/to/ptbxl --sampling_rate 500 --epochs 20 --viz_random
"""

import argparse
import ast
import json
import os
import time
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
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
from math import ceil

try:
    import ecg_plot
    ECG_PLOT_AVAILABLE = True
except ImportError:
    ECG_PLOT_AVAILABLE = False

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

# Jacobgil Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


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
# 2D CNN baseline
# -----------------------
class ECGConv2DBaseline(nn.Module):
    """
    Input:  x [B,12,L]
    Output: logits [B,2]  (class 0=NORM, 1=MI)
    """
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # downsample time only
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # downsample time only
        )
        # target layer for Grad-CAM: use last conv block output
        self.target_layer = self.conv3[0]  # the Conv2d(32->64)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.unsqueeze(1)            # [B,1,12,L]
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)             # [B,64,12,L']
        g = h.mean(dim=(2, 3))        # [B,64] global avg pool
        g = self.dropout(g)
        logits = self.fc(g)           # [B,2]
        return logits


# -----------------------
# Lightning Module (wraps ECGConv2DBaseline)
# -----------------------
class Conv2DBaselineLightning(pl.LightningModule):
    def __init__(
        self,
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

        self.model = ECGConv2DBaseline(dropout=dropout)

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

        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
# Metrics
# -----------------------
@torch.no_grad()
def simple_auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    y_true: [N] {0,1}
    y_score: [N] probability for class 1
    """
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
# Visualization helpers (lead selection, etc.)
# -----------------------
DEFAULT_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_lead_indices(leads_str: str | None, lead_names: list | None = None) -> list[int] | None:
    """
    Parse lead selection string into list of 0-based indices.
    Examples: "0,1,2,3" "I,II,III,V1" "0-5" "V1,V2,V3,V4,V5,V6"
    Returns None if leads_str is None/empty (meaning all leads).
    """
    if not leads_str or not str(leads_str).strip():
        return None
    lead_names = lead_names or DEFAULT_LEAD_NAMES
    name_to_idx = {n.upper(): i for i, n in enumerate(lead_names)}
    name_to_idx.update({n: i for i, n in enumerate(lead_names)})
    result = []
    for part in str(leads_str).replace(" ", "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("-"):
            lo, hi = part.split("-", 1)
            try:
                lo_i, hi_i = int(lo.strip()), int(hi.strip())
                result.extend(range(lo_i, hi_i + 1))
            except ValueError:
                pass
        elif part.upper() in name_to_idx:
            result.append(name_to_idx[part.upper()])
        else:
            try:
                result.append(int(part))
            except ValueError:
                if part.upper() in name_to_idx:
                    result.append(name_to_idx[part.upper()])
    return sorted(set(i for i in result if 0 <= i < 12)) if result else None


# -----------------------
# Grad-CAM helpers
# -----------------------
def cam_to_segments(cam_12L: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    cam_12L: [12,L] in [0,1]
    returns [12,T] in [0,1]
    """
    assert cam_12L.ndim == 2 and cam_12L.shape[0] == 12
    L = cam_12L.shape[1]
    assert window <= L and stride > 0
    T = (L - window) // stride + 1
    seg = np.zeros((12, T), dtype=np.float32)
    for t in range(T):
        s = t * stride
        e = min(s + window, L)
        seg[:, t] = cam_12L[:, s:e].mean(axis=1)
    # normalize per record for display
    mx = float(seg.max()) + 1e-6
    seg = seg / mx
    return seg


def ensure_cam_size(cam_hw: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    pytorch-grad-cam usually returns CAM in input resolution, but we make it robust:
    if cam is not [H,W], resize with torch interpolate.
    cam_hw: [Hc,Wc]
    """
    if cam_hw.shape == (H, W):
        return cam_hw
    t = torch.from_numpy(cam_hw).float()[None, None, :, :]  # [1,1,Hc,Wc]
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


# -----------------------
# Wrapper model for Grad-CAM (expects [B,C,H,W] input)
# -----------------------
class GradCAMWrapper(nn.Module):
    """Wrapper that accepts [B,C,H,W] and calls model's conv layers directly"""
    def __init__(self, base_model: ECGConv2DBaseline):
        super().__init__()
        self.conv1 = base_model.conv1
        self.conv2 = base_model.conv2
        self.conv3 = base_model.conv3
        self.dropout = base_model.dropout
        self.fc = base_model.fc
    
    def forward(self, x):
        # x is already [B,1,12,L] - no unsqueeze needed
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)             # [B,64,12,L']
        g = h.mean(dim=(2, 3))        # [B,64] global avg pool
        g = self.dropout(g)
        logits = self.fc(g)           # [B,2]
        return logits


# -----------------------
# Visualization to PDF using pytorch-grad-cam
# -----------------------
def visualize_pos_neg_to_pdf_gradcam(model, dataset, device, pdf_path: str,
                                     sampling_rate: int, window: int, stride: int,
                                     n_pos: int, n_neg: int,
                                     pos_class_name: str = "MI",
                                     random_pick: bool = False, seed: int = 123,
                                     lead_names=None,
                                     lead_indices: list[int] | None = None,
                                     heatmap_height: float = 1.0,
                                     ecg_height: float = 0.65):
    """
    Multi-page PDF:
      - N positives and M NORM from dataset (first or random)
      - Each page: heatmap (lead x segments) + 12 lead plots with shaded importance
    """
    model.eval()
    if lead_names is None:
        lead_names = list(DEFAULT_LEAD_NAMES)
    lead_indices = lead_indices if lead_indices is not None else list(range(12))
    n_leads = len(lead_indices)

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    
    # Create wrapper model for Grad-CAM (expects [B,C,H,W] format)
    gradcam_model = GradCAMWrapper(model).to(device)
    gradcam_model.eval()

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

    # GradCAM object (jacobgil) — reuse across samples
    # Use wrapper model which expects [B,C,H,W] format
    target_layers = [gradcam_model.conv3[0]]  # Conv2d layer for Grad-CAM
    with GradCAM(model=gradcam_model, target_layers=target_layers) as cam:
        with PdfPages(pdf_path) as pdf:
            for tag, idx in selected:
                x, y = dataset[idx]                      # x: [12,L], y: scalar int
                x_b = x.unsqueeze(0).to(device)          # [1,12,L]

                # Forward prob (no gradients needed here)
                with torch.no_grad():
                    logits = model(x_b)                      # [1,2]
                    prob = float(torch.softmax(logits, dim=1)[0, 1].item())

                # GradCAM expects [B,C,H,W] format (image-like)
                # Convert [1,12,L] to [1,1,12,L] for Grad-CAM library
                input_tensor = x_b.unsqueeze(1).requires_grad_(True)  # [1,1,12,L]

                # Explain class 1 (MI)
                targets = [ClassifierOutputTarget(1)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # [B,H,W] numpy

                cam_hw = grayscale_cam[0]                # [12,L] or possibly [12,L'] depending on internals
                cam_hw = ensure_cam_size(cam_hw, H=12, W=x.shape[1])

                # Normalize to [0,1] per record for display
                cam_hw = cam_hw - cam_hw.min()
                cam_hw = cam_hw / (cam_hw.max() + 1e-6)

                # Segment heatmap [12,T]
                seg_hm = cam_to_segments(cam_hw, window=window, stride=stride)
                Tseg = seg_hm.shape[1]
                Lsig = x.shape[1]

                # Convert to numpy (no gradients needed)
                with torch.no_grad():
                    x_np = x.detach().cpu().numpy()          # [12,L]

                is_norm_sample = (tag == "NORM")
                cmap = "Blues" if is_norm_sample else "Reds"
                shade_color = "blue" if is_norm_sample else "red"

                seg_hm_sel = seg_hm[lead_indices]
                fig = plt.figure(figsize=(11.7, max(8, heatmap_height + n_leads * ecg_height)))
                gs = fig.add_gridspec(n_leads + 2, 1, height_ratios=[heatmap_height] + [ecg_height] * n_leads + [0.4], hspace=0.01)

                ax0 = fig.add_subplot(gs[0, 0])
                im = ax0.imshow(seg_hm_sel, aspect="auto", vmin=0, vmax=1, cmap=cmap)
                ax0.set_yticks(range(n_leads))
                ax0.set_yticklabels([lead_names[i] for i in lead_indices])
                ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
                ax0.set_title(f"{tag} | true={int(y.item())} | P({pos_class_name})={prob:.3f} | idx={idx}")
                fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

                for k, lead in enumerate(lead_indices):
                    ax = fig.add_subplot(gs[k + 1, 0])
                    ax.plot(x_np[lead], linewidth=0.8, color='black', alpha=0.6)
                    ax.set_xlim(0, Lsig - 1)
                    ax.set_ylabel(lead_names[lead], rotation=0, labelpad=8, va="center", fontsize=8)
                    ax.set_yticklabels([])
                    ax.margins(y=0.02)

                    contrib = seg_hm[lead]  # [Tseg], 0..1
                    for t in range(Tseg):
                        a = float(contrib[t])
                        alpha = min(0.5, a * 0.6)
                        if alpha > 0.05:
                            start = t * stride
                            end = min(start + window, Lsig)
                            ax.axvspan(start, end, alpha=alpha, color=shade_color, linewidth=0)

                    ax.set_xticks([])

                axf = fig.add_subplot(gs[n_leads + 1, 0])
                axf.axis("off")
                axf.text(
                    0, 0.5,
                    f"Grad-CAM importance (0..1) for class {pos_class_name} (post-hoc). Darker red = higher importance.",
                    fontsize=10
                )

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


# -----------------------
# ECG Plot Visualization (ecg_plot library style)
# -----------------------
def _draw_ecg_plot_style(ax, ecg: np.ndarray, sample_rate: int, lead_names: list,
                         columns: int = 2, row_height: float = 0.5,
                         style: str | None = None, half_signal: bool = False) -> None:
    """
    Draw ecg_plot-style 12-lead ECG in the given axes.
    Replicates layout from https://github.com/dy1901/ecg_plot
    Paper-optimized: minimal gap between leads (attached), optional half-signal display.
    """
    if half_signal:
        ecg = ecg[:, : ecg.shape[1] // 2].copy()
    lead_order = list(range(len(ecg)))
    secs = len(ecg[0]) / sample_rate
    leads = len(lead_order)
    rows = int(ceil(leads / columns))
    display_factor = 1.0
    line_width = 0.5 * (display_factor ** 0.5)

    x_min, x_max = 0, columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    if style == "bw":
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line = (0, 0, 0)
    else:
        color_major = (1, 0, 0)
        color_minor = (1, 0.7, 0.7)
        color_line = (0, 0, 0.7)

    ax.set_xticks(np.arange(x_min, x_max + 0.01, 0.2))
    tick_step = 0.25 if row_height < 1.5 else 0.5
    ax.set_yticks(np.arange(y_min, y_max + 0.01, tick_step))
    ax.set_yticklabels([])
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(which="major", linestyle="-", linewidth=0.5 * (display_factor ** 0.5), color=color_major)
    ax.grid(which="minor", linestyle="-", linewidth=0.5 * (display_factor ** 0.5), color=color_minor)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.margins(0)
    ax.autoscale(enable=False)

    step = 1.0 / sample_rate
    for c in range(columns):
        for i in range(rows):
            if c * rows + i >= leads:
                break
            t_lead = lead_order[c * rows + i]
            y_offset = -(row_height / 2) * ceil(i % rows)
            x_offset = secs * c if c > 0 else 0
            sep_h = 0.12 * (row_height / 0.5)
            if c > 0:
                ax.plot(
                    [x_offset, x_offset],
                    [ecg[t_lead][0] + y_offset - sep_h, ecg[t_lead][0] + y_offset + sep_h],
                    linewidth=line_width,
                    color=color_line,
                )
            ax.text(x_offset + 0.07, y_offset - row_height * 0.3, lead_names[t_lead], fontsize=7 * (display_factor ** 0.5))
            ax.plot(
                np.arange(0, len(ecg[t_lead]) * step, step) + x_offset,
                ecg[t_lead] + y_offset,
                linewidth=line_width,
                color=color_line,
            )


def _draw_important_patches_on_ecg(
    ax,
    seg_hm: np.ndarray,
    window: int,
    stride: int,
    sampling_rate: int,
    shade_color: str = "red",
    importance_threshold: float = 0.2,
    max_alpha: float = 0.35,
    columns: int = 2,
    signal_len: int | None = None,
    half_signal: bool = False,
) -> None:
    """
    Overlay semi-transparent patches on ECG axes to mark important regions from Grad-CAM heatmap.
    seg_hm: (12, T) segment importance, already normalized 0-1.
    For 2-column ecg_plot layout, draws patches in both columns (same time range).
    """
    Tseg = seg_hm.shape[1]
    full_secs = (signal_len / sampling_rate) if signal_len else (
        max((Tseg - 1) * stride + window, window) / sampling_rate if Tseg > 0 else window / sampling_rate
    )
    secs = full_secs / 2 if half_signal else full_secs
    for t in range(Tseg):
        imp = float(np.max(seg_hm[:, t]))
        if imp < importance_threshold:
            continue
        alpha = min(max_alpha, imp * 0.5)
        start_sec = (t * stride) / sampling_rate
        end_sec = (t * stride + window) / sampling_rate
        if half_signal and end_sec > secs:
            continue
        ax.axvspan(start_sec, end_sec, alpha=alpha, color=shade_color, zorder=0, linewidth=0)
        if columns > 1 and end_sec <= secs:
            ax.axvspan(secs + start_sec, secs + end_sec, alpha=alpha, color=shade_color, zorder=0, linewidth=0)


def visualize_ecg_with_gradcam_heatmap_to_pdf(
    model,
    dataset,
    device,
    pdf_path: str,
    sampling_rate: int,
    window: int,
    stride: int,
    n_samples: int = 5,
    pos_class_name: str = "MI",
    random_pick: bool = False,
    seed: int = 42,
    lead_names: list | None = None,
    half_ecg: bool = True,
    row_height: float = 0.5,
    lead_indices: list[int] | None = None,
    heatmap_height: float = 0.5,
    ecg_height: float = 1.4,
) -> None:
    """
    Visualize ECG with Grad-CAM heatmap on top, saved to PDF.
    Combines ecg_plot-style 12-lead ECG with Grad-CAM feature attribution heatmap.
    Paper-optimized: half ECG, compressed lead spacing, one sample per file.
    """
    model.eval()
    if lead_names is None:
        lead_names = list(DEFAULT_LEAD_NAMES)
    lead_indices = lead_indices if lead_indices is not None else list(range(12))

    gradcam_model = GradCAMWrapper(model).to(device)
    gradcam_model.eval()
    target_layers = [gradcam_model.conv3[0]]

    out_dir = os.path.dirname(pdf_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    N = len(dataset)
    n_plot = min(n_samples, N)
    if random_pick:
        rng = np.random.default_rng(seed)
        indices = rng.choice(N, size=n_plot, replace=False)
    else:
        indices = np.arange(n_plot)

    saved_paths = []
    with GradCAM(model=gradcam_model, target_layers=target_layers) as cam:
        for idx in indices:
            x, y = dataset[idx]
            y_int = int(y.item())
            tag = pos_class_name if y_int == 1 else "NORM"
            x_b = x.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x_b)
                prob = float(torch.softmax(logits, dim=1)[0, 1].item())

            input_tensor = x_b.unsqueeze(1).requires_grad_(True)
            targets = [ClassifierOutputTarget(1)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            cam_hw = grayscale_cam[0]
            cam_hw = ensure_cam_size(cam_hw, H=12, W=x.shape[1])
            cam_hw = cam_hw - cam_hw.min()
            cam_hw = cam_hw / (cam_hw.max() + 1e-6)

            seg_hm = cam_to_segments(cam_hw, window=window, stride=stride)
            Lsig = x.shape[1]

            if half_ecg:
                n_seg_half = max(1, (Lsig // 2 - window) // stride + 1)
                seg_hm = seg_hm[:, :n_seg_half]

            x_np = x.detach().cpu().numpy()
            x_np_sel = x_np[lead_indices]
            seg_hm_sel = seg_hm[lead_indices]
            lead_names_sel = [lead_names[i] for i in lead_indices]

            shade_color = "blue" if tag == "NORM" else "red"

            fig = plt.figure(figsize=(7, 5.2))
            gs = fig.add_gridspec(2, 1, height_ratios=[heatmap_height, ecg_height], hspace=0.1)

            ax_hm = fig.add_subplot(gs[0, 0])
            im = ax_hm.imshow(seg_hm_sel, aspect="auto", vmin=0, vmax=1, cmap="Reds")
            ax_hm.set_yticks(range(len(lead_indices)))
            ax_hm.set_yticklabels(lead_names_sel, fontsize=7)
            ax_hm.set_xlabel(f"Segments (w={window}, s={stride}, fs={sampling_rate}Hz)", fontsize=8)
            ax_hm.set_title(f"Grad-CAM Heatmap | {tag} | P({pos_class_name})={prob:.3f} | idx={idx}", fontsize=9)
            fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.02, shrink=0.8)

            ax_ecg = fig.add_subplot(gs[1, 0])
            _draw_ecg_plot_style(
                ax_ecg, x_np_sel, sampling_rate, lead_names_sel,
                columns=2, row_height=row_height, half_signal=half_ecg,
            )
            _draw_important_patches_on_ecg(
                ax_ecg, seg_hm_sel, window, stride, sampling_rate,
                shade_color=shade_color, signal_len=Lsig, half_signal=half_ecg,
            )
            n_leads_str = f"{len(lead_indices)}-lead" if len(lead_indices) != 12 else "12-lead"
            ax_ecg.set_title(f"{n_leads_str} ECG (shaded = Grad-CAM important regions)", fontsize=8)

            fig.tight_layout(pad=0.5)
            sample_path = os.path.join(out_dir, f"{base_name}_{tag}_{idx:04d}.pdf")
            fig.savefig(sample_path, bbox_inches="tight", pad_inches=0.08)
            saved_paths.append(sample_path)
            plt.close(fig)

    print(f"Saved {len(saved_paths)} ECG+Grad-CAM heatmap PDFs to {out_dir}")


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path/to/ptbxl/")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500])

    # Binary classification task (NORM vs X)
    parser.add_argument("--task", type=str, default="norm_vs_mi",
                        choices=["norm_vs_mi", "norm_vs_sttc", "norm_vs_cd", "norm_vs_hyp"],
                        help="Binary task: norm_vs_mi, norm_vs_sttc, norm_vs_cd, norm_vs_hyp")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Visualization segmentation
    parser.add_argument("--window", type=int, default=None,
                        help="Window in samples. Default: 0.5s => 50@100Hz or 250@500Hz.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride in samples. Default: window//2 (50% overlap).")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="mesormorphic_ecg", help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB run name (default: timestamp)")
    parser.add_argument("--wandb_offline", action="store_true", help="Run WandB in offline mode")

    # Learning Rate Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "reduce_on_plateau", "cosine_restarts", "none"],
                        help="Learning rate scheduler type")
    parser.add_argument("--scheduler_params", type=str, default="{}",
                        help="JSON string for scheduler parameters (e.g., '{\"T_0\": 10, \"T_mult\": 2}')")

    # Early Stopping
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0,
                        help="Minimum change to qualify as improvement")
    parser.add_argument("--early_stop_monitor", type=str, default="val_auc",
                        choices=["val_loss", "val_auc", "val_acc"],
                        help="Metric to monitor for early stopping")
    parser.add_argument("--early_stop_mode", type=str, default="max",
                        choices=["min", "max"],
                        help="Whether to minimize or maximize the monitored metric")

    # Output
    parser.add_argument("--out_dir", type=str, default="runs/mi_vs_norm_baseline_2d_gradcam", help="Base output directory")

    # Visualization
    parser.add_argument("--viz_pdf", type=str, default="viz_mi_vs_norm_test_gradcam.pdf")
    parser.add_argument("--n_pos_viz", type=int, default=25)
    parser.add_argument("--n_neg_viz", type=int, default=25)
    parser.add_argument("--viz_random", action="store_true")
    parser.add_argument("--viz_seed", type=int, default=123)
    parser.add_argument("--viz_ecg_plot", action="store_true",
                        help="Visualize ECG samples with ecg_plot-style layout + Grad-CAM heatmap.")
    parser.add_argument("--viz_ecg_plot_n", type=int, default=5,
                        help="Number of ECG samples to plot with ecg_plot when --viz_ecg_plot.")
    parser.add_argument("--leads", type=str, default=None,
                        help="Leads to visualize: comma-separated indices (0-11) or names (I,II,III,aVR,aVL,aVF,V1-V6). "
                             "E.g. '0,1,2' or 'I,II,III' or 'V1,V2,V3,V4,V5,V6' or '0-5'. Default: all 12 leads.")
    parser.add_argument("--viz_heatmap_height", type=float, default=None,
                        help="Height ratio for the top heatmap panel. Default: 1.0 for main viz, 0.5 for ECG+heatmap viz.")
    parser.add_argument("--viz_ecg_height", type=float, default=None,
                        help="Height ratio for the bottom ECG/lead traces panel. Default: 0.65 per lead for main viz, 1.4 for ECG+heatmap viz.")

    # Inference-only (skip training, load checkpoint)
    parser.add_argument("--inference_only", action="store_true",
                        help="Skip training; load checkpoint and run inference + viz only.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint. Required when --inference_only.")

    args = parser.parse_args()

    if args.inference_only and not args.ckpt:
        parser.error("--ckpt is required when --inference_only")

    lead_indices = parse_lead_indices(args.leads)
    if lead_indices is not None:
        print(f"Visualizing leads: {lead_indices} ({[DEFAULT_LEAD_NAMES[i] for i in lead_indices]})")

    set_seed(args.seed)

    # Parse scheduler params
    try:
        scheduler_params = json.loads(args.scheduler_params)
    except json.JSONDecodeError:
        scheduler_params = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Timestamped run dir (include task for organization)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.inference_only:
        run_dir = os.path.join(os.path.dirname(args.ckpt), f"inference_{timestamp}")
    else:
        run_dir = os.path.join(args.out_dir, args.task, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 Run directory: {run_dir}")

    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    path = args.path if args.path.endswith("/") else args.path + "/"

    # Load metadata
    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    _, aggregate_diagnostic = build_superclasses(path)
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

    # Binary task: NORM vs pos_class (exclusive)
    TASK_TO_POS = {"norm_vs_mi": "MI", "norm_vs_sttc": "STTC", "norm_vs_cd": "CD", "norm_vs_hyp": "HYP"}
    pos_class = TASK_TO_POS[args.task]

    labels = Y["diagnostic_superclass"].values
    is_pos = np.array([pos_class in l for l in labels])
    is_norm = np.array(["NORM" in l for l in labels])
    keep = (is_pos ^ is_norm)
    Yf = Y[keep].copy()

    y_bin = np.array([1 if pos_class in l else 0 for l in Yf["diagnostic_superclass"].values], dtype=np.int64)
    print(f"Task {args.task}: N={len(Yf)} | {pos_class}={int(y_bin.sum())} | NORM={int((1-y_bin).sum())}")

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

    # Default window/stride for visualization segments
    if args.window is None:
        window = 50 if args.sampling_rate == 100 else 250  # 0.5s
    else:
        window = args.window
    if args.stride is None:
        stride = window // 2
    else:
        stride = args.stride

    assert window <= Lsig
    assert stride > 0
    Tseg = (Lsig - window) // stride + 1
    print(f"Visualization segments: window={window}, stride={stride}, T={Tseg}")

    # Torch tensors
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t   = torch.from_numpy(X_val).float()
    y_val_t   = torch.from_numpy(y_val).long()

    # Class weights (inverse frequency) for CE
    n_pos = float((y_train_t == 1).sum())
    n_neg = float((y_train_t == 0).sum())
    w0 = (n_pos + n_neg) / max(n_neg, 1.0)
    w1 = (n_pos + n_neg) / max(n_pos, 1.0)
    class_weights = [float(w0), float(w1)]
    print(f"Class weights CE: w0={w0:.3f}, w1={w1:.3f}")

    train_ds = PTBXLBinaryDatasetCE(X_train_t, y_train_t, per_lead_zscore=True)
    val_ds   = PTBXLBinaryDatasetCE(X_val_t, y_val_t, per_lead_zscore=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    if args.inference_only:
        print("Inference-only mode. Loading checkpoint:", args.ckpt)
        lit = Conv2DBaselineLightning.load_from_checkpoint(args.ckpt, map_location=device)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(device == "cuda"))

        # Lightning model (wraps ECGConv2DBaseline)
        lit = Conv2DBaselineLightning(
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
            scheduler_type=args.scheduler if args.scheduler != "none" else None,
            scheduler_params=scheduler_params,
        )

        # WandB logger
        wandb_name = args.wandb_name or timestamp
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=wandb_name,
            save_dir=run_dir,
            offline=args.wandb_offline,
        )
        wandb_logger.log_hyperparams(vars(args))
        wandb_logger.log_hyperparams({"window": window, "stride": stride, "signal_len": Lsig, "task": args.task, "pos_class": pos_class})

        # Callbacks
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

        # Test best
        best_path = ckpt_cb.best_model_path
        if best_path and os.path.exists(best_path):
            print("Loading best checkpoint:", best_path)
            lit = Conv2DBaselineLightning.load_from_checkpoint(best_path)
        wandb.finish()

    lit = lit.to(device)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices)
    trainer.test(lit, val_loader)

    # Best model validation metrics -> CSV
    lit = lit.to(device)
    lit.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = lit.model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
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

    # Save model text
    with open(os.path.join(run_dir, "model.txt"), "w") as f:
        f.write(str(lit.model))

    # Grad-CAM PDF
    pdf_path = args.viz_pdf
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(run_dir, pdf_path)

    print("Saving Grad-CAM PDF to:", pdf_path)
    heatmap_h = args.viz_heatmap_height if args.viz_heatmap_height is not None else 1.0
    ecg_h = args.viz_ecg_height if args.viz_ecg_height is not None else 0.65
    visualize_pos_neg_to_pdf_gradcam(
        model=lit.model,
        dataset=val_ds,
        device=device,
        pdf_path=pdf_path,
        sampling_rate=args.sampling_rate,
        window=window,
        stride=stride,
        n_pos=args.n_pos_viz,
        n_neg=args.n_neg_viz,
        pos_class_name=pos_class,
        random_pick=args.viz_random,
        seed=args.viz_seed,
        lead_indices=lead_indices,
        heatmap_height=heatmap_h,
        ecg_height=ecg_h,
    )

    # ECG plot visualization with Grad-CAM heatmap (PDF format)
    if args.viz_ecg_plot:
        ecg_pdf_path = os.path.join(run_dir, "ecg_with_gradcam_heatmap.pdf")
        print(f"Generating ECG+Grad-CAM heatmap PDF to {ecg_pdf_path}...")
        heatmap_h_ecg = args.viz_heatmap_height if args.viz_heatmap_height is not None else 0.5
        ecg_h_ecg = args.viz_ecg_height if args.viz_ecg_height is not None else 1.4
        visualize_ecg_with_gradcam_heatmap_to_pdf(
            model=lit.model,
            dataset=val_ds,
            device=device,
            pdf_path=ecg_pdf_path,
            sampling_rate=args.sampling_rate,
            window=window,
            stride=stride,
            n_samples=args.viz_ecg_plot_n,
            pos_class_name=pos_class,
            random_pick=args.viz_random,
            seed=args.viz_seed,
            lead_indices=lead_indices,
            heatmap_height=heatmap_h_ecg,
            ecg_height=ecg_h_ecg,
        )

    print("Done.")

    if not args.inference_only:
        wandb.finish()


if __name__ == "__main__":
    main()