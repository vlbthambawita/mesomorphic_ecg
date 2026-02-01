"""
Baseline: HyperNet-classifier (NO IMN linear dot-product) + Grad-CAM (jacobgil/pytorch-grad-cam) + PDF

- Model:
  1) Patchify ECG into overlapping patches (lead x time segments)
  2) Patch embedder -> tokens [B,P,D]
  3) Pool tokens -> g (mean + g_proj)
  4) HyperNet(g) -> logits [B,2]  (direct classification)  <-- BASELINE CHANGE

- Grad-CAM:
  Uses pytorch-grad-cam on the LAST conv layer inside patch_embed (Conv1d over patches).
  We compute Grad-CAM per patch, then map patch CAMs back into [12,L] timeline and save PDF.

Notes:
- This Grad-CAM is "patch-space CAM" (because patch_embed is Conv1d over patch samples).
  It produces importance over time within each patch, then we aggregate to global time axis.

Run:
python baseline_hypernet_classifier_gradcam.py --path /path/to/ptbxl --sampling_rate 500 --epochs 20 --viz_random
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

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
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

# Custom 1D Grad-CAM (pytorch-grad-cam expects 4D/5D; Conv1d outputs 3D)


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
    agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    return agg_df, aggregate_diagnostic


# -----------------------
# Dataset (CE, 2-class)
# -----------------------
class PTBXLBinaryDatasetCE(Dataset):
    """
    X: [N, 12, L]
    y: [N] int64 {0,1} where 1=MI, 0=NORM
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
        x = self.X[idx]  # [12,L]
        y = self.y[idx]  # scalar
        if self.per_lead_zscore:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std
        return x, y


# -----------------------
# Patchify (same as yours)
# -----------------------
def patchify_ecg_overlap(x: torch.Tensor, window: int, stride: int):
    """
    x: [B,12,L]
    returns patches: [B, 12*T, window], and T
    """
    B, C, L = x.shape
    assert C == 12
    assert window <= L
    assert stride > 0
    x_unf = x.unfold(dimension=2, size=window, step=stride)  # [B,12,T,window]
    T = x_unf.shape[2]
    patches = x_unf.contiguous().view(B, 12 * T, window)     # [B,P,W]
    return patches, T


# ============================================================
# BASELINE MODEL: HyperNet classifier (NO IMN linear dot-product)
# ============================================================
class ECGPatchHyperClassifier(nn.Module):
    """
    Baseline pipeline:
    - Patch ECG into lead×time windows
    - Patch embedder -> tokens
    - Pool tokens -> g
    - HyperNet(g) -> logits [B,2] directly
    """
    def __init__(
        self,
        signal_len: int,
        window: int,
        stride: int,
        token_dim: int = 64,
        hyper_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.signal_len = signal_len
        self.window = window
        self.stride = stride
        self.token_dim = token_dim

        self.T = (signal_len - window) // stride + 1
        self.P = 12 * self.T

        # Patch embedder (same spirit as your IMN)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, token_dim),
            nn.GELU(),
        )

        # Small projector after pooling
        self.g_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )

        # HyperNet produces class logits directly (2 classes)
        self.hyper_cls = nn.Sequential(
            nn.Linear(token_dim, hyper_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hyper_hidden, hyper_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hyper_hidden, 2),
        )

        # For Grad-CAM: choose a conv layer inside patch_embed
        # We'll target the second Conv1d for patch-level CAM
        self.gradcam_target_layer = self.patch_embed[2]  # nn.Conv1d(16,32,...)

    def forward(self, x):
        """
        x: [B,12,L]
        returns:
          logits: [B,2]
          tokens: [B,P,D]
          T: int
        """
        patches, T = patchify_ecg_overlap(x, self.window, self.stride)  # [B,P,W]
        B, P, W = patches.shape
        assert T == self.T and P == self.P and W == self.window

        patches_ = patches.view(B * P, 1, W)            # [B*P,1,W]
        tok = self.patch_embed(patches_)                # [B*P,D]
        tokens = tok.view(B, P, self.token_dim)         # [B,P,D]

        g = self.g_proj(tokens.mean(dim=1))             # [B,D]
        logits = self.hyper_cls(g)                      # [B,2]
        return logits, tokens, T


# -----------------------
# Lightning Module
# -----------------------
class HyperClassifierLightning(pl.LightningModule):
    def __init__(
        self,
        signal_len: int,
        window: int,
        stride: int,
        token_dim: int = 64,
        hyper_hidden: int = 256,
        dropout: float = 0.1,
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

        self.model = ECGPatchHyperClassifier(
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
                mode="max",  # Monitor validation AUC
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
        logits, tokens, T = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self._ce_weight())

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, tokens, T = self.model(x)
        loss = F.cross_entropy(logits, y, weight=self._ce_weight())

        prob = torch.softmax(logits, dim=1)[:, 1]  # P(MI)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_probs.append(prob.detach().cpu())
        self.val_y.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        # Simple AUC ROC implementation (same as your earlier scripts)
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
        logits, tokens, T = self.model(x)
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


# ============================================================
# Grad-CAM for this baseline (patch-space CAM -> time axis)
# ============================================================
def gradcam_patch_to_full_time(
    cam_patch: np.ndarray,  # [W] CAM over samples inside patch
    lead: int,
    t_idx: int,
    L: int,
    window: int,
    stride: int
):
    """
    Map a patch CAM [W] into global time axis [L] at correct location (t_idx).
    """
    start = t_idx * stride
    end = min(start + window, L)
    w_eff = end - start
    out = np.zeros((L,), dtype=np.float32)
    out[start:end] = cam_patch[:w_eff]
    return out


def gradcam_1d(model: nn.Module, input_tensor: torch.Tensor, target_layer: nn.Module, target_class: int) -> np.ndarray:
    """
    Compute Grad-CAM for 1D Conv layers. Returns CAM [B, L] normalized per sample.
    model: forward(x) -> logits [B, num_classes]
    input_tensor: [B, C, L] requires_grad=True
    """
    activations = []
    gradients = []

    def fwd_hook(m, inp, out):
        activations.append(out.detach())

    def bwd_hook(m, grad_inp, grad_out):
        gradients.append(grad_out[0].detach())

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(input_tensor)
        if logits.dim() == 1:
            loss = logits[target_class] if input_tensor.size(0) == 1 else logits[:, target_class].sum()
        else:
            loss = logits[:, target_class].sum()
        loss.backward()

        acts = activations[0].cpu().numpy()   # [B, C, L]
        grads = gradients[0].cpu().numpy()    # [B, C, L]

        # Grad-CAM: weights = global avg pool of gradients; cam = ReLU(sum_c(weights_c * acts_c))
        weights = np.mean(grads, axis=2)  # [B, C]
        cam = np.sum(weights[:, :, np.newaxis] * acts, axis=1)  # [B, L]
        cam = np.maximum(cam, 0)
        # Normalize per sample
        for i in range(cam.shape[0]):
            c = cam[i]
            c = c - c.min()
            if c.max() > 1e-8:
                c = c / c.max()
            cam[i] = c
        return cam.astype(np.float32)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


@torch.no_grad()
def visualize_pos_neg_to_pdf_gradcam_baseline(
    lightning_model: HyperClassifierLightning,
    dataset: Dataset,
    device: str,
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
    Uses 1D GradCAM on the patch_embed conv layer.
    Produces per-lead importance on full timeline by aggregating CAMs over patches.
    Saves a multi-page PDF. pos_class_name: label for positive class (e.g. MI, STTC, CD, HYP).
    """
    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    model = lightning_model.model.to(device)
    model.eval()

    # Collect indices
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
    selected = [(pos_class_name, i) for i in sel_pos] + [("NORM", i) for i in sel_neg]

    class PatchBatchWrapper(nn.Module):
        """Maps patch batch [P,1,W] -> logits [1,2] for Grad-CAM."""

        def __init__(self, full_model, window, stride):
            super().__init__()
            self.full_model = full_model
            self.window = window
            self.stride = stride

        def forward(self, patch_batch):
            tok = self.full_model.patch_embed(patch_batch)  # [P,D]
            tokens = tok.unsqueeze(0)
            g = self.full_model.g_proj(tokens.mean(dim=1))
            return self.full_model.hyper_cls(g)  # [1,2]

    wrapper = PatchBatchWrapper(model, window, stride).to(device).eval()

    with PdfPages(pdf_path) as pdf:
        for tag, idx in selected:
            x, y = dataset[idx]                 # x [12,L]
            x_b = x.unsqueeze(0).to(device)     # [1,12,L]
            Lsig = x.shape[1]

            # Forward probability (class 1 = positive)
            logits, tokens, T = model(x_b)
            prob_pos = float(torch.softmax(logits, dim=1)[0, 1].item())

            patches, T = patchify_ecg_overlap(x_b, window=window, stride=stride)  # [1,P,W]
            P = patches.shape[1]
            patches_in = patches.view(P, 1, window).clone().requires_grad_(True)  # [P,1,W]

            with torch.enable_grad():
                wrapper.zero_grad(set_to_none=True)
                grayscale_cam = gradcam_1d(
                    model=wrapper,
                    input_tensor=patches_in,
                    target_layer=model.gradcam_target_layer,
                    target_class=1,  # positive class
                )  # returns [P, W]

            # Aggregate patch CAMs into [12,L] by placing each patch CAM at its location
            cam_full = np.zeros((12, Lsig), dtype=np.float32)
            cam_count = np.zeros((12, Lsig), dtype=np.float32)

            # Patch index p corresponds to lead = p // T, segment = p % T
            for p in range(P):
                lead = p // T
                t_idx = p % T
                cam_patch = np.asarray(grayscale_cam[p]).ravel()  # [W]
                cam_line = gradcam_patch_to_full_time(
                    cam_patch=cam_patch.astype(np.float32),
                    lead=lead,
                    t_idx=t_idx,
                    L=Lsig,
                    window=window,
                    stride=stride
                )
                cam_full[lead] += cam_line
                cam_count[lead] += (cam_line > 0).astype(np.float32)

            cam_count = np.clip(cam_count, 1.0, None)
            cam_full = cam_full / cam_count

            # Normalize for display
            cam_full = cam_full - cam_full.min()
            cam_full = cam_full / (cam_full.max() + 1e-6)

            # Segment heatmap [12,T]
            seg_hm = cam_full.reshape(12, Lsig)
            Tseg = (Lsig - window) // stride + 1
            seg = np.zeros((12, Tseg), dtype=np.float32)
            for t in range(Tseg):
                s = t * stride
                e = min(s + window, Lsig)
                seg[:, t] = seg_hm[:, s:e].mean(axis=1)
            seg = seg / (seg.max() + 1e-6)

            x_np = x.detach().cpu().numpy()

            # ---- Plot page ----
            fig = plt.figure(figsize=(11.7, 16.5))
            gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1]*12 + [0.5])

            ax0 = fig.add_subplot(gs[0, 0])
            im = ax0.imshow(seg, aspect="auto", vmin=0, vmax=1, cmap="Reds")
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            ax0.set_title(f"{tag} | true={int(y.item())} | P({pos_class_name})={prob_pos:.3f} | idx={idx}")
            fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

            for lead in range(12):
                ax = fig.add_subplot(gs[lead + 1, 0])
                ax.plot(x_np[lead], linewidth=0.8)
                ax.set_xlim(0, Lsig - 1)
                ax.set_ylabel(lead_names[lead], rotation=0, labelpad=20, va="center")

                contrib = seg[lead]  # [Tseg], 0..1
                for t in range(Tseg):
                    a = float(contrib[t])
                    alpha = min(0.35, a * 0.35)
                    if alpha > 0:
                        start = t * stride
                        end = min(start + window, Lsig)
                        ax.axvspan(start, end, alpha=alpha, color="red", linewidth=0)

                ax.set_xticks([])

            axf = fig.add_subplot(gs[13, 0])
            axf.axis("off")
            axf.text(
                0, 0.5,
                f"Baseline HyperNet classifier: Grad-CAM computed at patch-embed conv layer. "
                f"Red = higher importance for {pos_class_name}.",
                fontsize=10
            )

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


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
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--hyper_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Patching
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
    parser.add_argument("--wandb_project", type=str, default="ecg-baseline-hypernet", help="WandB project name")
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
    parser.add_argument("--out_dir", type=str, default="runs/baseline_hypernet_classifier", help="Base output directory")

    # Visualization
    parser.add_argument("--viz_pdf", type=str, default="viz_gradcam_baseline.pdf")
    parser.add_argument("--n_pos_viz", type=int, default=25)
    parser.add_argument("--n_neg_viz", type=int, default=25)
    parser.add_argument("--viz_random", action="store_true")
    parser.add_argument("--viz_seed", type=int, default=123)

    args = parser.parse_args()

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
    keep = (is_pos ^ is_norm)  # exclusive: either pos or NORM, not both
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
    val_mask = (fold >= 9)  # folds 9 and 10 combined

    X_train, y_train = X[train_mask], y_bin[train_mask]
    X_val, y_val     = X[val_mask], y_bin[val_mask]

    print("Split sizes: train=", len(y_train), ", val=", len(y_val))

    # Default window/stride
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
    T = (Lsig - window) // stride + 1
    print(f"Patching: window={window}, stride={stride}, T={T}, P={12*T}")

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # Lightning model
    lit = HyperClassifierLightning(
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
        lit = HyperClassifierLightning.load_from_checkpoint(best_path)
    lit = lit.to(device)

    trainer.test(lit, val_loader)

    # Best model validation metrics -> CSV
    lit = lit.to(device)
    lit.eval()
    y_true_list, y_pred_list, y_prob_list = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits, _, _ = lit.model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]  # P(MI)
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
    visualize_pos_neg_to_pdf_gradcam_baseline(
        lightning_model=lit,
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
        seed=args.viz_seed
    )
    print("Done.")

    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    main()