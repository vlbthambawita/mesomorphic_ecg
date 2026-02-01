"""
ECG Patch IMN (Lightning) — GatedAttnPool variant.

Same as save_wgen_tokens script, but replaces tokens.mean(dim=1) with GatedAttnPool
(attention-weighted pooling over patches). Use --pool_hidden to set attention MLP hidden dim.

Use --save_wgen_tokens to save Wgen [N,P,D] and tokens [N,P,D] plus labels, logits, probs.
Formats: --save_wgen_tokens_format npz|pt|h5|all. Splits: --save_wgen_tokens_split test|val|both.
See "BEST OPTIONS FOR SAVING" in code for when to use each format.
"""
import argparse
import ast
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
class PTBXLBinaryDataset(Dataset):
    """
    X: [N, 12, L]
    y: [N] {0,1}
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, per_lead_zscore: bool = True):
        super().__init__()
        assert X.ndim == 3 and X.shape[1] == 12
        assert y.ndim == 1 and y.shape[0] == X.shape[0]
        self.X = X.float()
        self.y = y.float()
        self.per_lead_zscore = per_lead_zscore

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [12, L]
        y = self.y[idx]  # scalar

        if self.per_lead_zscore:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        return x, y


# -----------------------
# Overlapping patchify: [B,12,L] -> [B,P,W], P=12*T, T = floor((L-W)/stride)+1
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

    # Unfold over time dimension
    # x_unf: [B,12,T,window]
    x_unf = x.unfold(dimension=2, size=window, step=stride)
    T = x_unf.shape[2]
    patches = x_unf.contiguous().view(B, 12 * T, window)  # [B,P,W]
    return patches, T


# -----------------------
# Gated attention pooling (replaces mean pooling over tokens)
# -----------------------
class GatedAttnPool(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, tokens):  # [B,P,D]
        a = self.net(tokens).squeeze(-1)         # [B,P]
        a = torch.softmax(a, dim=1)              # [B,P]
        g = (tokens * a.unsqueeze(-1)).sum(dim=1)  # [B,D]
        return g, a


# -----------------------
# Model: Patch-wise IMN (binary) with overlap support
# -----------------------
class ECGPatchIMN(nn.Module):
    """
    - Patch ECG into lead×time windows (overlapping allowed)
    - Patch embedder (shared) -> token
    - Pool tokens -> g
    - Hypernet(g) -> instance-specific linear weights over tokens
    - logit = sum_{p,d} W_{p,d} * token_{p,d} + b
    """
    def __init__(self, signal_len: int, window: int, stride: int, token_dim: int = 64, hyper_hidden: int = 256, pool_hidden: int = 128, dropout: float = 0.1, verbose: bool = False):
        super().__init__()
        self.signal_len = signal_len
        self.window = window
        self.stride = stride
        self.token_dim = token_dim
        self.verbose = verbose

        # Compute T, P deterministically from L, window, stride
        self.T = (signal_len - window) // stride + 1
        self.P = 12 * self.T

        self.pool = GatedAttnPool(token_dim, hidden=pool_hidden)

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

        self.g_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
        )

        # Binary => output (P*D + 1)
        self.hyper = nn.Sequential(
            nn.Linear(token_dim, hyper_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hyper_hidden, hyper_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hyper_hidden, self.P * token_dim + 1),
        )

    def forward(self, x, verbose: bool = None):
        """
        x: [B,12,L]
        returns:
          logit: [B]
          Wgen: [B,P,D]
          tokens: [B,P,D]
          T: number of time segments
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print(f"[Forward] Input x shape: {x.shape}")
        
        patches, T = patchify_ecg_overlap(x, self.window, self.stride)  # [B,P,W]
        B, P, W = patches.shape
        assert T == self.T, f"T mismatch: got {T}, expected {self.T}"
        assert P == self.P, f"P mismatch: got {P}, expected {self.P}"
        assert W == self.window
        
        if verbose:
            print(f"[Forward] After patchify: patches shape: {patches.shape}, T: {T}")

        patches_ = patches.view(B * P, 1, W)
        if verbose:
            print(f"[Forward] After reshape: patches_ shape: {patches_.shape}")
        
        tok = self.patch_embed(patches_)                # [B*P,D]
        if verbose:
            print(f"[Forward] After patch_embed: tok shape: {tok.shape}")
        
        tokens = tok.view(B, P, self.token_dim)         # [B,P,D]
        if verbose:
            print(f"[Forward] After reshape to tokens: tokens shape: {tokens.shape}")

        g_pooled, _ = self.pool(tokens)                 # g_pooled [B,D]
        g = self.g_proj(g_pooled)                       # [B,D]
        if verbose:
            print(f"[Forward] After pool + g_proj: g shape: {g.shape}")
        
        out = self.hyper(g)                             # [B, P*D+1]
        if verbose:
            print(f"[Forward] After hyper: out shape: {out.shape}")
        
        Wgen = out[:, :-1].view(B, P, self.token_dim)   # [B,P,D]
        b = out[:, -1]                                  # [B]
        if verbose:
            print(f"[Forward] After splitting: Wgen shape: {Wgen.shape}, b shape: {b.shape}")

        logit = (Wgen * tokens).sum(dim=(1, 2)) + b
        if verbose:
            print(f"[Forward] Final output: logit shape: {logit.shape}")
        
        return logit, Wgen, tokens, T

    @torch.no_grad()
    def explain(self, x):
        """
        returns:
          prob: [B]
          heatmap: [B,12,T] signed contributions per lead×segment
        """
        logit, Wgen, tokens, T = self.forward(x)
        prob = torch.sigmoid(logit)
        patch_scores = (Wgen * tokens).sum(dim=-1)       # [B,P]
        heatmap = patch_scores.view(x.size(0), 12, T)    # [B,12,T]
        return prob, heatmap

    @torch.no_grad()
    def explain_with_topk(self, x, topk_leads: int = None, topk_patches: int = None):
        """
        returns:
          prob: [B]
          heatmap: [B,12,T] signed contributions per lead×segment
          top_lead_indices: [B, topk_leads] indices of top-k leads
          top_patch_indices: [B, topk_patches] indices of top-k patches (flattened lead×time)
        """
        logit, Wgen, tokens, T = self.forward(x)
        prob = torch.sigmoid(logit)
        patch_scores = (Wgen * tokens).sum(dim=-1)       # [B,P]
        heatmap = patch_scores.view(x.size(0), 12, T)    # [B,12,T]
        
        top_lead_indices = None
        top_patch_indices = None
        
        if topk_leads is not None and topk_leads > 0:
            # Compute per-lead importance (sum of absolute contributions)
            lead_importance = heatmap.abs().sum(dim=2)  # [B, 12]
            _, top_lead_indices = torch.topk(lead_importance, k=min(topk_leads, 12), dim=1)  # [B, topk_leads]
        
        if topk_patches is not None and topk_patches > 0:
            # Get top-k patches by absolute contribution
            patch_abs = patch_scores.abs()  # [B, P]
            _, top_patch_indices = torch.topk(patch_abs, k=min(topk_patches, patch_scores.size(1)), dim=1)  # [B, topk_patches]
        
        return prob, heatmap, top_lead_indices, top_patch_indices

    def forward_with_dropout(self, x, verbose: bool = None):
        """
        Forward pass with dropout enabled (for Monte Carlo dropout).
        """
        if verbose is None:
            verbose = self.verbose
        
        # Enable dropout layers
        self.train()  # This enables dropout
        
        patches, T = patchify_ecg_overlap(x, self.window, self.stride)  # [B,P,W]
        B, P, W = patches.shape
        
        patches_ = patches.view(B * P, 1, W)
        tok = self.patch_embed(patches_)                # [B*P,D]
        tokens = tok.view(B, P, self.token_dim)         # [B,P,D]

        g_pooled, _ = self.pool(tokens)                 # g_pooled [B,D]
        g = self.g_proj(g_pooled)                       # [B,D]
        out = self.hyper(g)                             # [B, P*D+1]

        Wgen = out[:, :-1].view(B, P, self.token_dim)   # [B,P,D]
        b = out[:, -1]                                  # [B]

        logit = (Wgen * tokens).sum(dim=(1, 2)) + b
        return logit, Wgen, tokens, T


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
# PyTorch Lightning Module
# -----------------------
class ECGPatchIMNLightning(pl.LightningModule):
    def __init__(
        self,
        signal_len: int,
        window: int,
        stride: int,
        token_dim: int = 64,
        hyper_hidden: int = 256,
        pool_hidden: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        l1_lambda: float = 0.05,
        pos_weight: float = 1.0,
        scheduler_type: str = "cosine",
        scheduler_params: dict = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ECGPatchIMN(
            signal_len=signal_len,
            window=window,
            stride=stride,
            token_dim=token_dim,
            hyper_hidden=hyper_hidden,
            pool_hidden=pool_hidden,
            dropout=dropout,
            verbose=verbose
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.pos_weight = pos_weight
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        
        # For tracking metrics
        self.pos_w = None
        
        # For storing outputs for epoch-end calculations
        self.validation_outputs = []
        self.test_outputs = []
        
        # Store verbose flag
        self.verbose = verbose

    def forward(self, x, verbose: bool = None):
        if verbose is None:
            verbose = self.verbose
        return self.model(x, verbose=verbose)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit, Wgen, tokens, T = self.model(x)
        
        if self.pos_w is None:
            self.pos_w = torch.tensor([self.pos_weight], device=self.device)
        
        bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=self.pos_w)
        l1 = Wgen.abs().mean()
        loss = bce + self.l1_lambda * l1
        
        with torch.no_grad():
            pred = (torch.sigmoid(logit) > 0.5).float()
            acc = (pred == y).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_bce", bce, on_step=False, on_epoch=True)
        self.log("train_l1", l1, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit, Wgen, tokens, T = self.model(x)
        
        bce = F.binary_cross_entropy_with_logits(logit, y)
        l1 = Wgen.abs().mean()
        loss = bce + self.l1_lambda * l1
        
        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).float()
        acc = (pred == y).float().mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_bce", bce, on_step=False, on_epoch=True)
        self.log("val_l1", l1, on_step=False, on_epoch=True)
        
        output = {"y": y, "prob": prob, "loss": loss, "acc": acc}
        self.validation_outputs.append(output)
        return output

    def test_step(self, batch, batch_idx):
        x, y = batch
        logit, Wgen, tokens, T = self.model(x)
        
        bce = F.binary_cross_entropy_with_logits(logit, y)
        l1 = Wgen.abs().mean()
        loss = bce + self.l1_lambda * l1
        
        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).float()
        acc = (pred == y).float().mean()
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_bce", bce, on_step=False, on_epoch=True)
        self.log("test_l1", l1, on_step=False, on_epoch=True)
        
        output = {"y": y, "prob": prob, "loss": loss, "acc": acc}
        self.test_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        # Compute AUC for validation
        if len(self.validation_outputs) > 0:
            ys = torch.cat([o["y"] for o in self.validation_outputs], dim=0)
            probs = torch.cat([o["prob"] for o in self.validation_outputs], dim=0)
            auc = simple_auc_roc(ys, probs)
            self.log("val_auc", auc, on_epoch=True, prog_bar=True, sync_dist=True)
            # Clear outputs for next epoch
            self.validation_outputs.clear()

    def on_test_epoch_end(self):
        # Compute AUC for test
        if len(self.test_outputs) > 0:
            ys = torch.cat([o["y"] for o in self.test_outputs], dim=0)
            probs = torch.cat([o["prob"] for o in self.test_outputs], dim=0)
            auc = simple_auc_roc(ys, probs)
            self.log("test_auc", auc, on_epoch=True, sync_dist=True)
            # Clear outputs
            self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        if self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
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

    @torch.no_grad()
    def explain(self, x):
        return self.model.explain(x)

    @torch.no_grad()
    def explain_with_topk(self, x, topk_leads: int = None, topk_patches: int = None):
        return self.model.explain_with_topk(x, topk_leads, topk_patches)

    @torch.no_grad()
    def monte_carlo_dropout_inference(self, x, n_samples: int = 50):
        """
        Perform Monte Carlo dropout inference.
        Returns:
          mean_prob: [B] mean probability across samples
          mean_heatmap: [B,12,T] mean heatmap across samples
          std_heatmap: [B,12,T] std of heatmap across samples
          mean_logit: [B] mean logit across samples
          std_logit: [B] std of logit across samples
        """
        B = x.size(0)
        T = self.model.T
        
        # Collect samples
        logits = []
        heatmaps = []
        
        for _ in range(n_samples):
            # Temporarily enable training mode for dropout, but keep no_grad
            self.model.train()
            logit, Wgen, tokens, _ = self.model.forward_with_dropout(x)
            self.model.eval()
            
            prob = torch.sigmoid(logit)
            patch_scores = (Wgen * tokens).sum(dim=-1)  # [B,P]
            heatmap = patch_scores.view(B, 12, T)       # [B,12,T]
            
            logits.append(logit)
            heatmaps.append(heatmap)
        
        # Stack and compute statistics
        logits_stack = torch.stack(logits, dim=0)  # [n_samples, B]
        heatmaps_stack = torch.stack(heatmaps, dim=0)  # [n_samples, B, 12, T]
        
        mean_logit = logits_stack.mean(dim=0)  # [B]
        std_logit = logits_stack.std(dim=0)    # [B]
        mean_prob = torch.sigmoid(mean_logit)  # [B]
        
        mean_heatmap = heatmaps_stack.mean(dim=0)  # [B, 12, T]
        std_heatmap = heatmaps_stack.std(dim=0)    # [B, 12, T]
        
        return mean_prob, mean_heatmap, std_heatmap, mean_logit, std_logit


# -----------------------
# Checkpointing / Logging
# -----------------------
def save_checkpoint(path, model, opt, epoch, best_val_auc, args, extra=None):
    ckpt = {
        "epoch": epoch,
        "best_val_auc": best_val_auc,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "args": vars(args),
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def append_metrics_csv(csv_path, row_dict):
    header = ",".join(row_dict.keys())
    row = ",".join([str(row_dict[k]) for k in row_dict.keys()])
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8") as f:
        if not exists:
            f.write(header + "\n")
        f.write(row + "\n")


# -----------------------
# Save final Wgen and tokens
# -----------------------
# BEST OPTIONS FOR SAVING:
#   - npz (compressed): Default. Small size, one file, numpy-native. Use for downstream
#     numpy/analysis, portability, or when you don't need PyTorch. Load: np.load(path)["Wgen"].
#   - pt: Keep tensors as-is for PyTorch. Use when you'll load back into PyTorch scripts.
#     Load: torch.load(path)["Wgen"].
#   - h5: For very large runs (many samples) or when you need chunked/partial loading.
#     Requires h5py. Use "all" to save all formats and compare sizes/load times.


@torch.no_grad()
def collect_wgen_tokens(model, loader, device, max_samples: int = None):
    """
    Run model on loader, collect Wgen [N,P,D], tokens [N,P,D], labels, logits, probs.
    Returns dict with numpy arrays (float32) and metadata P, T, token_dim.
    """
    model.eval()
    Wgen_list, tok_list, y_list, logit_list, prob_list = [], [], [], [], []
    n = 0
    for batch in loader:
        x, y = batch
        x = x.to(device)
        logit, Wgen, tokens, T = model.model(x)
        prob = torch.sigmoid(logit)
        B = x.size(0)
        Wgen_list.append(Wgen.cpu().numpy())
        tok_list.append(tokens.cpu().numpy())
        y_list.append(y.numpy())
        logit_list.append(logit.cpu().numpy())
        prob_list.append(prob.cpu().numpy())
        n += B
        if max_samples is not None and n >= max_samples:
            break
    Wgen = np.concatenate(Wgen_list, axis=0).astype(np.float32)
    tokens = np.concatenate(tok_list, axis=0).astype(np.float32)
    labels = np.concatenate(y_list, axis=0).astype(np.float32)
    logits = np.concatenate(logit_list, axis=0).astype(np.float32)
    probs = np.concatenate(prob_list, axis=0).astype(np.float32)
    if max_samples is not None and n > max_samples:
        Wgen = Wgen[:max_samples]
        tokens = tokens[:max_samples]
        labels = labels[:max_samples]
        logits = logits[:max_samples]
        probs = probs[:max_samples]
    P, D = Wgen.shape[1], Wgen.shape[2]
    return {
        "Wgen": Wgen,
        "tokens": tokens,
        "labels": labels,
        "logits": logits,
        "probs": probs,
        "P": P,
        "T": T,
        "token_dim": D,
        "N": Wgen.shape[0],
    }


def save_wgen_tokens_npz(data: dict, path: str):
    """Save as compressed npz. Load: d = np.load(path); d['Wgen'], d['tokens'], d['P'], etc."""
    out = {k: data[k] for k in ("Wgen", "tokens", "labels", "logits", "probs")}
    out["P"] = np.int64(data["P"])
    out["T"] = np.int64(data["T"])
    out["token_dim"] = np.int64(data["token_dim"])
    out["N"] = np.int64(data["N"])
    np.savez_compressed(path, **out)


def save_wgen_tokens_pt(data: dict, path: str):
    """Save as PyTorch .pt. Load: d = torch.load(path); d['Wgen'], d['tokens'], etc."""
    out = {
        "Wgen": torch.from_numpy(data["Wgen"]),
        "tokens": torch.from_numpy(data["tokens"]),
        "labels": torch.from_numpy(data["labels"]),
        "logits": torch.from_numpy(data["logits"]),
        "probs": torch.from_numpy(data["probs"]),
        "P": data["P"],
        "T": data["T"],
        "token_dim": data["token_dim"],
        "N": data["N"],
    }
    torch.save(out, path)


def save_wgen_tokens_h5(data: dict, path: str):
    """Save as HDF5. Good for large data; supports partial read. Requires h5py."""
    try:
        import h5py
    except ImportError:
        raise ImportError("save_wgen_tokens_h5 requires h5py. Install with: pip install h5py")
    with h5py.File(path, "w") as f:
        for k in ("Wgen", "tokens", "labels", "logits", "probs"):
            f.create_dataset(k, data=data[k], compression="gzip")
        for k in ("P", "T", "token_dim", "N"):
            f.attrs[k] = data[k]


# -----------------------
# Visualization to PDF (per-lead)
# -----------------------
@torch.no_grad()
def visualize_pos_neg_to_pdf(model, dataset, device, pdf_path: str, sampling_rate: int,
                             window: int, stride: int,
                             n_pos: int, n_neg: int,
                             random_pick: bool = False, seed: int = 123,
                             lead_names=None):
    """
    Saves a multi-page PDF:
      - First N positives (MI=1) and M negatives (NORM=0), or random if random_pick=True.
      - Each page: heatmap + 12 lead plots with shaded contributions.
    """
    model.eval()
    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    # Collect indices by label from dataset (assumes __getitem__ returns (x,y))
    pos_idx = []
    neg_idx = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        yv = int(float(y))
        if yv == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    if random_pick:
        rng = np.random.default_rng(seed)
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

    sel_pos = pos_idx[:n_pos]
    sel_neg = neg_idx[:n_neg]

    # If not enough samples, warn in console (still proceeds)
    if len(sel_pos) < n_pos:
        print(f"WARNING: requested n_pos={n_pos}, but only found {len(sel_pos)} positives in dataset.")
    if len(sel_neg) < n_neg:
        print(f"WARNING: requested n_neg={n_neg}, but only found {len(sel_neg)} negatives in dataset.")

    # You can choose the page order:
    #  - grouped: positives first then negatives (default here)
    selected = [("MI", i) for i in sel_pos] + [("NORM", i) for i in sel_neg]

    with PdfPages(pdf_path) as pdf:
        for tag, idx in selected:
            x, y = dataset[idx]
            x_b = x.unsqueeze(0).to(device)  # [1,12,L]

            prob, heatmap = model.explain(x_b)
            prob = float(prob.item())
            hm = heatmap.squeeze(0).detach().cpu().numpy()  # [12,T]
            x_np = x.detach().cpu().numpy()                 # [12,L]

            denom = np.max(np.abs(hm)) + 1e-6
            hm_disp = hm / denom  # [-1,1]

            L = x_np.shape[1]
            T = hm.shape[1]

            fig = plt.figure(figsize=(11.7, 16.5))
            gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1]*12 + [0.5])

            ax0 = fig.add_subplot(gs[0, 0])
            im = ax0.imshow(hm_disp, aspect="auto", vmin=-1, vmax=1, cmap="bwr")
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            ax0.set_title(f"{tag} sample | true={int(y.item())} | P(MI)={prob:.3f} | idx={idx}")
            fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

            for lead in range(12):
                ax = fig.add_subplot(gs[lead + 1, 0])
                ax.plot(x_np[lead], linewidth=0.8)
                ax.set_xlim(0, L - 1)
                ax.set_ylabel(lead_names[lead], rotation=0, labelpad=20, va="center")

                contrib = hm_disp[lead]  # [T]
                for t in range(T):
                    a = float(contrib[t])
                    alpha = min(0.30, abs(a) * 0.30)
                    if alpha > 0:
                        color = "red" if a > 0 else "blue"
                        start = t * stride
                        end = min(start + window, L)
                        ax.axvspan(start, end, alpha=alpha, color=color, linewidth=0)

                ax.set_xticks([])

            axf = fig.add_subplot(gs[13, 0])
            axf.axis("off")
            axf.text(0, 0.5, "Red=pushes toward MI, Blue=pushes toward NORM (signed contributions; normalized per record).",
                     fontsize=10)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# -----------------------
# Visualization: Top-k leads and patches
# -----------------------
@torch.no_grad()
def visualize_topk_leads_patches_to_pdf(model, dataset, device, pdf_path: str, sampling_rate: int,
                                       window: int, stride: int,
                                       n_pos: int, n_neg: int,
                                       topk_leads: int = 3,
                                       topk_patches: int = 10,
                                       random_pick: bool = False, seed: int = 123,
                                       lead_names=None):
    """
    Visualizes samples with top-k leads and patches highlighted.
    """
    model.eval()
    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    # Collect indices by label
    pos_idx = []
    neg_idx = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        yv = int(float(y))
        if yv == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    if random_pick:
        rng = np.random.default_rng(seed)
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

    sel_pos = pos_idx[:n_pos]
    sel_neg = neg_idx[:n_neg]

    selected = [("MI", i) for i in sel_pos] + [("NORM", i) for i in sel_neg]

    with PdfPages(pdf_path) as pdf:
        for tag, idx in selected:
            x, y = dataset[idx]
            x_b = x.unsqueeze(0).to(device)  # [1,12,L]

            prob, heatmap, top_lead_indices, top_patch_indices = model.explain_with_topk(
                x_b, topk_leads=topk_leads, topk_patches=topk_patches
            )
            prob = float(prob.item())
            hm = heatmap.squeeze(0).detach().cpu().numpy()  # [12,T]
            x_np = x.detach().cpu().numpy()                 # [12,L]
            
            # Get top-k indices
            top_leads = top_lead_indices.squeeze(0).cpu().numpy() if top_lead_indices is not None else np.array([])
            top_patches = top_patch_indices.squeeze(0).cpu().numpy() if top_patch_indices is not None else np.array([])
            
            # Convert patch indices to (lead, time) coordinates
            T = hm.shape[1]
            top_patch_coords = []
            for p_idx in top_patches:
                lead_idx = p_idx // T
                time_idx = p_idx % T
                top_patch_coords.append((lead_idx, time_idx))

            denom = np.max(np.abs(hm)) + 1e-6
            hm_disp = hm / denom

            L = x_np.shape[1]

            fig = plt.figure(figsize=(11.7, 16.5))
            gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1]*12 + [0.5])

            # Heatmap with top-k leads highlighted
            ax0 = fig.add_subplot(gs[0, 0])
            im = ax0.imshow(hm_disp, aspect="auto", vmin=-1, vmax=1, cmap="bwr")
            
            # Highlight top-k leads
            for lidx in top_leads:
                ax0.axhline(y=lidx-0.5, color='yellow', linewidth=3, alpha=0.7)
                ax0.axhline(y=lidx+0.5, color='yellow', linewidth=3, alpha=0.7)
            
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            title = f"{tag} sample | true={int(y.item())} | P(MI)={prob:.3f} | idx={idx}"
            if len(top_leads) > 0:
                top_lead_names = [lead_names[i] for i in top_leads]
                title += f"\nTop-{len(top_leads)} leads: {', '.join(top_lead_names)}"
            ax0.set_title(title)
            fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

            # Plot each lead
            for lead in range(12):
                ax = fig.add_subplot(gs[lead + 1, 0])
                ax.plot(x_np[lead], linewidth=0.8)
                ax.set_xlim(0, L - 1)
                ax.set_ylabel(lead_names[lead], rotation=0, labelpad=20, va="center")

                # Highlight if this is a top-k lead
                is_top_lead = lead in top_leads
                if is_top_lead:
                    ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.2, color='yellow', zorder=0)

                contrib = hm_disp[lead]  # [T]
                for t in range(T):
                    a = float(contrib[t])
                    alpha = min(0.30, abs(a) * 0.30)
                    if alpha > 0:
                        color = "red" if a > 0 else "blue"
                        start = t * stride
                        end = min(start + window, L)
                        
                        # Highlight top-k patches with thicker border
                        is_top_patch = (lead, t) in top_patch_coords
                        if is_top_patch:
                            ax.axvspan(start, end, alpha=alpha, color=color, linewidth=2, edgecolor='lime', zorder=1)
                        else:
                            ax.axvspan(start, end, alpha=alpha, color=color, linewidth=0)

                ax.set_xticks([])

            axf = fig.add_subplot(gs[13, 0])
            axf.axis("off")
            info_text = "Red=pushes toward MI, Blue=pushes toward NORM (signed contributions; normalized per record).\n"
            info_text += f"Yellow highlight: Top-{topk_leads} leads | Lime border: Top-{topk_patches} patches"
            axf.text(0, 0.5, info_text, fontsize=10)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


# -----------------------
# Visualization: Monte Carlo Dropout Uncertainty
# -----------------------
@torch.no_grad()
def visualize_uncertainty_to_pdf(model, dataset, device, pdf_path: str, sampling_rate: int,
                                 window: int, stride: int,
                                 n_pos: int, n_neg: int,
                                 n_mc_samples: int = 50,
                                 random_pick: bool = False, seed: int = 123,
                                 lead_names=None):
    """
    Visualizes uncertainty from Monte Carlo dropout sampling.
    """
    model.eval()
    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    # Collect indices by label
    pos_idx = []
    neg_idx = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        yv = int(float(y))
        if yv == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    if random_pick:
        rng = np.random.default_rng(seed)
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

    sel_pos = pos_idx[:n_pos]
    sel_neg = neg_idx[:n_neg]

    selected = [("MI", i) for i in sel_pos] + [("NORM", i) for i in sel_neg]

    with PdfPages(pdf_path) as pdf:
        for tag, idx in selected:
            x, y = dataset[idx]
            x_b = x.unsqueeze(0).to(device)  # [1,12,L]

            # Monte Carlo dropout inference
            mean_prob, mean_heatmap, std_heatmap, mean_logit, std_logit = model.monte_carlo_dropout_inference(
                x_b, n_samples=n_mc_samples
            )
            
            mean_prob_val = float(mean_prob.item())
            std_logit_val = float(std_logit.item())
            mean_hm = mean_heatmap.squeeze(0).detach().cpu().numpy()  # [12,T]
            std_hm = std_heatmap.squeeze(0).detach().cpu().numpy()    # [12,T]
            x_np = x.detach().cpu().numpy()                           # [12,L]

            denom_mean = np.max(np.abs(mean_hm)) + 1e-6
            mean_hm_disp = mean_hm / denom_mean
            
            # Normalize std by max std value
            denom_std = np.max(std_hm) + 1e-6
            std_hm_disp = std_hm / denom_std

            L = x_np.shape[1]
            T = mean_hm.shape[1]

            fig = plt.figure(figsize=(11.7, 20))
            gs = fig.add_gridspec(15, 1, height_ratios=[2, 2] + [1]*12 + [0.5])

            # Mean heatmap
            ax0 = fig.add_subplot(gs[0, 0])
            im0 = ax0.imshow(mean_hm_disp, aspect="auto", vmin=-1, vmax=1, cmap="bwr")
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            ax0.set_title(f"{tag} sample | true={int(y.item())} | P(MI)={mean_prob_val:.3f}±{std_logit_val:.3f} | idx={idx} | Mean Contributions")
            fig.colorbar(im0, ax=ax0, fraction=0.02, pad=0.01)

            # Uncertainty heatmap (std)
            ax1 = fig.add_subplot(gs[1, 0])
            im1 = ax1.imshow(std_hm_disp, aspect="auto", vmin=0, vmax=1, cmap="hot")
            ax1.set_yticks(range(12))
            ax1.set_yticklabels(lead_names)
            ax1.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            ax1.set_title(f"Uncertainty (Std) across {n_mc_samples} MC samples")
            fig.colorbar(im1, ax=ax1, fraction=0.02, pad=0.01)

            # Plot each lead with uncertainty shading
            for lead in range(12):
                ax = fig.add_subplot(gs[lead + 2, 0])
                ax.plot(x_np[lead], linewidth=0.8, color='black', label='Signal')
                ax.set_xlim(0, L - 1)
                ax.set_ylabel(lead_names[lead], rotation=0, labelpad=20, va="center")

                # Mean contribution shading
                mean_contrib = mean_hm_disp[lead]  # [T]
                std_contrib = std_hm_disp[lead]    # [T]
                
                for t in range(T):
                    a_mean = float(mean_contrib[t])
                    a_std = float(std_contrib[t])
                    
                    alpha_mean = min(0.30, abs(a_mean) * 0.30)
                    alpha_std = min(0.15, a_std * 0.15)  # Uncertainty shading
                    
                    start = t * stride
                    end = min(start + window, L)
                    
                    if alpha_mean > 0:
                        color = "red" if a_mean > 0 else "blue"
                        ax.axvspan(start, end, alpha=alpha_mean, color=color, linewidth=0, label='Mean contribution' if t == 0 else '')
                    
                    # Overlay uncertainty (darker = more uncertain)
                    if alpha_std > 0:
                        ax.axvspan(start, end, alpha=alpha_std, color='gray', linewidth=0, label='Uncertainty' if t == 0 else '')

                ax.set_xticks([])

            axf = fig.add_subplot(gs[14, 0])
            axf.axis("off")
            info_text = f"Mean contributions: Red=pushes toward MI, Blue=pushes toward NORM.\n"
            info_text += f"Gray overlay: Uncertainty (darker = more uncertain).\n"
            info_text += f"MC Dropout: {n_mc_samples} samples | P(MI)={mean_prob_val:.3f}±{std_logit_val:.3f}"
            axf.text(0, 0.5, info_text, fontsize=10)

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

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--l1_lambda", type=float, default=0.05)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--hyper_hidden", type=int, default=256)
    parser.add_argument("--pool_hidden", type=int, default=128, help="Hidden dim for GatedAttnPool")

    # Patching
    parser.add_argument("--window", type=int, default=None,
                        help="Window in samples. Default: 0.5s => 50@100Hz or 250@500Hz.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride in samples. Default: window//2 (overlap 50%%).")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto", help="accelerator: auto, gpu, cpu")
    parser.add_argument("--devices", type=int, default=1, help="number of devices")

    # WandB
    parser.add_argument("--wandb_project", type=str, default="ecg-patch-imn", help="WandB project name")
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

    # Visualization
    parser.add_argument("--viz_pdf", type=str, default="viz_mi_vs_norm_test.pdf", help="Output PDF filename")
    parser.add_argument("--n_pos_viz", type=int, default=25, help="Number of MI (positive) test samples to visualize")
    parser.add_argument("--n_neg_viz", type=int, default=25, help="Number of NORM (negative) test samples to visualize")
    parser.add_argument("--viz_random", action="store_true", help="Randomly sample positives/negatives instead of first N/M")
    parser.add_argument("--viz_seed", type=int, default=123, help="Seed for viz sampling when --viz_random is set")
    
    # Top-k visualization
    parser.add_argument("--viz_topk", action="store_true", help="Enable top-k leads/patches visualization")
    parser.add_argument("--topk_leads", type=int, default=3, help="Number of top-k leads to highlight")
    parser.add_argument("--topk_patches", type=int, default=10, help="Number of top-k patches to highlight")
    parser.add_argument("--viz_topk_pdf", type=str, default="viz_topk_test.pdf", help="Output PDF filename for top-k visualization")
    
    # Monte Carlo Dropout visualization
    parser.add_argument("--viz_mc_dropout", action="store_true", help="Enable Monte Carlo dropout uncertainty visualization")
    parser.add_argument("--mc_samples", type=int, default=50, help="Number of Monte Carlo samples for uncertainty estimation")
    parser.add_argument("--viz_uncertainty_pdf", type=str, default="viz_uncertainty_test.pdf", help="Output PDF filename for uncertainty visualization")
    
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")

    # Checkpoints / run directory
    parser.add_argument("--out_dir", type=str, default="runs/mi_vs_norm_imn", help="Directory for checkpoints/logs")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Optional experiment name; included in run dir as {script}_{exp_name}_{timestamp}")

    # Save final Wgen and tokens
    parser.add_argument("--save_wgen_tokens", action="store_true",
                        help="Save final Wgen and tokens after evaluation")
    parser.add_argument("--save_wgen_tokens_format", type=str, default="npz",
                        choices=["npz", "pt", "h5", "all"],
                        help="Format: npz (compressed, default), pt (PyTorch), h5 (HDF5), all")
    parser.add_argument("--save_wgen_tokens_split", type=str, default="test",
                        choices=["test", "val", "both"],
                        help="Which split to save: test, val, or both")
    parser.add_argument("--save_wgen_tokens_max_samples", type=int, default=None,
                        help="Cap number of samples to save (default: all)")

    args = parser.parse_args()

    set_seed(args.seed)
    
    # Parse scheduler params
    import json
    try:
        scheduler_params = json.loads(args.scheduler_params)
    except:
        scheduler_params = {}

    # -------------------------------------------------
    # Create run directory: {out_dir}/{script_name}[_{exp_name}]_{timestamp}
    # -------------------------------------------------
    script_basename = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = script_basename
    if args.exp_name is not None and args.exp_name.strip():
        exp = args.exp_name.strip().replace(" ", "_")
        exp = "".join(c for c in exp if c.isalnum() or c in "._-") or "exp"
        run_name = f"{script_basename}_{exp}"
    run_name = f"{run_name}_{timestamp}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"📁 Run directory: {run_dir}")

    # Save run configuration
    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    path = args.path if args.path.endswith("/") else args.path + "/"

    # Load metadata
    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    _, aggregate_diagnostic = build_superclasses(path)
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

    # MI vs NORM only (exclusive)
    labels = Y["diagnostic_superclass"].values
    is_mi = np.array(["MI" in l for l in labels])
    is_norm = np.array(["NORM" in l for l in labels])
    keep = (is_mi ^ is_norm)
    Yf = Y[keep].copy()

    y_bin = np.array([1.0 if "MI" in l else 0.0 for l in Yf["diagnostic_superclass"].values], dtype=np.float32)
    print(f"Kept MI vs NORM only: N={len(Yf)} | MI={int(y_bin.sum())} | NORM={int((1-y_bin).sum())}")

    # Load signals
    print(f"Loading signals at {args.sampling_rate}Hz (filtered set)...")
    X = load_raw_data(Yf, args.sampling_rate, path)  # [N, L, 12]
    X = np.transpose(X, (0, 2, 1))                   # [N, 12, L]
    N, C, L = X.shape
    print("X shape:", X.shape)

    # Folds: train 1-8, val 9, test 10
    fold = Yf["strat_fold"].values
    train_mask = (fold >= 1) & (fold <= 8)
    val_mask = (fold == 9)
    test_mask = (fold == 10)

    X_train, y_train = X[train_mask], y_bin[train_mask]
    X_val, y_val     = X[val_mask], y_bin[val_mask]
    X_test, y_test   = X[test_mask], y_bin[test_mask]

    print("Split sizes:", len(y_train), len(y_val), len(y_test))

    # Default window/stride
    if args.window is None:
        window = 50 if args.sampling_rate == 100 else 250  # 0.5s
    else:
        window = args.window
    if args.stride is None:
        stride = window // 2
    else:
        stride = args.stride

    assert window <= L, f"window={window} must be <= signal length L={L}"
    assert stride > 0, "stride must be > 0"
    T = (L - window) // stride + 1
    print(f"Patching: window={window}, stride={stride}, segments T={T}, patches P={12*T}")

    # Torch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val   = torch.from_numpy(X_val).float()
    y_val   = torch.from_numpy(y_val).float()
    X_test  = torch.from_numpy(X_test).float()
    y_test  = torch.from_numpy(y_test).float()

    # pos_weight for training imbalance
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = (n_neg / max(n_pos, 1.0))
    print(f"Train imbalance: pos_weight={pos_weight:.3f}")

    train_ds = PTBXLBinaryDataset(X_train, y_train, per_lead_zscore=True)
    val_ds   = PTBXLBinaryDataset(X_val, y_val, per_lead_zscore=True)
    test_ds  = PTBXLBinaryDataset(X_test, y_test, per_lead_zscore=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Create Lightning model
    model = ECGPatchIMNLightning(
        signal_len=L,
        window=window,
        stride=stride,
        token_dim=args.token_dim,
        hyper_hidden=args.hyper_hidden,
        pool_hidden=args.pool_hidden,
        dropout=0.1,
        lr=args.lr,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        pos_weight=pos_weight,
        scheduler_type=args.scheduler if args.scheduler != "none" else None,
        scheduler_params=scheduler_params,
        verbose=args.verbose,
    )

    # Setup WandB logger
    wandb_name = args.wandb_name or run_name
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=wandb_name,
        save_dir=run_dir,
        offline=args.wandb_offline,
    )
    
    # Log hyperparameters
    wandb_logger.log_hyperparams(vars(args))
    wandb_logger.log_hyperparams({
        "window": window,
        "stride": stride,
        "signal_len": L,
        "pos_weight": pos_weight,
        "pool_hidden": args.pool_hidden,
    })

    # Setup callbacks
    callbacks = []
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor=args.early_stop_monitor,
        mode=args.early_stop_mode,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        verbose=True,
    )
    callbacks.append(early_stop)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="best-{epoch:02d}-{val_auc:.4f}",
        monitor=args.early_stop_monitor,
        mode=args.early_stop_mode,
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=wandb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True,
    )

    # Train
    print("\n🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test
    print("\n🧪 Running test evaluation...")
    trainer.test(model, test_loader, ckpt_path="best")

    # Load best model for visualization
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\n📦 Loading best model from: {best_model_path}")
        model = ECGPatchIMNLightning.load_from_checkpoint(best_model_path)
        model.eval()

    # Visualization PDF on test set
    pdf_path = args.viz_pdf
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(run_dir, pdf_path)

    with open(os.path.join(run_dir, "model.txt"), "w") as f:
        f.write(str(model.model))

    print(f"\n📊 Saving per-lead visualizations to: {pdf_path}")
    device = next(model.parameters()).device
    visualize_pos_neg_to_pdf(
        model=model,
        dataset=test_ds,
        device=device,
        pdf_path=pdf_path,
        sampling_rate=args.sampling_rate,
        window=window,
        stride=stride,
        n_pos=args.n_pos_viz,
        n_neg=args.n_neg_viz,
        random_pick=args.viz_random,
        seed=args.viz_seed
    )
    print("Done.")
    
    # Top-k leads and patches visualization
    if args.viz_topk:
        topk_pdf_path = args.viz_topk_pdf
        if not os.path.isabs(topk_pdf_path):
            topk_pdf_path = os.path.join(run_dir, topk_pdf_path)
        
        print(f"\n📊 Saving top-k leads/patches visualizations to: {topk_pdf_path}")
        visualize_topk_leads_patches_to_pdf(
            model=model,
            dataset=test_ds,
            device=device,
            pdf_path=topk_pdf_path,
            sampling_rate=args.sampling_rate,
            window=window,
            stride=stride,
            n_pos=args.n_pos_viz,
            n_neg=args.n_neg_viz,
            topk_leads=args.topk_leads,
            topk_patches=args.topk_patches,
            random_pick=args.viz_random,
            seed=args.viz_seed
        )
        print("Top-k visualization done.")
    
    # Monte Carlo Dropout uncertainty visualization
    if args.viz_mc_dropout:
        uncertainty_pdf_path = args.viz_uncertainty_pdf
        if not os.path.isabs(uncertainty_pdf_path):
            uncertainty_pdf_path = os.path.join(run_dir, uncertainty_pdf_path)
        
        print(f"\n📊 Saving Monte Carlo dropout uncertainty visualizations to: {uncertainty_pdf_path}")
        print(f"   Running {args.mc_samples} MC samples per test sample...")
        visualize_uncertainty_to_pdf(
            model=model,
            dataset=test_ds,
            device=device,
            pdf_path=uncertainty_pdf_path,
            sampling_rate=args.sampling_rate,
            window=window,
            stride=stride,
            n_pos=args.n_pos_viz,
            n_neg=args.n_neg_viz,
            n_mc_samples=args.mc_samples,
            random_pick=args.viz_random,
            seed=args.viz_seed
        )
        print("Uncertainty visualization done.")

    # Save final Wgen and tokens
    if args.save_wgen_tokens:
        fmt = args.save_wgen_tokens_format
        split_choice = args.save_wgen_tokens_split
        max_samp = args.save_wgen_tokens_max_samples
        splits = []
        loaders = {}
        if split_choice in ("test", "both"):
            splits.append("test")
            loaders["test"] = test_loader
        if split_choice in ("val", "both"):
            splits.append("val")
            loaders["val"] = val_loader
        for sp in splits:
            loader = loaders[sp]
            print(f"\n📦 Collecting Wgen/tokens for {sp} split (max_samples={max_samp})...")
            data = collect_wgen_tokens(model, loader, device, max_samples=max_samp)
            print(f"   N={data['N']}, P={data['P']}, T={data['T']}, token_dim={data['token_dim']}")
            base = os.path.join(run_dir, f"wgen_tokens_{sp}")
            if fmt == "npz" or fmt == "all":
                p = base + ".npz"
                save_wgen_tokens_npz(data, p)
                print(f"   Saved npz: {p}")
            if fmt == "pt" or fmt == "all":
                p = base + ".pt"
                save_wgen_tokens_pt(data, p)
                print(f"   Saved pt:  {p}")
            if fmt == "h5" or fmt == "all":
                try:
                    p = base + ".h5"
                    save_wgen_tokens_h5(data, p)
                    print(f"   Saved h5: {p}")
                except ImportError as e:
                    print(f"   Skipped h5: {e}")
        print("Done saving Wgen/tokens.")

    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    main()
