"""
PTB-XL MI vs NORM - Interpretable Mesomorphic Neural Network (IMN) Implementation.
SINGLE LINEAR OUTPUT VERSION (Binary Classification with 1 Output Node).

Based on: "Interpretable Mesomorphic Neural Networks" (NeurIPS 2024)
Converted from black-box CNN + Grad-CAM baseline.

- Architecture: Deep Hypernetwork (CNN + Transition Decoder) -> Generates Weights W and Bias b.
- Prediction: Logit = sum(Input * Generated_Weights) + Generated_Bias (Shape: [B, 1])
- Loss: BinaryCrossEntropyWithLogits + Lambda * L1_Norm(Generated_Weights)
- XAI: Intrinsic. We visualize the generated weights * input.

Modifications:
- Added --top_k_pos and --top_k_neg to visualize segments contributing positively (Red) 
  or negatively (Blue) to the final logit decision.

Run:
python imn_ptbxl_1d.py --path /path/to/ptbxl --sampling_rate 500 --epochs 20 --lambda_l1 1e-4 --viz_random --top_k_pos 3 --top_k_neg 3
"""

import argparse
import ast
import json
import os
import random
from math import ceil
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
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator

try:
    import ecg_plot
    ECG_PLOT_AVAILABLE = True
except ImportError:
    ECG_PLOT_AVAILABLE = False

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
    y: [N] int64 {0,1}   0=NORM, 1=MI
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, per_lead_zscore: bool = True):
        super().__init__()
        assert X.ndim == 3 and X.shape[1] == 12
        self.X = X.float()
        self.y = y.float() # Changed to float for BCE
        self.per_lead_zscore = per_lead_zscore

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [12, L]
        y = self.y[idx]  # scalar float

        if self.per_lead_zscore:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        return x, y


# -----------------------
# IMN Architecture (Single Linear Output)
# -----------------------
class ECG_IMN(nn.Module):
    """
    Interpretable Mesomorphic Network for ECG.
    
    Generates ONE set of Weights W [B, 1, 12, L] and ONE Bias b [B, 1].
    Final Prediction: Logit = sum(W * x) + b
    
    Sigmoid(Logit) -> Probability of Positive Class.
    """
    def __init__(self, input_channels=12, signal_len=1000, dropout=0.2):
        super().__init__()
        self.C = input_channels
        self.L = signal_len
        
        # We output 1 feature map for the binary decision boundary
        output_dim = 1 

        # --- Hypernetwork Backbone (Encoder) ---
        # Input: [B, 1, 12, L]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        ) # Out: [B, 16, 12, L]

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
        ) # Out: [B, 32, 12, L/2]

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        ) # Out: [B, 64, 12, L/4]
        
        self.dropout = nn.Dropout(dropout)

        # --- Transition Network (Weight Generator) ---
        # Input: [B, 64, 12, L/4] -> Output: [B, 1, 12, L]
        self.transition = nn.Sequential(
            # Stage 1: Upsample L/4 -> L/2
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Upsample(scale_factor=(1, 2), mode='nearest'), 
            
            # Stage 2: Upsample L/2 -> L
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Upsample(scale_factor=(1, 2), mode='nearest'),

            # Final Projection: Map to 1 channel (Weight W)
            nn.Conv2d(16, output_dim, kernel_size=3, padding=1, bias=True) 
        )

        # --- Bias Generator ---
        # Generates scalar bias b
        self.bias_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias_head = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        x: [B, 12, L]
        Returns: logits [B, 1], generated_w [B, 1, 12, L], generated_b [B, 1]
        """
        B, C, L = x.shape
        
        # 1. Extract Features
        feat = x.unsqueeze(1)
        feat = self.conv1(feat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.dropout(feat)

        # 2. Generate Parameters
        # A) Weights W: [B, 1, 12, L]
        generated_w = self.transition(feat)
        
        # B) Bias b: [B, 1]
        b_feat = self.bias_pool(feat).view(B, -1)
        generated_b = self.bias_head(b_feat)

        # 3. Apply Single Linear Model
        # Logit = Sum(W * x) + b
        
        x_expanded = x.unsqueeze(1) # [B, 1, 12, L]
        
        # Element-wise multiplication
        weighted_input = generated_w * x_expanded
        
        # Sum over features -> [B, 1]
        logits = weighted_input.sum(dim=(2, 3)) + generated_b

        return logits, generated_w, generated_b.unsqueeze(-1)


# -----------------------
# Lightning Module
# -----------------------
class IMNLightning(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        signal_len: int,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_l1: float = 1e-4,
        pos_weight: float | None = None, # Scalar weight for BCE
        scheduler_type: str | None = "cosine",
        scheduler_params: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ECG_IMN(
            input_channels=input_channels,
            signal_len=signal_len,
            dropout=dropout
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_l1 = lambda_l1
        self.pos_weight_val = pos_weight
        
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        
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

    def _get_pos_weight(self):
        if self.pos_weight_val is None:
            return None
        return torch.tensor([self.pos_weight_val], device=self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch # y is [B]
        logits, gen_w, gen_b = self.model(x)
        logits = logits.squeeze(1) # Ensure [B]
        
        # 1. Prediction Loss (BCE With Logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self._get_pos_weight())
        
        # 2. Interpretability Loss (L1 norm on generated weights)
        l1_loss = gen_w.abs().mean()
        
        total_loss = bce_loss + (self.lambda_l1 * l1_loss)

        # Metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = (preds == y).float().mean()

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_bce", bce_loss, on_step=False, on_epoch=True)
        self.log("train_l1", l1_loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, gen_w, _ = self.model(x)
        logits = logits.squeeze(1)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self._get_pos_weight())
        
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        acc = (pred == y).float().mean()

        self.log("val_loss", bce_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_probs.append(prob.detach().cpu())
        self.val_y.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_y) if len(self.val_y) else None
        y_score = torch.cat(self.val_probs) if len(self.val_probs) else None
        if y_true is not None:
            auc = simple_auc_roc(y_true, y_score)
            self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_probs.clear()
        self.val_y.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, _, _ = self.model(x)
        logits = logits.squeeze(1)
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=self._get_pos_weight())
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()
        acc = (pred == y).float().mean()
        
        self.log("test_loss", bce_loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        return {"y": y.detach().cpu(), "p": prob.detach().cpu()}


# -----------------------
# Metrics Helper
# -----------------------
@torch.no_grad()
def simple_auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.detach().cpu().float()
    y_score = y_score.detach().cpu().float()
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true.numpy(), y_score.numpy()))


# -----------------------
# Visualization Helpers (Intrinsic IMN)
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
    name_to_idx.update({n: i for i, n in enumerate(lead_names)})  # case-sensitive fallback
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


def get_top_k_leads(impact: np.ndarray, k: int) -> np.ndarray:
    """
    Top-k leads by **signed** contribution to the positive logit.

    For the single-linear IMN we have:
        logit = sum(w * x) + b  =  sum(impact) + b
    so a lead's total contribution is impact[c].sum().

    We rank leads by this signed contribution (descending), which matches the
    Gradio app behaviour: top-k leads correspond to the strongest evidence
    for the positive class.
    """
    assert impact.ndim == 2
    # Signed total contribution per lead
    lead_imp = impact.sum(axis=1)  # [12]
    # Descending order: largest positive contributions first
    top_idx = np.argsort(lead_imp)[::-1][:k]
    return top_idx.astype(np.int64)


def get_top_k_segments(impact: np.ndarray, window: int, stride: int, k: int) -> np.ndarray:
    """
    Top-k segments by **signed** contribution to the positive logit.
    Returns top-k segment indices (0-based).
    """
    assert impact.ndim == 2
    L = impact.shape[1]
    T = (L - window) // stride + 1
    seg_contrib = np.zeros(T, dtype=np.float64)
    for t in range(T):
        s = t * stride
        e = min(s + window, L)
        # Signed contribution of this time window across all leads
        seg_contrib[t] = np.sum(impact[:, s:e])
    # Descending order: segments with largest positive contribution first
    top_idx = np.argsort(seg_contrib)[::-1][:k]
    return top_idx.astype(np.int64)


def get_top_k_segments_per_lead(impact: np.ndarray, window: int, stride: int, k: int) -> list[set[int]]:
    """
    Top-k segments per lead by that lead's **signed** contribution.
    Returns list of sets: top_seg_per_lead[lead] = set of segment indices.
    """
    assert impact.ndim == 2
    n_leads, L = impact.shape
    T = (L - window) // stride + 1
    result = []
    for c in range(n_leads):
        seg_contrib = np.zeros(T, dtype=np.float64)
        for t in range(T):
            s = t * stride
            e = min(s + window, L)
            # Signed contribution of this window on lead c
            seg_contrib[t] = np.sum(impact[c, s:e])
        # Descending: largest positive contributions first for this lead
        top_idx = np.argsort(seg_contrib)[::-1][:min(k, T)]
        result.append(set(top_idx.tolist()))
    return result

def get_top_pos_neg_segments_per_lead(impact: np.ndarray, window: int, stride: int, k_pos: int, k_neg: int) -> tuple[list[set[int]], list[set[int]]]:
    """
    Returns two lists of sets: (pos_segments, neg_segments) per lead.
    pos_segments: indices of top-k_pos segments with strongest positive contribution (supports MI).
    neg_segments: indices of top-k_neg segments with strongest negative contribution (supports NORM).
    """
    n_leads, L = impact.shape
    T = (L - window) // stride + 1
    pos_res = []
    neg_res = []
    
    for c in range(n_leads):
        seg_contrib = np.zeros(T, dtype=np.float64)
        for t in range(T):
            s = t * stride
            e = min(s + window, L)
            seg_contrib[t] = np.sum(impact[c, s:e])
        
        # Positive: Sort descending (largest positive first)
        sorted_idx_desc = np.argsort(seg_contrib)[::-1]
        # Filter > 0 and take top k_pos
        pos_indices = [i for i in sorted_idx_desc if seg_contrib[i] > 0][:k_pos]
        
        # Negative: Sort ascending (most negative first)
        sorted_idx_asc = np.argsort(seg_contrib)
        # Filter < 0 and take top k_neg
        neg_indices = [i for i in sorted_idx_asc if seg_contrib[i] < 0][:k_neg]
        
        pos_res.append(set(pos_indices))
        neg_res.append(set(neg_indices))
        
    return pos_res, neg_res


def imn_weights_to_segments(impact_12L: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Aggregates point-wise feature attribution (Impact) into segments.
    """
    assert impact_12L.ndim == 2
    L = impact_12L.shape[1]
    T = (L - window) // stride + 1
    seg = np.zeros((12, T), dtype=np.float32)
    
    for t in range(T):
        s = t * stride
        e = min(s + window, L)
        seg[:, t] = np.abs(impact_12L[:, s:e]).mean(axis=1)
        
    mx = seg.max() + 1e-9
    seg = seg / mx
    return seg

def visualize_imn_to_pdf(model, dataset, device, pdf_path: str,
                         sampling_rate: int, window: int, stride: int,
                         n_pos: int, n_neg: int,
                         pos_class_name: str = "MI",
                         random_pick: bool = False, seed: int = 123,
                         lead_names=None, lambda_l1: float = 1e-4,
                         viz_negative_class: bool = False,
                         lead_indices: list[int] | None = None,
                         heatmap_height: float = 1.0,
                         ecg_height: float = 0.65,
                         top_k_leads: int | None = None,
                         top_k_segments: int | None = None,
                         top_k_pos: int = 0,
                         top_k_neg: int = 0):
    """
    Visualizes IMN Feature Attributions for SINGLE LINEAR OUTPUT.
    Equation: Logit = sum(w * x) + b
    
    - If y=1 (MI): Positive contributions (Red) increase the logit.
    - If y=0 (NORM): Negative contributions (Blue) decrease the logit.
    """
    model.eval()
    if lead_names is None:
        lead_names = list(DEFAULT_LEAD_NAMES)
    base_lead_indices = lead_indices if lead_indices is not None else list(range(12))

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    
    # Select indices
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

    with PdfPages(pdf_path) as pdf:
        for tag, idx in selected:
            x, y = dataset[idx]               # x: [12, L]
            y_int = int(y.item())
            x_b = x.unsqueeze(0).to(device)   # [1, 12, L]
            
            with torch.no_grad():
                # Forward IMN (Single Linear Output)
                logits, gen_w, gen_b = model(x_b)
                prob = float(torch.sigmoid(logits)[0, 0].item())
                
                # gen_w is [1, 1, 12, L] -> Squeeze to [12, L]
                w_used = gen_w[0, 0, :, :].cpu().numpy()
                
            x_np = x.numpy()
            
            # --- Feature Attribution Strategy ---
            # Paper Eq: Impact = w * x
            impact = w_used * x_np
            
            # Logic: 
            # If visualizing MI (Pos): We care about what pushed the score UP (Positive Impact).
            # If visualizing NORM (Neg): We care about what pushed the score DOWN (Negative Impact).
            
            is_norm_sample = (tag == "NORM")
            
            # Aggregation for Heatmap (Magnitude)
            seg_hm = imn_weights_to_segments(impact, window=window, stride=stride)
            Tseg = seg_hm.shape[1]
            Lsig = x.shape[1]

            # Per-sample top-k: compute indices for marking; use full 12 leads when marking
            top_k_lead_indices = None
            if top_k_leads is not None:
                top_k_lead_indices = set(get_top_k_leads(impact, min(top_k_leads, 12)).tolist())
            
            # Determine if we are doing Top-K Segments logic
            use_pos_neg_split = (top_k_pos > 0 or top_k_neg > 0)
            use_full_12 = top_k_leads is not None or top_k_segments is not None or use_pos_neg_split
            lead_indices = list(range(12)) if use_full_12 else list(base_lead_indices)
            n_leads = len(lead_indices)

            # --- Calculate Segment Sets ---
            top_seg_per_lead = None # Old logic
            pos_seg_per_lead = None # New logic
            neg_seg_per_lead = None # New logic

            if use_pos_neg_split:
                pos_seg_per_lead, neg_seg_per_lead = get_top_pos_neg_segments_per_lead(impact, window, stride, top_k_pos, top_k_neg)
            elif top_k_segments is not None:
                # Fallback to old signed logic (highlights most positive/signed contribution)
                top_seg_per_lead = get_top_k_segments_per_lead(impact, window, stride, min(top_k_segments, Tseg))

            # Lead labels: add ★ for top-k leads when marking
            def _lead_label(i: int) -> str:
                lbl = lead_names[i]
                if top_k_lead_indices is not None and i in top_k_lead_indices:
                    lbl = f"{lbl}★"
                return lbl

            # Colors for fallback legacy
            shade_color = "blue" if is_norm_sample else "red"
            
            title_prob = f"P({pos_class_name})={prob:.3f}"
            topk_leads_str = f" | top-{top_k_leads} leads" if top_k_leads else ""
            if use_pos_neg_split:
                topk_seg_str = f" | top(+{top_k_pos}, -{top_k_neg}) segs"
            else:
                topk_seg_str = f" | top-{top_k_segments} segs" if top_k_segments else ""

            # Top-k mode: no heatmap, only ECG with marks. Otherwise: heatmap + ECG.
            show_heatmap = not use_full_12
            if show_heatmap:
                fig = plt.figure(figsize=(11.7, max(8, heatmap_height + n_leads * ecg_height)))
                gs = fig.add_gridspec(n_leads + 2, 1, height_ratios=[heatmap_height] + [ecg_height] * n_leads + [0.4], hspace=0.01)
                cmap = "Blues" if is_norm_sample else "Reds"
                seg_hm_plot = seg_hm[lead_indices]
                
                # If using split, aggregate indices for heatmap cropping
                if use_pos_neg_split:
                    all_top = set()
                    for s in pos_seg_per_lead: all_top |= s
                    for s in neg_seg_per_lead: all_top |= s
                    if all_top: seg_hm_plot = seg_hm_plot[:, sorted(all_top)]
                elif top_seg_per_lead is not None:
                    all_top = set()
                    for s in top_seg_per_lead: all_top |= s
                    if all_top: seg_hm_plot = seg_hm_plot[:, sorted(all_top)]

                ax0 = fig.add_subplot(gs[0, 0])
                im = ax0.imshow(seg_hm_plot, aspect="auto", vmin=0, vmax=1, cmap=cmap)
                ax0.set_yticks(range(n_leads))
                ax0.set_yticklabels([_lead_label(i) for i in lead_indices])
                seg_label = f"Top Segments" if (top_seg_per_lead or use_pos_neg_split) else f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)"
                ax0.set_xlabel(seg_label)
                ax0.set_title(f"IMN Intrinsic Explanation (Single Linear) | {tag} | True={y_int} | {title_prob} | idx={idx}{topk_leads_str}")
                fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)
                gs_offset = 1
            else:
                fig = plt.figure(figsize=(11.7, max(8, n_leads * ecg_height + 0.5)))
                gs = fig.add_gridspec(n_leads + 1, 1, height_ratios=[ecg_height] * n_leads + [0.4], hspace=0.01)
                gs_offset = 0

            # Signal traces with shading
            for k, lead in enumerate(lead_indices):
                ax = fig.add_subplot(gs[k + gs_offset, 0])
                is_top = top_k_lead_indices is not None and lead in top_k_lead_indices
                ax.plot(x_np[lead], linewidth=1.0 if is_top else 0.8, color='black', alpha=0.6)
                if is_top:
                    ax.set_facecolor((1, 0.95, 0.9) if not is_norm_sample else (0.95, 0.97, 1))
                ax.set_xlim(0, Lsig - 1)
                ax.set_ylabel(_lead_label(lead), rotation=0, labelpad=8, va="center", fontsize=8, fontweight="bold" if is_top else "normal")
                ax.set_yticklabels([])
                ax.margins(y=0.02)
                
                # --- Drawing Boxes ---
                # Helper to draw box
                def draw_box(t_idx, color):
                    alpha = 0.25
                    start = t_idx * stride
                    end = min(start + window, Lsig)
                    ax.axvspan(start, end, alpha=alpha, color=color, linewidth=0, zorder=0)
                    ylo, yhi = ax.get_ylim()
                    rect = Rectangle((start, ylo), end - start, yhi - ylo,
                                     fill=False, edgecolor=color, linewidth=1.5, zorder=2)
                    ax.add_patch(rect)

                if use_pos_neg_split:
                    # Draw Positive (Red)
                    if pos_seg_per_lead:
                        for t in pos_seg_per_lead[lead]:
                            draw_box(t, "red")
                    # Draw Negative (Blue)
                    if neg_seg_per_lead:
                        for t in neg_seg_per_lead[lead]:
                            draw_box(t, "blue")
                elif top_seg_per_lead:
                    # Old fallback
                    segs_for_lead = top_seg_per_lead[lead]
                    for t in segs_for_lead:
                        draw_box(t, shade_color)
                
                ax.set_xticks([])

            # Footer
            axf = fig.add_subplot(gs[n_leads + gs_offset, 0])
            axf.axis("off")
            footer = f"IMN Feature Attribution: $|w \cdot x|$. Single Linear Function. L1 Reg={lambda_l1}"
            if not show_heatmap:
                viz_note = ""
                if use_pos_neg_split:
                    viz_note = "Red=Positive Contribution (MI), Blue=Negative Contribution (NORM)."
                else:
                    viz_note = f"Boxed = top-k segments per lead ({shade_color})."
                footer = f"{tag} | True={y_int} | P({pos_class_name})={prob:.3f} | idx={idx}{topk_leads_str}{topk_seg_str}. ★ = top-k leads. {viz_note} {footer}"
            axf.text(0, 0.5, footer, fontsize=10)

            fig.tight_layout(pad=0.3)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)


# -----------------------
# ECG Plot Visualization (ecg_plot library)
# -----------------------
def visualize_ecg_with_ecg_plot(
    ecg: np.ndarray,
    sample_rate: int = 500,
    title: str = "ECG",
    save_path: str | None = None,
    lead_names: list | None = None,
    columns: int = 2,
    style: str | None = None,
) -> None:
    """
    Visualize ECG using the ecg_plot library (https://github.com/dy1901/ecg_plot).
    """
    if not ECG_PLOT_AVAILABLE:
        raise ImportError("ecg_plot is not installed. Run: pip install ecg_plot")

    if ecg.ndim == 3:
        ecg = ecg[0]  # Take first sample from batch
    assert ecg.ndim == 2 and ecg.shape[0] == 12, "ecg must be (12, L) or (N, 12, L)"

    plot_kwargs = dict(
        sample_rate=sample_rate,
        title=title,
        columns=columns,
    )
    if lead_names is not None:
        plot_kwargs["lead_index"] = lead_names
    if style is not None:
        plot_kwargs["style"] = style
    ecg_plot.plot(ecg, **plot_kwargs)

    if save_path:
        out_dir = os.path.dirname(save_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        path_for_ecg = out_dir + "/" if out_dir and not out_dir.endswith("/") else (out_dir or "./")
        ecg_plot.save_as_png(base_name, path_for_ecg)
    else:
        ecg_plot.show()


def visualize_ecg_batch_with_ecg_plot(
    ecg_batch: np.ndarray,
    sample_rate: int = 500,
    out_dir: str = "ecg_plots",
    n_samples: int = 5,
    title_prefix: str = "ECG",
    random_indices: bool = False,
    seed: int = 42,
) -> list[str]:
    """
    Visualize multiple ECG samples using ecg_plot and save to PNG files.
    """
    if not ECG_PLOT_AVAILABLE:
        raise ImportError("ecg_plot is not installed. Run: pip install ecg_plot")

    os.makedirs(out_dir, exist_ok=True)
    N = ecg_batch.shape[0]
    n_plot = min(n_samples, N)

    if random_indices:
        rng = np.random.default_rng(seed)
        indices = rng.choice(N, size=n_plot, replace=False)
    else:
        indices = np.arange(n_plot)

    saved_paths = []
    for i, idx in enumerate(indices):
        ecg = ecg_batch[idx]  # (12, L)
        title = f"{title_prefix} sample {idx}"
        fname = os.path.join(out_dir, f"ecg_sample_{idx:04d}.png")
        visualize_ecg_with_ecg_plot(
            ecg,
            sample_rate=sample_rate,
            title=title,
            save_path=fname,
        )
        saved_paths.append(fname)
    return saved_paths


def _draw_ecg_plot_style(ax, ecg: np.ndarray, sample_rate: int, lead_names: list,
                         columns: int = 2, row_height: float = 0.5,
                         style: str | None = None, half_signal: bool = False) -> None:
    """
    Draw ecg_plot-style 12-lead ECG in the given axes.
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
            sep_h = 0.12 * (row_height / 0.5)  # Scale separator with row_height
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
    draw_bbox: bool = False,
    per_lead: bool = False,
    row_height: float = 0.5,
) -> None:
    """
    Overlay semi-transparent patches on ECG axes to mark important regions from IMN heatmap.
    seg_hm: (n_leads, T) segment importance, 0-1.
    For 2-column ecg_plot layout. If per_lead=True, draws only in each lead's y band.
    """
    n_leads, Tseg = seg_hm.shape
    full_secs = (signal_len / sampling_rate) if signal_len else (
        max((Tseg - 1) * stride + window, window) / sampling_rate if Tseg > 0 else window / sampling_rate
    )
    secs = full_secs / 2 if half_signal else full_secs
    ylo, yhi = ax.get_ylim()
    rows = int(ceil(n_leads / columns))
    band_h = row_height / 2

    def _draw_rect(x_lo: float, y_lo: float, w: float, h: float, alpha: float) -> None:
        ax.add_patch(Rectangle((x_lo, y_lo), w, h, facecolor=shade_color, alpha=alpha, zorder=0, linewidth=0))
        if draw_bbox:
            ax.add_patch(Rectangle((x_lo, y_lo), w, h, fill=False, edgecolor=shade_color, linewidth=1.5, zorder=2))

    for t in range(Tseg):
        start_sec = (t * stride) / sampling_rate
        end_sec = (t * stride + window) / sampling_rate
        if half_signal and end_sec > secs:
            continue
        w = end_sec - start_sec

        if per_lead:
            for lead_idx in range(n_leads):
                imp = float(seg_hm[lead_idx, t])
                if imp < importance_threshold:
                    continue
                row = lead_idx % rows
                col = lead_idx // rows
                y_offset = -(row_height / 2) * row
                y_lead_lo = y_offset - band_h
                y_lead_hi = y_offset + band_h
                h_lead = y_lead_hi - y_lead_lo
                alpha = min(max_alpha, 0.25) # Fixed visibility if binary mask
                if col == 0:
                    _draw_rect(start_sec, y_lead_lo, w, h_lead, alpha)
                else:
                    _draw_rect(secs + start_sec, y_lead_lo, w, h_lead, alpha)
        else:
            imp = float(np.max(seg_hm[:, t]))
            if imp < importance_threshold:
                continue
            alpha = min(max_alpha, imp * 0.5)
            h = yhi - ylo
            _draw_rect(start_sec, ylo, w, h, alpha)
            if columns > 1 and end_sec <= secs:
                _draw_rect(secs + start_sec, ylo, w, h, alpha)


def visualize_ecg_with_imn_heatmap_to_pdf(
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
    lambda_l1: float = 1e-4,
    half_ecg: bool = True,
    row_height: float = 0.5,
    lead_indices: list[int] | None = None,
    heatmap_height: float = 0.5,
    ecg_height: float = 1.4,
    top_k_leads: int | None = None,
    top_k_segments: int | None = None,
    top_k_pos: int = 0,
    top_k_neg: int = 0,
) -> None:
    """
    Visualize ECG with IMN heatmap on top, saved to PDF.
    Combines ecg_plot-style 12-lead ECG with IMN feature attribution heatmap.
    """
    model.eval()
    if lead_names is None:
        lead_names = list(DEFAULT_LEAD_NAMES)
    base_lead_indices = lead_indices if lead_indices is not None else list(range(12))

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
    for idx in indices:
        x, y = dataset[idx]
        y_int = int(y.item())
        tag = pos_class_name if y_int == 1 else "NORM"
        x_b = x.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, gen_w, _ = model(x_b)
            prob = float(torch.sigmoid(logits)[0, 0].item())
            w_used = gen_w[0, 0, :, :].cpu().numpy()

        x_np = x.numpy()
        impact = w_used * x_np
        seg_hm = imn_weights_to_segments(impact, window=window, stride=stride)
        Lsig = x.shape[1]

        # Trim heatmap to first half when using half ECG
        if half_ecg:
            n_seg_half = max(1, (Lsig // 2 - window) // stride + 1)
            seg_hm = seg_hm[:, :n_seg_half]
            impact_for_seg = impact[:, : Lsig // 2]  # top-k from displayed region only
        else:
            impact_for_seg = impact
        Tseg = seg_hm.shape[1]

        # Per-sample top-k: compute for marking; use full 12 leads when marking
        top_k_lead_indices = None
        if top_k_leads is not None:
            top_k_lead_indices = set(get_top_k_leads(impact, min(top_k_leads, 12)).tolist())

        # Determine if we use pos/neg split logic
        use_pos_neg_split = (top_k_pos > 0 or top_k_neg > 0)
        use_full_12 = top_k_leads is not None or top_k_segments is not None or use_pos_neg_split
        lead_indices = list(range(12)) if use_full_12 else list(base_lead_indices)

        # Calculate segments
        pos_seg_per_lead = None
        neg_seg_per_lead = None
        top_seg_per_lead = None
        
        if use_pos_neg_split:
            pos_seg_per_lead, neg_seg_per_lead = get_top_pos_neg_segments_per_lead(impact_for_seg, window, stride, top_k_pos, top_k_neg)
        elif top_k_segments is not None:
            top_seg_per_lead = get_top_k_segments_per_lead(impact_for_seg, window, stride, min(top_k_segments, Tseg))

        # Lead labels: add ★ for top-k leads
        def _lead_label(i: int) -> str:
            lbl = lead_names[i]
            if top_k_lead_indices is not None and i in top_k_lead_indices:
                lbl = f"{lbl}★"
            return lbl

        x_np_sel = x_np[lead_indices]
        lead_names_sel = [_lead_label(i) for i in lead_indices]

        # For heatmap: full 12 or subset; optionally top-k segments
        seg_hm_plot = seg_hm[lead_indices]
        if use_pos_neg_split:
             all_top = set()
             for s in pos_seg_per_lead: all_top |= s
             for s in neg_seg_per_lead: all_top |= s
             if all_top: seg_hm_plot = seg_hm_plot[:, sorted(all_top)]
        elif top_seg_per_lead is not None:
            all_top = set()
            for s in top_seg_per_lead:
                all_top |= s
            if all_top:
                seg_hm_plot = seg_hm_plot[:, sorted(all_top)]
        
        # Prepare shading masks
        # 1. Pos/Neg Split Logic
        seg_hm_pos = np.zeros_like(seg_hm[lead_indices])
        seg_hm_neg = np.zeros_like(seg_hm[lead_indices])
        # 2. Legacy fallback
        seg_hm_shade = seg_hm[lead_indices].copy()

        if use_pos_neg_split:
            for i, lead_idx in enumerate(lead_indices):
                if pos_seg_per_lead:
                    for t in pos_seg_per_lead[lead_idx]:
                        seg_hm_pos[i, t] = 1.0
                if neg_seg_per_lead:
                    for t in neg_seg_per_lead[lead_idx]:
                        seg_hm_neg[i, t] = 1.0
        elif top_seg_per_lead is not None:
            for i, lead_idx in enumerate(lead_indices):
                segs = top_seg_per_lead[lead_idx]
                for t in range(Tseg):
                    seg_hm_shade[i, t] = 1.0 if t in segs else 0

        is_norm_sample = tag == "NORM"
        shade_color = "blue" if is_norm_sample else "red"

        # Top-k mode: no heatmap, only ECG with marks. Otherwise: heatmap + ECG.
        show_heatmap = not use_full_12
        if show_heatmap:
            fig = plt.figure(figsize=(7, 5.2))
            gs = fig.add_gridspec(2, 1, height_ratios=[heatmap_height, ecg_height], hspace=0.1)
            cmap = "Blues" if is_norm_sample else "Reds"
            ax_hm = fig.add_subplot(gs[0, 0])
            im = ax_hm.imshow(seg_hm_plot, aspect="auto", vmin=0, vmax=1, cmap=cmap)
            ax_hm.set_yticks(range(len(lead_indices)))
            ax_hm.set_yticklabels(lead_names_sel, fontsize=7)
            seg_label = f"Top Segments" if (top_seg_per_lead or use_pos_neg_split) else f"Segments (w={window}, s={stride}, fs={sampling_rate}Hz)"
            ax_hm.set_xlabel(seg_label, fontsize=8)
            topk_str = (f" | top-{top_k_leads} leads" if top_k_leads else "")
            if use_pos_neg_split:
                topk_str += f" | top(+{top_k_pos}, -{top_k_neg}) segs"
            else:
                topk_str += (f" | top-{top_k_segments} segs" if top_k_segments else "")
            ax_hm.set_title(f"IMN Heatmap | {tag} | P({pos_class_name})={prob:.3f} | idx={idx}{topk_str}", fontsize=9)
            fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.02, shrink=0.8)
            ax_ecg = fig.add_subplot(gs[1, 0])
        else:
            fig = plt.figure(figsize=(7, 4.5))
            gs = fig.add_gridspec(1, 1)
            ax_ecg = fig.add_subplot(gs[0, 0])

        # ECG (ecg_plot style)
        _draw_ecg_plot_style(
            ax_ecg, x_np_sel, sampling_rate, lead_names_sel,
            columns=2, row_height=row_height, half_signal=half_ecg,
        )
        
        # Apply Shading
        if use_pos_neg_split:
            # Draw Positive (Red)
            _draw_important_patches_on_ecg(
                ax_ecg, seg_hm_pos, window, stride, sampling_rate,
                shade_color="red", signal_len=Lsig, half_signal=half_ecg,
                draw_bbox=use_full_12, per_lead=use_full_12, row_height=row_height,
                importance_threshold=0.5 # Binary mask 1.0
            )
            # Draw Negative (Blue)
            _draw_important_patches_on_ecg(
                ax_ecg, seg_hm_neg, window, stride, sampling_rate,
                shade_color="blue", signal_len=Lsig, half_signal=half_ecg,
                draw_bbox=use_full_12, per_lead=use_full_12, row_height=row_height,
                importance_threshold=0.5 # Binary mask 1.0
            )
        else:
            # Legacy fallback
            _draw_important_patches_on_ecg(
                ax_ecg, seg_hm_shade, window, stride, sampling_rate,
                shade_color=shade_color, signal_len=Lsig, half_signal=half_ecg,
                draw_bbox=use_full_12, per_lead=use_full_12, row_height=row_height,
            )

        if use_full_12:
            expl = "Red=Pos(MI), Blue=Neg(NORM)" if use_pos_neg_split else "boxed=top-k segs"
            ax_ecg.set_title(f"{tag} | P({pos_class_name})={prob:.3f} | idx={idx} | ★=top-k leads, {expl}", fontsize=8)
        else:
            n_leads_str = f"{len(lead_indices)}-lead" if len(lead_indices) != 12 else "12-lead"
            shade_str = "top-k segments per lead" if (top_seg_per_lead or use_pos_neg_split) else "IMN important regions"
            ax_ecg.set_title(f"{n_leads_str} ECG (shaded = {shade_str})", fontsize=8)

        fig.tight_layout(pad=0.5)
        sample_path = os.path.join(out_dir, f"{base_name}_{tag}_{idx:04d}.pdf")
        fig.savefig(sample_path, bbox_inches="tight", pad_inches=0.08)
        saved_paths.append(sample_path)
        plt.close(fig)

    print(f"Saved {len(saved_paths)} ECG+IMN heatmap PDFs to {out_dir}")


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path/to/ptbxl/")
    parser.add_argument("--sampling_rate", type=int, default=500, choices=[100, 500])
    parser.add_argument("--task", type=str, default="norm_vs_mi",
                        choices=["norm_vs_mi", "norm_vs_sttc", "norm_vs_cd", "norm_vs_hyp"])

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # IMN Specific
    parser.add_argument("--lambda_l1", type=float, default=1e-4, 
                        help="Regularization strength for sparsity in generated weights.")

    # Inference-only (skip training, load checkpoint)
    parser.add_argument("--inference_only", action="store_true",
                        help="Skip training; load checkpoint and run inference + viz only.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to checkpoint. Required when --inference_only.")

    # Viz
    parser.add_argument("--window", type=int, nargs="*", default=None,
                        help="Window size(s) for segment aggregation.")
    parser.add_argument("--stride", type=int, nargs="*", default=None,
                        help="Stride(s) for segment aggregation.")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    
    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--scheduler_params", type=str, default="{}")

    # Early Stopping
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--early_stop_monitor", type=str, default="val_auc")
    parser.add_argument("--early_stop_mode", type=str, default="max")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="mesomorphic_ecg")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_offline", action="store_true")
    
    # Out
    parser.add_argument("--out_dir", type=str, default="runs/imn_ecg")
    parser.add_argument("--viz_pdf", type=str, default="imn_explanation.pdf")
    parser.add_argument("--n_pos_viz", type=int, default=25)
    parser.add_argument("--n_neg_viz", type=int, default=25)
    parser.add_argument("--viz_random", action="store_true")
    parser.add_argument("--viz_negative_class", action="store_true", help="Deprecated in single-linear mode, but kept for arg compatibility.")
    parser.add_argument("--viz_ecg_plot", action="store_true",
                        help="Visualize ECG samples using ecg_plot library (pip install ecg_plot).")
    parser.add_argument("--viz_ecg_plot_n", type=int, default=5,
                        help="Number of ECG samples to plot with ecg_plot when --viz_ecg_plot.")
    # Reserved CLI flags for ECG+IMN plots (half-length ECG and one sample per file are defaults).
    parser.add_argument(
        "--half_ecg",
        action="store_true",
        help="Reserved flag for ECG+IMN plots; currently half-length ECG is used by default.",
    )
    parser.add_argument(
        "--one_per_file",
        action="store_true",
        help="Reserved flag; ECG+IMN plots are saved one sample per file by default.",
    )
    parser.add_argument(
        "--leads",
        type=str,
        default=None,
        help="Leads to visualize: comma-separated indices (0-11) or names (I,II,III,aVR,aVL,aVF,V1-V6). "
             "E.g. '0,1,2' or 'I,II,III' or 'V1,V2,V3,V4,V5,V6' or '0-5'. Default: all 12 leads.",
    )
    parser.add_argument(
        "--viz_heatmap_height",
        type=float,
        default=None,
        help="Height ratio for the top heatmap panel. Lower = shorter heatmap. "
             "Default: 1.0 for IMN viz, 0.5 for ECG+heatmap viz.",
    )
    parser.add_argument(
        "--viz_ecg_height",
        type=float,
        default=None,
        help="Height ratio for the bottom ECG/lead traces panel. Lower = shorter. "
             "Default: 0.65 per lead for IMN viz, 1.4 for ECG+heatmap viz.",
    )
    parser.add_argument(
        "--top_k_leads",
        type=int,
        default=None,
        help="Visualize only the top-k most important leads per sample (based on mean |impact|).",
    )
    parser.add_argument(
        "--top_k_segments",
        type=int,
        default=None,
        help="Visualize only the top-k most important segments per sample (based on max importance over leads).",
    )
    # NEW ARGUMENTS FOR POSITIVE/NEGATIVE SEGMENTS
    parser.add_argument(
        "--top_k_pos",
        type=int,
        default=0,
        help="Visualize the top-k segments per lead that contribute POSITIVELY (towards MI) to the linear output. "
             "These will be marked in RED. Overrides --top_k_segments if > 0."
    )
    parser.add_argument(
        "--top_k_neg",
        type=int,
        default=0,
        help="Visualize the top-k segments per lead that contribute NEGATIVELY (towards NORM) to the linear output. "
             "These will be marked in BLUE. Overrides --top_k_segments if > 0."
    )

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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.inference_only:
        run_dir = os.path.join(os.path.dirname(args.ckpt), f"inference_{timestamp}")
    else:
        run_dir = os.path.join(args.out_dir, args.task, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    # Load Data
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
    keep = (is_pos ^ is_norm)
    Yf = Y[keep].copy()
    y_bin = np.array([1 if pos_class in l else 0 for l in Yf["diagnostic_superclass"].values], dtype=np.int64)

    print(f"Loading signals at {args.sampling_rate}Hz ...")
    X = load_raw_data(Yf, args.sampling_rate, path)
    X = np.transpose(X, (0, 2, 1))  # [N, 12, L]
    N, C, Lsig = X.shape
    print(f"Data Shape: {X.shape}")

    fold = Yf["strat_fold"].values
    train_mask = (fold >= 1) & (fold <= 8)
    val_mask = (fold >= 9)

    X_train, y_train = X[train_mask], y_bin[train_mask]
    X_val, y_val     = X[val_mask], y_bin[val_mask]

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t   = torch.from_numpy(X_val).float()
    y_val_t   = torch.from_numpy(y_val).float()

    # Weighting for BCE (pos_weight)
    n_pos = float((y_train_t == 1).sum())
    n_neg = float((y_train_t == 0).sum())
    pos_weight = n_neg / max(n_pos, 1.0)
    print(f"Positive Class Weight: {pos_weight:.4f}")

    train_ds = PTBXLBinaryDataset(X_train_t, y_train_t)
    val_ds   = PTBXLBinaryDataset(X_val_t, y_val_t)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=(device=="cuda"))

    if args.inference_only:
        print("Inference-only mode. Loading checkpoint:", args.ckpt)
        lit = IMNLightning.load_from_checkpoint(args.ckpt, map_location=device)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, pin_memory=(device=="cuda"))

        lit = IMNLightning(
            input_channels=C,
            signal_len=Lsig,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_l1=args.lambda_l1,
            pos_weight=pos_weight, # Scalar weight
            scheduler_type=args.scheduler if args.scheduler != "none" else None,
            scheduler_params=scheduler_params,
        )

        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name or timestamp,
            save_dir=run_dir,
            offline=args.wandb_offline
        )

        ckpt_cb = ModelCheckpoint(
            dirpath=run_dir,
            filename="best-imn-{epoch:02d}-{val_auc:.4f}",
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
            print("Loading best:", best_path)
            lit = IMNLightning.load_from_checkpoint(best_path, map_location=device)

        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices)
        trainer.test(lit, val_loader)

    lit = lit.to(device)
    lit.model.to(device)
    lit.eval()
    
    # Save Metrics
    model_device = next(lit.model.parameters()).device
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(model_device)
            logits, _, _ = lit.model(x)
            prob = torch.sigmoid(logits.squeeze(1))
            pred = (prob > 0.5).long()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    print("Metrics:", metrics)

    # Visualization
    default_window = 50 if args.sampling_rate == 100 else 250
    windows = args.window if args.window else [default_window]
    strides = args.stride if args.stride else [w // 2 for w in windows]
    if len(strides) < len(windows):
        strides = strides + [strides[-1] if strides else default_window // 2] * (len(windows) - len(strides))
    elif len(strides) > len(windows):
        strides = strides[:len(windows)]
    viz_pairs = list(zip(windows, strides))

    for viz_window, viz_stride in viz_pairs:
        base_name = os.path.splitext(args.viz_pdf)[0]
        ext = os.path.splitext(args.viz_pdf)[1] or ".pdf"
        pdf_name = f"{base_name}_w{viz_window}_s{viz_stride}{ext}" if len(viz_pairs) > 1 else args.viz_pdf
        pdf_path = pdf_name if os.path.isabs(pdf_name) else os.path.join(run_dir, pdf_name)

        print(f"Generating IMN explanations (window={viz_window}, stride={viz_stride}) to {pdf_path}...")
        heatmap_h = args.viz_heatmap_height if args.viz_heatmap_height is not None else 1.0
        ecg_h = args.viz_ecg_height if args.viz_ecg_height is not None else 0.65
        visualize_imn_to_pdf(
            model=lit.model,
            dataset=val_ds,
            device=model_device,
            pdf_path=pdf_path,
            sampling_rate=args.sampling_rate,
            window=viz_window,
            stride=viz_stride,
            n_pos=args.n_pos_viz,
            n_neg=args.n_neg_viz,
            pos_class_name=pos_class,
            random_pick=args.viz_random,
            seed=123,
            lambda_l1=args.lambda_l1,
            viz_negative_class=args.viz_negative_class,
            lead_indices=lead_indices,
            heatmap_height=heatmap_h,
            ecg_height=ecg_h,
            top_k_leads=args.top_k_leads,
            top_k_segments=args.top_k_segments,
            top_k_pos=args.top_k_pos,
            top_k_neg=args.top_k_neg,
        )

    # ECG plot visualization with IMN heatmap (PDF format)
    if args.viz_ecg_plot:
        viz_window, viz_stride = viz_pairs[0] if viz_pairs else (default_window, default_window // 2)
        ecg_pdf_path = os.path.join(run_dir, "ecg_with_imn_heatmap.pdf")
        print(f"Generating ECG+IMN heatmap PDF to {ecg_pdf_path}...")
        heatmap_h_ecg = args.viz_heatmap_height if args.viz_heatmap_height is not None else 0.5
        ecg_h_ecg = args.viz_ecg_height if args.viz_ecg_height is not None else 1.4
        visualize_ecg_with_imn_heatmap_to_pdf(
            model=lit.model,
            dataset=val_ds,
            device=model_device,
            pdf_path=ecg_pdf_path,
            sampling_rate=args.sampling_rate,
            window=viz_window,
            stride=viz_stride,
            n_samples=args.viz_ecg_plot_n,
            pos_class_name=pos_class,
            random_pick=args.viz_random,
            seed=args.seed,
            lambda_l1=args.lambda_l1,
            lead_indices=lead_indices,
            heatmap_height=heatmap_h_ecg,
            ecg_height=ecg_h_ecg,
            top_k_leads=args.top_k_leads,
            top_k_segments=args.top_k_segments,
            top_k_pos=args.top_k_pos,
            top_k_neg=args.top_k_neg,
        )

    print("Done.")

if __name__ == "__main__":
    main()