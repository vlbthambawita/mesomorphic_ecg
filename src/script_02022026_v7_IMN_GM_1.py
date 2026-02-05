"""
PTB-XL MI vs NORM - Interpretable Mesomorphic Neural Network (IMN) Implementation.

Based on: "Interpretable Mesomorphic Neural Networks" (NeurIPS 2024)
Converted from black-box CNN + Grad-CAM baseline.

- Architecture: Deep Hypernetwork (CNN) -> Generates Linear Weights for the input instance.
- Prediction: Logits = dot(Input, Generated_Weights) + Generated_Bias
- Loss: CrossEntropy + Lambda * L1_Norm(Generated_Weights)
- XAI: Intrinsic. We visualize the generated weights * input (Feature Attribution).

Run:
python imn_ptbxl_2d.py --path /path/to/ptbxl --sampling_rate 500 --epochs 20 --lambda_l1 1e-4 --viz_random
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
import matplotlib.colors as mcolors

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
class PTBXLBinaryDatasetCE(Dataset):
    """
    X: [N, 12, L]
    y: [N] int64 {0,1}   0=NORM, 1=MI
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, per_lead_zscore: bool = True):
        super().__init__()
        assert X.ndim == 3 and X.shape[1] == 12
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
# IMN Architecture
# -----------------------
class ECG_IMN(nn.Module):
    """
    Interpretable Mesomorphic Network for ECG.
    
    Hypernetwork: A 2D CNN that processes the signal.
    Output: Instead of logits, it outputs Weights W and Bias b.
    Final Prediction: y = x^T * W + b
    
    This enforces that the decision boundary is locally linear (per instance),
    making W directly interpretable as feature importance.
    """
    def __init__(self, input_channels=12, signal_len=1000, num_classes=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_channels * signal_len
        self.num_classes = num_classes
        self.C = input_channels
        self.L = signal_len

        # --- Hypernetwork Backbone (CNN feature extractor) ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.dropout = nn.Dropout(dropout)

        # --- Hypernetwork Head (Generates W and b) ---
        # Input: 64 (from global average pooling of CNN backbone)
        # Output: num_classes * (input_dim + 1) -> Weights + Bias for every feature per class
        # Note: This layer can be very large. 
        # For 500Hz (L=2500): 12*2500 = 30k inputs. Output ~ 60k parameters.
        self.hyper_head = nn.Linear(64, num_classes * (self.input_dim + 1))

    def forward(self, x):
        """
        x: [B, 12, L]
        """
        B = x.shape[0]
        
        # 1. Extract Features via Hypernetwork
        # Input to CNN needs [B, 1, 12, L]
        feat = x.unsqueeze(1)
        feat = self.conv1(feat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)       # [B, 64, 12, L/4]
        
        # Global Average Pooling
        feat = feat.mean(dim=(2, 3))  # [B, 64]
        feat = self.dropout(feat)

        # 2. Generate Linear Model Parameters (Weights & Biases)
        # params: [B, num_classes * (M + 1)]
        params = self.hyper_head(feat)
        
        # Reshape to [B, num_classes, M + 1]
        params = params.view(B, self.num_classes, self.input_dim + 1)
        
        # Split into Weights W [B, C, M] and Bias b [B, C, 1]
        generated_w = params[:, :, :-1]
        generated_b = params[:, :, -1]

        # 3. Apply Local Linear Model
        # Flatten input x: [B, 12, L] -> [B, M]
        x_flat = x.reshape(B, -1)
        
        # Logits = (W * x) + b
        # Perform batched dot product
        # x_flat.unsqueeze(1): [B, 1, M]
        # generated_w: [B, C, M]
        # product: [B, C, M] -> sum over M -> [B, C]
        logits = (generated_w * x_flat.unsqueeze(1)).sum(dim=2) + generated_b

        return logits, generated_w, generated_b


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
        lambda_l1: float = 1e-4,  # Coefficient for interpretability regularization
        class_weights: list[float] | None = None,
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
        self._class_weights = class_weights
        
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
        return torch.tensor(self._class_weights, dtype=torch.float32, device=self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, gen_w, gen_b = self.model(x)
        
        # 1. Prediction Loss (Cross Entropy)
        ce_loss = F.cross_entropy(logits, y, weight=self._ce_weight())
        
        # 2. Interpretability Loss (L1 norm on generated weights)
        # Encourages sparsity in the explanation
        l1_loss = gen_w.abs().mean()
        
        total_loss = ce_loss + (self.lambda_l1 * l1_loss)

        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ce", ce_loss, on_step=False, on_epoch=True)
        self.log("train_l1", l1_loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, gen_w, _ = self.model(x)
        
        ce_loss = F.cross_entropy(logits, y, weight=self._ce_weight())
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()

        self.log("val_loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True)
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
        logits, _, _ = self.model(x)
        ce_loss = F.cross_entropy(logits, y, weight=self._ce_weight())
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        
        self.log("test_loss", ce_loss, on_step=False, on_epoch=True)
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
def imn_weights_to_segments(impact_12L: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    Aggregates point-wise feature attribution (Impact) into segments for cleaner visualization.
    impact_12L: [12, L]
    """
    assert impact_12L.ndim == 2
    L = impact_12L.shape[1]
    T = (L - window) // stride + 1
    seg = np.zeros((12, T), dtype=np.float32)
    
    for t in range(T):
        s = t * stride
        e = min(s + window, L)
        # Mean absolute impact in this segment
        seg[:, t] = np.abs(impact_12L[:, s:e]).mean(axis=1)
        
    # Normalize per record [0, 1] for visualization
    mx = seg.max() + 1e-9
    seg = seg / mx
    return seg

def visualize_imn_to_pdf(model, dataset, device, pdf_path: str,
                         sampling_rate: int, window: int, stride: int,
                         n_pos: int, n_neg: int,
                         pos_class_name: str = "MI",
                         random_pick: bool = False, seed: int = 123,
                         lead_names=None, lambda_l1: float = 1e-4):
    """
    Visualizes IMN Feature Attributions.
    Calculation: Impact = w(x) * x
    """
    model.eval()
    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        
    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)
    
    # 1. Select indices
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
            x_b = x.unsqueeze(0).to(device)   # [1, 12, L]
            
            with torch.no_grad():
                # Forward IMN
                logits, gen_w, gen_b = model(x_b)
                prob = float(torch.softmax(logits, dim=1)[0, 1].item())
                
                # gen_w shape: [1, 2, M], where M = 12 * L
                # We want the weights for Class 1 (Positive/MI)
                # Reshape back to [12, L]
                w_pos = gen_w[0, 1, :].view(12, -1).cpu().numpy()
                
            x_np = x.numpy()
            
            # --- Feature Attribution Strategy ---
            # Paper Eq (2): Impact = w * x
            # This shows the contribution of the feature to the specific prediction.
            impact = w_pos * x_np
            
            # Create segment heatmap for visualization overview
            seg_hm = imn_weights_to_segments(impact, window=window, stride=stride)
            Tseg = seg_hm.shape[1]
            Lsig = x.shape[1]

            # Plotting
            fig = plt.figure(figsize=(11.7, 16.5))
            gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1]*12 + [0.5])

            # Heatmap Top
            ax0 = fig.add_subplot(gs[0, 0])
            im = ax0.imshow(seg_hm, aspect="auto", vmin=0, vmax=1, cmap="Reds")
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            ax0.set_title(f"IMN Intrinsic Explanation | {tag} | True={int(y.item())} | P({pos_class_name})={prob:.3f} | idx={idx}")
            fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

            # Signal traces with shading
            for lead in range(12):
                ax = fig.add_subplot(gs[lead + 1, 0])
                ax.plot(x_np[lead], linewidth=0.8, color='black', alpha=0.6)
                ax.set_xlim(0, Lsig - 1)
                ax.set_ylabel(lead_names[lead], rotation=0, labelpad=20, va="center")
                
                # Shade based on aggregated segment importance
                contrib = seg_hm[lead]
                for t in range(Tseg):
                    a = float(contrib[t])
                    # Thresholding to avoid clutter
                    alpha = min(0.5, a * 0.6)
                    if alpha > 0.05:
                        start = t * stride
                        end = min(start + window, Lsig)
                        ax.axvspan(start, end, alpha=alpha, color="red", linewidth=0)
                
                ax.set_xticks([])

            # Footer
            axf = fig.add_subplot(gs[13, 0])
            axf.axis("off")
            axf.text(0, 0.5, 
                     f"IMN Feature Attribution: $|w(x) \cdot x|$ aggregated by segment. L1 Reg={lambda_l1}", 
                     fontsize=10)

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
    parser.add_argument("--task", type=str, default="norm_vs_mi",
                        choices=["norm_vs_mi", "norm_vs_sttc", "norm_vs_cd", "norm_vs_hyp"])

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
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

    # Viz (window/stride can be repeated for multiple configs, e.g. --window 100 200 500 --stride 50 100 250)
    parser.add_argument("--window", type=int, nargs="*", default=None,
                        help="Window size(s) for segment aggregation. Default: 250 for 500Hz, 50 for 100Hz.")
    parser.add_argument("--stride", type=int, nargs="*", default=None,
                        help="Stride(s) for segment aggregation. Default: window//2 per window.")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    
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

    args = parser.parse_args()

    if args.inference_only and not args.ckpt:
        parser.error("--ckpt is required when --inference_only")

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
    
    # Save config
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
    
    # Filter classes
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

    # Splits
    fold = Yf["strat_fold"].values
    train_mask = (fold >= 1) & (fold <= 8)
    val_mask = (fold >= 9)

    X_train, y_train = X[train_mask], y_bin[train_mask]
    X_val, y_val     = X[val_mask], y_bin[val_mask]

    # Convert to Tensor
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t   = torch.from_numpy(X_val).float()
    y_val_t   = torch.from_numpy(y_val).long()

    # Weighting
    n_pos = float((y_train_t == 1).sum())
    n_neg = float((y_train_t == 0).sum())
    w0 = (n_pos + n_neg) / max(n_neg, 1.0)
    w1 = (n_pos + n_neg) / max(n_pos, 1.0)
    class_weights = [float(w0), float(w1)]

    # Datasets
    train_ds = PTBXLBinaryDatasetCE(X_train_t, y_train_t)
    val_ds   = PTBXLBinaryDatasetCE(X_val_t, y_val_t)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=(device=="cuda"))

    if args.inference_only:
        # Load checkpoint and skip training
        print("Inference-only mode. Loading checkpoint:", args.ckpt)
        lit = IMNLightning.load_from_checkpoint(args.ckpt, map_location=device)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, pin_memory=(device=="cuda"))

        # Lightning IMN
        lit = IMNLightning(
            input_channels=C,
            signal_len=Lsig,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lambda_l1=args.lambda_l1,
            class_weights=class_weights,
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

        # Test Best
        best_path = ckpt_cb.best_model_path
        if best_path and os.path.exists(best_path):
            print("Loading best:", best_path)
            lit = IMNLightning.load_from_checkpoint(best_path, map_location=device)

        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices)
        trainer.test(lit, val_loader)

    lit = lit.to(device)
    lit.model.to(device)  # ensure inner model is on device
    
    # Save validation metrics (use model's actual device; trainer.test may have moved it)
    lit.eval()
    model_device = next(lit.model.parameters()).device
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(model_device)
            logits, _, _ = lit.model(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(dim=1)
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

    # Build (window, stride) pairs for visualization
    default_window = 50 if args.sampling_rate == 100 else 250
    windows = args.window if args.window else [default_window]
    strides = args.stride if args.stride else [w // 2 for w in windows]
    # Match lengths: if one list is shorter, extend with its last value
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
            lambda_l1=args.lambda_l1
        )
    print("Done.")

if __name__ == "__main__":
    main()