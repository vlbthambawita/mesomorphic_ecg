"""
Baseline (no IMN) MI vs NORM on PTB-XL + Grad-CAM explanations + PDF report.

Key points:
- Model: simple 2D CNN classifier (treat ECG as 1x12xL "image")
- XAI: Grad-CAM on the last Conv2d feature map
- PDF: selects N positives (MI) and M negatives (NORM) from TEST set,
       shows (1) Grad-CAM heatmap (lead x segments) and (2) 12 lead plots with red shading.
- Splits: train folds 1-8, val fold 9, test fold 10
- Sampling rate: 100 or 500
- Window/stride define visualization segments (and Grad-CAM is averaged into same segments)

Run:
python baseline_ptbxl_gradcam.py --path /path/to/ptbxl --sampling_rate 500 --epochs 20 --viz_random
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
# Baseline model (no IMN): 2D CNN on [B,1,12,L]
# -----------------------
class ECGConv2DBaseline(nn.Module):
    """
    Input:  x [B,12,L]
    Output: logit [B]
    """
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7)),
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 32, kernel_size=(3, 15), padding=(1, 7)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # downsample time only

            nn.Conv2d(32, 64, kernel_size=(3, 15), padding=(1, 7)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # downsample time only
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)             # [B,1,12,L]
        feats = self.features(x)       # [B,64,12,L']
        pooled = feats.mean(dim=(2, 3))  # [B,64]
        pooled = self.dropout(pooled)
        logit = self.classifier(pooled).squeeze(-1)  # [B]
        return logit


# -----------------------
# Grad-CAM for Conv2D model: returns heatmap over [B,12,L]
# -----------------------
class GradCAM2D:
    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out  # [B,C,H,W]
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]  # [B,C,H,W]
        self.target_module.register_forward_hook(fwd_hook)
        self.target_module.register_full_backward_hook(bwd_hook)

    @staticmethod
    def _normalize_01(cam: torch.Tensor) -> torch.Tensor:
        # cam: [B,H,W] or [B,12,L] (non-negative)
        cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
        cam = cam / (cam.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return cam

    def __call__(self, x: torch.Tensor):
        """
        x: [B,12,L]
        returns:
          prob: [B]
          cam_up: [B,12,L] in [0,1]
        """
        # Ensure input requires gradients
        if not x.requires_grad:
            x = x.requires_grad_(True)
        
        self.model.zero_grad(set_to_none=True)
        
        # Forward pass with gradients enabled (needed even in eval mode)
        with torch.enable_grad():
            logit = self.model(x)                # [B]
        
        prob = torch.sigmoid(logit).detach()

        # Backprop through summed logit (binary)
        score = logit.sum()
        score.backward(retain_graph=False)

        A = self.activations                 # [B,C,12,L']
        dA = self.gradients                  # [B,C,12,L']

        # Channel weights
        w = dA.mean(dim=(2, 3), keepdim=True)   # [B,C,1,1]
        cam = (w * A).sum(dim=1)                # [B,12,L']
        cam = F.relu(cam)

        # Upsample to original [12,L]
        cam_up = F.interpolate(
            cam.unsqueeze(1),                   # [B,1,12,L']
            size=(x.shape[1], x.shape[2]),      # (12, L)
            mode="bilinear",
            align_corners=False
        ).squeeze(1)                            # [B,12,L]
        cam_up = self._normalize_01(cam_up)
        return prob, cam_up


# -----------------------
# Helper: convert Grad-CAM [12,L] to segment heatmap [12,T]
# -----------------------
def cam_12L_to_segments(cam_12L: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    cam_12L: [12,L] in [0,1]
    returns seg_heatmap: [12,T] in [0,1]
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
    # Normalize per-record for display robustness
    m = np.max(seg) + 1e-6
    seg = seg / m
    return seg


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
# Train/Eval
# -----------------------
def train_one_epoch(model, loader, opt, device, pos_weight: float):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    pos_w = torch.tensor([pos_weight], device=device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = (torch.sigmoid(logit) > 0.5).float()
            acc = (pred == y).float().mean().item()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ys, ps = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y)

        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).float()
        acc = (pred == y).float().mean().item()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        n += bs

        ys.append(y)
        ps.append(prob)

    ys = torch.cat(ys, dim=0)
    ps = torch.cat(ps, dim=0)
    auc = simple_auc_roc(ys, ps)
    return total_loss / n, total_acc / n, auc


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
# Visualization to PDF (per-lead) using Grad-CAM segments
# -----------------------
def visualize_pos_neg_to_pdf_gradcam(model, cam_explainer: GradCAM2D, dataset, device, pdf_path: str,
                                     sampling_rate: int, window: int, stride: int,
                                     n_pos: int, n_neg: int,
                                     random_pick: bool = False, seed: int = 123,
                                     lead_names=None):
    """
    Saves a multi-page PDF:
      - First N positives (MI=1) and M negatives (NORM=0), or random if random_pick=True.
      - Each page: segment heatmap (Grad-CAM averaged into segments) + 12 lead plots shaded by importance.
    """
    model.eval()
    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    pos_idx, neg_idx = [], []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if int(float(y)) == 1:
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
        print(f"WARNING: requested n_pos={n_pos}, but only found {len(sel_pos)} positives in dataset.")
    if len(sel_neg) < n_neg:
        print(f"WARNING: requested n_neg={n_neg}, but only found {len(sel_neg)} negatives in dataset.")

    selected = [("MI", i) for i in sel_pos] + [("NORM", i) for i in sel_neg]

    with PdfPages(pdf_path) as pdf:
        for tag, idx in selected:
            x, y = dataset[idx]
            x_b = x.unsqueeze(0).to(device).requires_grad_(True)  # [1,12,L]

            prob, cam_12L = cam_explainer(x_b)  # cam_12L: [1,12,L] in [0,1]
            prob = float(prob.item())
            # Detach and convert to numpy (no gradients needed after CAM computation)
            with torch.no_grad():
                cam_np = cam_12L.squeeze(0).detach().cpu().numpy()  # [12,L]
                x_np = x.detach().cpu().numpy()                     # [12,L]

            seg_hm = cam_12L_to_segments(cam_np, window=window, stride=stride)  # [12,Tseg]
            Tseg = seg_hm.shape[1]
            Lsig = x_np.shape[1]

            fig = plt.figure(figsize=(11.7, 16.5))
            gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1]*12 + [0.5])

            # Heatmap: lead x segments (0..1)
            ax0 = fig.add_subplot(gs[0, 0])
            im = ax0.imshow(seg_hm, aspect="auto", vmin=0, vmax=1, cmap="Reds")
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
            ax0.set_title(f"{tag} sample | true={int(y.item())} | P(MI)={prob:.3f} | idx={idx}")
            fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

            # Per-lead plots with shading by segment importance
            for lead in range(12):
                ax = fig.add_subplot(gs[lead + 1, 0])
                ax.plot(x_np[lead], linewidth=0.8)
                ax.set_xlim(0, Lsig - 1)
                ax.set_ylabel(lead_names[lead], rotation=0, labelpad=20, va="center")

                contrib = seg_hm[lead]  # [Tseg], 0..1
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
                "Grad-CAM importance (0..1). Darker red = higher importance. (Post-hoc XAI baseline; not signed.)",
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

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Visualization segmentation
    parser.add_argument("--window", type=int, default=None,
                        help="Window in samples for segment visualization. Default: 0.5s => 50@100Hz or 250@500Hz.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride in samples. Default: window//2 (50% overlap).")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    # Visualization
    parser.add_argument("--viz_pdf", type=str, default="viz_mi_vs_norm_test_gradcam.pdf", help="Output PDF filename")
    parser.add_argument("--n_pos_viz", type=int, default=25, help="Number of MI (positive) test samples to visualize")
    parser.add_argument("--n_neg_viz", type=int, default=25, help="Number of NORM (negative) test samples to visualize")
    parser.add_argument("--viz_random", action="store_true", help="Randomly sample positives/negatives instead of first N/M")
    parser.add_argument("--viz_seed", type=int, default=123, help="Seed for viz sampling when --viz_random is set")

    # Checkpoints / logs
    parser.add_argument("--out_dir", type=str, default="runs/mi_vs_norm_baseline_gradcam", help="Directory for checkpoints/logs")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # -------------------------------------------------
    # Create timestamped run directory
    # -------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"📁 Run directory: {run_dir}")

    # Save run configuration
    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    ckpt_last = os.path.join(run_dir, "last.pt")
    ckpt_best = os.path.join(run_dir, "best.pt")

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
    N, C, Lsig = X.shape
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

    # Default window/stride for visualization segments
    if args.window is None:
        window = 50 if args.sampling_rate == 100 else 250  # 0.5s
    else:
        window = args.window
    if args.stride is None:
        stride = window // 2
    else:
        stride = args.stride

    assert window <= Lsig, f"window={window} must be <= signal length L={Lsig}"
    assert stride > 0, "stride must be > 0"
    Tseg = (Lsig - window) // stride + 1
    print(f"Visualization segments: window={window}, stride={stride}, segments T={Tseg}")

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
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # Model + optimizer
    model = ECGConv2DBaseline(dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Grad-CAM target module: last Conv2d in features
    # features layout: [Conv,BN,Act, Conv,BN,Act,Pool, Conv,BN,Act,Pool]
    target_module = model.features[7]  # nn.Conv2d(32,64,...)
    cam_explainer = GradCAM2D(model, target_module)

    best_val_auc = -1.0

    # Train with checkpoints
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, pos_weight)
        va_loss, va_acc, va_auc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f} | {dt:.1f}s")

        # save last
        save_checkpoint(ckpt_last, model, opt, epoch, best_val_auc, args, extra={"window": window, "stride": stride, "L": Lsig})

        # update best
        if not np.isnan(va_auc) and va_auc > best_val_auc:
            best_val_auc = va_auc
            save_checkpoint(ckpt_best, model, opt, epoch, best_val_auc, args, extra={"window": window, "stride": stride, "L": Lsig})
            print(f"  ✅ best updated: val_auc={best_val_auc:.4f} -> saved {ckpt_best}")

        append_metrics_csv(metrics_csv, {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_auc": va_auc,
            "best_val_auc": best_val_auc,
            "seconds": dt
        })

    # Load best for testing if exists
    if os.path.exists(ckpt_best):
        ckpt = torch.load(ckpt_best, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"\nLoaded BEST checkpoint from epoch {ckpt['epoch']} with best_val_auc={ckpt['best_val_auc']:.4f}")

    te_loss, te_acc, te_auc = evaluate(model, test_loader, device)
    print(f"\nTEST | loss {te_loss:.4f} acc {te_acc:.4f} auc {te_auc:.4f}")

    # Save model text
    with open(os.path.join(run_dir, "model.txt"), "w") as f:
        f.write(str(model))

    # Visualization PDF on test set
    pdf_path = args.viz_pdf
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(run_dir, pdf_path)

    print(f"\nSaving Grad-CAM visualizations to: {pdf_path}")
    visualize_pos_neg_to_pdf_gradcam(
        model=model,
        cam_explainer=cam_explainer,
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


if __name__ == "__main__":
    main()