import argparse
import ast
import os
import time
import random
import numpy as np
import pandas as pd
import wfdb

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
    def __init__(self, signal_len: int, window: int, stride: int, token_dim: int = 64, hyper_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.signal_len = signal_len
        self.window = window
        self.stride = stride
        self.token_dim = token_dim

        # Compute T, P deterministically from L, window, stride
        self.T = (signal_len - window) // stride + 1
        self.P = 12 * self.T

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

    def forward(self, x):
        """
        x: [B,12,L]
        returns:
          logit: [B]
          Wgen: [B,P,D]
          tokens: [B,P,D]
          T: number of time segments
        """
        patches, T = patchify_ecg_overlap(x, self.window, self.stride)  # [B,P,W]
        B, P, W = patches.shape
        assert T == self.T, f"T mismatch: got {T}, expected {self.T}"
        assert P == self.P, f"P mismatch: got {P}, expected {self.P}"
        assert W == self.window

        patches_ = patches.view(B * P, 1, W)
        tok = self.patch_embed(patches_)                # [B*P,D]
        tokens = tok.view(B, P, self.token_dim)         # [B,P,D]

        g = self.g_proj(tokens.mean(dim=1))             # [B,D]
        out = self.hyper(g)                             # [B, P*D+1]
        Wgen = out[:, :-1].view(B, P, self.token_dim)   # [B,P,D]
        b = out[:, -1]                                  # [B]

        logit = (Wgen * tokens).sum(dim=(1, 2)) + b
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
def train_one_epoch(model, loader, opt, device, l1_lambda: float, pos_weight: float):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    pos_w = torch.tensor([pos_weight], device=device)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logit, Wgen, tokens, T = model(x)
        bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
        l1 = Wgen.abs().mean()
        loss = bce + l1_lambda * l1

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
def evaluate(model, loader, device, l1_lambda: float):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ys, ps = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logit, Wgen, tokens, T = model(x)
        bce = F.binary_cross_entropy_with_logits(logit, y)
        l1 = Wgen.abs().mean()
        loss = bce + l1_lambda * l1

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
# Visualization to PDF (per-lead)
# -----------------------
@torch.no_grad()
def visualize_to_pdf(model, dataset, device, pdf_path: str, sampling_rate: int,
                     window: int, stride: int, n_samples: int = 10, start_idx: int = 0,
                     lead_names=None):
    """
    Saves a multi-page PDF.
    Each page = one ECG:
      - top: lead×segment heatmap (12 x T)
      - then: 12 subplots, one per lead, waveform with segment shading
    """
    model.eval()
    if lead_names is None:
        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        max_idx = min(len(dataset), start_idx + n_samples)
        for idx in range(start_idx, max_idx):
            x, y = dataset[idx]
            x_b = x.unsqueeze(0).to(device)  # [1,12,L]

            prob, heatmap = model.explain(x_b)
            prob = float(prob.item())
            hm = heatmap.squeeze(0).detach().cpu().numpy()  # [12,T]
            x_np = x.detach().cpu().numpy()                 # [12,L]

            # Normalize heatmap for display
            denom = np.max(np.abs(hm)) + 1e-6
            hm_disp = hm / denom  # [-1,1]

            L = x_np.shape[1]
            T = hm.shape[1]

            # Figure layout
            fig = plt.figure(figsize=(11.7, 16.5))  # A3-ish portrait
            gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1]*12 + [0.5])

            # Heatmap
            ax0 = fig.add_subplot(gs[0, 0])
            im = ax0.imshow(hm_disp, aspect="auto", vmin=-1, vmax=1, cmap="bwr")
            ax0.set_yticks(range(12))
            ax0.set_yticklabels(lead_names)
            ax0.set_xlabel(f"Segments (window={window} samples, stride={stride}, fs={sampling_rate}Hz)")
            ax0.set_title(f"MI vs NORM | true={int(y.item())} | P(MI)={prob:.3f} | idx={idx}")
            fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

            # Per-lead waveforms with segment shading
            for lead in range(12):
                ax = fig.add_subplot(gs[lead + 1, 0])
                ax.plot(x_np[lead], linewidth=0.8)
                ax.set_xlim(0, L - 1)
                ax.set_ylabel(lead_names[lead], rotation=0, labelpad=20, va="center")

                contrib = hm_disp[lead]  # [T]
                # Shade each window span: [t*stride, t*stride+window)
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
            axf.text(0, 0.5, "Red=pushes toward MI, Blue=pushes toward NORM (signed contributions, normalized per record).",
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

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--l1_lambda", type=float, default=0.05)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--hyper_hidden", type=int, default=256)

    # Patching
    parser.add_argument("--window", type=int, default=None,
                        help="Window in samples. Default: 0.5s => 50@100Hz or 250@500Hz.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride in samples. Default: window//2 (overlap 50%).")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    # Visualization
    parser.add_argument("--n_viz", type=int, default=25, help="Number of TEST samples to visualize into PDF")
    parser.add_argument("--viz_start", type=int, default=0, help="Start index within test dataset subset")
    parser.add_argument("--viz_pdf", type=str, default="viz_mi_vs_norm_test.pdf", help="Output PDF filename")

    # Checkpoints
    parser.add_argument("--out_dir", type=str, default="runs/mi_vs_norm_imn", help="Directory for checkpoints/logs")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    ckpt_last = os.path.join(args.out_dir, "last.pt")
    ckpt_best = os.path.join(args.out_dir, "best.pt")

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
                              num_workers=args.num_workers, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device=="cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device=="cuda"))

    # Model
    model = ECGPatchIMN(
        signal_len=L, window=window, stride=stride,
        token_dim=args.token_dim, hyper_hidden=args.hyper_hidden, dropout=0.1
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = -1.0

    # Train with checkpoints
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, args.l1_lambda, pos_weight)
        va_loss, va_acc, va_auc = evaluate(model, val_loader, device, args.l1_lambda)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f} | {dt:.1f}s")

        # save last
        save_checkpoint(ckpt_last, model, opt, epoch, best_val_auc, args, extra={"window": window, "stride": stride, "L": L})

        # update best
        if not np.isnan(va_auc) and va_auc > best_val_auc:
            best_val_auc = va_auc
            save_checkpoint(ckpt_best, model, opt, epoch, best_val_auc, args, extra={"window": window, "stride": stride, "L": L})
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

    te_loss, te_acc, te_auc = evaluate(model, test_loader, device, args.l1_lambda)
    print(f"\nTEST | loss {te_loss:.4f} acc {te_acc:.4f} auc {te_auc:.4f}")

    # Visualization PDF on test set
    pdf_path = args.viz_pdf
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(args.out_dir, pdf_path)

    print(f"\nSaving per-lead visualizations to: {pdf_path}")
    visualize_to_pdf(
        model=model,
        dataset=test_ds,
        device=device,
        pdf_path=pdf_path,
        sampling_rate=args.sampling_rate,
        window=window,
        stride=stride,
        n_samples=args.n_viz,
        start_idx=args.viz_start
    )
    print("Done.")


if __name__ == "__main__":
    main()