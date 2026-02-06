import argparse
import ast
import os
import time
import random
import numpy as np
import pandas as pd

# Fix for pandas 2.0+ compatibility with wfdb
# Monkey patch DataFrame.set_index to handle StringArray
_original_set_index = pd.DataFrame.set_index
def _patched_set_index(self, keys, drop=True, append=False, inplace=False, verify_integrity=False):
    # Convert StringArray to list if needed
    if isinstance(keys, pd.arrays.StringArray):
        keys = keys.tolist()
    elif hasattr(keys, 'tolist') and not isinstance(keys, (list, tuple, np.ndarray)):
        keys = keys.tolist()
    return _original_set_index(self, keys, drop=drop, append=append, inplace=inplace, verify_integrity=verify_integrity)
pd.DataFrame.set_index = _patched_set_index

import wfdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


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
# Patchify ECG
# -----------------------
def patchify_ecg(x: torch.Tensor, window: int):
    B, Ld, L = x.shape
    assert Ld == 12
    assert L % window == 0, f"window={window} must divide signal length L={L}"
    T = L // window
    patches = x.view(B, Ld, T, window).contiguous().view(B, Ld * T, window)
    return patches, T


# -----------------------
# Model: Patch-wise IMN for ECG (binary)
# -----------------------
class ECGPatchIMN(nn.Module):
    def __init__(self, signal_len: int, window: int, token_dim: int = 64, hyper_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.signal_len = signal_len
        self.window = window
        self.token_dim = token_dim

        assert signal_len % window == 0
        self.T = signal_len // window
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
        patches, T = patchify_ecg(x, self.window)  # [B,P,W]
        B, P, W = patches.shape
        assert P == self.P and T == self.T and W == self.window

        patches_ = patches.view(B * P, 1, W)
        tok = self.patch_embed(patches_)                 # [B*P,D]
        tokens = tok.view(B, P, self.token_dim)          # [B,P,D]

        g = self.g_proj(tokens.mean(dim=1))              # [B,D]

        out = self.hyper(g)                              # [B, P*D+1]
        Wgen = out[:, :-1].view(B, P, self.token_dim)    # [B,P,D]
        b = out[:, -1]                                   # [B]

        logit = (Wgen * tokens).sum(dim=(1, 2)) + b
        return logit, Wgen, tokens, T

    @torch.no_grad()
    def explain(self, x):
        logit, Wgen, tokens, T = self.forward(x)
        prob = torch.sigmoid(logit)
        patch_scores = (Wgen * tokens).sum(dim=-1)            # [B,P]
        heatmap = patch_scores.view(x.size(0), 12, T)         # [B,12,T]
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
# Visualization
# -----------------------
@torch.no_grad()
def visualize_sample(model, dataset, device, idx: int, sampling_rate: int, window: int, output_path: str = None):
    model.eval()
    lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    x, y = dataset[idx]
    x = x.unsqueeze(0).to(device)
    prob, heatmap = model.explain(x)
    prob = prob.item()
    hm = heatmap.squeeze(0).detach().cpu().numpy()  # [12,T]
    x_cpu = x.squeeze(0).detach().cpu().numpy()     # [12,L]

    denom = np.max(np.abs(hm)) + 1e-6
    hm_disp = hm / denom

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(hm_disp, aspect="auto", vmin=-1, vmax=1, cmap="bwr")
    plt.yticks(range(12), lead_names)
    plt.xlabel(f"Time segments (window={window} samples, fs={sampling_rate}Hz)")
    plt.title(f"MI vs NORM explanation (signed)\ntrue={int(y.item())}, P(MI)={prob:.3f}")
    plt.colorbar()

    lead = 1  # II
    plt.subplot(1, 2, 2)
    plt.plot(x_cpu[lead], linewidth=1.0)
    T = hm.shape[1]
    seg_len = window
    contrib = hm_disp[lead]

    for t in range(T):
        a = float(contrib[t])
        alpha = min(0.35, abs(a) * 0.35)
        if alpha > 0:
            color = "red" if a > 0 else "blue"
            plt.axvspan(t*seg_len, (t+1)*seg_len, alpha=alpha, color=color)

    plt.title("Lead II with contribution bands")
    plt.xlabel("Samples")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    plt.close()


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="path/to/ptbxl/")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--l1_lambda", type=float, default=0.05)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--hyper_hidden", type=int, default=256)
    parser.add_argument("--window", type=int, default=None,
                        help="Patch window in samples. Default: 0.5s -> 50@100Hz or 250@500Hz.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--viz_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="Directory to save visualization PDFs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    path = args.path if args.path.endswith("/") else args.path + "/"

    # Load metadata
    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Diagnostic superclasses
    _, aggregate_diagnostic = build_superclasses(path)
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

    # Filter to MI-only and NORM-only (exclude others)
    labels = Y["diagnostic_superclass"].values
    is_mi = np.array(["MI" in l for l in labels])
    is_norm = np.array(["NORM" in l for l in labels])

    keep = (is_mi ^ is_norm)  # exactly one of them
    Yf = Y[keep].copy()

    # Binary label: MI=1, NORM=0
    y_bin = np.array([1.0 if "MI" in l else 0.0 for l in Yf["diagnostic_superclass"].values], dtype=np.float32)

    print(f"Kept MI vs NORM only: N={len(Yf)} | MI={int(y_bin.sum())} | NORM={int((1-y_bin).sum())}")

    # Load signals for filtered rows
    print(f"Loading signals at {args.sampling_rate}Hz (filtered set)...")
    X = load_raw_data(Yf, args.sampling_rate, path)  # [N, L, 12]
    X = np.transpose(X, (0, 2, 1))                   # [N, 12, L]
    N, C, L = X.shape
    print("X shape:", X.shape)

    # Split by strat_fold: train 1-8, val 9, test 10
    fold = Yf["strat_fold"].values
    train_mask = (fold >= 1) & (fold <= 8)
    val_mask = (fold == 9)
    test_mask = (fold == 10)

    X_train, y_train = X[train_mask], y_bin[train_mask]
    X_val, y_val     = X[val_mask], y_bin[val_mask]
    X_test, y_test   = X[test_mask], y_bin[test_mask]

    print("Split sizes:", len(y_train), len(y_val), len(y_test))

    # Default window: 0.5s patches
    if args.window is None:
        window = 50 if args.sampling_rate == 100 else 250
    else:
        window = args.window
    assert L % window == 0, f"Signal length L={L} must be divisible by window={window}"

    # Torch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val   = torch.from_numpy(X_val).float()
    y_val   = torch.from_numpy(y_val).float()
    X_test  = torch.from_numpy(X_test).float()
    y_test  = torch.from_numpy(y_test).float()

    # Pos weight for imbalance (used only in training)
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

    model = ECGPatchIMN(signal_len=L, window=window, token_dim=args.token_dim,
                        hyper_hidden=args.hyper_hidden, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_auc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, args.l1_lambda, pos_weight)
        va_loss, va_acc, va_auc = evaluate(model, val_loader, device, args.l1_lambda)
        dt = time.time() - t0

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f} | {dt:.1f}s")

        if not np.isnan(va_auc) and va_auc > best_val_auc:
            best_val_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Save checkpoint every N epochs
        if epoch % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, 
                                          f"checkpoint_epoch{epoch:03d}_valauc{va_auc:.4f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': va_loss,
                'val_acc': va_acc,
                'val_auc': va_auc,
                'train_loss': tr_loss,
                'train_acc': tr_acc,
                'best_val_auc': best_val_auc,
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
        # Save best model checkpoint
        best_checkpoint_path = os.path.join(args.checkpoint_dir, 
                                          f"best_model_valauc{best_val_auc:.4f}.pt")
        torch.save({
            'epoch': 'best',
            'model_state_dict': best_state,
            'best_val_auc': best_val_auc,
        }, best_checkpoint_path)
        print(f"\nBest model checkpoint saved to: {best_checkpoint_path}")

    te_loss, te_acc, te_auc = evaluate(model, test_loader, device, args.l1_lambda)
    print(f"\nTEST | loss {te_loss:.4f} acc {te_acc:.4f} auc {te_auc:.4f}")

    print("\nVisualizing explanation on TEST set...")
    viz_idx = min(args.viz_idx, len(test_ds) - 1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    x, y_true = test_ds[viz_idx]
    x_viz = x.unsqueeze(0).to(device)
    with torch.no_grad():
        prob, _ = model.explain(x_viz)
        prob_val = prob.item()
    
    output_filename = f"visualization_idx{viz_idx}_true{int(y_true.item())}_prob{prob_val:.3f}.pdf"
    output_path = os.path.join(args.output_dir, output_filename)
    
    visualize_sample(model, test_ds, device, idx=viz_idx, sampling_rate=args.sampling_rate, 
                     window=window, output_path=output_path)


if __name__ == "__main__":
    main()