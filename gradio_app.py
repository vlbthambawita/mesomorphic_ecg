"""
Gradio app for ECG Patch IMN: load checkpoint, predict, visualize segments/leads,
and optionally ablate top leads/patches to see prediction change.

Usage:
  pip install gradio   # if not installed
  python gradio_app.py

Then open http://localhost:7860. Upload an ECG CSV (12xL or Lx12, e.g. 12x5000 @ 500 Hz),
or use the Example if sample_ecg_12x5000.csv exists. Set top-k leads/patches, optionally
enable ablation (remove top leads/patches) to see how P(MI) changes.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Add project root and import from training script (no main execution)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import script_25012026_v3_Selected_01_lightning_save_wgen_tokens as train

ECGPatchIMNLightning = train.ECGPatchIMNLightning
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs/mi_vs_norm_imn/2026-01-26_13-51-19/best-epoch=10-val_auc=0.9379.ckpt",
)
_MODEL_CACHE = {"path": None, "model": None, "device": None}


def load_model(ckpt_path: str, device: str | None = None):
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if _MODEL_CACHE["path"] == ckpt_path and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["device"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ECGPatchIMNLightning.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    _MODEL_CACHE["path"] = ckpt_path
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["device"] = device
    return model, device


def load_ecg_from_csv(csv_path: str, layout: str, signal_len: int) -> np.ndarray:
    df = pd.read_csv(csv_path, header=None)
    arr = df.values.astype(np.float32)
    is_12xL = "12xL" in layout or layout.strip().lower().startswith("12")
    if is_12xL:
        if arr.shape[0] != 12:
            raise ValueError(f"Expected 12 rows (leads), got {arr.shape[0]}.")
        x = arr
    else:
        if arr.shape[1] != 12:
            raise ValueError(f"Expected 12 columns (leads), got {arr.shape[1]}.")
        x = arr.T
    L = x.shape[1]
    if L != signal_len:
        raise ValueError(f"ECG length L={L} != model signal_len={signal_len}.")
    return x


def zscore_per_lead(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True).clip(min=1e-6)
    return ((x - mean) / std).astype(np.float32)


def ablate_and_recompute(
    patch_scores: torch.Tensor,
    b: torch.Tensor,
    T: int,
    top_lead_idx: list[int],
    top_patch_idx: list[int],
    n_remove_leads: int,
    n_remove_patches: int,
) -> float:
    ps = patch_scores.clone()
    P = ps.shape[1]
    lead_dim = P // 12
    if n_remove_leads > 0 and top_lead_idx:
        for lead_idx in top_lead_idx[:n_remove_leads]:
            start, end = lead_idx * lead_dim, min((lead_idx + 1) * lead_dim, P)
            ps[:, start:end] = 0.0
    if n_remove_patches > 0 and top_patch_idx:
        for p in top_patch_idx[:n_remove_patches]:
            if p < P:
                ps[:, p] = 0.0
    logit_abl = ps.sum(dim=1) + b
    return float(torch.sigmoid(logit_abl).item())


def build_fig(
    x_np: np.ndarray,
    heatmap: np.ndarray,
    window: int,
    stride: int,
    top_leads: list[int] | None,
    top_patches: list[int] | None,
    removed_leads: list[int] | None,
    removed_patches: list[tuple[int, int]] | None,
    T: int,
    pred: str,
    prob: float,
    prob_abl: float | None,
) -> plt.Figure:
    import matplotlib.patches as mpatches

    L = x_np.shape[1]
    hm = heatmap / (np.max(np.abs(heatmap)) + 1e-6)
    lead_abs = np.abs(heatmap).sum(axis=1)
    total_abs = lead_abs.sum() + 1e-9
    lead_pct = 100.0 * lead_abs / total_abs

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1] * 12 + [0.5])
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(hm, aspect="auto", vmin=-1, vmax=1, cmap="bwr")
    ax0.set_yticks(range(12))
    ax0.set_yticklabels(LEAD_NAMES)
    ax0.set_xlabel(f"Segments (window={window}, stride={stride})")
    t = f"{pred} | P(MI) = {prob:.3f}"
    if prob_abl is not None:
        t += f" -> Ablated P(MI) = {prob_abl:.3f}"
    ax0.set_title(t)
    fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

    rem_lead_set = set(removed_leads or [])
    rem_patch_set = set(removed_patches or [])
    for rl in rem_lead_set:
        rect = mpatches.Rectangle((-0.5, rl - 0.5), T, 1, fill=False, edgecolor="red", linewidth=2.5, zorder=10)
        ax0.add_patch(rect)

    top_set = set()
    if top_patches:
        for p in top_patches:
            top_set.add((p // T, p % T))

    for lead in range(12):
        ax = fig.add_subplot(gs[lead + 1, 0])
        ax.plot(x_np[lead], linewidth=0.8, color="black")
        ax.set_xlim(0, L - 1)
        lbl = f"{LEAD_NAMES[lead]} {lead_pct[lead]:.1f}%"
        ax.set_ylabel(lbl, rotation=0, labelpad=20, va="center")
        ylo, yhi = ax.get_ylim()
        if top_leads and lead in top_leads:
            ax.axhspan(ylo, yhi, alpha=0.15, color="gold", zorder=0)
        if lead in rem_lead_set:
            rect = mpatches.Rectangle((0, ylo), L - 1, yhi - ylo, fill=False, edgecolor="red", linewidth=2.5, zorder=10)
            ax.add_patch(rect)
        for ti in range(T):
            a = float(hm[lead, ti])
            alpha = min(0.35, abs(a) * 0.35)
            if alpha > 0:
                c = "red" if a > 0 else "blue"
                s, e = ti * stride, min(ti * stride + window, L)
                hi = (lead, ti) in top_set
                is_rem = (lead, ti) in rem_patch_set
                if is_rem:
                    ax.axvspan(s, e, alpha=alpha, color=c, linewidth=0, zorder=0)
                    rect = mpatches.Rectangle((s, ylo), e - s, yhi - ylo, fill=False, edgecolor="red", linewidth=2, zorder=10)
                    ax.add_patch(rect)
                elif hi:
                    ax.axvspan(s, e, alpha=alpha, color=c, linewidth=1.5, edgecolor="lime", zorder=1)
                else:
                    ax.axvspan(s, e, alpha=alpha, color=c, linewidth=0, zorder=0)
        ax.set_xticks([])

    axf = fig.add_subplot(gs[13, 0])
    axf.axis("off")
    leg = "Red=MI, Blue=NORM. Gold=top leads, Lime=top patches. "
    if rem_lead_set or rem_patch_set:
        leg += "Red boxes=removed (ablation). "
    leg += "Lead % = contribution share."
    axf.text(0, 0.5, leg, fontsize=9)
    fig.tight_layout()
    return fig


def predict(
    ckpt_path: str,
    ecg_file,
    csv_layout: str,
    topk_leads: int,
    topk_patches: int,
    remove_leads: bool,
    n_remove_leads: int,
    remove_patches: bool,
    n_remove_patches: int,
):
    err = "Upload an ECG CSV (12 x L or L x 12)."
    if ecg_file is None:
        return err, None, "—", "—", "—"
    path = ecg_file if isinstance(ecg_file, str) else (ecg_file.name if hasattr(ecg_file, "name") else None)
    if not path or not os.path.isfile(path):
        return err, None, "—", "—", "—"
    try:
        model, device = load_model(ckpt_path)
    except Exception as e:
        return f"Checkpoint error: {e}", None, "—", "—", "—"
    hp = model.hparams
    signal_len = int(hp["signal_len"])
    window = int(hp["window"] or (250 if signal_len >= 2500 else 50))
    stride = int(hp["stride"] or (window // 2))
    try:
        x = load_ecg_from_csv(path, csv_layout, signal_len)
    except Exception as e:
        return f"ECG error: {e}", None, "—", "—", "—"
    x = zscore_per_lead(x)
    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logit, Wgen, tokens, T = model.model(x_t)
        prob = torch.sigmoid(logit)
        patch_scores = (Wgen * tokens).sum(dim=-1)
        b = logit - patch_scores.sum(dim=1)
    prob_val = float(prob.item())
    pred = "MI" if prob_val >= 0.5 else "NORM"
    hm = patch_scores.view(1, 12, T).squeeze(0).cpu().numpy()
    lead_imp = np.abs(hm).sum(axis=1)
    k = min(max(0, int(topk_leads)), 12)
    top_leads = np.argsort(lead_imp)[::-1][:k].tolist() if k else []
    pa = np.abs(patch_scores.squeeze(0).cpu().numpy())
    kp = min(max(0, int(topk_patches)), pa.size)
    top_patches = np.argsort(pa)[::-1][:kp].tolist() if kp else []
    prob_abl = None
    nr, np_ = (n_remove_leads if remove_leads else 0), (n_remove_patches if remove_patches else 0)
    removed_leads: list[int] = []
    removed_patches: list[tuple[int, int]] = []
    if (nr or np_) and (top_leads or top_patches):
        prob_abl = ablate_and_recompute(patch_scores, b, T, top_leads, top_patches, nr, np_)
        if nr > 0 and top_leads:
            removed_leads = top_leads[:nr]
        if np_ > 0 and top_patches:
            removed_patches = [(p // T, p % T) for p in top_patches[:np_]]
    fig = build_fig(
        x, hm, window, stride,
        top_leads or None, top_patches or None,
        removed_leads or None, removed_patches or None,
        T, pred, prob_val, prob_abl,
    )
    tl = ", ".join(LEAD_NAMES[i] for i in top_leads) if top_leads else "—"
    tp = ", ".join(f"({LEAD_NAMES[p // T]},{p % T})" for p in top_patches[:12])
    if len(top_patches) > 12:
        tp += " ..."
    abl = f"Ablated P(MI) = {prob_abl:.3f}" if prob_abl is not None else "—"
    return f"**{pred}** | P(MI) = {prob_val:.3f}", fig, tl, tp, abl


def main():
    demo = gr.Blocks(title="ECG Patch IMN", theme=gr.themes.Soft())
    with demo:
        gr.Markdown("# ECG Patch IMN – Predict & Explain\nLoad checkpoint, upload ECG CSV, get prediction and segment/lead importance. Optionally remove top leads/patches (ablation).")
        with gr.Row():
            ckpt = gr.Textbox(label="Checkpoint path", value=DEFAULT_CKPT if os.path.isfile(DEFAULT_CKPT) else "", placeholder="path/to/best.ckpt")
            ecg = gr.File(label="ECG CSV", file_count="single", type="filepath")
        layout = gr.Radio(label="CSV layout", choices=["12xL (leads x time)", "Lx12 (time x leads)"], value="12xL (leads x time)")
        with gr.Row():
            topk_leads = gr.Number(label="Top-k leads", value=3, minimum=0, maximum=12, step=1)
            topk_patches = gr.Number(label="Top-k patches", value=10, minimum=0, maximum=200, step=1)
        gr.Markdown("### Ablation")
        with gr.Row():
            remove_leads = gr.Checkbox(label="Remove top leads", value=False)
            n_remove_leads = gr.Number(label="Num. leads to remove", value=1, minimum=0, maximum=12, step=1)
            remove_patches = gr.Checkbox(label="Remove top patches", value=False)
            n_remove_patches = gr.Number(label="Num. patches to remove", value=5, minimum=0, maximum=100, step=1)
        btn = gr.Button("Run", variant="primary")
        out_summary = gr.Markdown()
        out_plot = gr.Plot()
        out_leads = gr.Textbox(label="Top leads", interactive=False)
        out_patches = gr.Textbox(label="Top patches (lead, seg)", interactive=False)
        out_abl = gr.Textbox(label="Ablated", interactive=False)
        btn.click(
            predict,
            inputs=[ckpt, ecg, layout, topk_leads, topk_patches, remove_leads, n_remove_leads, remove_patches, n_remove_patches],
            outputs=[out_summary, out_plot, out_leads, out_patches, out_abl],
        )
        ex_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_ecg_12x5000.csv")
        if os.path.isfile(ex_path):
            gr.Examples(
                examples=[[DEFAULT_CKPT if os.path.isfile(DEFAULT_CKPT) else "", ex_path]],
                inputs=[ckpt, ecg],
                label="Example (sample ECG)",
            )
        gr.Markdown("CSV: 12 leads x L samples (e.g. 5000 @ 500 Hz). Red=MI, Blue=NORM.")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
