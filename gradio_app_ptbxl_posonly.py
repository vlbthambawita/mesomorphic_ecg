"""
Gradio app for ECG Patch IMN (positive-only contributions) with PTB-XL.

This app:
  - Loads ECGs from the PTB-XL dataset path.
  - Shows clinical notes and ground truth (MI vs NORM).
  - Runs the *positive-only* patch-contribution model
    (`script_25012026_v3_Selected_01_lightning_save_wgen_tokens_posonly.py`).
  - Visualizes *only positive (MI-pushing)* contributions as red segments
    overlaid on each lead and as a heatmap.
"""

from __future__ import annotations

import ast
import os
import sys
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import wfdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import script_25012026_v3_Selected_01_lightning_save_wgen_tokens_posonly as train_pos


ECGPatchIMNLightning = train_pos.ECGPatchIMNLightning
build_superclasses = train_pos.build_superclasses

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_PTBXL = "/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"

# Default to a recent positive-only checkpoint if present; otherwise leave blank.
DEFAULT_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs/mi_vs_norm_imn/script_25012026_v3_Selected_01_lightning_save_wgen_tokens_posonly_test_run_25012026_v3_Full_with_pos_only_2026-01-28_12-15-38/best-epoch=10-val_auc=0.9379.ckpt",
)

_MODEL_CACHE: dict[str, Any] = {"path": None, "model": None, "device": None}


def _ensure_trailing_slash(p: str) -> str:
    return p if p.endswith("/") else p + "/"


def load_model(ckpt_path: str, device: Optional[str] = None):
    """Load and cache the positive-only Lightning model from a checkpoint."""
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


def zscore_per_lead(x: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalization."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True).clip(min=1e-6)
    return ((x - mean) / std).astype(np.float32)


def ablate_and_recompute(
    ps: torch.Tensor,
    b: torch.Tensor,
    T: int,
    top_lead_idx: list[int],
    top_patch_idx: list[int],
    n_remove_leads: int,
    n_remove_patches: int,
) -> float:
    """
    Zero out selected leads/patches in the (non-negative) patch scores and
    recompute the probability via the logit + bias.
    """
    ps = ps.clone()
    P = ps.shape[1]
    lead_dim = P // 12

    # Remove whole leads
    if n_remove_leads > 0 and top_lead_idx:
        for li in top_lead_idx[:n_remove_leads]:
            s, e = li * lead_dim, min((li + 1) * lead_dim, P)
            ps[:, s:e] = 0.0

    # Remove individual patches
    if n_remove_patches > 0 and top_patch_idx:
        for p in top_patch_idx[:n_remove_patches]:
            if p < P:
                ps[:, p] = 0.0

    return float(torch.sigmoid(ps.sum(dim=1) + b).item())


def build_fig_posonly(
    x_np: np.ndarray,
    heatmap: np.ndarray,
    window: int,
    stride: int,
    top_leads: Optional[list[int]],
    top_patches: Optional[list[int]],
    removed_leads: Optional[list[int]],
    removed_patches: Optional[list[tuple[int, int]]],
    T: int,
    pred: str,
    prob: float,
    prob_abl: Optional[float],
) -> plt.Figure:
    """
    Build a matplotlib figure similar to the full model app, but using the
    positive-only heatmap (non-negative contributions).
    """
    import matplotlib.patches as mpatches

    L = x_np.shape[1]

    # Normalize heatmap (already >=0) for display
    hm = heatmap / (np.max(np.abs(heatmap)) + 1e-6)

    # Per-lead contribution share
    lead_abs = np.abs(heatmap).sum(axis=1)
    lead_pct = 100.0 * lead_abs / (lead_abs.sum() + 1e-9)

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1] * 12 + [0.5])

    # Top heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(hm, aspect="auto", vmin=0.0, vmax=1.0, cmap="Reds")
    ax0.set_yticks(range(12))
    ax0.set_yticklabels(LEAD_NAMES)
    ax0.set_xlabel(f"Segments (window={window}, stride={stride})")
    t = f"{pred} | P(MI) = {prob:.3f} (positive-only)"
    if prob_abl is not None:
        t += f" -> Ablated P(MI) = {prob_abl:.3f}"
    ax0.set_title(t)
    fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

    rem_lead_set = set(removed_leads or [])
    rem_patch_set = set(removed_patches or [])

    # Highlight removed leads in the heatmap
    for rl in rem_lead_set:
        rect = mpatches.Rectangle(
            (-0.5, rl - 0.5),
            T,
            1,
            fill=False,
            edgecolor="red",
            linewidth=2.5,
            zorder=10,
        )
        ax0.add_patch(rect)

    # For quick membership check of top patches
    top_set = set((p // T, p % T) for p in (top_patches or []))

    # Lead-wise ECG traces with overlays
    for lead in range(12):
        ax = fig.add_subplot(gs[lead + 1, 0])
        ax.plot(x_np[lead], linewidth=0.8, color="black")
        ax.set_xlim(0, L - 1)
        ax.set_ylabel(
            f"{LEAD_NAMES[lead]} {lead_pct[lead]:.1f}%",
            rotation=0,
            labelpad=20,
            va="center",
        )

        ylo, yhi = ax.get_ylim()

        # Shade top leads
        if top_leads and lead in top_leads:
            ax.axhspan(ylo, yhi, alpha=0.15, color="gold", zorder=0)

        # Mark removed whole leads
        if lead in rem_lead_set:
            ax.add_patch(
                mpatches.Rectangle(
                    (0, ylo),
                    L - 1,
                    yhi - ylo,
                    fill=False,
                    edgecolor="red",
                    linewidth=2.5,
                    zorder=10,
                )
            )

        # Overlay patch contributions
        for ti in range(T):
            a = float(hm[lead, ti])
            alpha = min(0.35, abs(a) * 0.35)
            if alpha <= 0:
                continue
            c = "red"
            s, e = ti * stride, min(ti * stride + window, L)
            hi = (lead, ti) in top_set
            is_rem = (lead, ti) in rem_patch_set

            if is_rem:
                ax.axvspan(s, e, alpha=alpha, color=c, linewidth=0, zorder=0)
                ax.add_patch(
                    mpatches.Rectangle(
                        (s, ylo),
                        e - s,
                        yhi - ylo,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                        zorder=10,
                    )
                )
            elif hi:
                ax.axvspan(
                    s,
                    e,
                    alpha=alpha,
                    color=c,
                    linewidth=1.5,
                    edgecolor="lime",
                    zorder=1,
                )
            else:
                ax.axvspan(s, e, alpha=alpha, color=c, linewidth=0, zorder=0)

        ax.set_xticks([])

    # Legend / explanatory text
    axf = fig.add_subplot(gs[13, 0])
    axf.axis("off")
    leg = (
        "Red = positive contribution toward MI (negative contributions clipped to 0). "
        "Gold = top leads, Lime = top patches. "
    )
    if rem_lead_set or rem_patch_set:
        leg += "Red boxes = removed (ablation). "
    leg += "Lead % = contribution share."
    axf.text(0, 0.5, leg, fontsize=9)

    fig.tight_layout()
    return fig


def load_ptbxl_meta(ptbxl_path: str, sampling_rate: int):
    """Load PTB-XL metadata and restrict to MI vs NORM test fold."""
    path = _ensure_trailing_slash(ptbxl_path)
    db_path = path + "ptbxl_database.csv"
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Not found: {db_path}")
    Y = pd.read_csv(db_path, index_col="ecg_id")
    Y["scp_codes"] = Y["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    _, aggregate_diagnostic = build_superclasses(path)
    Y["diagnostic_superclass"] = Y["scp_codes"].apply(aggregate_diagnostic)
    labels = Y["diagnostic_superclass"].values
    is_mi = np.array(["MI" in str(l) for l in labels])
    is_norm = np.array(["NORM" in str(l) for l in labels])
    keep = is_mi ^ is_norm
    Yf = Y[keep].copy()
    y_bin = np.array([1.0 if "MI" in str(l) else 0.0 for l in Yf["diagnostic_superclass"].values], dtype=np.float32)
    fold = Yf["strat_fold"].values
    test_mask = fold == 10
    Y_test = Yf[test_mask].copy()
    y_test = y_bin[test_mask]
    records = []
    for i, (idx, row) in enumerate(Y_test.iterrows()):
        fn = row["filename_hr"] if sampling_rate == 500 else row["filename_lr"]
        report = str(row.get("report", "") or "")
        age = row.get("age", "")
        sex = str(row.get("sex", "")) if pd.notna(row.get("sex")) else ""
        gt = "MI" if y_test[i] >= 0.5 else "NORM"
        records.append(
            {
                "ecg_id": int(idx),
                "filename": fn,
                "report": report,
                "gt": gt,
                "y": float(y_test[i]),
                "age": age,
                "sex": sex,
            }
        )
    return records, path, sampling_rate


def load_ecg_from_ptbxl(ptbxl_path: str, filename: str, sampling_rate: int) -> np.ndarray:
    path = _ensure_trailing_slash(ptbxl_path)
    sig, _ = wfdb.rdsamp(path + filename)
    return np.transpose(sig.astype(np.float32), (1, 0))


def on_load_ptbxl(ptbxl_path: str, sampling_rate: int, state: Optional[dict]):
    """Button handler: load PTB-XL test fold and populate record dropdown."""
    if not ptbxl_path or not ptbxl_path.strip():
        return "Enter PTB-XL path.", gr.update(choices=[], value=None), state or {}, "—", "—"
    try:
        records, path, fs = load_ptbxl_meta(ptbxl_path.strip(), int(sampling_rate))
    except Exception as e:
        return f"Load error: {e}", gr.update(choices=[], value=None), state or {}, "—", "—"
    choices = [f"{r['ecg_id']} | {r['gt']} | age {r['age']} {r['sex']}" for r in records]
    value = choices[0] if choices else None
    state = {"records": records, "path": path, "fs": fs}
    report = (records[0]["report"] or "(no clinical notes)") if records else "—"
    gt = records[0]["gt"] if records else "—"
    return (
        f"Loaded {len(records)} test records (MI vs NORM).",
        gr.update(choices=choices, value=value),
        state,
        report,
        gt,
    )


def on_select_record(choice: str, state: Optional[dict]):
    """Dropdown handler: update clinical notes and GT when a record is selected."""
    if not state or not state.get("records") or not choice:
        return "—", "—"
    try:
        ecg_id = int(choice.split("|")[0].strip())
    except Exception:
        return "—", "—"
    for r in state["records"]:
        if r["ecg_id"] == ecg_id:
            return r["report"] or "(no clinical notes)", r["gt"]
    return "—", "—"


def predict_ptbxl_posonly(
    ckpt_path: str,
    ptbxl_path: str,
    sampling_rate: int,
    record_choice: str,
    state: Optional[dict],
    topk_leads: int,
    topk_patches: int,
    remove_leads: bool,
    n_remove_leads: int,
    remove_patches: bool,
    n_remove_patches: int,
):
    """
    Run positive-only model inference on a selected PTB-XL record and
    visualize non-negative contributions, with optional ablation of
    top leads/patches.
    """
    err = "Select a record and Load PTB-XL first.", None, "—", "—", "—", "—", "—"
    if not state or not state.get("records") or not record_choice:
        return err
    try:
        ecg_id = int(record_choice.split("|")[0].strip())
    except Exception:
        return err
    rec = next((r for r in state["records"] if r["ecg_id"] == ecg_id), None)
    if not rec:
        return err

    path, fs = state["path"], state["fs"]
    report = rec["report"] or "(no clinical notes)"
    gt = rec["gt"]

    try:
        model, device = load_model(ckpt_path)
    except Exception as e:
        return f"Checkpoint error: {e}", None, report, gt, "—", "—", "—"

    hp = model.hparams
    signal_len = int(hp["signal_len"])
    window = int(hp["window"] or (250 if signal_len >= 2500 else 50))
    stride = int(hp["stride"] or (window // 2))

    try:
        x = load_ecg_from_ptbxl(path, rec["filename"], fs)
    except Exception as e:
        return f"ECG load error: {e}", None, report, gt, "—", "—", "—"

    if x.shape[1] != signal_len:
        return (
            f"ECG length {x.shape[1]} != model {signal_len}. Use matching sampling rate.",
            None,
            report,
            gt,
            "—",
            "—",
            "—",
        )

    x = zscore_per_lead(x)
    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward through the underlying positive-only model to get
        # logits and non-negative patch scores.
        logit, Wgen, tokens, T = model.model(x_t)
        prob = torch.sigmoid(logit)
        patch_scores = (Wgen * tokens).sum(dim=-1)  # [B,P]
        patch_scores = patch_scores.clamp_min(0.0)
        b = logit - patch_scores.sum(dim=1)

    prob_val = float(prob.item())
    pred = "MI" if prob_val >= 0.5 else "NORM"

    # Heatmap [12, T] from non-negative patch scores
    hm = patch_scores.view(1, 12, T).squeeze(0).cpu().numpy()

    # Top-k leads and patches (by non-negative contribution)
    lead_imp = np.abs(hm).sum(axis=1)
    k = min(max(0, int(topk_leads)), 12)
    top_leads = np.argsort(lead_imp)[::-1][:k].tolist() if k else []

    pa = np.abs(patch_scores.squeeze(0).cpu().numpy())
    kp = min(max(0, int(topk_patches)), pa.size)
    top_patches = np.argsort(pa)[::-1][:kp].tolist() if kp else []

    # Optional ablation
    prob_abl = None
    nr = n_remove_leads if remove_leads else 0
    np_ = n_remove_patches if remove_patches else 0
    removed_leads = []
    removed_patches = []

    if (nr or np_) and (top_leads or top_patches):
        prob_abl = ablate_and_recompute(
            patch_scores,
            b,
            T,
            top_leads,
            top_patches,
            nr,
            np_,
        )
        if nr > 0 and top_leads:
            removed_leads = top_leads[:nr]
        if np_ > 0 and top_patches:
            removed_patches = [(p // T, p % T) for p in top_patches[:np_]]

    fig = build_fig_posonly(
        x,
        hm,
        window,
        stride,
        top_leads or None,
        top_patches or None,
        removed_leads or None,
        removed_patches or None,
        T,
        pred,
        prob_val,
        prob_abl,
    )

    tl = ", ".join(LEAD_NAMES[i] for i in top_leads) if top_leads else "—"
    tp = (
        ", ".join(f"({LEAD_NAMES[p // T]},{p % T})" for p in top_patches[:12])
        if top_patches
        else "—"
    )
    if len(top_patches) > 12:
        tp += " ..."

    abl = f"Ablated P(MI) = {prob_abl:.3f}" if prob_abl is not None else "—"
    summary = (
        f"**{pred}** | P(MI) = {prob_val:.3f} | "
        f"Ground truth: **{gt}** (positive-only contributions)"
    )
    return summary, fig, report, gt, tl, tp, abl


def main():
    demo = gr.Blocks(
        title="ECG Patch IMN – PTB-XL (Positive-only Contributions)",
        theme=gr.themes.Soft(),
    )
    with demo:
        gr.Markdown(
            "# ECG Patch IMN – PTB-XL (Positive-only)\n"
            "Load ECGs from **PTB-XL** path. View **clinical notes** and **ground truth**, "
            "then run prediction with **positive-only (MI-pushing) patch contributions**."
        )

        # PTB-XL loading
        with gr.Row():
            ptbxl_path = gr.Textbox(
                label="PTB-XL path",
                value=DEFAULT_PTBXL if os.path.isdir(DEFAULT_PTBXL) else "",
                placeholder="path/to/ptb-xl/1_0_1",
            )
            fs = gr.Radio(label="Sampling rate", choices=[100, 500], value=500)
            load_btn = gr.Button("Load PTB-XL", variant="secondary")

        load_status = gr.Markdown()
        ptbxl_state = gr.State(value=None)

        with gr.Row():
            record_dd = gr.Dropdown(
                label="Test record (ecg_id | GT | age sex)",
                choices=[],
                value=None,
            )

        with gr.Row():
            clinical_notes = gr.Textbox(
                label="Clinical notes (report)",
                value="",
                lines=4,
                max_lines=8,
                interactive=False,
            )
            ground_truth = gr.Textbox(
                label="Ground truth",
                value="—",
                interactive=False,
            )

        load_btn.click(
            fn=on_load_ptbxl,
            inputs=[ptbxl_path, fs, ptbxl_state],
            outputs=[load_status, record_dd, ptbxl_state, clinical_notes, ground_truth],
        )
        record_dd.change(
            fn=on_select_record,
            inputs=[record_dd, ptbxl_state],
            outputs=[clinical_notes, ground_truth],
        )

        # Model & prediction
        gr.Markdown("### Model & positive-only prediction")
        with gr.Row():
            ckpt = gr.Textbox(
                label="Checkpoint path (positive-only model)",
                value=DEFAULT_CKPT if os.path.isfile(DEFAULT_CKPT) else "",
                placeholder="path/to/best_posonly.ckpt",
            )

        with gr.Row():
            topk_leads = gr.Number(
                label="Top-k leads",
                value=3,
                minimum=0,
                maximum=12,
                step=1,
            )
            topk_patches = gr.Number(
                label="Top-k patches",
                value=10,
                minimum=0,
                maximum=200,
                step=1,
            )

        gr.Markdown("### Ablation")
        with gr.Row():
            remove_leads = gr.Checkbox(label="Remove top leads", value=False)
            n_remove_leads = gr.Number(
                label="Num. leads to remove",
                value=1,
                minimum=0,
                maximum=12,
                step=1,
            )
            remove_patches = gr.Checkbox(label="Remove top patches", value=False)
            n_remove_patches = gr.Number(
                label="Num. patches to remove",
                value=5,
                minimum=0,
                maximum=100,
                step=1,
            )

        run_btn = gr.Button("Run positive-only prediction", variant="primary")

        out_summary = gr.Markdown()
        out_plot = gr.Plot()
        out_notes = gr.Textbox(label="Clinical notes", lines=3, interactive=False)
        out_gt = gr.Textbox(label="Ground truth", interactive=False)
        out_leads = gr.Textbox(label="Top positive leads", interactive=False)
        out_patches = gr.Textbox(
            label="Top positive patches (lead, seg)",
            interactive=False,
        )
        out_abl = gr.Textbox(label="Ablated", interactive=False)

        run_btn.click(
            fn=predict_ptbxl_posonly,
            inputs=[
                ckpt,
                ptbxl_path,
                fs,
                record_dd,
                ptbxl_state,
                topk_leads,
                topk_patches,
                remove_leads,
                n_remove_leads,
                remove_patches,
                n_remove_patches,
            ],
            outputs=[
                out_summary,
                out_plot,
                out_notes,
                out_gt,
                out_leads,
                out_patches,
                out_abl,
            ],
        )

        gr.Markdown(
            "PTB-XL path = dir with `ptbxl_database.csv`, `scp_statements.csv`. Test fold only. "
            "Red shading = **positive** contribution toward MI; negative contributions are clipped to 0."
        )

    demo.launch(server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    main()

