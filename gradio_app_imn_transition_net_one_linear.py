"""
Gradio app for ECG IMN (Interpretable Mesomorphic Network) with Transition Net
and a SINGLE LINEAR OUTPUT (binary logit: logit = sum(w * x) + b, P(pos) = sigmoid(logit)).

This app:
  - Loads ECGs from the PTB-XL dataset path.
  - Lets you choose a PTB-XL binary task: POS_CLASS vs NORM.
  - Shows clinical notes and ground truth.
  - Runs the single-linear IMN model (script_02022026_v7_IMN_GM_2_with_transition_net_with_one_linear_eq.py).
  - Visualizes feature attribution (Impact = w * x) aggregated by segments.
  - Lets you select window and stride to explore different segment granularities.
  - Highlights top-k leads and top-k segments (by signed contribution to the positive logit).
  - Optional ablation: remove top leads/segments and recompute P(pos_class).
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
import script_02022026_v7_IMN_GM_2_with_transition_net_with_one_linear_eq as imn_script


IMNLightning = imn_script.IMNLightning
build_superclasses = imn_script.build_superclasses
imn_weights_to_segments = imn_script.imn_weights_to_segments

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_PTBXL = "/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1"

# Default mapping from task name to positive diagnostic superclass
TASK_TO_POS = {
    "norm_vs_mi": "MI",
    "norm_vs_sttc": "STTC",
    "norm_vs_cd": "CD",
    "norm_vs_hyp": "HYP",
}

# Leave default checkpoint empty; user should point to a checkpoint trained
# with the single-linear script for the chosen task and sampling rate.
DEFAULT_CKPT = ""

_MODEL_CACHE: dict[str, Any] = {"path": None, "model": None, "device": None}


def _ensure_trailing_slash(p: str) -> str:
    return p if p.endswith("/") else p + "/"


def load_model(ckpt_path: str, device: Optional[str] = None):
    """Load and cache the IMN Lightning model from a checkpoint (single-linear version)."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if _MODEL_CACHE["path"] == ckpt_path and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["device"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IMNLightning.load_from_checkpoint(ckpt_path, map_location=device)
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


def ablate_and_recompute_imn_single(
    gen_w: torch.Tensor,
    x_t: torch.Tensor,
    gen_b: torch.Tensor,
    top_leads: list[int],
    top_segments: list[tuple[int, int]],
    n_remove_leads: int,
    n_remove_segments: int,
    window: int,
    stride: int,
    L: int,
) -> float:
    """
    Ablation for SINGLE-LINEAR IMN.

    gen_w: [1, 1, 12, L], x_t: [1, 12, L], gen_b: [1, 1] or [1, 1, 1].
    We zero weights at selected leads/segments, keep the scalar bias unchanged,
    then recompute the single logit and its sigmoid P(pos_class).
    """
    # Extract [12, L] weight map
    w_single = gen_w[0, 0].clone()  # [12, L]

    # Remove whole leads
    if n_remove_leads > 0 and top_leads:
        for li in top_leads[:n_remove_leads]:
            w_single[li, :] = 0.0

    # Remove segments
    if n_remove_segments > 0 and top_segments:
        for lead, t in top_segments[:n_remove_segments]:
            s = t * stride
            e = min(s + window, L)
            w_single[lead, s:e] = 0.0

    x_exp = x_t[0]  # [12, L]

    # Bias: handle [1, 1] or [1, 1, 1]
    if gen_b.ndim == 3:
        b = gen_b[0, 0, 0]
    else:
        b = gen_b[0, 0]

    new_logit = (w_single.to(x_exp.device) * x_exp).sum() + b
    prob_pos = float(torch.sigmoid(new_logit).item())
    return prob_pos


def build_fig_imn_single(
    x_np: np.ndarray,
    seg_hm: np.ndarray,
    window: int,
    stride: int,
    T: int,
    pred: str,
    prob_pos: float,
    pos_class_name: str,
    sampling_rate: int,
    top_leads: Optional[list[int]] = None,
    top_segments: Optional[list[tuple[int, int]]] = None,
    removed_leads: Optional[list[int]] = None,
    removed_segments: Optional[list[tuple[int, int]]] = None,
    prob_abl: Optional[float] = None,
    lead_imp_signed: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Build a matplotlib figure for IMN single-linear feature attribution visualization.
    seg_hm: [12, T] segment heatmap from imn_weights_to_segments (normalized magnitudes).
    lead_imp_signed: [12] signed contribution per lead (impact.sum(axis=1)); used for
        percentages so they match the top-leads ranking (by signed contribution).
    """
    import matplotlib.patches as mpatches

    L = x_np.shape[1]

    # Per-lead contribution share: use signed contribution so percentages match top leads.
    # Use sum of |lead_imp_signed| as denominator so ranking is preserved when total < 0
    # (e.g. NORM prediction); otherwise negative total would invert the percentages.
    if lead_imp_signed is not None:
        denom = np.abs(lead_imp_signed).sum() + 1e-9
        lead_pct = 100.0 * lead_imp_signed / denom
    else:
        lead_abs = np.abs(seg_hm).sum(axis=1)
        lead_pct = 100.0 * lead_abs / (lead_abs.sum() + 1e-9)

    cmap = "Reds"
    shade_color = "red"

    rem_lead_set = set(removed_leads or [])
    rem_seg_set = set(removed_segments or [])
    top_seg_set = set(top_segments or [])

    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(14, 1, height_ratios=[2] + [1] * 12 + [0.5])

    # Top heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(seg_hm, aspect="auto", vmin=0.0, vmax=1.0, cmap=cmap)
    ax0.set_yticks(range(12))
    ax0.set_yticklabels(LEAD_NAMES)
    ax0.set_xlabel(f"Segments (window={window}, stride={stride}, fs={sampling_rate}Hz)")
    prob_str = f"P({pos_class_name})={prob_pos:.3f}"
    title = f"IMN Intrinsic Explanation (Single Linear) | {pred} | {prob_str}"
    if prob_abl is not None:
        p_str = f"{prob_abl:.4f}" if prob_abl < 0.001 else f"{prob_abl:.3f}"
        title += f" -> Ablated P({pos_class_name}) = {p_str}"
    ax0.set_title(title)
    fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.01)

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

    # Lead-wise ECG traces with overlays
    for lead in range(12):
        ax = fig.add_subplot(gs[lead + 1, 0])
        ax.plot(x_np[lead], linewidth=0.8, color="black", alpha=0.6)
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

        contrib = seg_hm[lead]
        for t in range(T):
            a = float(contrib[t])
            alpha = min(0.5, a * 0.6)
            if alpha <= 0.05:
                continue
            start = t * stride
            end = min(start + window, L)
            hi = (lead, t) in top_seg_set
            is_rem = (lead, t) in rem_seg_set

            if is_rem:
                ax.axvspan(start, end, alpha=alpha, facecolor=shade_color, zorder=0)
                ax.add_patch(
                    mpatches.Rectangle(
                        (start, ylo),
                        end - start,
                        yhi - ylo,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                        zorder=10,
                    )
                )
            elif hi:
                ax.axvspan(
                    start,
                    end,
                    alpha=alpha,
                    facecolor=shade_color,
                    edgecolor="lime",
                    linewidth=1.5,
                    zorder=1,
                )
            else:
                ax.axvspan(start, end, alpha=alpha, facecolor=shade_color, zorder=0)

        ax.set_xticks([])

    # Footer
    axf = fig.add_subplot(gs[13, 0])
    axf.axis("off")
    leg = (
        f"IMN Feature Attribution (Single Linear): |w(x)·x| aggregated by segment "
        f"(window={window}, stride={stride}). Gold/Lime = top leads/segments by signed contribution "
        f"(highest positive = most evidence for {pos_class_name}). "
    )
    if rem_lead_set or rem_seg_set:
        leg += "Red boxes = removed (ablation)."
    axf.text(0.5, 0.5, leg, fontsize=9, wrap=True, transform=axf.transAxes, ha="center", va="center")

    fig.tight_layout()
    return fig


def load_ptbxl_meta(ptbxl_path: str, sampling_rate: int, task: str):
    """
    Load PTB-XL metadata and restrict to POS_CLASS vs NORM on the test fold (fold=10),
    where POS_CLASS is determined by the selected task (norm_vs_mi, norm_vs_cd, ...).
    """
    path = _ensure_trailing_slash(ptbxl_path)
    db_path = path + "ptbxl_database.csv"
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Not found: {db_path}")

    Y = pd.read_csv(db_path, index_col="ecg_id")
    Y["scp_codes"] = Y["scp_codes"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    _, aggregate_diagnostic = build_superclasses(path)
    Y["diagnostic_superclass"] = Y["scp_codes"].apply(aggregate_diagnostic)

    pos_class = TASK_TO_POS.get(task, "MI")
    labels = Y["diagnostic_superclass"].values
    is_pos = np.array([pos_class in l for l in labels])
    is_norm = np.array(["NORM" in l for l in labels])
    keep = is_pos ^ is_norm
    Yf = Y[keep].copy()

    y_bin = np.array(
        [
            1.0 if pos_class in l else 0.0
            for l in Yf["diagnostic_superclass"].values
        ],
        dtype=np.float32,
    )

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
        gt = pos_class if y_test[i] >= 0.5 else "NORM"
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

    return records, path, sampling_rate, pos_class


def load_ecg_from_ptbxl(ptbxl_path: str, filename: str, sampling_rate: int) -> np.ndarray:
    path = _ensure_trailing_slash(ptbxl_path)
    sig, _ = wfdb.rdsamp(path + filename)
    return np.transpose(sig.astype(np.float32), (1, 0))


def on_load_ptbxl(ptbxl_path: str, sampling_rate: int, task: str, state: Optional[dict]):
    """Button handler: load PTB-XL test fold and populate record dropdown."""
    if not ptbxl_path or not ptbxl_path.strip():
        return "Enter PTB-XL path.", gr.update(choices=[], value=None), state or {}, "—", "—"
    try:
        records, path, fs, pos_class = load_ptbxl_meta(ptbxl_path.strip(), int(sampling_rate), task)
    except Exception as e:
        return f"Load error: {e}", gr.update(choices=[], value=None), state or {}, "—", "—"

    choices = [f"{r['ecg_id']} | {r['gt']} | age {r['age']} {r['sex']}" for r in records]
    value = choices[0] if choices else None
    state = {"records": records, "path": path, "fs": fs, "pos_class": pos_class, "task": task}
    report = (records[0]["report"] or "(no clinical notes)") if records else "—"
    gt = records[0]["gt"] if records else "—"
    return (
        f"Loaded {len(records)} test records ({pos_class} vs NORM).",
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


def predict_imn_single(
    ckpt_path: str,
    ptbxl_path: str,
    sampling_rate: int,
    record_choice: str,
    state: Optional[dict],
    window: int,
    stride: int,
    topk_leads: int,
    topk_segments: int,
    remove_leads: bool,
    n_remove_leads: int,
    remove_segments: bool,
    n_remove_segments: int,
):
    """
    Run single-linear IMN inference on a selected PTB-XL record and visualize
    feature attribution with user-selected window and stride. Optionally show
    top-k leads/segments and ablate them to see effect on P(pos_class).
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
    pos_class_name = state.get("pos_class", "MI")
    report = rec["report"] or "(no clinical notes)"
    gt = rec["gt"]

    try:
        model, device = load_model(ckpt_path)
    except Exception as e:
        return f"Checkpoint error: {e}", None, report, gt, "—", "—", "—"

    signal_len = int(model.hparams["signal_len"])

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
        logits, gen_w, gen_b = model.model(x_t)
        # logits: [1, 1] -> scalar logit
        logit = logits.squeeze()
        prob_pos = float(torch.sigmoid(logit).item())
        y_int = int(rec["y"])

        # Single-linear: only one weight map [1, 1, 12, L]
        w_used = gen_w[0, 0, :, :].cpu().numpy()  # [12, L]

    x_np = x.astype(np.float64)
    impact = w_used * x_np  # Feature attribution: w * x (signed contribution)

    # Aggregate into segments using user-selected window and stride (magnitude)
    seg_hm = imn_weights_to_segments(impact, window=window, stride=stride)
    T = seg_hm.shape[1]
    L = x_np.shape[1]

    # Top-k leads: rank by SIGNED contribution (impact) to the positive logit.
    lead_imp_signed = impact.sum(axis=1)  # [12] total contribution per lead
    k_leads = min(max(0, int(topk_leads)), 12)
    top_leads = np.argsort(lead_imp_signed)[::-1][:k_leads].tolist() if k_leads else []

    # Top-k segments: rank by SIGNED segment contribution (mean impact per segment)
    seg_signed = np.zeros((12, T), dtype=np.float64)
    for t in range(T):
        s, e = t * stride, min(t * stride + window, L)
        seg_signed[:, t] = impact[:, s:e].mean(axis=1)
    seg_flat = seg_signed.flatten()
    k_seg = min(max(0, int(topk_segments)), seg_flat.size)
    top_flat_idx = np.argsort(seg_flat)[::-1][:k_seg]
    top_segments = [(idx // T, idx % T) for idx in top_flat_idx] if k_seg else []

    # Optional ablation
    prob_abl = None
    nr = n_remove_leads if remove_leads else 0
    ns = n_remove_segments if remove_segments else 0
    removed_leads = []
    removed_segments = []

    if (nr or ns) and (top_leads or top_segments):
        prob_abl = ablate_and_recompute_imn_single(
            gen_w,
            x_t,
            gen_b,
            top_leads,
            top_segments,
            nr,
            ns,
            window,
            stride,
            L,
        )
        if nr > 0 and top_leads:
            removed_leads = top_leads[:nr]
        if ns > 0 and top_segments:
            removed_segments = top_segments[:ns]

    pred = pos_class_name if prob_pos >= 0.5 else "NORM"

    fig = build_fig_imn_single(
        x_np,
        seg_hm,
        window,
        stride,
        T,
        pred,
        prob_pos,
        pos_class_name,
        fs,
        top_leads=top_leads or None,
        top_segments=top_segments or None,
        removed_leads=removed_leads or None,
        removed_segments=removed_segments or None,
        prob_abl=prob_abl,
        lead_imp_signed=lead_imp_signed,
    )

    tl = ", ".join(LEAD_NAMES[i] for i in top_leads) if top_leads else "—"
    tp = (
        ", ".join(f"({LEAD_NAMES[l]},{t})" for l, t in top_segments[:12])
        if top_segments
        else "—"
    )
    if len(top_segments) > 12:
        tp += " ..."

    def _fmt_prob(p: float) -> str:
        """Format probability with enough precision to avoid misleading 0.000."""
        return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"

    abl = f"Ablated P({pos_class_name}) = {_fmt_prob(prob_abl)}" if prob_abl is not None else "—"
    summary = (
        f"**{pred}** | P({pos_class_name}) = {prob_pos:.3f}"
        + (f" → **{_fmt_prob(prob_abl)}** (after ablation)" if prob_abl is not None else "")
        + f" | Ground truth: **{gt}** | window={window}, stride={stride}"
    )
    return summary, fig, report, gt, tl, tp, abl


def main():
    demo = gr.Blocks(
        title="ECG IMN – PTB-XL (Transition Net, Single Linear)",
        theme=gr.themes.Soft(),
    )
    with demo:
        gr.Markdown(
            "# ECG IMN – PTB-XL (Transition Net, Single Linear)\n"
            "Load ECGs from **PTB-XL** path. Choose a binary task (**POS_CLASS vs NORM**), "
            "view **clinical notes** and **ground truth**, then run the single-linear IMN "
            "prediction with **feature attribution** (Impact = w·x). "
            "Select **window** and **stride** to explore different segment granularities."
        )

        # PTB-XL loading
        with gr.Row():
            ptbxl_path = gr.Textbox(
                label="PTB-XL path",
                value=DEFAULT_PTBXL if os.path.isdir(DEFAULT_PTBXL) else "",
                placeholder="path/to/ptb-xl/1_0_1",
            )
            fs = gr.Radio(label="Sampling rate", choices=[100, 500], value=100)
            task = gr.Radio(
                label="Task (positive class vs NORM)",
                choices=list(TASK_TO_POS.keys()),
                value="norm_vs_mi",
            )
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
            inputs=[ptbxl_path, fs, task, ptbxl_state],
            outputs=[load_status, record_dd, ptbxl_state, clinical_notes, ground_truth],
        )
        record_dd.change(
            fn=on_select_record,
            inputs=[record_dd, ptbxl_state],
            outputs=[clinical_notes, ground_truth],
        )

        # Model & prediction
        gr.Markdown("### Model & IMN prediction (single-linear)")
        with gr.Row():
            ckpt = gr.Textbox(
                label="Checkpoint path (IMN transition net, single-linear)",
                value=DEFAULT_CKPT if DEFAULT_CKPT and os.path.isfile(DEFAULT_CKPT) else "",
                placeholder="path/to/best-imn-single-linear.ckpt",
            )

        gr.Markdown("### Window & stride (segment aggregation)")
        with gr.Row():
            window = gr.Slider(
                label="Window size",
                minimum=10,
                maximum=500,
                value=50,
                step=5,
                info="Segment width for aggregating point-wise attributions",
            )
            stride = gr.Slider(
                label="Stride",
                minimum=5,
                maximum=250,
                value=25,
                step=5,
                info="Step between segments (typically window/2)",
            )

        gr.Markdown(
            "### Top-k leads & segments\n"
            "Ranked by **signed** contribution (w·x) to the positive logit: "
            "highest positive = most evidence for the selected positive class. "
            "Removing them should decrease P(pos_class) for positive-predicting leads."
        )
        with gr.Row():
            topk_leads = gr.Number(
                label="Top-k leads",
                value=3,
                minimum=0,
                maximum=12,
                step=1,
            )
            topk_segments = gr.Number(
                label="Top-k segments",
                value=10,
                minimum=0,
                maximum=200,
                step=1,
            )

        gr.Markdown("### Ablation (remove top leads/segments and recompute)")
        with gr.Row():
            remove_leads = gr.Checkbox(label="Remove top leads", value=False)
            n_remove_leads = gr.Number(
                label="Num. leads to remove",
                value=1,
                minimum=0,
                maximum=12,
                step=1,
            )
            remove_segments = gr.Checkbox(label="Remove top segments", value=False)
            n_remove_segments = gr.Number(
                label="Num. segments to remove",
                value=5,
                minimum=0,
                maximum=100,
                step=1,
            )

        run_btn = gr.Button("Run IMN prediction", variant="primary")

        out_summary = gr.Markdown()
        out_plot = gr.Plot()
        out_notes = gr.Textbox(label="Clinical notes", lines=3, interactive=False)
        out_gt = gr.Textbox(label="Ground truth", interactive=False)
        out_leads = gr.Textbox(label="Top leads", interactive=False)
        out_segments = gr.Textbox(
            label="Top segments (lead, seg_idx)",
            interactive=False,
        )
        out_abl = gr.Textbox(label="Ablated", interactive=False)

        run_btn.click(
            fn=predict_imn_single,
            inputs=[
                ckpt,
                ptbxl_path,
                fs,
                record_dd,
                ptbxl_state,
                window,
                stride,
                topk_leads,
                topk_segments,
                remove_leads,
                n_remove_leads,
                remove_segments,
                n_remove_segments,
            ],
            outputs=[
                out_summary,
                out_plot,
                out_notes,
                out_gt,
                out_leads,
                out_segments,
                out_abl,
            ],
        )

        gr.Markdown(
            "IMN (single-linear) generates weights w(x) per sample point and a scalar bias b. "
            "Logit = sum(w(x)·x) + b, with P(pos_class) = sigmoid(logit). "
            "Window/stride control segment aggregation of |w·x|. "
            "Top-k leads/segments are highlighted (gold/lime). "
            "Enable ablation to remove top contributors and see how P(pos_class) changes. "
            "Ensure your checkpoint was trained with the same sampling rate and task as selected above."
        )

    demo.launch(server_name="0.0.0.0", server_port=7863)


if __name__ == "__main__":
    main()

