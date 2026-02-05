import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import wfdb
import ast
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# -------------------------------------------------------------------------
# 1. Model Architecture (Matches script)
# -------------------------------------------------------------------------
class ECG_IMN(nn.Module):
    def __init__(self, input_channels=12, signal_len=1000, num_classes=2, dropout=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.C = input_channels
        self.L = signal_len

        # Backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(16), nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.dropout = nn.Dropout(dropout)

        # Transition Network (Weight Generator)
        self.transition = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Upsample(scale_factor=(1, 2), mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Upsample(scale_factor=(1, 2), mode='nearest'),
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1, bias=True)
        )

        # Bias Generator
        self.bias_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias_head = nn.Linear(64, num_classes)

    def forward(self, x):
        B, C, L = x.shape
        feat = x.unsqueeze(1)
        feat = self.conv1(feat)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.dropout(feat)

        generated_w = self.transition(feat) 
        b_feat = self.bias_pool(feat).view(B, -1)
        generated_b = self.bias_head(b_feat)

        x_expanded = x.unsqueeze(1)
        weighted_input = generated_w * x_expanded
        logits = weighted_input.sum(dim=(2, 3)) + generated_b

        return logits, generated_w, generated_b.unsqueeze(-1)

# -------------------------------------------------------------------------
# 2. Data Helpers
# -------------------------------------------------------------------------
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def build_superclasses(path):
    scp_path = os.path.join(path, 'scp_statements.csv')
    if not os.path.exists(scp_path):
        raise FileNotFoundError(f"scp_statements.csv not found in {path}")
        
    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    return agg_df, aggregate_diagnostic

def imn_weights_to_segments(impact_12L: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Aggregates point-wise impact into segments for heatmap."""
    L = impact_12L.shape[1]
    T = (L - window) // stride + 1
    if T <= 0: return np.zeros((12, 1)) # Handle edge case
    
    seg = np.zeros((12, T), dtype=np.float32)
    for t in range(T):
        s = t * stride
        e = min(s + window, L)
        seg[:, t] = np.abs(impact_12L[:, s:e]).mean(axis=1)
        
    # Normalize [0, 1]
    mx = seg.max() + 1e-9
    seg = seg / mx
    return seg

# -------------------------------------------------------------------------
# 3. Application State & Logic
# -------------------------------------------------------------------------
CACHE = {"model": None, "X": None, "Y": None, "task": None}
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TASK_MAP = {
    "norm_vs_mi": "MI",
    "norm_vs_sttc": "STTC",
    "norm_vs_cd": "CD",
    "norm_vs_hyp": "HYP"
}

def load_system(ptb_path, ckpt_path, task_name, sampling_rate=500):
    """Loads data filtered by task (Folds 9,10 only) and model."""
    try:
        print(f"Loading DB from {ptb_path}...")
        db_path = os.path.join(ptb_path, "ptbxl_database.csv")
        Y = pd.read_csv(db_path, index_col="ecg_id")
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        
        _, aggregate_diagnostic = build_superclasses(ptb_path)
        Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)
        
        # Filter 1: Validation Folds (9, 10)
        Y = Y[Y.strat_fold >= 9]
        
        # Filter 2: Task Specific
        pos_class = TASK_MAP[task_name]
        
        def is_relevant(labels):
            has_pos = pos_class in labels
            has_norm = "NORM" in labels
            # Keep if it is strictly one or the other (binary task)
            return (has_pos and not has_norm) or (has_norm and not has_pos)

        mask = Y["diagnostic_superclass"].apply(is_relevant)
        Y_task = Y[mask].copy()
        
        if len(Y_task) == 0:
            return "Error: No data found for this task/fold configuration.", False

        print(f"Loading signals for {len(Y_task)} samples...")
        X = load_raw_data(Y_task, sampling_rate, ptb_path)
        X = np.transpose(X, (0, 2, 1)) # [N, 12, L]

        # Normalize
        mean = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2, keepdims=True) + 1e-6
        X = (X - mean) / std

        CACHE["X"] = torch.from_numpy(X).float()
        CACHE["Y"] = Y_task
        CACHE["task"] = task_name
        
        # Load Model
        print(f"Loading model from {ckpt_path}...")
        model = ECG_IMN(input_channels=12, signal_len=X.shape[2], num_classes=2)
        
        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            return "Error: Checkpoint not found.", False
            
        model.eval()
        CACHE["model"] = model
        
        return f"Loaded {len(Y_task)} {task_name} records (Folds 9-10). Ready.", True
        
    except Exception as e:
        return f"Error: {str(e)}", False

def run_analysis(sample_idx, k_leads, k_wins, viz_window, viz_stride):
    if CACHE["model"] is None:
        return None, "System not loaded.", "N/A"

    # 1. Get Data
    idx = int(sample_idx)
    if idx < 0 or idx >= len(CACHE["X"]):
        return None, "Index out of bounds.", "N/A"

    x_raw = CACHE["X"][idx].unsqueeze(0) # [1, 12, L]
    meta = CACHE["Y"].iloc[idx]
    
    # Ground Truth & Clinical Notes
    true_labels = meta["diagnostic_superclass"]
    clinical_notes = meta["scp_codes"] # Dictionary of codes
    report_text = f"**Diagnostic Class**: {true_labels}\n**Clinical Notes (SCP)**: {clinical_notes}"

    # 2. Inference
    pos_class_name = TASK_MAP[CACHE["task"]]
    
    with torch.no_grad():
        logits, gen_w, gen_b = CACHE["model"](x_raw)
        probs = torch.softmax(logits, dim=1)[0]
        p_neg, p_pos = float(probs[0]), float(probs[1])
        
        # Attribution: visualize weights for the PREDICTED class
        pred_idx = torch.argmax(logits).item() 
        w_used = gen_w[0, pred_idx].numpy()
        x_np = x_raw[0].numpy()
        impact = w_used * x_np # [12, L]

    pred_str = "Positive" if pred_idx == 1 else "NORM"
    result_text = f"**Original**: Pred={pred_str} | P({pos_class_name})={p_pos:.1%} | P(NORM)={p_neg:.1%}"

    # 3. Intervention (Masking)
    # Rank Leads (.copy() avoids negative strides from [::-1] that break PyTorch indexing)
    imp_leads = np.sum(np.abs(impact), axis=1)
    top_leads = np.argsort(imp_leads)[::-1][:k_leads].copy()

    # Rank Windows (using user provided viz_window for masking logic as well for consistency)
    L = x_np.shape[1]
    n_seg = L // viz_window
    win_imps = []
    for i in range(n_seg):
        s, e = i*viz_window, (i+1)*viz_window
        win_imps.append(np.sum(np.abs(impact[:, s:e])))
    top_wins = np.argsort(win_imps)[::-1][:k_wins].copy()

    # Apply Mask
    x_masked = x_raw.clone()
    mask_regions = []
    
    # Mask Leads
    x_masked[0, top_leads, :] = 0
    
    # Mask Windows
    for w_idx in top_wins:
        s, e = w_idx*viz_window, (w_idx+1)*viz_window
        x_masked[0, :, s:e] = 0
        mask_regions.append((s, e))

    # Re-Inference
    with torch.no_grad():
        logits_new, _, _ = CACHE["model"](x_masked)
        probs_new = torch.softmax(logits_new, dim=1)[0]
        p_pos_new = float(probs_new[1])
        
    new_pred = "Positive" if p_pos_new > 0.5 else "NORM"
    delta = p_pos - p_pos_new
    result_text += f"\n\n**After Masking (Top {k_leads} Leads, Top {k_wins} Windows)**:\nPred={new_pred} | P({pos_class_name})={p_pos_new:.1%} | Delta={delta:.1%}"

    # 4. Visualization
    # Generate Heatmap Segments
    seg_map = imn_weights_to_segments(impact, viz_window, viz_stride)
    
    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(13, 1, height_ratios=[1.5] + [1]*12)
    
    # Heatmap
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(seg_map, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax0.set_title(f"Feature Attribution (Window={viz_window}, Stride={viz_stride})")
    ax0.set_yticks(range(12))
    ax0.set_yticklabels(LEAD_NAMES)
    plt.colorbar(im, ax=ax0, fraction=0.01, pad=0.01)

    # Signals
    # Normalize for plotting
    vmax_sig = np.abs(x_np).max()
    
    for i in range(12):
        ax = fig.add_subplot(gs[i+1], sharex=ax0)
        # Plot Original
        ax.plot(x_np[i], color='black', linewidth=0.8, alpha=0.8)
        
        # Highlight Masked Regions (Windows)
        for (s, e) in mask_regions:
            ax.axvspan(s, e, color='red', alpha=0.3, label='Masked Window' if i==0 else "")
            
        # Highlight Masked Leads
        if i in top_leads:
            ax.set_facecolor('#ffcccc')
            ax.text(0, 0, " LEAD REMOVED ", color='red', fontweight='bold', transform=ax.transAxes)

        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=20, va="center")
        ax.set_ylim(-vmax_sig, vmax_sig)
        ax.set_yticks([])
        if i < 11: ax.set_xticks([])

    plt.tight_layout()
    return fig, result_text, report_text

# -------------------------------------------------------------------------
# 4. Gradio UI
# -------------------------------------------------------------------------
def app_main():
    with gr.Blocks(title="IMN ECG Analysis") as demo:
        gr.Markdown("## Interpretable Mesomorphic Neural Networks (IMN) - Clinical Validation Tool")
        
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                # Configuration
                gr.Markdown("### 1. Load Data")
                ptb_input = gr.Textbox(value="./ptbxl/", label="PTB-XL Directory")
                ckpt_input = gr.Textbox(value="model.ckpt", label="Model Checkpoint")
                task_input = gr.Dropdown(
                    choices=["norm_vs_mi", "norm_vs_sttc", "norm_vs_cd", "norm_vs_hyp"],
                    value="norm_vs_mi", label="Classification Task"
                )
                load_btn = gr.Button("Load Validation Set (Folds 9-10)", variant="primary")
                status_txt = gr.Textbox(label="System Status", interactive=False)

                gr.Markdown("### 2. Select Sample & Visualize")
                idx_input = gr.Number(value=0, label="Sample Index", precision=0)
                
                with gr.Row():
                    win_size = gr.Number(value=100, label="Viz Window Size")
                    stride_size = gr.Number(value=50, label="Viz Stride")
                
                gr.Markdown("### 3. Intervention (Explainability)")
                top_leads = gr.Slider(0, 12, value=0, step=1, label="Mask Top K Leads")
                top_wins = gr.Slider(0, 10, value=0, step=1, label="Mask Top K Windows")
                
                run_btn = gr.Button("Analyze Sample", variant="primary")
                
                gr.Markdown("### 4. Ground Truth")
                clinical_output = gr.Markdown(label="Clinical Data")
                
                gr.Markdown("### 5. Prediction Stats")
                res_output = gr.Markdown()

            with gr.Column(scale=2):
                plot_output = gr.Plot(label="Signal & Attribution")

        # Callbacks
        load_btn.click(
            load_system,
            inputs=[ptb_input, ckpt_input, task_input],
            outputs=[status_txt, load_btn] # dummy output for btn
        )
        
        run_btn.click(
            run_analysis,
            inputs=[idx_input, top_leads, top_wins, win_size, stride_size],
            outputs=[plot_output, res_output, clinical_output]
        )

    return demo

if __name__ == "__main__":
    app = app_main()
    app.launch()