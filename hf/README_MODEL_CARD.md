---
license: cc-by-4.0
---
## mesomorphicECG

### Model overview

The **mesomorphicECG** repository hosts a family of binary ECG classification models trained on 12‑lead ECG signals at two sampling rates (100 Hz and 500 Hz). Each model predicts whether an ECG segment belongs to a normal control patient (`norm`) or to one of four diagnostic categories:

- **norm_vs_cd**: Normal vs coronary artery disease (CD)
- **norm_vs_hyp**: Normal vs hypertensive heart disease (HYP)
- **norm_vs_mi**: Normal vs myocardial infarction (MI)
- **norm_vs_sttc**: Normal vs ST‑T abnormalities (STTC)

Five architectural variants are provided:

- **Categorical IMN**: A multi‑layer “IMN transition net” that predicts a binary label.
- **Single‑Linear IMN**: A simplified variant with a single linear decision head (useful for interpretability analyses and probing feature representations).
- **GradCAM baseline (100 Hz)**: A 2D CNN with post‑hoc Grad‑CAM explanations at 100 Hz sampling.
- **GradCAM baseline (500 Hz)**: A 2D CNN with post‑hoc Grad‑CAM explanations at 500 Hz sampling.
- **IMN Direct**: A basic IMN implementation with direct classification (useful for ablation and comparison).

Models are provided for both **100 Hz** and **500 Hz** sampling rates where applicable, yielding **4 (tasks) × 7 (architectures) = 28 model configurations**.

### Available checkpoints and structure

Checkpoints are organized in the repository as:

| Category | Path | Checkpoint pattern | Metrics file |
|----------|------|--------------------|--------------|
| Categorical IMN 100 Hz | `categorical_imn_100hz/<task>/` | `best-imn-epoch=E-val_auc=A.ckpt` | `metrics.csv` |
| Categorical IMN 500 Hz | `categorical_imn_500hz/<task>/` | `best-imn-epoch=E-val_auc=A.ckpt` | `metrics.csv` |
| Single‑Linear IMN 100 Hz | `single_linear_imn_100hz/<task>/` | `best-imn-epoch=E-val_auc=A.ckpt` | `metrics.csv` |
| Single‑Linear IMN 500 Hz | `single_linear_imn_500hz/<task>/` | `best-imn-epoch=E-val_auc=A.ckpt` | `metrics.csv` |
| GradCAM 100 Hz | `gradcam_100hz/<task>/` | `best-epoch=E-val_auc=A.ckpt` | `validation_metrics.csv` |
| GradCAM 500 Hz | `gradcam_500hz/<task>/` | `best-epoch=E-val_auc=A.ckpt` | `validation_metrics.csv` |
| IMN Direct | `imn_direct/<task>/` | `best-imn-epoch=E-val_auc=A.ckpt` | `metrics.csv` |

Where `<task>` is one of:

- `norm_vs_cd`
- `norm_vs_hyp`
- `norm_vs_mi`
- `norm_vs_sttc`

Each task directory typically contains:

- **Checkpoint file**: Best validation checkpoint (by AUC). Naming varies by architecture (see table above).
- **`args.yaml`**: Training configuration and hyperparameters.
- **Metrics file**: Summary metrics (e.g. accuracy, balanced accuracy, precision, recall, F1, MCC, AUROC) for the best model. GradCAM models use `validation_metrics.csv`; IMN models use `metrics.csv`.

### Intended use

- **Primary use**: Research on ECG‑based risk stratification, disease detection, and model interpretability.
- **Tasks**:
  - Binary classification of individual ECG windows or segments.
  - Comparison of model behavior across sampling rates (100 Hz vs 500 Hz).
  - Comparison of full categorical vs single‑linear decision heads.
  - Comparison of IMN vs Grad‑CAM baseline architectures.
- **Users**:
  - ML and signal processing researchers working on cardiovascular AI.
  - Clinician‑scientists exploring interpretable ECG models.
  - Developers building proof‑of‑concept ECG classification systems.

These models are **not** intended for direct clinical decision making without further validation and regulatory clearance.

### Out‑of‑scope uses

- **Do not use for**:
  - Real‑time clinical diagnosis or triage without extensive external validation.
  - Deployment in medical devices or hospital systems without regulatory approval.
  - Populations, devices, or acquisition protocols that differ significantly from the training data, unless carefully re‑evaluated.

### Data

- **Input**:
  - Multi‑lead ECG time series (e.g. 12 leads).
  - Sampling rate: **100 Hz** or **500 Hz**, depending on the model.
  - Input windows are fixed‑length ECG segments (exact window length and stride are defined in `args.yaml` for each checkpoint).
- **Labels**:
  - Binary labels: `0` for normal, `1` for the target diagnostic group (CD / HYP / MI / STTC), per task.

The data used for training and validation consists of de‑identified ECG records. For details on cohort selection, preprocessing, and labeling, please refer to the associated project documentation or publication (if available) or contact the authors.

### Training and architecture

- **Base architectures**:
  - **IMN variants**: IMN‑based “transition net” encoder for ECG time series. Convolutional / temporal feature extraction followed by fully‑connected layers.
  - **GradCAM baseline**: 2D CNN with post‑hoc Grad‑CAM for spatial attribution.
  - **IMN Direct**: Basic IMN with direct classification head.
- **Variants**:
  - **Categorical IMN**: Standard deep classifier with non‑linear layers before the output.
  - **Single‑Linear IMN**: Same encoder, but with a single linear output layer (no hidden layers after the encoder) to support interpretability and linear probing.
  - **GradCAM**: Standard 2D CNN trained for classification; Grad‑CAM heatmaps provide spatial explanations.
  - **IMN Direct**: Simplified IMN for ablation studies.
- **Optimization**:
  - Binary classification objective (e.g. cross‑entropy).
  - Model selection based on **validation AUROC**; upload scripts select the checkpoint with the highest `val_auc` across runs (see metrics files).

Exact hyperparameters (learning rate, batch size, input window length, etc.) are stored per‑run in the accompanying `args.yaml` files.

### Evaluation

- **Metrics**:
  - `accuracy`
  - `balanced_accuracy`
  - `precision`
  - `recall`
  - `f1_score`
  - `mcc`
  - `auroc` (used for model selection)
- **Performance**:
  - The best checkpoints typically achieve **high AUROC (≈0.90–0.97)** on validation data for IMN and GradCAM models, with task‑dependent variation.
  - IMN Direct models are early/ablation checkpoints and may show lower AUROC.
  - Per‑task, per‑model metrics are available in the corresponding metrics files.

These metrics reflect performance on the internal validation splits and **may not generalize** to other datasets, institutions, or devices.

### How to use

#### 1. Download a checkpoint

```python
from huggingface_hub import hf_hub_download

repo_id = "SEARCH-IHI/mesomorphicECG"

# Example: Categorical IMN 500 Hz, norm_vs_mi
ckpt_path = hf_hub_download(
    repo_id=repo_id,
    filename="categorical_imn_500hz/norm_vs_mi/best-imn-epoch=18-val_auc=0.9555.ckpt",
)
print(ckpt_path)

# Example: GradCAM 100 Hz, norm_vs_mi
ckpt_path = hf_hub_download(
    repo_id=repo_id,
    filename="gradcam_100hz/norm_vs_mi/best-epoch=25-val_auc=0.9699.ckpt",
)

# Example: IMN Direct, norm_vs_cd
ckpt_path = hf_hub_download(
    repo_id=repo_id,
    filename="imn_direct/norm_vs_cd/best-imn-epoch=08-val_auc=0.5155.ckpt",
)
```

#### 2. Run inference

Use the corresponding inference scripts from the [project repository](https://github.com/SEARCH-IHI/mesomorphicECG) (or equivalent), passing the downloaded checkpoint path via `--ckpt`. See the project README for task‑specific arguments (window, stride, leads, etc.).
