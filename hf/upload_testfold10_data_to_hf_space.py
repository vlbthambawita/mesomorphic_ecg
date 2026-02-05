#!/usr/bin/env python
"""
Create and upload PTB-XL test-fold(=10) data binaries to a Hugging Face Space.

This script:
- Reads PTB-XL metadata and raw ECG signals.
- For each combination of:
    - sampling_rate: 100 Hz, 500 Hz
    - task: norm_vs_cd, norm_vs_hyp, norm_vs_mi, norm_vs_sttc
  it:
    - Restricts to POS_CLASS vs NORM on strat_fold == 10 (test fold).
    - Loads ECG waveforms at the chosen sampling rate.
    - Packs arrays into a `.npz` file with keys:
        * signals: [N, 12, L] (float32)
        * labels:  [N] (0=NORM, 1=POS_CLASS)
        * reports: [N] (object/str)
        * age:     [N]
        * sex:     [N] (object/str)
        * ecg_id:  [N] (int)
    - Uploads the `.npz` into the specified Space repo under:
        data/ptbxl_{sampling_rate}hz_{task}_test.npz

Defaults:
- Space repo: SEARCH-IHI/mesomorphicECG_XAI
- Expects HF token from:
    * --token argument, or
    * HUGGINGFACE_HUB_TOKEN / HF_TOKEN env vars, optionally loaded from .env

Example:
    python upload_testfold10_data_to_hf_space.py \\
        --ptbxl-path /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1 \\
        --repo-id SEARCH-IHI/mesomorphicECG_XAI \\
        --dry-run
"""

from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


TASK_TO_POS: Dict[str, str] = {
    "norm_vs_mi": "MI",
    "norm_vs_sttc": "STTC",
    "norm_vs_cd": "CD",
    "norm_vs_hyp": "HYP",
}


def _ensure_trailing_slash(p: str) -> str:
    return p if p.endswith("/") else p + "/"


def build_superclasses(path: str):
    """
    Load PTB-XL scp_statements.csv and build diagnostic superclass aggregator.
    """
    agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    return agg_df, aggregate_diagnostic


def load_raw_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
    """
    Official PTB-XL loading helper:
    - sampling_rate == 100: use filename_lr
    - sampling_rate == 500: use filename_hr
    Returns: [N, L, 12]
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, _ in data])  # [N, L, 12]
    return data


@dataclass
class Fold10Data:
    sampling_rate: int
    task: str
    local_path: str
    n_examples: int
    signal_len: int


def collect_test_fold10_data(
    ptbxl_path: str,
    sampling_rate: int,
    task: str,
    out_dir: str,
) -> Optional[Fold10Data]:
    """
    Build the test-fold(=10) dataset for one (sampling_rate, task) combination,
    save to .npz, and return metadata. Returns None if no matching samples.
    """
    path = _ensure_trailing_slash(ptbxl_path)
    db_path = path + "ptbxl_database.csv"
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"PTB-XL database not found at {db_path}")

    Y = pd.read_csv(db_path, index_col="ecg_id")
    Y["scp_codes"] = Y["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    _, aggregate_diagnostic = build_superclasses(path)
    Y["diagnostic_superclass"] = Y["scp_codes"].apply(aggregate_diagnostic)

    if task not in TASK_TO_POS:
        raise ValueError(f"Unknown task: {task}. Expected one of {list(TASK_TO_POS.keys())}")
    pos_class = TASK_TO_POS[task]

    labels = Y["diagnostic_superclass"].values
    is_pos = np.array([pos_class in l for l in labels])
    is_norm = np.array(["NORM" in l for l in labels])
    keep = is_pos ^ is_norm
    Yf = Y[keep].copy()
    if Yf.empty:
        print(f"[WARN] No POS vs NORM samples for task={task} at sampling_rate={sampling_rate}")
        return None

    y_bin = np.array(
        [1 if pos_class in l else 0 for l in Yf["diagnostic_superclass"].values],
        dtype=np.int64,
    )

    fold = Yf["strat_fold"].values
    test_mask = fold == 10
    Y_test = Yf[test_mask].copy()
    y_test = y_bin[test_mask]

    if Y_test.empty:
        print(f"[WARN] No fold=10 samples for task={task} at sampling_rate={sampling_rate}")
        return None

    print(
        f"[INFO] Loading raw ECGs for fs={sampling_rate}Hz, task={task}, "
        f"test fold size={len(Y_test)} ..."
    )
    X = load_raw_data(Y_test, sampling_rate, path)  # [N, L, 12]
    X = np.transpose(X, (0, 2, 1)).astype(np.float32)  # -> [N, 12, L]
    N, C, L = X.shape

    # Metadata arrays
    if "report" in Y_test.columns:
        reports_series = Y_test["report"].astype("object").fillna("")
        reports = reports_series.values
    else:
        reports = np.array([""] * N, dtype=object)

    if "age" in Y_test.columns:
        age = Y_test["age"].values
    else:
        age = np.full(N, np.nan)

    if "sex" in Y_test.columns:
        sex_series = Y_test["sex"].astype("object").fillna("")
        sex = sex_series.values
    else:
        sex = np.array([""] * N, dtype=object)
    ecg_id = Y_test.index.values.astype(np.int64)

    os.makedirs(out_dir, exist_ok=True)
    fname = f"ptbxl_{sampling_rate}hz_{task}_test.npz"
    local_path = os.path.join(out_dir, fname)

    np.savez_compressed(
        local_path,
        signals=X,
        labels=y_test.astype(np.int64),
        reports=reports,
        age=age,
        sex=sex,
        ecg_id=ecg_id,
    )

    print(f"[OK] Saved {N} examples to {local_path} (L={L})")
    return Fold10Data(
        sampling_rate=sampling_rate,
        task=task,
        local_path=local_path,
        n_examples=N,
        signal_len=L,
    )


def build_commit_operations_for_space(
    bundles: List[Fold10Data],
) -> List[CommitOperationAdd]:
    """
    Map local .npz files into `data/` paths inside the Space repo.
    """
    ops: List[CommitOperationAdd] = []
    for b in bundles:
        if not os.path.isfile(b.local_path):
            print(f"[WARN] Missing local file (skipping): {b.local_path}")
            continue
        fname = os.path.basename(b.local_path)
        repo_path = os.path.join("data", fname)
        ops.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=b.local_path,
            )
        )
    return ops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and upload PTB-XL test-fold(10) data binaries to a Hugging Face Space.",
    )
    parser.add_argument(
        "--ptbxl-path",
        type=str,
        required=True,
        help="Path to PTB-XL root (directory containing ptbxl_database.csv).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="SEARCH-IHI/mesomorphicECG_XAI",
        help="Hugging Face Space repo id (e.g. 'SEARCH-IHI/mesomorphicECG_XAI').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token. If omitted, will use env / cache.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "testfold10_npz"),
        help="Local directory to store generated .npz files before upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create local .npz files and print planned uploads; do not modify the Space.",
    )

    args = parser.parse_args()

    # Load optional .env from PROJECT_ROOT so HF_TOKEN / HUGGINGFACE_HUB_TOKEN are available.
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path, override=False)

    sampling_rates = [100, 500]
    tasks = list(TASK_TO_POS.keys())

    all_bundles: List[Fold10Data] = []
    for fs in sampling_rates:
        for task in tasks:
            try:
                bundle = collect_test_fold10_data(
                    ptbxl_path=args.ptbxl_path,
                    sampling_rate=fs,
                    task=task,
                    out_dir=args.out_dir,
                )
            except Exception as e:
                print(f"[ERROR] Failed for fs={fs}, task={task}: {e}")
                continue
            if bundle is not None:
                all_bundles.append(bundle)

    if not all_bundles:
        print("[ERROR] No .npz bundles were created; nothing to upload.")
        return

    print("\nSummary of created .npz bundles:")
    for b in all_bundles:
        print(
            f"  - fs={b.sampling_rate}Hz, task={b.task}, "
            f"N={b.n_examples}, L={b.signal_len}, file={os.path.basename(b.local_path)}"
        )

    ops = build_commit_operations_for_space(all_bundles)
    if not ops:
        print("[ERROR] No commit operations built; aborting.")
        return

    if args.dry_run:
        print(
            f"\n[DRY RUN] Would create a commit to Space '{args.repo_id}' "
            f"with {len(ops)} file additions:"
        )
        for op in ops:
            print(f"    add {op.path_in_repo}")
        return

    # Resolve token precedence: CLI arg > env vars
    token = args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token) if token is not None else HfApi()

    print(
        f"\nCreating commit on Hugging Face Space '{args.repo_id}' "
        f"with {len(ops)} file additions..."
    )
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="space",
        operations=ops,
        commit_message="Add PTB-XL test-fold(10) data binaries for mesomorphicECG XAI app.",
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()

