#!/usr/bin/env python
"""
Upload best checkpoints for GradCAM (100Hz, 500Hz) and IMN Direct to Hugging Face Hub.

This script scans the following three main categories:

Main categories (local base directories):
- gradcam_100hz -> runs/baseline_2d_gradcam_100Hz
- gradcam_500hz -> runs/baseline_2d_gradcam
- imn_direct -> runs/imn_ecg_simple_implementation_GM_Basic

Subcategories (subdirectories under each base directory):
- norm_vs_cd
- norm_vs_hyp
- norm_vs_mi
- norm_vs_sttc

For each (category, subcategory), it:
1. Locates all timestamped run subdirectories.
2. Within each run, finds checkpoint files:
   - GradCAM: best-epoch=E-val_auc=A.ckpt
   - IMN Direct: best-imn-epoch=E-val_auc=A.ckpt
3. Selects the checkpoint with the highest val_auc across all runs.
4. Uploads the chosen checkpoint, together with that run's args.yaml and
   metrics file (validation_metrics.csv for GradCAM, metrics.csv for IMN Direct)
   into the specified Hugging Face model repo under:
   <category>/<subcategory>/

Default repo: SEARCH-IHI/mesomorphicECG
  (see: https://huggingface.co/SEARCH-IHI/mesomorphicECG)

Authentication:
- Either run `huggingface-cli login` beforehand, OR
- Set the environment variable HUGGINGFACE_HUB_TOKEN or HF_TOKEN, OR
- Put HUGGINGFACE_HUB_TOKEN or HF_TOKEN in a `.env` file in the project root
  (it will be loaded automatically via `python-dotenv`), OR
- Pass --token YOUR_TOKEN on the command line.

Usage examples:
  python upload_gradcam_imn_direct_checkpoints_to_hf.py --dry-run
  python upload_gradcam_imn_direct_checkpoints_to_hf.py --repo-id SEARCH-IHI/mesomorphicECG
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from huggingface_hub import HfApi, CommitOperationAdd


PROJECT_ROOT = "/work/vajira/DL2025/mesormorphic_ecg"


# Local base directories for each main category, keyed by the path
# that will be used inside the Hugging Face repo.
# Each entry: (local_path, ckpt_pattern_regex, metrics_filename)
CATEGORY_CONFIG: Dict[str, Tuple[str, re.Pattern, str]] = {
    # 1. GradCAM 100Hz
    "gradcam_100hz": (
        os.path.join(PROJECT_ROOT, "runs", "baseline_2d_gradcam_100Hz"),
        re.compile(
            r"^best-epoch=(?P<epoch>\d+)-val_auc=(?P<val_auc>[0-9.]+)\.ckpt$"
        ),
        "validation_metrics.csv",
    ),
    # 2. GradCAM 500Hz
    "gradcam_500hz": (
        os.path.join(PROJECT_ROOT, "runs", "baseline_2d_gradcam"),
        re.compile(
            r"^best-epoch=(?P<epoch>\d+)-val_auc=(?P<val_auc>[0-9.]+)\.ckpt$"
        ),
        "validation_metrics.csv",
    ),
    # 3. IMN Direct
    "imn_direct": (
        os.path.join(PROJECT_ROOT, "runs", "imn_ecg_simple_implementation_GM_Basic"),
        re.compile(
            r"^best-imn-epoch=(?P<epoch>\d+)-val_auc=(?P<val_auc>[0-9.]+)\.ckpt$"
        ),
        "metrics.csv",
    ),
}

# Subcategories under each main category.
SUBCATEGORIES: List[str] = ["norm_vs_cd", "norm_vs_hyp", "norm_vs_mi", "norm_vs_sttc"]


@dataclass
class BestCheckpoint:
    """Information about the best checkpoint for a (category, subcategory)."""

    category_key: str
    subcategory: str
    local_path: str
    epoch: int
    val_auc: float
    run_dir: str
    metrics_filename: str


def find_best_checkpoint_in_run(
    run_dir: str, ckpt_pattern: re.Pattern
) -> Optional[Tuple[str, int, float]]:
    """
    Find the best checkpoint in a single run directory.

    Returns:
        (checkpoint_path, epoch, val_auc) or None if not found.
    """
    best: Optional[Tuple[str, int, float]] = None

    for entry in os.listdir(run_dir):
        full_path = os.path.join(run_dir, entry)
        if not os.path.isfile(full_path):
            continue

        match = ckpt_pattern.match(entry)
        if not match:
            continue

        epoch = int(match.group("epoch"))
        val_auc = float(match.group("val_auc"))

        if best is None or val_auc > best[2]:
            best = (full_path, epoch, val_auc)

    return best


def find_best_checkpoint_for_subcategory(
    base_dir: str,
    category_key: str,
    subcategory: str,
    ckpt_pattern: re.Pattern,
    metrics_filename: str,
) -> Optional[BestCheckpoint]:
    """
    Find the best checkpoint for a given (category, subcategory) across all runs.
    """
    subcat_dir = os.path.join(base_dir, subcategory)
    if not os.path.isdir(subcat_dir):
        print(f"[WARN] Missing subcategory directory: {subcat_dir}")
        return None

    best_overall: Optional[BestCheckpoint] = None

    for run_name in sorted(os.listdir(subcat_dir)):
        run_dir = os.path.join(subcat_dir, run_name)
        if not os.path.isdir(run_dir):
            continue

        found = find_best_checkpoint_in_run(run_dir, ckpt_pattern)
        if not found:
            continue

        ckpt_path, epoch, val_auc = found
        candidate = BestCheckpoint(
            category_key=category_key,
            subcategory=subcategory,
            local_path=ckpt_path,
            epoch=epoch,
            val_auc=val_auc,
            run_dir=run_dir,
            metrics_filename=metrics_filename,
        )

        if best_overall is None or candidate.val_auc > best_overall.val_auc:
            best_overall = candidate

    if best_overall is None:
        print(
            f"[WARN] No best checkpoint found for "
            f"{category_key}/{subcategory} under {subcat_dir}"
        )

    return best_overall


def discover_all_best_checkpoints() -> List[BestCheckpoint]:
    """
    Discover best checkpoints for all (category, subcategory) combinations.
    """
    results: List[BestCheckpoint] = []

    for category_key, (base_dir, ckpt_pattern, metrics_filename) in CATEGORY_CONFIG.items():
        if not os.path.isdir(base_dir):
            print(f"[WARN] Base directory does not exist: {base_dir}")
            continue

        for subcategory in SUBCATEGORIES:
            best = find_best_checkpoint_for_subcategory(
                base_dir=base_dir,
                category_key=category_key,
                subcategory=subcategory,
                ckpt_pattern=ckpt_pattern,
                metrics_filename=metrics_filename,
            )
            if best is not None:
                results.append(best)

    return results


def build_commit_operations(
    best_checkpoints: List[BestCheckpoint],
) -> List[CommitOperationAdd]:
    """
    Build a list of CommitOperationAdd objects for uploading to the Hub.

    For each best checkpoint, we include:
      - the checkpoint file itself
      - args.yaml from the same run directory (if present)
      - metrics file (validation_metrics.csv or metrics.csv) from the same run (if present)
    """
    operations: List[CommitOperationAdd] = []

    for bc in best_checkpoints:
        category_dir = bc.category_key
        subcat_dir = bc.subcategory

        # 1) Checkpoint file
        ckpt_repo_path = os.path.join(
            category_dir, subcat_dir, os.path.basename(bc.local_path)
        )
        if os.path.isfile(bc.local_path):
            operations.append(
                CommitOperationAdd(
                    path_in_repo=ckpt_repo_path,
                    path_or_fileobj=bc.local_path,
                )
            )
        else:
            print(f"[WARN] Checkpoint file missing on disk: {bc.local_path}")

        # 2) args.yaml (optional)
        args_path = os.path.join(bc.run_dir, "args.yaml")
        if os.path.isfile(args_path):
            args_repo_path = os.path.join(category_dir, subcat_dir, "args.yaml")
            operations.append(
                CommitOperationAdd(
                    path_in_repo=args_repo_path,
                    path_or_fileobj=args_path,
                )
            )
        else:
            print(f"[WARN] args.yaml not found in run dir: {bc.run_dir}")

        # 3) metrics file (optional)
        metrics_path = os.path.join(bc.run_dir, bc.metrics_filename)
        if os.path.isfile(metrics_path):
            metrics_repo_path = os.path.join(
                category_dir, subcat_dir, bc.metrics_filename
            )
            operations.append(
                CommitOperationAdd(
                    path_in_repo=metrics_repo_path,
                    path_or_fileobj=metrics_path,
                )
            )
        else:
            print(
                f"[WARN] {bc.metrics_filename} not found in run dir: {bc.run_dir}"
            )

    return operations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload best GradCAM and IMN Direct checkpoints to Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="SEARCH-IHI/mesomorphicECG",
        help="Hugging Face model repo id (e.g. 'SEARCH-IHI/mesomorphicECG').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help=(
            "Hugging Face access token. If omitted, the token will be taken "
            "from the environment or the local Hugging Face cache."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be uploaded without modifying the repo.",
    )

    args = parser.parse_args()

    # Load .env from the project root, if present, so that HF_TOKEN or
    # HUGGINGFACE_HUB_TOKEN defined there are available. Uses python-dotenv.
    env_path = os.path.join(PROJECT_ROOT, ".env")
    load_dotenv(env_path, override=False)

    print("Discovering best checkpoints across categories and subcategories...")
    best_checkpoints = discover_all_best_checkpoints()

    if not best_checkpoints:
        print("[ERROR] No best checkpoints found. Nothing to upload.")
        return

    print("\nBest checkpoints selected:")
    for bc in best_checkpoints:
        print(
            f"  - {bc.category_key}/{bc.subcategory}: "
            f"{os.path.basename(bc.local_path)} "
            f"(epoch={bc.epoch}, val_auc={bc.val_auc:.4f})"
        )

    operations = build_commit_operations(best_checkpoints)

    if not operations:
        print("[ERROR] No files to upload (no commit operations built).")
        return

    if args.dry_run:
        print(
            f"\n[DRY RUN] Would create a commit to repo '{args.repo_id}' with "
            f"{len(operations)} file additions."
        )
        for op in operations:
            print(f"    add {op.path_in_repo}")
        return

    print(
        f"\nCreating commit on Hugging Face Hub in repo '{args.repo_id}' with "
        f"{len(operations)} file additions..."
    )

    # Resolve token precedence:
    # 1) --token argument
    # 2) HUGGINGFACE_HUB_TOKEN from env/.env
    # 3) HF_TOKEN from env/.env
    token = args.token
    if token is None:
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    api = HfApi(token=token) if token is not None else HfApi()
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Add best GradCAM (100Hz, 500Hz) and IMN Direct checkpoints.",
    )

    print("Upload complete.")


if __name__ == "__main__":
    main()
