"""5-fold stratified cross-validation for the IDH classifier.

Pools every labelled case from ``artifacts/manifest.json`` (across train / val
/ test splits), splits them into 5 stratified folds by IDH label, and trains
a fresh model on each fold. Reports mean ± std of best smoothed val AUC,
which is more robust than the single-split numbers reported in the PR.

Each fold runs as a subprocess call to ``uv run train-idh`` with a temp
manifest, so the training pipeline is exactly the same recipe as production.

Usage::

    uv run python scripts/cv_idh.py --output-dir artifacts/cv

Resumability: skips folds whose checkpoint already exists. Delete
``artifacts/cv/`` to start over.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from idh_glioma.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cv"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--profile", default="a6000")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.manifest)
    all_records: list[dict] = []
    for split in ("train", "val", "test"):
        all_records.extend(
            r for r in manifest.get(split, []) if r.get("idh_label") is not None
        )

    labels = np.array([int(r["idh_label"]) for r in all_records])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"Total labelled cases: {len(all_records)} (WT={n_neg}, Mutant={n_pos})")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_results: list[dict] = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(all_records, labels)):
        train_recs = [all_records[i] for i in train_idx]
        val_recs = [all_records[i] for i in val_idx]
        train_lbls = [int(r["idh_label"]) for r in train_recs]
        val_lbls = [int(r["idh_label"]) for r in val_recs]
        print(f"\n=== Fold {fold_i + 1}/{args.n_folds} ===")
        print(
            f"  train: {len(train_recs)} (WT={train_lbls.count(0)}, "
            f"Mut={train_lbls.count(1)})"
        )
        print(
            f"  val:   {len(val_recs)} (WT={val_lbls.count(0)}, "
            f"Mut={val_lbls.count(1)})"
        )

        fold_manifest = {"train": train_recs, "val": val_recs, "test": []}
        manifest_path = args.output_dir / f"cv_fold_{fold_i}.json"
        manifest_path.write_text(json.dumps(fold_manifest))

        ckpt_path = args.output_dir / f"cv_fold_{fold_i}.pt"
        if ckpt_path.exists():
            print(f"  ckpt exists -> skipping training.")
        else:
            cmd = [
                "uv", "run", "train-idh",
                "--manifest", str(manifest_path),
                "--epochs", str(args.epochs),
                "--profile", args.profile,
                "--output", str(ckpt_path),
            ]
            print(f"  cmd: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        smoothed_auc = float(state.get("smoothed_auc", float("nan")))
        val_auc = float(state.get("val_auc", float("nan")))
        epoch = int(state.get("epoch", -1))
        print(
            f"  best: smoothed AUC {smoothed_auc:.4f} "
            f"(raw {val_auc:.4f}) @ epoch {epoch}"
        )
        fold_results.append(
            {
                "fold": fold_i,
                "smoothed_auc": smoothed_auc,
                "val_auc": val_auc,
                "epoch": epoch,
                "n_train": len(train_recs),
                "n_val": len(val_recs),
                "train_class_balance": [train_lbls.count(0), train_lbls.count(1)],
                "val_class_balance": [val_lbls.count(0), val_lbls.count(1)],
            }
        )

    smoothed_aucs = np.array([r["smoothed_auc"] for r in fold_results])
    raw_aucs = np.array([r["val_auc"] for r in fold_results])
    print(f"\n=== {args.n_folds}-fold CV summary ===")
    print(f"Smoothed val AUC: {smoothed_aucs.mean():.4f} ± {smoothed_aucs.std():.4f}")
    print(f"Raw val AUC:      {raw_aucs.mean():.4f} ± {raw_aucs.std():.4f}")
    print(f"Per-fold smoothed: {[f'{a:.3f}' for a in smoothed_aucs]}")

    summary_path = args.output_dir / "cv_results.json"
    summary_path.write_text(
        json.dumps(
            {
                "folds": fold_results,
                "smoothed_auc_mean": float(smoothed_aucs.mean()),
                "smoothed_auc_std": float(smoothed_aucs.std()),
                "raw_auc_mean": float(raw_aucs.mean()),
                "raw_auc_std": float(raw_aucs.std()),
            },
            indent=2,
        )
    )
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()
