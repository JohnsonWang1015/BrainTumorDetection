"""Evaluate the MONAI 3D IDH classifier on the held-out test split.

Mirrors eval_idh.py but uses 3D tumor ROI cropping + DenseNet121 3D.
Reports case-level AUC + confusion (slice-level is meaningless here
because each case maps to a single 3D crop).

Outputs
-------
- AUC + classification report printed
- Confusion matrix PNG  -> outputs/eval_idh_monai_confusion.png
- ROC curve PNG         -> outputs/eval_idh_monai_roc.png
- Per-case CSV          -> outputs/eval_idh_monai_cases.csv

Usage::

    uv run eval-idh-monai
    uv run eval-idh-monai --ckpt checkpoints/densenet3d_idh.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from idh_glioma.utils import load_json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _zscore(x):
    m, s = x.mean(), x.std()
    return (x - m) / (s + 1e-6)


def _tumor_bbox_3d(mask, margin=4):
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        return None
    mn = nz.min(axis=0)
    mx = nz.max(axis=0) + 1
    h, w, d = mask.shape
    return (
        max(int(mn[0]) - margin, 0),
        min(int(mx[0]) + margin, h),
        max(int(mn[1]) - margin, 0),
        min(int(mx[1]) + margin, w),
        max(int(mn[2]) - margin, 0),
        min(int(mx[2]) + margin, d),
    )


@torch.inference_mode()
def _predict_case(record, model, device, target_size, margin):
    mods = record["modalities"]
    flair = nib.load(mods["flair"]).get_fdata().astype(np.float32)
    t1 = nib.load(mods["t1"]).get_fdata().astype(np.float32)
    t1gd = nib.load(mods["t1Gd"]).get_fdata().astype(np.float32)
    t2 = nib.load(mods["t2"]).get_fdata().astype(np.float32)
    mask = (nib.load(record["mask_path"]).get_fdata() > 0).astype(np.float32)

    bbox = _tumor_bbox_3d(mask, margin=margin)
    if bbox is None:
        return float("nan")
    y0, y1, x0, x1, z0, z1 = bbox
    flair = _zscore(flair[y0:y1, x0:x1, z0:z1])
    t1 = _zscore(t1[y0:y1, x0:x1, z0:z1])
    t1gd = _zscore(t1gd[y0:y1, x0:x1, z0:z1])
    t2 = _zscore(t2[y0:y1, x0:x1, z0:z1])
    vol = np.stack([flair, t1, t1gd, t2], axis=0)

    t = torch.from_numpy(vol).unsqueeze(0)
    t = F.interpolate(t, size=target_size, mode="trilinear", align_corners=False)
    t = t.to(device)
    if device.type == "cuda":
        t = t.contiguous(memory_format=torch.channels_last_3d)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logit = model(t)
    return float(torch.sigmoid(logit.float()).cpu().item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/densenet3d_idh.pt"))
    p.add_argument("--target-size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--margin", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/eval_idh_monai"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_json(args.manifest)
    records = [r for r in manifest["test"] if r.get("idh_label") is not None]
    if not records:
        raise ValueError("Test split has no labelled cases.")

    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    target_size = tuple(state.get("target_size", args.target_size))
    margin = int(state.get("margin", args.margin))
    threshold = float(state.get("threshold", args.threshold))
    print(f"[eval-idh-monai] target_size={target_size} margin={margin} threshold={threshold:.4f}")

    model = DenseNet121(spatial_dims=3, in_channels=4, out_channels=1, dropout_prob=0.2).to(device)
    model.load_state_dict(state["model"])
    model.train(False)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)

    case_results = []
    for r in tqdm(records, desc="eval-idh-monai"):
        prob = _predict_case(r, model, device, target_size, margin)
        case_results.append(
            {
                "case_id": r.get("case_id", ""),
                "label": int(r["idh_label"]),
                "case_prob": prob,
                "case_pred": int(prob >= threshold),
            }
        )

    probs = np.array([r["case_prob"] for r in case_results])
    labels = np.array([r["label"] for r in case_results])
    preds = (probs >= threshold).astype(int)

    print(f"\n{'=' * 60}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Test cases  : {len(case_results)} ({int(labels.sum())} mutant, {int((1-labels).sum())} WT)")
    print(f"{'=' * 60}")

    if len(np.unique(labels)) >= 2:
        auc = roc_auc_score(labels, probs)
        print(f"\n[CASE-LEVEL]  AUC = {auc:.4f}")
        print(classification_report(labels, preds, target_names=["WT", "Mutant"], digits=4, zero_division=0))
    else:
        auc = float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "eval_idh_monai_cases.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label", "case_pred", "case_prob"])
        w.writeheader()
        for r in case_results:
            w.writerow({**r, "case_prob": f"{r['case_prob']:.6f}"})
    print(f"Per-case CSV -> {csv_path}")

    if _HAS_MPL and not np.isnan(auc):
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["WT", "Mutant"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"3D IDH Classifier (AUC={auc:.3f})")
        fig.tight_layout()
        fig.savefig(args.output_dir / "eval_idh_monai_confusion.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(labels, probs, ax=ax, name=f"3D DenseNet (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC -- 3D IDH classifier")
        fig.tight_layout()
        fig.savefig(args.output_dir / "eval_idh_monai_roc.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
