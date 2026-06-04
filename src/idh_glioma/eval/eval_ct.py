"""Evaluate the CT/MRI binary tumor classifier on the held-out test split.

Outputs
-------
- Accuracy, Precision, Recall, F1, AUC-ROC printed to stdout
- Confusion matrix PNG  →  outputs/eval_ct_confusion.png
- ROC curve PNG         →  outputs/eval_ct_roc.png
- Per-image predictions →  outputs/eval_ct_predictions.csv

Usage::

    uv run eval-ct
    uv run eval-ct --modality ct --ckpt checkpoints/mobilenetv3_ct_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from idh_glioma.data.ct_datasets import BrainImageDataset
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


@torch.inference_mode()
def evaluate(
    manifest: Path,
    ckpt: Path,
    modality: str,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    img_size: int,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = BrainImageDataset(manifest, split="test", modality=modality, img_size=img_size, augment=False)
    if len(ds) == 0:
        raise ValueError("Test split is empty — check manifest path and modality filter.")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = build_mobilenetv3_binary(num_input_channels=3).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    all_probs: list[float] = []
    all_labels: list[int] = []

    for images, labels in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().reshape(-1).astype(int).tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= 0.5).astype(int)

    acc = accuracy_score(labels_arr, preds_arr)
    auc = roc_auc_score(labels_arr, probs_arr)
    report = classification_report(labels_arr, preds_arr, target_names=["Healthy", "Tumor"], digits=4)

    print(f"\n{'=' * 50}")
    print(f"Checkpoint : {ckpt}")
    print(f"Test images: {len(ds)}  (modality={modality})")
    print(f"{'=' * 50}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"AUC-ROC    : {auc:.4f}")
    print()
    print(report)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    import csv
    csv_path = output_dir / "eval_ct_predictions.csv"
    paths = [ds.records[i]["path"] for i in range(len(ds))]
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "pred", "prob_tumor"])
        for p, lbl, pred, prob in zip(paths, labels_arr, preds_arr, probs_arr):
            writer.writerow([p, lbl, pred, f"{prob:.6f}"])
    print(f"Predictions → {csv_path}")

    if _HAS_MPL:
        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(labels_arr, preds_arr)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Tumor"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"CT/MRI Classifier  (acc={acc:.3f})")
        fig.tight_layout()
        cm_path = output_dir / "eval_ct_confusion.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix → {cm_path}")

        # ROC curve
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(labels_arr, probs_arr, ax=ax, name=f"MobileNetV3 (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC Curve — CT/MRI Tumor Detection")
        fig.tight_layout()
        roc_path = output_dir / "eval_ct_roc.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"ROC curve        → {roc_path}")

    return {"accuracy": acc, "auc_roc": auc}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CT/MRI binary tumor classifier")
    p.add_argument("--manifest", type=Path, default=Path("artifacts/ct_manifest.json"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/mobilenetv3_ct_best.pt"))
    p.add_argument("--modality", choices=["ct", "mri", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        args.manifest,
        args.ckpt,
        args.modality,
        args.batch_size,
        args.num_workers,
        args.output_dir,
        args.img_size,
    )


if __name__ == "__main__":
    main()
