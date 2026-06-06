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


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    """Expected Calibration Error for a binary classifier.

    ``probs`` are P(tumor). For each sample the model's *confidence* is the
    probability of its predicted class (``max(p, 1-p)``). Samples are grouped
    into ``n_bins`` equal-width confidence bins; ECE is the population-weighted
    gap between accuracy and mean confidence per bin:

        ECE = Σ_b (|b| / N) · |acc(b) − conf(b)|

    Returns ``(ece, bins)`` where ``bins`` carries per-bin stats for a
    reliability diagram.
    """
    preds = (probs >= 0.5).astype(int)
    confidence = np.maximum(probs, 1.0 - probs)
    correct = (preds == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    bins: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        # Bin 0 includes its left edge; every bin is closed on the right so the
        # boundary value (and confidence == 1.0 in the last bin) is counted once.
        if i == 0:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0, "acc": 0.0, "conf": 0.0})
            continue
        bin_acc = float(correct[in_bin].mean())
        bin_conf = float(confidence[in_bin].mean())
        ece += (count / n) * abs(bin_acc - bin_conf)
        bins.append(
            {"lo": float(lo), "hi": float(hi), "count": count, "acc": bin_acc, "conf": bin_conf}
        )
    return float(ece), bins


@torch.inference_mode()
def evaluate(
    manifest: Path,
    ckpt: Path,
    modality: str,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    img_size: int,
    tta: bool = False,
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
    # Checkpoints are self-describing; legacy ones default to the Small backbone.
    variant = state.get("variant", "small")
    # Temperature scaling (Guo et al. 2017): p = sigmoid(logit / T). T>0 leaves
    # the 0.5 decision unchanged (accuracy/AUC fixed) and only softens confidence.
    temperature = float(state.get("temperature", 1.0))
    model = build_mobilenetv3_binary(num_input_channels=3, variant=variant).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded {variant} backbone  (TTA={'on' if tta else 'off'}, T={temperature:.3f})")

    all_probs: list[float] = []
    all_labels: list[int] = []

    for images, labels in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if tta:
            # Average logits with the horizontal-flip view (brain L/R symmetry).
            logits = 0.5 * (logits + model(torch.flip(images, dims=[3])))
        probs = torch.sigmoid(logits / temperature).cpu().numpy().reshape(-1)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().reshape(-1).astype(int).tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= 0.5).astype(int)

    acc = accuracy_score(labels_arr, preds_arr)
    auc = roc_auc_score(labels_arr, probs_arr)
    ece, ece_bins = expected_calibration_error(probs_arr, labels_arr, n_bins=15)
    report = classification_report(labels_arr, preds_arr, target_names=["Healthy", "Tumor"], digits=4)

    print(f"\n{'=' * 50}")
    print(f"Checkpoint : {ckpt}")
    print(f"Test images: {len(ds)}  (modality={modality})")
    print(f"{'=' * 50}")
    print(f"Accuracy   : {acc:.4f}")
    print(f"AUC-ROC    : {auc:.4f}")
    print(f"ECE (15-bin): {ece:.4f}   (lower = better calibrated)")
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

        # Reliability diagram — per-bin accuracy vs confidence. The diagonal is
        # perfect calibration; bars below it = overconfident, above = under.
        populated = [b for b in ece_bins if b["count"] > 0]
        if populated:
            centers = [(b["lo"] + b["hi"]) / 2 for b in populated]
            accs = [b["acc"] for b in populated]
            confs = [b["conf"] for b in populated]
            width = 1.0 / len(ece_bins)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.bar(centers, accs, width=width * 0.9, edgecolor="black",
                   alpha=0.75, label="accuracy")
            ax.plot(confs, accs, "o-", color="C1", lw=1, ms=4, label="confidence")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Reliability Diagram  (ECE={ece:.4f})")
            ax.legend(loc="upper left")
            fig.tight_layout()
            rel_path = output_dir / "eval_ct_reliability.png"
            fig.savefig(rel_path, dpi=150)
            plt.close(fig)
            print(f"Reliability      → {rel_path}")

    return {"accuracy": acc, "auc_roc": auc, "ece": ece}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CT/MRI binary tumor classifier")
    p.add_argument("--manifest", type=Path, default=Path("artifacts/ct_manifest.json"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/mobilenetv3_ct_best.pt"))
    p.add_argument("--modality", choices=["ct", "mri", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Test-time augmentation: average original + horizontal-flip logits",
    )
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
        tta=args.tta,
    )


if __name__ == "__main__":
    main()
