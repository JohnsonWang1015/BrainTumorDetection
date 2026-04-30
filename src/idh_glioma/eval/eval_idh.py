"""Evaluate the MobileNetV3 IDH classifier on the held-out test split.

Slice-level aggregation: every tumor-bearing slice in each test case is
scored, then averaged per case to produce a single case-level probability.
Reports both slice-level and case-level metrics.

Outputs
-------
- AUC, classification report, per-case predictions printed to stdout
- Confusion matrix PNG  -> outputs/eval_idh_confusion.png
- ROC curve PNG         -> outputs/eval_idh_roc.png
- Per-case CSV          -> outputs/eval_idh_cases.csv

Usage::

    uv run eval-idh
    uv run eval-idh --ckpt checkpoints/mobilenetv3_idh_best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from idh_glioma.data.datasets import _crop_roi, _tumor_bbox_2d, load_nifti, zscore
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary
from idh_glioma.utils import load_json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

@torch.inference_mode()
def _predict_case(
    record: dict,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
    img_size: int,
    use_roi: bool,
    roi_margin: int,
) -> tuple[float, np.ndarray]:
    """Returns (case_prob, slice_probs) -- mean probability across tumor slices."""
    flair = zscore(load_nifti(record["modalities"]["flair"]))
    t1gd = zscore(load_nifti(record["modalities"]["t1Gd"]))
    t2 = zscore(load_nifti(record["modalities"]["t2"]))
    mask = (load_nifti(record["mask_path"]) > 0).astype(np.float32)

    tumor_z = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    if len(tumor_z) == 0:
        return float("nan"), np.array([])

    slice_probs: list[float] = []
    for start in range(0, len(tumor_z), batch_size):
        batch_z = tumor_z[start : start + batch_size]
        cropped: list[torch.Tensor] = []
        for z in batch_z:
            stacked = np.stack(
                [flair[:, :, z], t1gd[:, :, z], t2[:, :, z]], axis=0
            )  # (3, H, W)
            if use_roi:
                bbox = _tumor_bbox_2d(mask[:, :, z], roi_margin)
                stacked = _crop_roi(stacked, bbox)
            t = torch.from_numpy(stacked.copy()).unsqueeze(0)  # (1, 3, h, w)
            t = F.interpolate(
                t, size=(img_size, img_size), mode="bilinear", align_corners=False
            )
            cropped.append(t.squeeze(0))
        tensor = torch.stack(cropped, dim=0).to(device, non_blocking=True)
        if device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        slice_probs.extend(probs.tolist())

    return float(np.mean(slice_probs)), np.array(slice_probs)


def score_test_split(
    manifest_path: Path,
    ckpt: Path,
    batch_size: int,
    output_dir: Path,
    variant: str,
    img_size: int,
    use_roi: bool,
    roi_margin: int,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    manifest = load_json(manifest_path)
    records = [r for r in manifest["test"] if r.get("idh_label") is not None]
    if not records:
        raise ValueError("Test split has no labelled cases.")

    state = torch.load(ckpt, map_location=device, weights_only=True)
    # Honour the checkpoint's recorded settings when present so eval matches training.
    ckpt_variant = state.get("variant", variant)
    ckpt_use_roi = state.get("use_roi", use_roi)
    ckpt_roi_margin = state.get("roi_margin", roi_margin)
    ckpt_img_size = state.get("img_size", img_size)
    if ckpt_variant != variant or ckpt_use_roi != use_roi:
        print(
            f"[eval-idh] Overriding CLI args with checkpoint metadata: "
            f"variant={ckpt_variant} use_roi={ckpt_use_roi} "
            f"roi_margin={ckpt_roi_margin} img_size={ckpt_img_size}"
        )
    variant = ckpt_variant
    use_roi = ckpt_use_roi
    roi_margin = ckpt_roi_margin
    img_size = ckpt_img_size
    threshold = float(state.get("threshold", 0.5))
    if threshold != 0.5:
        print(
            f"[eval-idh] Using calibrated threshold={threshold:.4f} "
            f"(method={state.get('threshold_method', '?')}, "
            f"split={state.get('threshold_split', '?')})"
        )

    model = build_mobilenetv3_binary(variant=variant).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    case_results: list[dict] = []
    slice_probs_all: list[float] = []
    slice_labels_all: list[int] = []

    for record in tqdm(records, desc="eval-idh"):
        case_prob, slice_probs = _predict_case(
            record, model, device, use_amp, batch_size,
            img_size=img_size, use_roi=use_roi, roi_margin=roi_margin,
        )
        label = int(record["idh_label"])
        case_results.append(
            {
                "case_id": record.get("case_id", ""),
                "label": label,
                "case_prob": case_prob,
                "case_pred": int(case_prob >= threshold),
                "n_slices": len(slice_probs),
            }
        )
        slice_probs_all.extend(slice_probs.tolist())
        slice_labels_all.extend([label] * len(slice_probs))

    case_probs = np.array([r["case_prob"] for r in case_results])
    case_labels = np.array([r["label"] for r in case_results])
    case_preds = (case_probs >= threshold).astype(int)

    slice_probs_arr = np.array(slice_probs_all)
    slice_labels_arr = np.array(slice_labels_all)
    slice_preds_arr = (slice_probs_arr >= threshold).astype(int)

    print(f"\n{'=' * 60}")
    print(f"Checkpoint  : {ckpt}")
    print(f"Test cases  : {len(case_results)} ({int(case_labels.sum())} mutant, {int((1 - case_labels).sum())} WT)")
    print(f"Test slices : {len(slice_probs_all)}")
    print(f"{'=' * 60}")

    if len(np.unique(case_labels)) >= 2:
        case_auc = roc_auc_score(case_labels, case_probs)
        print(f"\n[CASE-LEVEL]  AUC = {case_auc:.4f}")
        print(classification_report(case_labels, case_preds, target_names=["WT", "Mutant"], digits=4, zero_division=0))
    else:
        case_auc = float("nan")
        print("\n[CASE-LEVEL]  AUC undefined (only one class in test split)")

    if len(np.unique(slice_labels_arr)) >= 2:
        slice_auc = roc_auc_score(slice_labels_arr, slice_probs_arr)
        print(f"[SLICE-LEVEL] AUC = {slice_auc:.4f}")
        print(classification_report(slice_labels_arr, slice_preds_arr, target_names=["WT", "Mutant"], digits=4, zero_division=0))
    else:
        slice_auc = float("nan")

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "eval_idh_cases.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "label", "case_pred", "case_prob", "n_slices"])
        writer.writeheader()
        for r in case_results:
            r_out = {**r, "case_prob": f"{r['case_prob']:.6f}"}
            writer.writerow(r_out)
    print(f"\nPer-case CSV -> {csv_path}")

    if _HAS_MPL and not np.isnan(case_auc):
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(case_labels, case_preds, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["WT", "Mutant"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"IDH Classifier (case-level, AUC={case_auc:.3f})")
        fig.tight_layout()
        fig.savefig(output_dir / "eval_idh_confusion.png", dpi=150)
        plt.close(fig)
        print(f"Confusion matrix -> {output_dir / 'eval_idh_confusion.png'}")

        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(case_labels, case_probs, ax=ax, name=f"Case-level (AUC={case_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC Curve -- IDH Mutation Classifier (case-level)")
        fig.tight_layout()
        fig.savefig(output_dir / "eval_idh_roc.png", dpi=150)
        plt.close(fig)
        print(f"ROC curve        -> {output_dir / 'eval_idh_roc.png'}")

    return {"case_auc": float(case_auc), "slice_auc": float(slice_auc)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate IDH classifier on TCGA-LGG test split")
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/mobilenetv3_idh_best.pt"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/eval_idh"))
    p.add_argument(
        "--variant", choices=["small", "large"], default="large",
        help="Backbone size. Overridden by checkpoint metadata if present.",
    )
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument(
        "--use-roi", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--roi-margin", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    score_test_split(
        args.manifest,
        args.ckpt,
        args.batch_size,
        args.output_dir,
        variant=args.variant,
        img_size=args.img_size,
        use_roi=args.use_roi,
        roi_margin=args.roi_margin,
    )


if __name__ == "__main__":
    main()
