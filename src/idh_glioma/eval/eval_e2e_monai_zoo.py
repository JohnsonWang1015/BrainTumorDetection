"""End-to-end evaluation: MONAI Model Zoo bundle (seg) + 3D DenseNet (cls).

Computes both segmentation Dice (vs ground truth) and IDH classification
metrics (AUC, classification report) across the test split, with the
calibrated threshold from the classifier checkpoint.

Outputs
-------
- Per-case CSV   -> outputs/eval_e2e_zoo_cases.csv
- Confusion PNG  -> outputs/eval_e2e_zoo_confusion.png
- ROC PNG        -> outputs/eval_e2e_zoo_roc.png

Usage::

    uv run eval-e2e-zoo
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import DenseNet121, SegResNet
from monai.transforms import KeepLargestConnectedComponent, NormalizeIntensity
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


def _bbox3d(mask, margin=4):
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        return None
    mn, mx = nz.min(0), nz.max(0) + 1
    h, w, d = mask.shape
    return (
        max(int(mn[0]) - margin, 0), min(int(mx[0]) + margin, h),
        max(int(mn[1]) - margin, 0), min(int(mx[1]) + margin, w),
        max(int(mn[2]) - margin, 0), min(int(mx[2]) + margin, d),
    )


def _zsc(c):
    return (c - c.mean()) / (c.std() + 1e-6)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument(
        "--bundle-ckpt", type=Path,
        default=Path("checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt"),
    )
    p.add_argument(
        "--cls-ckpt", type=Path,
        default=Path("checkpoints/densenet3d_idh_jitter.pt"),
    )
    p.add_argument("--seg-roi-size", type=int, nargs=3, default=(240, 240, 160))
    p.add_argument("--cls-target-size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--cls-margin", type=int, default=4)
    p.add_argument("--seg-overlap", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/eval_e2e_zoo"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    seg = SegResNet(
        spatial_dims=3, in_channels=4, out_channels=3, init_filters=16,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
    ).to(device)
    state = torch.load(args.bundle_ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    seg.load_state_dict(state)
    seg.train(False)

    cls_state = torch.load(args.cls_ckpt, map_location=device, weights_only=True)
    cls = DenseNet121(spatial_dims=3, in_channels=4, out_channels=1, dropout_prob=0.2).to(device)
    cls.load_state_dict(cls_state["model"])
    cls.train(False)
    target_size = tuple(cls_state.get("target_size", args.cls_target_size))
    margin = int(cls_state.get("margin", args.cls_margin))
    threshold = float(cls_state.get("threshold", 0.5))
    print(
        f"[eval-e2e-zoo] cls cfg: target_size={target_size} margin={margin} "
        f"threshold={threshold:.4f}"
    )

    keep_largest = KeepLargestConnectedComponent(applied_labels=[1])
    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)

    manifest = load_json(args.manifest)
    records = [r for r in manifest["test"] if r.get("idh_label") is not None]
    if not records:
        raise ValueError("Test split has no labelled cases.")

    case_results = []
    for r in tqdm(records, desc="eval-e2e-zoo"):
        flair = nib.load(r["modalities"]["flair"]).get_fdata().astype(np.float32)
        t1 = nib.load(r["modalities"]["t1"]).get_fdata().astype(np.float32)
        t1gd = nib.load(r["modalities"]["t1Gd"]).get_fdata().astype(np.float32)
        t2 = nib.load(r["modalities"]["t2"]).get_fdata().astype(np.float32)
        gt = (nib.load(r["mask_path"]).get_fdata() > 0).astype(np.uint8)

        # Segmentation pass with bundle (channel order T1c, T1, T2, FLAIR)
        img4 = np.stack([t1gd, t1, t2, flair], axis=0)
        img_t = normalize(torch.from_numpy(img4)).unsqueeze(0).to(device).float()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            seg_logits = sliding_window_inference(
                img_t, roi_size=tuple(args.seg_roi_size), sw_batch_size=1,
                predictor=seg, overlap=args.seg_overlap,
            )
        pp = torch.sigmoid(seg_logits).cpu().numpy()[0]
        raw = (pp[1] > 0.5).astype(np.uint8)  # WT channel
        pred_mask = keep_largest(torch.from_numpy(raw).unsqueeze(0)).squeeze(0).numpy().astype(np.uint8)

        inter = (pred_mask * gt).sum()
        dice = (2.0 * inter + 1e-6) / (pred_mask.sum() + gt.sum() + 1e-6)

        # Classification pass on predicted-mask ROI
        bb = _bbox3d(pred_mask, margin=margin)
        if bb is None:
            idh_prob = float("nan")
        else:
            f, t1c, tg, t2c = (
                _zsc(v[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]])
                for v in (flair, t1, t1gd, t2)
            )
            crop = np.stack([f, t1c, tg, t2c], axis=0)
            ct = torch.from_numpy(crop).unsqueeze(0)
            ct = F.interpolate(ct, size=target_size, mode="trilinear", align_corners=False).to(device)
            if device.type == "cuda":
                ct = ct.contiguous(memory_format=torch.channels_last_3d)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logit = cls(ct)
            idh_prob = float(torch.sigmoid(logit.float()).cpu().item())

        case_results.append({
            "case_id": r.get("case_id", ""),
            "label": int(r["idh_label"]),
            "case_prob": idh_prob,
            "case_pred": int(idh_prob >= threshold) if not np.isnan(idh_prob) else -1,
            "seg_dice": float(dice),
            "pred_voxels": int(pred_mask.sum()),
            "gt_voxels": int(gt.sum()),
        })

    dices = np.array([r["seg_dice"] for r in case_results])
    probs = np.array([r["case_prob"] for r in case_results])
    labels = np.array([r["label"] for r in case_results])
    preds = np.array([r["case_pred"] for r in case_results])

    print(f"\n{'=' * 60}")
    print(f"Bundle      : {args.bundle_ckpt}")
    print(f"Classifier  : {args.cls_ckpt}")
    print(f"Threshold   : {threshold:.4f}")
    print(f"Test cases  : {len(records)}")
    print(f"{'=' * 60}")
    print(f"[Seg] Dice  mean : {dices.mean():.4f} +/- {dices.std():.4f}")
    print(f"[Seg] Dice median: {np.median(dices):.4f}")
    if len(np.unique(labels)) >= 2 and not np.isnan(probs).all():
        auc = roc_auc_score(labels, probs)
        print(f"[Cls] AUC        : {auc:.4f}")
        print(classification_report(labels, preds, target_names=["WT", "Mutant"], digits=4, zero_division=0))
    else:
        auc = float("nan")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "eval_e2e_zoo_cases.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label", "case_pred", "case_prob", "seg_dice", "pred_voxels", "gt_voxels"])
        w.writeheader()
        for r in case_results:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"Per-case CSV -> {csv_path}")

    if _HAS_MPL and not np.isnan(auc):
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["WT", "Mutant"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"E2E (bundle + jitter) AUC={auc:.3f}")
        fig.tight_layout()
        fig.savefig(args.output_dir / "eval_e2e_zoo_confusion.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 5))
        RocCurveDisplay.from_predictions(labels, probs, ax=ax, name=f"E2E (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC -- end-to-end MONAI pipeline")
        fig.tight_layout()
        fig.savefig(args.output_dir / "eval_e2e_zoo_roc.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
