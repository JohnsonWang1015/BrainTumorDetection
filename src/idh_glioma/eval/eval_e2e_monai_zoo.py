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
from monai.inferers import sliding_window_inference
from monai.networks.nets import DenseNet121, SegResNet
from monai.transforms import NormalizeIntensity
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tqdm import tqdm

from idh_glioma.infer.e2e_roi import (
    apply_mask_postprocess,
    merge_e2e_config,
    predict_multi_view_idh,
)
from idh_glioma.utils import load_json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

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
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--e2e-config", type=Path, default=Path("artifacts/e2e_idh_config.json"))
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
    threshold = float(cls_state.get("threshold", 0.5))
    base_margin = int(cls_state.get("margin", args.cls_margin))
    e2e_cfg = merge_e2e_config(
        args.e2e_config,
        fallback_threshold=threshold,
        fallback_base_margin=base_margin,
    )
    print(
        f"[eval-e2e-zoo] cls cfg: target_size={target_size} "
        f"base_margin={e2e_cfg['base_margin']} threshold={e2e_cfg['threshold']:.4f} "
        f"aggregation={e2e_cfg['aggregation']} views={e2e_cfg['view_margins']} "
        f"dilate_iters={e2e_cfg['dilate_iters']}"
    )

    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)

    manifest = load_json(args.manifest)
    records = [r for r in manifest[args.split] if r.get("idh_label") is not None]
    if not records:
        raise ValueError(f"{args.split!r} split has no labelled cases.")

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
        pred_mask = apply_mask_postprocess(
            raw,
            keep_largest=bool(e2e_cfg["keep_largest"]),
            dilate_iters=int(e2e_cfg["dilate_iters"]),
        )

        inter = (pred_mask * gt).sum()
        dice = (2.0 * inter + 1e-6) / (pred_mask.sum() + gt.sum() + 1e-6)

        # Classification pass on predicted-mask ROI
        if not pred_mask.any():
            idh_prob = float("nan")
            view_probs: list[float] = []
        else:
            idh_prob, view_probs, _ = predict_multi_view_idh(
                flair=flair,
                t1=t1,
                t1gd=t1gd,
                t2=t2,
                pred_mask=pred_mask,
                cls_model=cls,
                device=device,
                target_size=target_size,
                cfg=e2e_cfg,
                use_amp=use_amp,
            )

        case_results.append({
            "case_id": r.get("case_id", ""),
            "label": int(r["idh_label"]),
            "case_prob": idh_prob,
            "case_pred": int(idh_prob >= float(e2e_cfg["threshold"])) if not np.isnan(idh_prob) else -1,
            "seg_dice": float(dice),
            "pred_voxels": int(pred_mask.sum()),
            "gt_voxels": int(gt.sum()),
            "num_views": len(view_probs),
        })

    dices = np.array([r["seg_dice"] for r in case_results])
    probs = np.array([r["case_prob"] for r in case_results])
    labels = np.array([r["label"] for r in case_results])
    preds = np.array([r["case_pred"] for r in case_results])

    print(f"\n{'=' * 60}")
    print(f"Bundle      : {args.bundle_ckpt}")
    print(f"Classifier  : {args.cls_ckpt}")
    print(f"E2E config  : {args.e2e_config}")
    print(f"Threshold   : {float(e2e_cfg['threshold']):.4f}")
    print(f"Split       : {args.split}")
    print(f"Cases       : {len(records)}")
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
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "label",
                "case_pred",
                "case_prob",
                "seg_dice",
                "pred_voxels",
                "gt_voxels",
                "num_views",
            ],
        )
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
