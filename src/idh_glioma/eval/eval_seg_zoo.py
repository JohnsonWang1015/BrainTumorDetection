"""Evaluate the MONAI Model Zoo `brats_mri_segmentation` bundle on TCGA-LGG.

Zero-shot: no fine-tuning. Uses the bundle's exact architecture (SegResNet,
init_filters=16, out_channels=3 for TC/WT/ET) and preprocessing (just
NormalizeIntensityd, channel_wise + nonzero). The bundle was trained on
the BraTS challenge cohort which is in the same coordinate space and
spacing as BraTS-TCGA-LGG, so no Orientationd / Spacingd is needed.

Channel order in the bundle: T1c, T1, T2, FLAIR (so we feed t1Gd, t1, t2,
flair). Output channel 1 is whole-tumor; we report Dice on that channel
against the binarised ground-truth mask.

Usage::

    uv run eval-seg-zoo
    uv run eval-seg-zoo --bundle-ckpt checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import NormalizeIntensity
from tqdm import tqdm

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
        "--bundle-ckpt",
        type=Path,
        default=Path("checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt"),
    )
    p.add_argument("--roi-size", type=int, nargs=3, default=(240, 240, 160))
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegResNet(
        spatial_dims=3, in_channels=4, out_channels=3, init_filters=16,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
    ).to(device)
    state = torch.load(args.bundle_ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.train(False)

    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)
    manifest = load_json(args.manifest)
    records = manifest["test"]
    if not records:
        raise ValueError("Test split is empty.")

    case_results = []
    for r in tqdm(records, desc="eval-seg-zoo"):
        flair = nib.load(r["modalities"]["flair"]).get_fdata().astype(np.float32)
        t1 = nib.load(r["modalities"]["t1"]).get_fdata().astype(np.float32)
        t1gd = nib.load(r["modalities"]["t1Gd"]).get_fdata().astype(np.float32)
        t2 = nib.load(r["modalities"]["t2"]).get_fdata().astype(np.float32)
        gt_mask = (nib.load(r["mask_path"]).get_fdata() > 0).astype(np.uint8)

        # Bundle channel order: T1c, T1, T2, FLAIR
        img4 = np.stack([t1gd, t1, t2, flair], axis=0)
        img_t = torch.from_numpy(img4)
        img_t = normalize(img_t).unsqueeze(0).to(device).float()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = sliding_window_inference(
                img_t, roi_size=tuple(args.roi_size), sw_batch_size=args.sw_batch_size,
                predictor=model, overlap=args.overlap,
            )
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (3, H, W, D)
        pred_wt = (probs[1] > 0.5).astype(np.uint8)

        inter = (pred_wt * gt_mask).sum()
        dice = (2.0 * inter + 1e-6) / (pred_wt.sum() + gt_mask.sum() + 1e-6)
        union = pred_wt.sum() + gt_mask.sum() - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        case_results.append(
            {
                "case_id": r.get("case_id", ""),
                "dice": float(dice),
                "iou": float(iou),
                "pred_voxels": int(pred_wt.sum()),
                "gt_voxels": int(gt_mask.sum()),
            }
        )

    dices = np.array([r["dice"] for r in case_results])
    ious = np.array([r["iou"] for r in case_results])
    print(f"\n{'=' * 60}")
    print(f"Bundle      : {args.bundle_ckpt}")
    print(f"Test cases  : {len(case_results)}")
    print(f"{'=' * 60}")
    print(f"WT Dice  mean : {dices.mean():.4f} +/- {dices.std():.4f}")
    print(f"WT Dice  median: {np.median(dices):.4f}")
    print(f"WT IoU   mean : {ious.mean():.4f} +/- {ious.std():.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "eval_seg_zoo_cases.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "dice", "iou", "pred_voxels", "gt_voxels"])
        w.writeheader()
        for r in case_results:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"Per-case CSV -> {csv_path}")

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dices, bins=20, edgecolor="white", color="seagreen")
        ax.axvline(dices.mean(), color="tomato", lw=2, label=f"mean={dices.mean():.3f}")
        ax.set_xlabel("WT Dice")
        ax.set_ylabel("Cases")
        ax.set_title("MONAI Model Zoo (zero-shot) -- WT Dice on TCGA-LGG test")
        ax.legend()
        fig.tight_layout()
        hist_path = args.output_dir / "eval_seg_zoo_dice_hist.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Dice histogram -> {hist_path}")


if __name__ == "__main__":
    main()
