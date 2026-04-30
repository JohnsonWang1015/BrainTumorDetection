"""Evaluate the MONAI SegResNet checkpoint on the held-out test split.

Mirrors eval_seg.py but uses MONAI loaders + sliding-window inference at
full resolution. Computes per-case Dice + IoU on the binarised whole-tumor
mask.

Outputs
-------
- Mean / median Dice + IoU printed to stdout
- Per-case CSV       -> outputs/eval_seg_monai_cases.csv
- Dice histogram PNG -> outputs/eval_seg_monai_dice_hist.png

Usage::

    uv run eval-seg-monai
    uv run eval-seg-monai --ckpt checkpoints/segresnet_tcga.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToTensord,
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


def _records_to_monai(records, project_root):
    out = []
    for r in records:
        mods = r["modalities"]
        out.append(
            {
                "image": [
                    str(project_root / mods["flair"]),
                    str(project_root / mods["t1"]),
                    str(project_root / mods["t1Gd"]),
                    str(project_root / mods["t2"]),
                ],
                "label": str(project_root / r["mask_path"]),
                "case_id": r.get("case_id", ""),
            }
        )
    return out


def _build_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
        ]
    )


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MONAI SegResNet on TCGA-LGG test split")
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/segresnet_tcga.pt"))
    p.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--sw-batch-size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path.cwd()
    manifest = load_json(args.manifest)
    test_data = _records_to_monai(manifest["test"], project_root)
    if not test_data:
        raise ValueError("Test split is empty.")

    tf = _build_transforms()
    test_ds = CacheDataset(data=test_data, transform=tf, cache_rate=1.0, num_workers=args.num_workers)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    init_filters = int(state.get("init_filters", args.init_filters))
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        init_filters=init_filters,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    ).to(device)
    model.load_state_dict(state["model"])
    model.train(False)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False, reduction="mean_batch", get_not_nans=False)

    case_results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval-seg-monai"):
            x = batch["image"].to(device, non_blocking=True).float()
            y = (batch["label"] > 0).float().to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = sliding_window_inference(
                    x,
                    roi_size=tuple(args.roi_size),
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.overlap,
                )
            preds = [post_pred(p) for p in decollate_batch(logits)]
            labels = [post_label(t) for t in decollate_batch(y)]
            dice_metric.reset()
            iou_metric.reset()
            dice_metric(y_pred=preds, y=labels)
            iou_metric(y_pred=preds, y=labels)
            dice_val = float(dice_metric.aggregate().mean().item())
            iou_val = float(iou_metric.aggregate().mean().item())
            case_id = batch.get("case_id", [""])[0]
            case_results.append({"case_id": str(case_id), "dice": dice_val, "iou": iou_val})

    dices = np.array([r["dice"] for r in case_results])
    ious = np.array([r["iou"] for r in case_results])
    print(f"\n{'=' * 60}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Test cases  : {len(case_results)}")
    print(f"{'=' * 60}")
    print(f"Dice  mean  : {dices.mean():.4f}  +/- {dices.std():.4f}")
    print(f"Dice  median: {np.median(dices):.4f}")
    print(f"IoU   mean  : {ious.mean():.4f}  +/- {ious.std():.4f}")
    print(f"{'=' * 60}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "eval_seg_monai_cases.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "dice", "iou"])
        w.writeheader()
        for r in case_results:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"Per-case CSV -> {csv_path}")

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dices, bins=20, edgecolor="white", color="steelblue")
        ax.axvline(dices.mean(), color="tomato", lw=2, label=f"mean={dices.mean():.3f}")
        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Cases")
        ax.set_title("MONAI SegResNet -- Dice distribution (test set)")
        ax.legend()
        fig.tight_layout()
        hist_path = args.output_dir / "eval_seg_monai_dice_hist.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Dice histogram -> {hist_path}")


if __name__ == "__main__":
    main()
