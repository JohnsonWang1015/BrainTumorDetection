"""Evaluate the UNet2D segmentation model on the held-out test split.

Computes per-case Dice and IoU across the test set by aggregating
predictions slice-by-slice.

Outputs
-------
- Mean ± std Dice and IoU printed to stdout
- Per-case CSV       →  outputs/eval_seg_cases.csv
- Dice histogram PNG →  outputs/eval_seg_dice_hist.png

Usage::

    uv run eval-seg
    uv run eval-seg --manifest artifacts/manifest.json --ckpt checkpoints/unet2d_tcga_v1.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from idh_glioma.data.datasets import load_nifti, zscore
from idh_glioma.models.unet2d import UNet2D
from idh_glioma.utils import load_json

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

INFER_IMG_SIZE = 240


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = (pred * target).sum()
    return float((2 * intersection + eps) / (pred.sum() + target.sum() + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


@torch.inference_mode()
def _predict_case(
    record: dict,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pred_mask_3d, gt_mask_3d) both binary uint8."""
    flair = zscore(load_nifti(record["modalities"]["flair"]))
    t1 = zscore(load_nifti(record["modalities"]["t1"]))
    t1gd = zscore(load_nifti(record["modalities"]["t1Gd"]))
    t2 = zscore(load_nifti(record["modalities"]["t2"]))
    gt_mask = (load_nifti(record["mask_path"]) > 0).astype(np.uint8)

    depth = flair.shape[-1]
    pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)

    for z_start in range(0, depth, batch_size):
        z_end = min(depth, z_start + batch_size)
        slices = np.stack(
            [
                np.stack([flair[:, :, z], t1[:, :, z], t1gd[:, :, z], t2[:, :, z]], axis=0)
                for z in range(z_start, z_end)
            ],
            axis=0,
        )  # (B, 4, H, W)
        tensor = torch.from_numpy(slices).to(device, non_blocking=True)
        if tensor.shape[-1] != INFER_IMG_SIZE or tensor.shape[-2] != INFER_IMG_SIZE:
            tensor = F.interpolate(tensor, size=(INFER_IMG_SIZE, INFER_IMG_SIZE), mode="bilinear", align_corners=False)
        if device.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logit = model(tensor)
        prob = torch.sigmoid(logit).cpu().numpy()[:, 0]  # (B, H, W)

        for i, z in enumerate(range(z_start, z_end)):
            pred_slice = (prob[i] > 0.5).astype(np.uint8)
            # Resize prediction back to original spatial size if needed
            orig_h, orig_w = flair.shape[0], flair.shape[1]
            if pred_slice.shape != (orig_h, orig_w):
                pred_t = torch.from_numpy(pred_slice[None, None].astype(np.float32))
                pred_t = F.interpolate(pred_t, size=(orig_h, orig_w), mode="nearest")
                pred_slice = pred_t.numpy()[0, 0].astype(np.uint8)
            pred_mask[:, :, z] = pred_slice

    return pred_mask, gt_mask


def evaluate(
    manifest_path: Path,
    ckpt: Path,
    batch_size: int,
    output_dir: Path,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    manifest = load_json(manifest_path)
    records = manifest["test"]
    if not records:
        raise ValueError("Test split is empty.")

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = UNet2D().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    case_results: list[dict] = []
    for record in tqdm(records, desc="eval-seg"):
        pred, gt = _predict_case(record, model, device, use_amp, batch_size)
        dice = dice_score(pred, gt)
        iou = iou_score(pred, gt)
        case_results.append({
            "case_id": record.get("case_id", Path(record["modalities"]["flair"]).parent.name),
            "dice": dice,
            "iou": iou,
        })

    dices = np.array([r["dice"] for r in case_results])
    ious = np.array([r["iou"] for r in case_results])

    print(f"\n{'=' * 50}")
    print(f"Checkpoint  : {ckpt}")
    print(f"Test cases  : {len(records)}")
    print(f"{'=' * 50}")
    print(f"Dice  mean  : {dices.mean():.4f}  ± {dices.std():.4f}")
    print(f"Dice  median: {np.median(dices):.4f}")
    print(f"IoU   mean  : {ious.mean():.4f}  ± {ious.std():.4f}")
    print(f"{'=' * 50}")

    output_dir.mkdir(parents=True, exist_ok=True)

    import csv
    csv_path = output_dir / "eval_seg_cases.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "dice", "iou"])
        writer.writeheader()
        for r in case_results:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"Per-case CSV → {csv_path}")

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dices, bins=20, edgecolor="white", color="steelblue")
        ax.axvline(dices.mean(), color="tomato", lw=2, label=f"mean={dices.mean():.3f}")
        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Cases")
        ax.set_title("UNet2D Segmentation — Dice Distribution (test set)")
        ax.legend()
        fig.tight_layout()
        hist_path = output_dir / "eval_seg_dice_hist.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"Dice histogram → {hist_path}")

    return {"dice_mean": float(dices.mean()), "iou_mean": float(ious.mean())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate UNet2D segmentation on BraTS test split")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifest.json"),
        help="Path to manifest. Must match the cohort the checkpoint was trained on.",
    )
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/unet2d_tcga_v1.pt"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.manifest, args.ckpt, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
