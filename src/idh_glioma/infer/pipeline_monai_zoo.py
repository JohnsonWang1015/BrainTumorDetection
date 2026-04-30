"""End-to-end inference combining the MONAI Model Zoo brats bundle (seg)
with our 3D DenseNet IDH classifier.

The bundle is used zero-shot: SegResNet (init_filters=16, out_channels=3)
trained by MONAI on the BraTS challenge cohort. We take the WT channel of
its 3-channel output (TC / WT / ET) as the binary tumor mask. Bundle
preprocessing is just NormalizeIntensityd (channel_wise + nonzero) -- no
Orientationd / Spacingd because the BraTS-TCGA-LGG cohort is already in
the BraTS coordinate space.

Channel order in the bundle: T1c, T1, T2, FLAIR. We reorder our manifest's
flair / t1 / t1Gd / t2 accordingly.

Usage::

    uv run infer-monai-zoo \
        --case-dir datasets/.../TCGA-CS-4942 \
        --output-mask outputs/TCGA-CS-4942_pred_mask_zoo.nii.gz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import DenseNet121, SegResNet
from monai.transforms import KeepLargestConnectedComponent, NormalizeIntensity


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


def _zscore_crop(vol, bbox):
    y0, y1, x0, x1, z0, z1 = bbox
    crop = vol[y0:y1, x0:x1, z0:z1]
    return (crop - crop.mean()) / (crop.std() + 1e-6)


def predict(
    case_dir: Path,
    bundle_ckpt: Path,
    cls_ckpt: Path,
    output_mask: Path,
    seg_roi_size=(240, 240, 160),
    cls_target_size=(96, 96, 96),
    cls_margin=4,
    seg_overlap=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    flair_path = next(case_dir.glob("*_flair.nii.gz"))
    t1_path = next(case_dir.glob("*_t1.nii.gz"))
    t1gd_path = next(case_dir.glob("*_t1Gd.nii.gz"))
    t2_path = next(case_dir.glob("*_t2.nii.gz"))

    flair_img = cast(nib.Nifti1Image, nib.load(str(flair_path)))
    flair = flair_img.get_fdata().astype(np.float32)
    t1 = nib.load(str(t1_path)).get_fdata().astype(np.float32)
    t1gd = nib.load(str(t1gd_path)).get_fdata().astype(np.float32)
    t2 = nib.load(str(t2_path)).get_fdata().astype(np.float32)

    seg = SegResNet(
        spatial_dims=3, in_channels=4, out_channels=3, init_filters=16,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
    ).to(device)
    state = torch.load(bundle_ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    seg.load_state_dict(state)
    seg.train(False)

    cls_state = torch.load(cls_ckpt, map_location=device, weights_only=True)
    cls = DenseNet121(spatial_dims=3, in_channels=4, out_channels=1, dropout_prob=0.2).to(device)
    cls.load_state_dict(cls_state["model"])
    cls.train(False)
    cls_target_size = tuple(cls_state.get("target_size", cls_target_size))
    cls_margin = int(cls_state.get("margin", cls_margin))
    cls_threshold = float(cls_state.get("threshold", 0.5))

    keep_largest = KeepLargestConnectedComponent(applied_labels=[1])
    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)

    # ---- Pass 1: bundle segmentation ----
    t0 = time.perf_counter()
    img4 = np.stack([t1gd, t1, t2, flair], axis=0)  # bundle order
    img_t = torch.from_numpy(img4)
    img_t = normalize(img_t).unsqueeze(0).to(device).float()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        seg_logits = sliding_window_inference(
            img_t, roi_size=tuple(seg_roi_size), sw_batch_size=1, predictor=seg, overlap=seg_overlap
        )
    probs = torch.sigmoid(seg_logits).cpu().numpy()[0]  # (3, H, W, D)
    raw_mask = (probs[1] > 0.5).astype(np.uint8)  # WT channel
    mask_t = torch.from_numpy(raw_mask).unsqueeze(0)
    pred_mask = keep_largest(mask_t).squeeze(0).numpy().astype(np.uint8)
    seg_latency = time.perf_counter() - t0

    # ---- Pass 2: 3D classifier on the predicted-mask ROI ----
    bb = _bbox3d(pred_mask, margin=cls_margin)
    if bb is None:
        idh_prob = float("nan")
        idh_pred = -1
        cls_latency = 0.0
    else:
        # IDH classifier expects channel order: FLAIR, T1, T1Gd, T2 (the train script's order).
        flair_c = _zscore_crop(flair, bb)
        t1_c = _zscore_crop(t1, bb)
        t1gd_c = _zscore_crop(t1gd, bb)
        t2_c = _zscore_crop(t2, bb)
        crop = np.stack([flair_c, t1_c, t1gd_c, t2_c], axis=0)
        ct = torch.from_numpy(crop).unsqueeze(0)
        ct = F.interpolate(ct, size=tuple(cls_target_size), mode="trilinear", align_corners=False).to(device)
        if device.type == "cuda":
            ct = ct.contiguous(memory_format=torch.channels_last_3d)
        t1c = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logit = cls(ct)
        idh_prob = float(torch.sigmoid(logit.float()).cpu().item())
        idh_pred = int(idh_prob >= cls_threshold)
        cls_latency = time.perf_counter() - t1c

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(pred_mask.astype(np.uint8), affine=flair_img.affine, header=flair_img.header),
        str(output_mask),
    )

    print(f"Saved segmentation: {output_mask}")
    print(f"Segmentation: {int(pred_mask.sum())} predicted-tumor voxels, {seg_latency*1000:.0f} ms")
    if np.isnan(idh_prob):
        print("Predicted IDH mutation probability: n/a (no tumor predicted)")
    else:
        print(f"Predicted IDH mutation probability: {idh_prob:.4f} ({cls_latency*1000:.0f} ms)")
        print(f"Predicted IDH class: {idh_pred} (threshold {cls_threshold:.4f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument(
        "--bundle-ckpt", type=Path,
        default=Path("checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt"),
    )
    p.add_argument(
        "--cls-ckpt", type=Path,
        default=Path("checkpoints/densenet3d_idh_jitter.pt"),
        help="3D IDH classifier ckpt. The jitter-trained one is the recommended default.",
    )
    p.add_argument(
        "--output-mask", type=Path,
        default=Path("outputs/pred_mask_monai_zoo.nii.gz"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    predict(args.case_dir, args.bundle_ckpt, args.cls_ckpt, args.output_mask)


if __name__ == "__main__":
    main()
