"""End-to-end inference with MONAI 3D models (SegResNet + DenseNet121).

Mirrors infer/pipeline.py but uses the 3D MONAI checkpoints:

* SegResNet 3D for segmentation (sliding-window inference @ 96^3, overlap 0.5)
* DenseNet121 3D for IDH classification on the predicted-mask 3D ROI

Critical detail: the segmentation model was trained on RAS-oriented, 1mm
isotropic volumes (MONAI Spacingd / Orientationd transforms), so we must
apply the same preprocessing at inference time. Feeding raw NIfTI without
those transforms collapses the prediction.

Run::

    uv run python -m idh_glioma.infer.pipeline_monai \
        --case-dir datasets/.../TCGA-CS-4942 \
        --output-mask outputs/TCGA-CS-4942_pred_mask_monai.nii.gz
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
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    KeepLargestConnectedComponent,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToTensord,
)


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


def predict(
    case_dir: Path,
    seg_ckpt: Path,
    cls_ckpt: Path,
    output_mask: Path,
    seg_roi_size=(96, 96, 96),
    cls_target_size=(96, 96, 96),
    cls_margin=4,
    sw_batch_size=2,
    overlap=0.5,
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

    tf = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )
    item = tf({"image": [str(flair_path), str(t1_path), str(t1gd_path), str(t2_path)]})
    img = item["image"]  # MetaTensor (4, h, w, d) in RAS @ 1mm

    seg_state = torch.load(seg_ckpt, map_location=device, weights_only=True)
    init_filters = int(seg_state.get("init_filters", 32))
    seg = SegResNet(
        spatial_dims=3, in_channels=4, out_channels=1, init_filters=init_filters,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
    ).to(device)
    seg.load_state_dict(seg_state["model"])
    seg.train(False)

    cls_state = torch.load(cls_ckpt, map_location=device, weights_only=True)
    cls = DenseNet121(spatial_dims=3, in_channels=4, out_channels=1, dropout_prob=0.2).to(device)
    cls.load_state_dict(cls_state["model"])
    cls.train(False)
    cls_target_size = tuple(cls_state.get("target_size", cls_target_size))
    cls_margin = int(cls_state.get("margin", cls_margin))
    cls_threshold = float(cls_state.get("threshold", 0.5))

    keep_largest = KeepLargestConnectedComponent(applied_labels=[1])

    t0 = time.perf_counter()
    img_t = img.unsqueeze(0).to(device).float()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        seg_logit = sliding_window_inference(
            img_t, roi_size=tuple(seg_roi_size), sw_batch_size=sw_batch_size, predictor=seg, overlap=overlap
        )
    raw_mask = (torch.sigmoid(seg_logit).cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
    mask_t = torch.from_numpy(raw_mask).unsqueeze(0)
    pred_mask = keep_largest(mask_t).squeeze(0).numpy().astype(np.uint8)
    seg_latency = time.perf_counter() - t0

    bb = _bbox3d(pred_mask, margin=cls_margin)
    if bb is None:
        idh_prob = float("nan")
        idh_pred = -1
        cls_latency = 0.0
    else:
        y0, y1, x0, x1, z0, z1 = bb
        img_np = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
        crop = img_np[:, y0:y1, x0:x1, z0:z1]
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

    # Save mask in RAS@1mm space (the MONAI-preprocessed orientation).
    flair_ref = cast(nib.Nifti1Image, nib.load(str(flair_path)))
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    # NB: pred_mask is in RAS@1mm spacing, NOT the original image space. Use the
    # MetaTensor's affine for a faithful save; alternatively, MONAI Invertd
    # could resample back to the original grid.
    affine = img.affine.numpy() if hasattr(img.affine, "numpy") else np.asarray(img.affine)
    nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine=affine), str(output_mask))

    print(f"Saved segmentation (RAS@1mm): {output_mask}")
    print(f"Segmentation: {int(pred_mask.sum())} predicted-tumor voxels, {seg_latency*1000:.0f} ms")
    if np.isnan(idh_prob):
        print("Predicted IDH mutation probability: n/a (no tumor predicted)")
    else:
        print(f"Predicted IDH mutation probability: {idh_prob:.4f} ({cls_latency*1000:.0f} ms)")
        print(f"Predicted IDH class: {idh_pred} (threshold {cls_threshold:.4f})")
        print(f"BraTS-style reference: {flair_ref.shape}; reported in RAS@1mm space")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--seg-ckpt", type=Path, default=Path("checkpoints/segresnet_brats2021.pt"))
    p.add_argument("--cls-ckpt", type=Path, default=Path("checkpoints/densenet3d_idh.pt"))
    p.add_argument("--output-mask", type=Path, default=Path("outputs/pred_mask_monai.nii.gz"))
    return p.parse_args()


def main():
    args = parse_args()
    predict(args.case_dir, args.seg_ckpt, args.cls_ckpt, args.output_mask)


if __name__ == "__main__":
    main()
