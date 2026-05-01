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
from monai.inferers import sliding_window_inference
from monai.networks.nets import DenseNet121, SegResNet
from monai.transforms import NormalizeIntensity

from idh_glioma.infer.e2e_roi import (
    apply_mask_postprocess,
    merge_e2e_config,
    predict_multi_view_idh,
)


def predict(
    case_dir: Path,
    bundle_ckpt: Path,
    cls_ckpt: Path,
    output_mask: Path,
    seg_roi_size=(240, 240, 160),
    cls_target_size=(96, 96, 96),
    cls_margin=4,
    seg_overlap=0.5,
    e2e_config: Path | None = None,
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
    cls_threshold = float(cls_state.get("threshold", 0.5))
    e2e_cfg = merge_e2e_config(
        e2e_config,
        fallback_threshold=cls_threshold,
        fallback_base_margin=int(cls_state.get("margin", cls_margin)),
    )

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
    pred_mask = apply_mask_postprocess(
        raw_mask,
        keep_largest=bool(e2e_cfg["keep_largest"]),
        dilate_iters=int(e2e_cfg["dilate_iters"]),
    )
    seg_latency = time.perf_counter() - t0

    # ---- Pass 2: 3D classifier on the predicted-mask ROI ----
    if not pred_mask.any():
        idh_prob = float("nan")
        idh_pred = -1
        cls_latency = 0.0
        view_probs: list[float] = []
    else:
        t1c = time.perf_counter()
        idh_prob, view_probs, _ = predict_multi_view_idh(
            flair=flair,
            t1=t1,
            t1gd=t1gd,
            t2=t2,
            pred_mask=pred_mask,
            cls_model=cls,
            device=device,
            target_size=tuple(cls_target_size),
            cfg=e2e_cfg,
            use_amp=use_amp,
        )
        idh_pred = int(idh_prob >= float(e2e_cfg["threshold"]))
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
        print(
            f"Predicted IDH class: {idh_pred} "
            f"(threshold {float(e2e_cfg['threshold']):.4f}, "
            f"aggregation={e2e_cfg['aggregation']}, views={len(view_probs)})"
        )


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
    p.add_argument(
        "--e2e-config",
        type=Path,
        default=Path("artifacts/e2e_idh_config.json"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    predict(args.case_dir, args.bundle_ckpt, args.cls_ckpt, args.output_mask, e2e_config=args.e2e_config)


if __name__ == "__main__":
    main()
