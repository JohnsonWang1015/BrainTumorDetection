from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from idh_glioma.data.datasets import _crop_roi, _tumor_bbox_2d
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary
from idh_glioma.models.unet2d import UNet2D


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def _load_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flair_img = cast(
        nib.Nifti1Image, nib.load(str(next(case_dir.glob("*_flair.nii.gz"))))
    )
    t1_img = cast(nib.Nifti1Image, nib.load(str(next(case_dir.glob("*_t1.nii.gz")))))
    t1gd_img = cast(
        nib.Nifti1Image, nib.load(str(next(case_dir.glob("*_t1Gd.nii.gz"))))
    )
    t2_img = cast(nib.Nifti1Image, nib.load(str(next(case_dir.glob("*_t2.nii.gz")))))

    flair = flair_img.get_fdata().astype(np.float32)
    t1 = t1_img.get_fdata().astype(np.float32)
    t1gd = t1gd_img.get_fdata().astype(np.float32)
    t2 = t2_img.get_fdata().astype(np.float32)
    return _zscore(flair), _zscore(t1), _zscore(t1gd), _zscore(t2)


@torch.inference_mode()
def predict(
    case_dir: Path,
    seg_ckpt: Path,
    cls_ckpt: Path,
    output_mask: Path,
    batch_size: int,
    amp: bool,
    cudnn_benchmark: bool,
    tf32: bool,
    compile_model: bool,
    cls_variant: str,
    cls_img_size: int,
    cls_use_roi: bool,
    cls_roi_margin: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

    flair, t1, t1gd, t2 = _load_case(case_dir)

    seg_model = UNet2D().to(device)
    seg_state = torch.load(seg_ckpt, map_location=device)
    seg_model.load_state_dict(seg_state["model"])
    seg_model.eval()
    if device.type == "cuda":
        seg_model = seg_model.to(memory_format=torch.channels_last)
    if compile_model:
        try:
            seg_model = torch.compile(seg_model)
        except Exception as exc:
            print(f"Segmentation torch.compile disabled: {exc}")

    cls_state = torch.load(cls_ckpt, map_location=device)
    # Honour checkpoint-recorded settings so inference matches training.
    cls_variant = cls_state.get("variant", cls_variant)
    cls_use_roi = cls_state.get("use_roi", cls_use_roi)
    cls_roi_margin = cls_state.get("roi_margin", cls_roi_margin)
    cls_img_size = cls_state.get("img_size", cls_img_size)
    cls_threshold = float(cls_state.get("threshold", 0.5))

    cls_model = build_mobilenetv3_binary(variant=cls_variant).to(device)
    cls_model.load_state_dict(cls_state["model"])
    cls_model.eval()
    if device.type == "cuda":
        cls_model = cls_model.to(memory_format=torch.channels_last)
    if compile_model:
        try:
            cls_model = torch.compile(cls_model)
        except Exception as exc:
            print(f"Classification torch.compile disabled: {exc}")

    pred_mask = np.zeros_like(flair, dtype=np.uint8)
    non_blocking = device.type == "cuda"

    # Pass 1: run the segmentation model over every z-slice; store predictions.
    for z_start in range(0, flair.shape[-1], batch_size):
        z_end = min(flair.shape[-1], z_start + batch_size)
        z_indices = range(z_start, z_end)

        seg_batch = np.stack(
            [
                np.stack(
                    [flair[:, :, z], t1[:, :, z], t1gd[:, :, z], t2[:, :, z]], axis=0
                )
                for z in z_indices
            ],
            axis=0,
        )
        seg_tensor = torch.from_numpy(seg_batch).to(
            device,
            non_blocking=non_blocking,
        )
        if device.type == "cuda":
            seg_tensor = seg_tensor.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            seg_logit = seg_model(seg_tensor)
        seg_prob = torch.sigmoid(seg_logit).cpu().numpy()[:, 0]
        for batch_idx, z in enumerate(z_indices):
            pred_mask[:, :, z] = (seg_prob[batch_idx] > 0.5).astype(np.uint8)

    # Pass 2: classify only tumor-bearing slices, using the predicted mask to
    # crop the ROI so train/inference distributions match.
    tumor_z = np.where(pred_mask.sum(axis=(0, 1)) > 0)[0]
    cls_logits: list[float] = []
    if len(tumor_z) == 0:
        idh_prob = float("nan")
    else:
        for z_start in range(0, len(tumor_z), batch_size):
            batch_z = tumor_z[z_start : z_start + batch_size]
            cropped: list[torch.Tensor] = []
            for z in batch_z:
                stacked = np.stack(
                    [flair[:, :, z], t1gd[:, :, z], t2[:, :, z]], axis=0
                )
                if cls_use_roi:
                    bbox = _tumor_bbox_2d(pred_mask[:, :, z], cls_roi_margin)
                    stacked = _crop_roi(stacked, bbox)
                t = torch.from_numpy(stacked.copy()).unsqueeze(0)
                t = F.interpolate(
                    t,
                    size=(cls_img_size, cls_img_size),
                    mode="bilinear",
                    align_corners=False,
                )
                cropped.append(t.squeeze(0))
            cls_tensor = torch.stack(cropped, dim=0).to(
                device, non_blocking=non_blocking
            )
            if device.type == "cuda":
                cls_tensor = cls_tensor.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                cls_logit_batch = cls_model(cls_tensor)
            cls_logits.extend(cls_logit_batch.cpu().numpy().reshape(-1).tolist())
        idh_prob = float(
            (1.0 / (1.0 + np.exp(-np.array(cls_logits)))).mean()
        )
    idh_pred = int(idh_prob >= cls_threshold) if not np.isnan(idh_prob) else -1

    ref = cast(nib.Nifti1Image, nib.load(str(next(case_dir.glob("*_flair.nii.gz")))))
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            pred_mask.astype(np.uint8), affine=ref.affine, header=ref.header
        ),
        str(output_mask),
    )

    print(f"Saved segmentation: {output_mask}")
    if np.isnan(idh_prob):
        print("Predicted IDH mutation probability: n/a (no tumor predicted)")
    else:
        print(f"Predicted IDH mutation probability: {idh_prob:.4f}")
    print(
        f"Predicted IDH class (0=WT,1=Mut,-1=undetermined): {idh_pred} "
        f"(threshold={cls_threshold:.4f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end inference for segmentation + IDH classification"
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--seg-ckpt", type=Path, default=Path("checkpoints/unet2d_tcga_v1.pt")
    )
    parser.add_argument(
        "--cls-ckpt",
        type=Path,
        default=Path("checkpoints/mobilenetv3_idh_best.pt"),
        help="IDH classifier checkpoint. Defaults to the new training output. "
        "Falls back to checkpoints/mobilenetv3_idh_v3.pt if you haven't retrained.",
    )
    parser.add_argument(
        "--output-mask", type=Path, default=Path("outputs/pred_mask.nii.gz")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", choices=["default", "a6000"], default="default")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--cls-variant",
        choices=["small", "large"],
        default="large",
        help="Classifier backbone size. Overridden by checkpoint metadata.",
    )
    parser.add_argument("--cls-img-size", type=int, default=224)
    parser.add_argument(
        "--cls-use-roi", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cls-roi-margin", type=int, default=10)
    return parser.parse_args()


def apply_profile(args: argparse.Namespace, device: torch.device) -> argparse.Namespace:
    if args.profile != "a6000" or device.type != "cuda":
        return args

    if args.batch_size == 16:
        args.batch_size = 32
    args.amp = True
    args.cudnn_benchmark = True
    args.tf32 = True
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = apply_profile(args, device)
    predict(
        args.case_dir,
        args.seg_ckpt,
        args.cls_ckpt,
        args.output_mask,
        batch_size=args.batch_size,
        amp=args.amp,
        cudnn_benchmark=args.cudnn_benchmark,
        tf32=args.tf32,
        compile_model=args.compile,
        cls_variant=args.cls_variant,
        cls_img_size=args.cls_img_size,
        cls_use_roi=args.cls_use_roi,
        cls_roi_margin=args.cls_roi_margin,
    )


if __name__ == "__main__":
    main()
