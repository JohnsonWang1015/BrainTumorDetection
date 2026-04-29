from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch

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

    cls_model = build_mobilenetv3_binary().to(device)
    cls_state = torch.load(cls_ckpt, map_location=device)
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
    cls_logits: list[float] = []
    non_blocking = device.type == "cuda"

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

        cls_batch = np.stack(
            [
                np.stack([flair[:, :, z], t1gd[:, :, z], t2[:, :, z]], axis=0)
                for z in z_indices
            ],
            axis=0,
        )
        cls_tensor = torch.from_numpy(cls_batch).to(
            device,
            non_blocking=non_blocking,
        )
        if device.type == "cuda":
            cls_tensor = cls_tensor.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            cls_logit_batch = cls_model(cls_tensor)
        cls_logits.extend(cls_logit_batch.cpu().numpy().reshape(-1).tolist())

    idh_prob = 1.0 / (1.0 + np.exp(-np.array(cls_logits))).mean()
    idh_pred = int(idh_prob >= 0.5)

    ref = cast(nib.Nifti1Image, nib.load(str(next(case_dir.glob("*_flair.nii.gz")))))
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            pred_mask.astype(np.uint8), affine=ref.affine, header=ref.header
        ),
        str(output_mask),
    )

    print(f"Saved segmentation: {output_mask}")
    print(f"Predicted IDH mutation probability: {idh_prob:.4f}")
    print(f"Predicted IDH class (0=WT,1=Mut): {idh_pred}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end inference for segmentation + IDH classification"
    )
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--seg-ckpt", type=Path, default=Path("checkpoints/unet2d_tcga_v1.pt")
    )
    parser.add_argument(
        "--cls-ckpt", type=Path, default=Path("checkpoints/mobilenetv3_idh_v3.pt")
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
    )


if __name__ == "__main__":
    main()
