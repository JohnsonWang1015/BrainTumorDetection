from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np


def _load_sam3_model(checkpoint: str, model_type: str):
    try:
        from sam3 import SamPredictor, sam_model_registry  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SAM3 package is not installed. Install official Meta SAM3 package on your training host."
        ) from e

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    predictor = SamPredictor(model)
    return predictor


def _normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    x = slice_2d.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (x * 255).astype(np.uint8)


def run_sam3_inference(
    volume_path: Path, checkpoint: str, model_type: str = "vit_h"
) -> np.ndarray:
    predictor = _load_sam3_model(checkpoint=checkpoint, model_type=model_type)
    vol_img = cast(nib.Nifti1Image, nib.load(str(volume_path)))
    vol = vol_img.get_fdata().astype(np.float32)

    pred_mask = np.zeros_like(vol, dtype=np.uint8)
    for z in range(vol.shape[-1]):
        image = _normalize_to_uint8(vol[:, :, z])
        rgb = np.stack([image, image, image], axis=-1)
        predictor.set_image(rgb)
        h, w = image.shape
        box = np.array([w * 0.2, h * 0.2, w * 0.8, h * 0.8])
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        pred_mask[:, :, z] = masks[0].astype(np.uint8)
    return pred_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 slice-by-slice inference for NIfTI volume"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="vit_h")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = run_sam3_inference(args.image, args.checkpoint, args.model_type)
    ref = cast(nib.Nifti1Image, nib.load(str(args.image)))
    out_nii = nib.Nifti1Image(
        pred.astype(np.uint8), affine=ref.affine, header=ref.header
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nii, str(args.output))
    print(f"Saved SAM3 mask: {args.output}")


if __name__ == "__main__":
    main()
