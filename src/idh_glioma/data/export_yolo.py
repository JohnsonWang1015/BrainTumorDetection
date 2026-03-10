from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import cv2
import nibabel as nib
import numpy as np
import yaml

from idh_glioma.utils import ensure_dir, load_json


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return (x * 255).astype(np.uint8)


def mask_to_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = mask.shape
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1 + 1) / w
    bh = (y2 - y1 + 1) / h
    return xc, yc, bw, bh


def export_split(records: list[dict[str, object]], split: str, out_dir: Path) -> None:
    image_dir = ensure_dir(out_dir / "images" / split)
    label_dir = ensure_dir(out_dir / "labels" / split)

    for rec in records:
        modalities = cast(dict[str, str], rec["modalities"])
        flair_img = cast(nib.Nifti1Image, nib.load(str(modalities["flair"])))
        mask_img = cast(nib.Nifti1Image, nib.load(str(rec["mask_path"])))
        flair = flair_img.get_fdata().astype(np.float32)
        mask = (mask_img.get_fdata() > 0).astype(np.uint8)

        tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
        if len(tumor_slices) == 0:
            continue

        for z in tumor_slices[:: max(1, len(tumor_slices) // 6)]:
            img = normalize_to_uint8(flair[:, :, z])
            m = mask[:, :, z]
            bbox = mask_to_bbox(m)
            if bbox is None:
                continue

            stem = f"{rec['case_id']}_{z:03d}"
            img_path = image_dir / f"{stem}.png"
            txt_path = label_dir / f"{stem}.txt"
            cv2.imwrite(str(img_path), img)

            xc, yc, bw, bh = bbox
            txt_path.write_text(
                f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BraTS slices to YOLO detection format"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/manifest.json")
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/yolo"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)

    export_split(manifest["train"], "train", args.out)
    export_split(manifest["val"], "val", args.out)
    export_split(manifest["test"], "test", args.out)

    yaml_path = args.out / "dataset.yaml"
    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "glioma"},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"YOLO dataset exported to: {args.out}")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()
