from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from idh_glioma.models.unet2d import UNet2D


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def find_modality(case_dir: Path, modality: str) -> Path:
    matches = sorted(case_dir.glob(f"*_{modality}.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"Missing modality '{modality}' in {case_dir}")
    return matches[0]


def load_case(
    case_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, nib.Nifti1Image]:
    flair_path = find_modality(case_dir, "flair")
    t1_path = find_modality(case_dir, "t1")
    t1gd_path = find_modality(case_dir, "t1Gd")
    t2_path = find_modality(case_dir, "t2")

    flair_img = nib.load(str(flair_path))
    flair = flair_img.get_fdata().astype(np.float32)
    t1 = nib.load(str(t1_path)).get_fdata().astype(np.float32)
    t1gd = nib.load(str(t1gd_path)).get_fdata().astype(np.float32)
    t2 = nib.load(str(t2_path)).get_fdata().astype(np.float32)

    return zscore(flair), zscore(t1), zscore(t1gd), zscore(t2), flair_img


@torch.inference_mode()
def infer_case(
    model: UNet2D,
    case_dir: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    flair, t1, t1gd, t2, flair_img = load_case(case_dir)
    pred = np.zeros_like(flair, dtype=np.uint8)

    for z_start in range(0, flair.shape[-1], batch_size):
        z_end = min(flair.shape[-1], z_start + batch_size)
        z_indices = range(z_start, z_end)

        batch = np.stack(
            [
                np.stack(
                    [flair[:, :, z], t1[:, :, z], t1gd[:, :, z], t2[:, :, z]], axis=0
                )
                for z in z_indices
            ],
            axis=0,
        )
        x = torch.from_numpy(batch).to(device)
        y = torch.sigmoid(model(x)).cpu().numpy()[:, 0]

        for batch_idx, z in enumerate(z_indices):
            pred[:, :, z] = (y[batch_idx] > 0.5).astype(np.uint8)

    return flair, pred, flair_img


def save_outputs(
    case_dir: Path,
    flair: np.ndarray,
    pred: np.ndarray,
    flair_img: nib.Nifti1Image,
    output_dir: Path,
    alpha: float,
) -> tuple[Path, Path, int]:
    case_id = case_dir.name
    mask_path = output_dir / f"{case_id}_pred_mask.nii.gz"
    png_path = output_dir / f"{case_id}_overlay.png"

    nib.save(
        nib.Nifti1Image(
            pred.astype(np.uint8), affine=flair_img.affine, header=flair_img.header
        ),
        str(mask_path),
    )

    areas = pred.reshape(-1, pred.shape[-1]).sum(axis=0)
    z = int(np.argmax(areas)) if int(areas.max()) > 0 else pred.shape[-1] // 2

    plt.figure(figsize=(6, 6))
    plt.imshow(flair[:, :, z], cmap="gray")
    plt.imshow(
        np.ma.masked_where(pred[:, :, z] == 0, pred[:, :, z]),
        cmap="autumn",
        alpha=alpha,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close()

    return mask_path, png_path, z


def collect_cases(cases_root: Path, case_glob: str, limit: int | None) -> list[Path]:
    cases = sorted(p for p in cases_root.glob(case_glob) if p.is_dir())
    if limit is not None:
        cases = cases[:limit]
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch U-Net inference for BraTS cases and save NIfTI mask + overlay PNG"
    )
    parser.add_argument(
        "--cases-root",
        type=Path,
        default=Path(
            "datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"
        ),
    )
    parser.add_argument("--case-glob", type=str, default="TCGA-*")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/unet2d_mri.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch_unet2d"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.cases_root.exists():
        raise FileNotFoundError(f"cases-root not found: {args.cases_root}")

    cases = collect_cases(args.cases_root, args.case_glob, args.limit)
    if not cases:
        raise RuntimeError(
            "No case directories found. Check --cases-root and --case-glob"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = UNet2D().to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Cases: {len(cases)}")
    print(f"Output dir: {args.output_dir}")

    success = 0
    failed = 0

    for case_dir in cases:
        try:
            flair, pred, flair_img = infer_case(
                model=model,
                case_dir=case_dir,
                batch_size=args.batch_size,
                device=device,
            )
            mask_path, png_path, z = save_outputs(
                case_dir=case_dir,
                flair=flair,
                pred=pred,
                flair_img=flair_img,
                output_dir=args.output_dir,
                alpha=args.alpha,
            )
            print(f"[OK] {case_dir.name} -> {mask_path.name}, {png_path.name}, z={z}")
            success += 1
        except Exception as exc:
            print(f"[FAIL] {case_dir.name}: {exc}")
            failed += 1

    print(f"Done. success={success}, failed={failed}")


if __name__ == "__main__":
    main()
