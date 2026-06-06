"""Generate native-space SegResNet 3D segmentation masks for chosen cases.

`eval-seg-monai` computes Dice in the model's RAS / 1 mm resampled space and never
maps the prediction back, so the masks it could dump do NOT align with the native GT
(or the MONAI-bundle E2E masks in outputs/e2e_*_pred.nii.gz). For the same-case,
different-method 3D orbit (SegResNet 3D vs MONAI bundle) we need the SegResNet mask in
the SAME native grid as GT. This script reuses the eval transforms but adds MONAI's
``Invertd`` to undo Spacingd/Orientationd/EnsureChannelFirstd, then writes the mask on
the original image affine.

Usage::

    uv run python scripts/infer_segresnet_native.py --cases TCGA-DU-7301
    uv run python scripts/infer_segresnet_native.py            # default case list
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
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

from idh_glioma.utils import load_json


def _records_to_monai(records, project_root):
    out = []
    for r in records:
        mods = r["modalities"]
        out.append({
            "image": [
                str(project_root / mods["flair"]),
                str(project_root / mods["t1"]),
                str(project_root / mods["t1Gd"]),
                str(project_root / mods["t2"]),
            ],
            "case_id": r.get("case_id", ""),
            "native_ref": str(project_root / r["mask_path"]),
        })
    return out


def _build_transforms():
    # Mirror eval_seg_monai exactly, but on the image only (no GT needed) and keep the
    # meta so Invertd can reconstruct the native grid.
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear",)),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])


def parse_args():
    p = argparse.ArgumentParser(description="SegResNet 3D native-space inference")
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/segresnet_brats2021.pt"))
    p.add_argument("--cases", nargs="+",
                   default=["TCGA-DU-7301", "TCGA-CS-6669"])
    p.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--sw-batch-size", type=int, default=2)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path.cwd()
    manifest = load_json(args.manifest)
    pool = {c["case_id"]: c for c in manifest["test"] + manifest["val"]}
    records = [pool[c] for c in args.cases if c in pool]
    missing = [c for c in args.cases if c not in pool]
    if missing:
        print(f"[warn] not in manifest test/val: {missing}")
    if not records:
        raise SystemExit("No requested cases found in manifest.")

    data = _records_to_monai(records, project_root)
    tf = _build_transforms()
    ds = CacheDataset(data=data, transform=tf, cache_rate=1.0, num_workers=2)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    init_filters = int(state.get("init_filters", args.init_filters))
    model = SegResNet(
        spatial_dims=3, in_channels=4, out_channels=1, init_filters=init_filters,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
    ).to(device)
    model.load_state_dict(state["model"])
    model.train(False)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            cid = str(batch["case_id"][0])
            x = batch["image"].to(device, non_blocking=True).float()
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = sliding_window_inference(
                    x, roi_size=tuple(args.roi_size), sw_batch_size=args.sw_batch_size,
                    predictor=model, overlap=args.overlap,
                )
            # Store logits back in the batch so the discrete prediction keeps the RAS /
            # 1 mm affine MONAI assigned it; then resample that onto the native GT grid
            # with nibabel (nearest, order=0) so it aligns with GT and the bundle mask.
            # (Invertd can't be used here: the prediction loses the Spacingd/Orientationd
            # inverse metadata through sliding-window inference + decollate.)
            batch["pred"] = logits
            for item in decollate_batch(batch):
                item["pred"] = post(item["pred"])
                pred = item["pred"]
                arr = np.squeeze(pred.detach().cpu().numpy()).astype(np.float32)
                pred_aff = np.asarray(pred.affine.detach().cpu().numpy())
                ref = nib.load(str(item["native_ref"]))
                res = resample_from_to(
                    nib.Nifti1Image(arr, pred_aff),
                    (ref.shape, ref.affine),
                    order=0,
                )
                mask = np.asarray(res.dataobj).astype(np.uint8)
                out_path = args.output_dir / f"segresnet_{cid}_pred.nii.gz"
                nib.save(nib.Nifti1Image(mask, ref.affine), str(out_path))
                print(f"[segresnet] {cid}: vox={int((mask > 0).sum())} -> {out_path}")


if __name__ == "__main__":
    main()
