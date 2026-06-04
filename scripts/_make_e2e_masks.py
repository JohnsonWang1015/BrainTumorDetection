"""Generate and save MONAI-bundle segmentation masks for every case in a split.

Reuses the exact bundle seg pass + post-processing from `eval-e2e-zoo` so the
saved masks match the reported Dice. Writes `outputs/e2e_<case_id>_pred.nii.gz`
(skipping any that already exist) so the report gallery / 3D GIF can use them.

Usage::

    uv run python scripts/_make_e2e_masks.py --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import NormalizeIntensity
from tqdm import tqdm

from idh_glioma.infer.e2e_roi import apply_mask_postprocess, merge_e2e_config
from idh_glioma.utils import load_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument(
        "--bundle-ckpt", type=Path,
        default=Path("checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt"),
    )
    p.add_argument("--seg-roi-size", type=int, nargs=3, default=(240, 240, 160))
    p.add_argument("--seg-overlap", type=float, default=0.5)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--e2e-config", type=Path, default=Path("artifacts/e2e_idh_config.json"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    seg = SegResNet(
        spatial_dims=3, in_channels=4, out_channels=3, init_filters=16,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
    ).to(device)
    state = torch.load(args.bundle_ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    seg.load_state_dict(state)
    seg.train(False)

    e2e_cfg = merge_e2e_config(args.e2e_config, fallback_threshold=0.5, fallback_base_margin=4)
    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)

    manifest = load_json(args.manifest)
    records = [r for r in manifest[args.split] if r.get("idh_label") is not None]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for r in tqdm(records, desc="seg-masks"):
        cid = r.get("case_id", "")
        out_path = args.out_dir / f"e2e_{cid}_pred.nii.gz"
        if out_path.exists() and not args.overwrite:
            continue
        flair = nib.load(r["modalities"]["flair"]).get_fdata().astype(np.float32)
        t1 = nib.load(r["modalities"]["t1"]).get_fdata().astype(np.float32)
        t1gd = nib.load(r["modalities"]["t1Gd"]).get_fdata().astype(np.float32)
        t2 = nib.load(r["modalities"]["t2"]).get_fdata().astype(np.float32)

        img4 = np.stack([t1gd, t1, t2, flair], axis=0)  # bundle order: T1c, T1, T2, FLAIR
        img_t = normalize(torch.from_numpy(img4)).unsqueeze(0).to(device).float()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            seg_logits = sliding_window_inference(
                img_t, roi_size=tuple(args.seg_roi_size), sw_batch_size=1,
                predictor=seg, overlap=args.seg_overlap,
            )
        pp = torch.sigmoid(seg_logits).cpu().numpy()[0]
        raw = (pp[1] > 0.5).astype(np.uint8)  # WT channel
        pred_mask = apply_mask_postprocess(
            raw,
            keep_largest=bool(e2e_cfg["keep_largest"]),
            dilate_iters=int(e2e_cfg["dilate_iters"]),
        )
        ref = nib.load(r["mask_path"])
        nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), ref.affine, ref.header), str(out_path))
        gt = (ref.get_fdata() > 0).astype(np.uint8)
        inter = (pred_mask * gt).sum()
        dice = (2.0 * inter + 1e-6) / (pred_mask.sum() + gt.sum() + 1e-6)
        tqdm.write(f"[mask] {cid}: dice={dice:.4f} voxels={int(pred_mask.sum())} -> {out_path.name}")


if __name__ == "__main__":
    main()
