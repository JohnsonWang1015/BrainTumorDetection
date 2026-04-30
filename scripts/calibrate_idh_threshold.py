"""Pick the optimal IDH classification threshold on the val split.

Uses case-level probabilities (slice probs averaged per case) and Youden's J
statistic (TPR - FPR maximisation) to choose a threshold, then writes it back
into the checkpoint metadata. ``eval-idh`` and ``infer`` will pick it up
automatically.

Run:
    uv run python scripts/calibrate_idh_threshold.py \
        --ckpt checkpoints/mobilenetv3_idh_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve

from idh_glioma.data.datasets import _crop_roi, _tumor_bbox_2d, load_nifti, zscore
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary
from idh_glioma.utils import load_json


@torch.inference_mode()
def _case_prob(record: dict, model, device, img_size, use_roi, roi_margin) -> float:
    flair = zscore(load_nifti(record["modalities"]["flair"]))
    t1gd = zscore(load_nifti(record["modalities"]["t1Gd"]))
    t2 = zscore(load_nifti(record["modalities"]["t2"]))
    mask = (load_nifti(record["mask_path"]) > 0).astype(np.float32)
    tumor_z = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    if len(tumor_z) == 0:
        return float("nan")
    probs: list[float] = []
    for z in tumor_z:
        stacked = np.stack([flair[:, :, z], t1gd[:, :, z], t2[:, :, z]], axis=0)
        if use_roi:
            bbox = _tumor_bbox_2d(mask[:, :, z], roi_margin)
            stacked = _crop_roi(stacked, bbox)
        t = torch.from_numpy(stacked.copy()).unsqueeze(0)
        t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
        t = t.to(device)
        if device.type == "cuda":
            t = t.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logit = model(t)
        probs.append(float(torch.sigmoid(logit.float()).cpu().item()))
    return float(np.mean(probs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    variant = state.get("variant", "large")
    img_size = state.get("img_size", 224)
    use_roi = state.get("use_roi", True)
    roi_margin = state.get("roi_margin", 10)
    print(f"[calibrate] ckpt cfg: variant={variant} img_size={img_size} use_roi={use_roi} roi_margin={roi_margin}")

    model = build_mobilenetv3_binary(variant=variant).to(device)
    model.load_state_dict(state["model"])
    model.train(False)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    manifest = load_json(args.manifest)
    records = [r for r in manifest[args.split] if r.get("idh_label") is not None]
    print(f"[calibrate] running on {len(records)} {args.split} cases...")

    probs: list[float] = []
    labels: list[int] = []
    for r in records:
        p = _case_prob(r, model, device, img_size, use_roi, roi_margin)
        probs.append(p)
        labels.append(int(r["idh_label"]))
        print(f"  {r['case_id']:20s} label={labels[-1]} prob={p:.4f}")

    probs_arr = np.array(probs)
    labels_arr = np.array(labels)
    if len(np.unique(labels_arr)) < 2:
        raise ValueError(f"{args.split} split is single-class; need both 0 and 1 to calibrate.")

    fpr, tpr, thresholds = roc_curve(labels_arr, probs_arr)
    j = tpr - fpr
    best_i = int(np.argmax(j))
    best_thresh = float(thresholds[best_i])
    # roc_curve sometimes returns thresholds > 1 for the first entry; clip.
    if best_thresh > 1.0:
        best_thresh = 1.0
    print(f"\n[calibrate] Youden's J max = {j[best_i]:.4f} @ threshold {best_thresh:.4f}")
    print(f"  -> TPR (Mutant recall)  = {tpr[best_i]:.4f}")
    print(f"  -> 1-FPR (WT specificity) = {1 - fpr[best_i]:.4f}")

    state["threshold"] = best_thresh
    state["threshold_split"] = args.split
    state["threshold_method"] = "youden_j"
    torch.save(state, args.ckpt)
    print(f"[calibrate] saved threshold={best_thresh:.4f} to {args.ckpt}")


if __name__ == "__main__":
    main()
