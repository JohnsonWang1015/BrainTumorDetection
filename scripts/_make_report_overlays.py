"""Generate figures for the model report:
  1. Segmentation overlays (FLAIR + GT green + Pred red) for a success and a failure case.
  2. CT/MRI classification montage (success + failure inputs with predicted prob).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nibabel as nib
import numpy as np

ROOT = Path(".")
OUT = ROOT / "docs" / "report_assets"
OUT.mkdir(parents=True, exist_ok=True)

manifest = json.load(open("artifacts/manifest.json"))
cases = {c["case_id"]: c for c in manifest["test"] + manifest["val"]}


def load(path):
    return np.asarray(nib.load(str(path)).dataobj)


def dice(a, b):
    a, b = a > 0, b > 0
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return 2.0 * inter / s if s else 1.0


def best_slice(gt):
    return int(np.argmax((gt > 0).sum(axis=(0, 1))))


def seg_overlay(case_id, pred_path, banner, fname, model_name):
    rec = cases[case_id]
    flair = load(rec["modalities"]["flair"])
    gt = load(rec["mask_path"])
    pred = load(pred_path)
    z = best_slice(gt)
    img = np.rot90(flair[:, :, z])
    g = np.rot90(gt[:, :, z] > 0)
    p = np.rot90(pred[:, :, z] > 0)
    d = dice(gt, pred)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("FLAIR (input)")
    axes[1].imshow(img, cmap="gray")
    axes[1].contour(g, colors="lime", linewidths=1.3)
    axes[1].set_title("Ground Truth (green)")
    axes[2].imshow(img, cmap="gray")
    axes[2].contour(g, colors="lime", linewidths=1.0)
    if p.any():
        axes[2].contour(p, colors="red", linewidths=1.0)
    axes[2].set_title("GT (green) vs Pred (red)")
    note = "" if p.any() else "  [predicted mask is EMPTY]"
    fig.suptitle(f"{banner} — {model_name}\n{case_id}  |  Dice = {d:.3f}{note}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[seg] {case_id}: dice={d:.4f} -> {fname}")


# --- Segmentation: success = MONAI bundle E2E; failure = legacy 2D U-Net empty pred ---
seg_overlay("TCGA-CS-6669", "outputs/e2e_TCGA-CS-6669_pred.nii.gz",
            "SEGMENTATION SUCCESS", "seg_success.png", "MONAI Model Zoo bundle (3D)")
seg_overlay("TCGA-DU-7301", "outputs/batch_unet2d/TCGA-DU-7301_pred_mask.nii.gz",
            "SEGMENTATION FAILURE", "seg_failure.png", "Legacy U-Net 2D")


# --- IDH end-to-end montage: success (TN) + false-positive + false-negative ---
def idh_montage():
    # Operating threshold 0.13 (val-selected by macro-F1, artifacts/e2e_idh_config.json).
    # (case_id, e2e pred mask, true label, pred label, P(mutant), tag)
    items = [
        ("TCGA-HT-8111", "outputs/e2e_TCGA-HT-8111_pred.nii.gz", 1, 1, 0.202,
         "SUCCESS (TP)\nMut -> Mut"),
        ("TCGA-CS-6669", "outputs/e2e_TCGA-CS-6669_pred.nii.gz", 0, 1, 0.138,
         "FAILURE (FP)\nWT -> Mut"),
        ("TCGA-DU-8162", "outputs/e2e_TCGA-DU-8162_pred.nii.gz", 0, 1, 0.216,
         "FAILURE (FP)\nWT -> Mut"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, (cid, pp, lab, pred, prob, tag) in zip(axes, items):
        ax.axis("off")
        rec = cases[cid]
        flair = load(rec["modalities"]["flair"])
        gt = load(rec["mask_path"])
        pmask = load(pp)
        z = best_slice(gt)
        img = np.rot90(flair[:, :, z])
        g = np.rot90(gt[:, :, z] > 0)
        p = np.rot90(pmask[:, :, z] > 0)
        ax.imshow(img, cmap="gray")
        ax.contour(g, colors="lime", linewidths=1.0)
        if p.any():
            ax.contour(p, colors="red", linewidths=1.0)
        color = "green" if lab == pred else "red"
        ax.set_title(f"{tag}\n{cid}\nP(mut)={prob:.3f}",
                     color=color, fontsize=10)
    fig.suptitle("IDH End-to-End (MONAI bundle seg + DenseNet121, threshold 0.13)\n"
                 "GT (green) vs Pred mask (red)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "idh_e2e_montage.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[idh] E2E montage -> idh_e2e_montage.png")


idh_montage()


# --- CT/MRI classification montage ---
ct = [
    ("datasets/Kaggle_multimodal/Dataset/Brain Tumor CT scan Images/Tumor/ct_tumor (917).jpg",
     1, 1, 1.000, "SUCCESS\nTumor -> Tumor"),
    ("datasets/Kaggle_multimodal/Dataset/Brain Tumor MRI images/Healthy/mri_healthy (10).jpg",
     0, 0, 0.02, "SUCCESS\nHealthy -> Healthy"),
    ("datasets/Kaggle_multimodal/Dataset/Brain Tumor CT scan Images/Healthy/ct_healthy (264).jpg",
     0, 1, 0.999, "FAILURE (FP)\nHealthy -> Tumor"),
    ("datasets/Kaggle_multimodal/Dataset/Brain Tumor CT scan Images/Tumor/ct_tumor (291).jpg",
     1, 0, 0.006, "FAILURE (FN)\nTumor -> Healthy"),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
for ax, (p, lab, pred, prob, tag) in zip(axes, ct):
    fp = ROOT / p
    ax.axis("off")
    if fp.exists():
        ax.imshow(mpimg.imread(str(fp)), cmap="gray")
    color = "green" if lab == pred else "red"
    ax.set_title(f"{tag}\nP(tumor)={prob:.3f}", color=color, fontsize=11)
fig.suptitle("CT/MRI Tumor Classification (MobileNetV3-Small) — success vs failure", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "cls_ct_montage.png", dpi=110, bbox_inches="tight")
plt.close(fig)
print("[cls] CT montage -> cls_ct_montage.png")
