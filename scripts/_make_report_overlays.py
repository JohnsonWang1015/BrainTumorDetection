"""Generate figures for the model report:
  1. Segmentation overlays (FLAIR + GT green + Pred red) for a success and a failure case.
  2. CT/MRI classification montage (success + failure inputs with predicted prob).
"""
from __future__ import annotations

import csv
import json
import os
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


# --- Extended CT/MRI classification gallery: 2x4 real examples across both modalities ---
def ct_gallery_extended():
    """8-panel gallery (CT + MRI x TP/TN/FP/FN) with REAL probabilities pulled
    from outputs/eval_ct_predictions.csv so the figure matches the eval run exactly."""
    pred_csv = ROOT / "outputs" / "eval_ct_predictions.csv"
    by_name = {}
    if pred_csv.exists():
        for r in csv.DictReader(open(pred_csv)):
            by_name[os.path.basename(r["path"])] = (
                int(r["label"]), int(r["pred"]), float(r["prob_tumor"]))

    D = "datasets/Kaggle_multimodal/Dataset"
    # (relative path, modality tag, outcome tag) — probs/labels come from the CSV.
    picks = [
        (f"{D}/Brain Tumor CT scan Images/Tumor/ct_tumor (917).jpg",   "CT",  "TP"),
        (f"{D}/Brain Tumor CT scan Images/Healthy/ct_healthy (955).jpg","CT",  "TN"),
        (f"{D}/Brain Tumor MRI images/Tumor/glioma (246).jpg",          "MRI", "TP"),
        (f"{D}/Brain Tumor MRI images/Healthy/mri_healthy (196).jpg",   "MRI", "TN"),
        (f"{D}/Brain Tumor CT scan Images/Healthy/ct_healthy (264).jpg","CT",  "FP"),
        (f"{D}/Brain Tumor CT scan Images/Tumor/ct_tumor (291).jpg",    "CT",  "FN"),
        (f"{D}/Brain Tumor MRI images/Healthy/mri_healthy (1005).jpg",  "MRI", "FP"),
        (f"{D}/Brain Tumor MRI images/Tumor/meningioma (55).jpg",       "MRI", "FN"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax, (p, modtag, outcome) in zip(axes.ravel(), picks):
        ax.axis("off")
        fp = ROOT / p
        name = os.path.basename(p)
        if fp.exists():
            ax.imshow(mpimg.imread(str(fp)), cmap="gray")
        lab, pred, prob = by_name.get(name, (None, None, float("nan")))
        truth = {1: "Tumor", 0: "Healthy"}.get(lab, "?")
        guess = {1: "Tumor", 0: "Healthy"}.get(pred, "?")
        ok = (lab == pred)
        color = "green" if ok else "red"
        mark = "OK" if ok else "X"
        ax.set_title(f"[{mark}] {modtag} {outcome}: {truth} -> {guess}\n{name}\nP(tumor)={prob:.3f}",
                     color=color, fontsize=9)
    fig.suptitle("CT/MRI Tumor Classification (MobileNetV3-Small) — 8-case inference gallery\n"
                 "top: correct (TP/TN)   bottom: errors (FP/FN)   |   real probs from eval_ct_predictions.csv",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "cls_ct_gallery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[cls] extended CT/MRI gallery -> cls_ct_gallery.png")


ct_gallery_extended()


# --- Segmentation gallery: all 3 available E2E predicted masks (MONAI bundle) ---
def seg_gallery():
    """Row of 3 MONAI-bundle segmentation results (FLAIR + GT green + Pred red)."""
    items = ["TCGA-HT-8111", "TCGA-DU-8162", "TCGA-CS-6669"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
    for ax, cid in zip(axes, items):
        ax.axis("off")
        rec = cases[cid]
        flair = load(rec["modalities"]["flair"])
        gt = load(rec["mask_path"])
        pred = load(f"outputs/e2e_{cid}_pred.nii.gz")
        z = best_slice(gt)
        img = np.rot90(flair[:, :, z])
        ax.imshow(img, cmap="gray")
        ax.contour(np.rot90(gt[:, :, z] > 0), colors="lime", linewidths=1.1)
        ax.contour(np.rot90(pred[:, :, z] > 0), colors="red", linewidths=1.1)
        ax.set_title(f"{cid}\nDice = {dice(gt, pred):.3f}", fontsize=11)
    fig.suptitle("MONAI Model Zoo bundle segmentation — multi-case inference\n"
                 "GT (green) vs Pred (red), best tumor slice", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "seg_gallery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[seg] multi-case gallery -> seg_gallery.png")


seg_gallery()


# --- Multi-slice volumetric view of the best segmentation case ---
def seg_multislice(cid="TCGA-HT-8111", n=5):
    """Show n evenly spaced axial slices through the tumor to demonstrate
    volumetric consistency of the 3D bundle prediction."""
    rec = cases[cid]
    flair = load(rec["modalities"]["flair"])
    gt = load(rec["mask_path"])
    pred = load(f"outputs/e2e_{cid}_pred.nii.gz")
    zs = np.where((gt > 0).sum(axis=(0, 1)) > 0)[0]
    if len(zs) == 0:
        return
    sel = np.linspace(zs[0], zs[-1], n).astype(int)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
    for ax, z in zip(axes, sel):
        ax.axis("off")
        img = np.rot90(flair[:, :, z])
        ax.imshow(img, cmap="gray")
        g = np.rot90(gt[:, :, z] > 0)
        p = np.rot90(pred[:, :, z] > 0)
        if g.any():
            ax.contour(g, colors="lime", linewidths=1.0)
        if p.any():
            ax.contour(p, colors="red", linewidths=1.0)
        ax.set_title(f"slice z={z}", fontsize=10)
    fig.suptitle(f"{cid} — volumetric segmentation across tumor slices "
                 f"(overall Dice = {dice(gt, pred):.3f})\nGT (green) vs Pred (red)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "seg_multislice.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[seg] multi-slice view ({cid}) -> seg_multislice.png")


seg_multislice()


# --- Segmentation error decomposition: TP / FN / FP per case (where Dice loss comes from) ---
def seg_error_decomp(cids=("TCGA-HT-8111", "TCGA-DU-8162", "TCGA-CS-6669")):
    """Color-code each E2E prediction into TP (correct overlap), FN (missed GT),
    and FP (over-segmentation) so the figure literally shows where the Dice loss
    lives — reinforcing that the loss is peripheral FP, not main-body contour error."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, len(cids), figsize=(4.5 * len(cids), 5.0))
    if len(cids) == 1:
        axes = [axes]
    for ax, cid in zip(axes, cids):
        ax.axis("off")
        rec = cases[cid]
        flair = load(rec["modalities"]["flair"])
        gt = load(rec["mask_path"])
        pred = load(f"outputs/e2e_{cid}_pred.nii.gz")
        z = best_slice(gt)
        img = np.rot90(flair[:, :, z])
        g = np.rot90(gt[:, :, z] > 0)
        p = np.rot90(pred[:, :, z] > 0)
        lbl = np.zeros(g.shape, dtype=int)          # 0 bg, 1 TP, 2 FN, 3 FP
        lbl[g & p] = 1
        lbl[g & ~p] = 2
        lbl[~g & p] = 3
        ax.imshow(img, cmap="gray")
        masked = np.ma.masked_where(lbl == 0, lbl)
        cmap = ListedColormap(["#ffd400", "#00d400", "#ff2b2b"])  # TP / FN / FP
        ax.imshow(masked, cmap=cmap, vmin=1, vmax=3, alpha=0.45)
        tp = int((g & p).sum())
        fn = int((g & ~p).sum())
        fp = int((~g & p).sum())
        ax.set_title(f"{cid}\nDice={dice(gt, pred):.3f}  (slice z={z})\n"
                     f"TP={tp}  FN={fn}  FP={fp}", fontsize=10)
    handles = [Patch(color="#ffd400", label="TP (correct overlap)"),
               Patch(color="#00d400", label="FN (missed GT)"),
               Patch(color="#ff2b2b", label="FP (over-segmentation)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle("Segmentation error decomposition (MONAI bundle E2E, best tumor slice)\n"
                 "where the Dice loss comes from: correct overlap vs missed vs extra voxels",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(OUT / "seg_error_decomp.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[seg] error decomposition -> seg_error_decomp.png")


seg_error_decomp()


# --- Cross-modality consistency: same 3D prediction overlaid on all 4 input modalities ---
def seg_modalities(cid="TCGA-HT-8111"):
    """Overlay the identical GT/Pred contours on FLAIR / T1 / T1Gd / T2 to show the
    3D prediction tracks the anatomical tumor boundary, not one modality's intensity."""
    rec = cases[cid]
    mods = [("flair", "FLAIR"), ("t1", "T1"), ("t1Gd", "T1Gd"), ("t2", "T2")]
    gt = load(rec["mask_path"])
    pred = load(f"outputs/e2e_{cid}_pred.nii.gz")
    z = best_slice(gt)
    g = np.rot90(gt[:, :, z] > 0)
    p = np.rot90(pred[:, :, z] > 0)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    for ax, (key, name) in zip(axes, mods):
        ax.axis("off")
        img = np.rot90(load(rec["modalities"][key])[:, :, z])
        ax.imshow(img, cmap="gray")
        ax.contour(g, colors="lime", linewidths=1.0)
        if p.any():
            ax.contour(p, colors="red", linewidths=1.0)
        ax.set_title(name, fontsize=12)
    fig.suptitle(f"{cid} — one 3D prediction overlaid on all 4 input modalities "
                 f"(slice z={z}, Dice = {dice(gt, pred):.3f})\nGT (green) vs Pred (red)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "seg_modalities.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[seg] cross-modality panel ({cid}) -> seg_modalities.png")


seg_modalities()
