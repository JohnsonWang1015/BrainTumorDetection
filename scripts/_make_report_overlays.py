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


# --- CT/MRI classification montage (MobileNetV3-Large) ---
# Probabilities are read from the live eval CSV so the montage always matches the
# current checkpoint. Under the Large model the error set is 0 FP / 5 FN, so both
# failure panels are missed tumors (the Small-era false positives are now correct).
_ct_pred = {}
_ct_csv = ROOT / "outputs" / "eval_ct_predictions.csv"
if _ct_csv.exists():
    for _r in csv.DictReader(open(_ct_csv)):
        _ct_pred[os.path.basename(_r["path"])] = (
            int(_r["label"]), int(_r["pred"]), float(_r["prob_tumor"]))

ct = [
    ("datasets/Kaggle_multimodal/Dataset/Brain Tumor CT scan Images/Tumor/ct_tumor (917).jpg",
     "SUCCESS"),
    ("datasets/Kaggle_multimodal/Dataset/Brain Tumor MRI images/Healthy/mri_healthy (196).jpg",
     "SUCCESS"),
    ("datasets/Kaggle_multimodal/Dataset/Brain Tumor CT scan Images/Tumor/ct_tumor (1908).jpg",
     "FAILURE"),
    ("datasets/Kaggle_multimodal/Dataset/Brain Tumor MRI images/Tumor/meningioma (55).jpg",
     "FAILURE"),
]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
for ax, (p, kind) in zip(axes, ct):
    fp = ROOT / p
    ax.axis("off")
    if fp.exists():
        ax.imshow(mpimg.imread(str(fp)), cmap="gray")
    lab, pred, prob = _ct_pred.get(os.path.basename(p), (None, None, float("nan")))
    truth = {1: "Tumor", 0: "Healthy"}.get(lab, "?")
    guess = {1: "Tumor", 0: "Healthy"}.get(pred, "?")
    ok = (lab == pred)
    outcome = "" if ok else ("(FP)" if pred == 1 else "(FN)")
    color = "green" if ok else "red"
    ax.set_title(f"{kind} {outcome}\n{truth} -> {guess}\nP(tumor)={prob:.3f}",
                 color=color, fontsize=11)
fig.suptitle("CT/MRI Tumor Classification (MobileNetV3-Large) — success vs failure\n"
             "real probs from eval_ct_predictions.csv | both failures are missed tumors (0 FP / 5 FN)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.92))
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
    # (relative path, modality tag) — labels/preds/probs come from the live CSV and the
    # outcome (TP/TN/FP/FN) is derived, never asserted. Under MobileNetV3-Large the test
    # error set is 0 FP / 5 FN, so the bottom row is all missed tumors (2 CT + 2 MRI;
    # the 3 MRI misses are meningiomas, a non-glioma subtype under-represented in training).
    picks = [
        (f"{D}/Brain Tumor CT scan Images/Tumor/ct_tumor (917).jpg",     "CT"),   # TP
        (f"{D}/Brain Tumor CT scan Images/Healthy/ct_healthy (955).jpg", "CT"),   # TN
        (f"{D}/Brain Tumor MRI images/Tumor/glioma (246).jpg",           "MRI"),  # TP
        (f"{D}/Brain Tumor MRI images/Healthy/mri_healthy (196).jpg",    "MRI"),  # TN
        (f"{D}/Brain Tumor CT scan Images/Tumor/ct_tumor (7).jpg",       "CT"),   # FN
        (f"{D}/Brain Tumor CT scan Images/Tumor/ct_tumor (1908).jpg",    "CT"),   # FN
        (f"{D}/Brain Tumor MRI images/Tumor/meningioma (55).jpg",        "MRI"),  # FN
        (f"{D}/Brain Tumor MRI images/Tumor/meningioma (914).jpg",       "MRI"),  # FN
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax, (p, modtag) in zip(axes.ravel(), picks):
        ax.axis("off")
        fp = ROOT / p
        name = os.path.basename(p)
        if fp.exists():
            ax.imshow(mpimg.imread(str(fp)), cmap="gray")
        lab, pred, prob = by_name.get(name, (None, None, float("nan")))
        truth = {1: "Tumor", 0: "Healthy"}.get(lab, "?")
        guess = {1: "Tumor", 0: "Healthy"}.get(pred, "?")
        ok = (lab == pred)
        outcome = ("TP" if lab == 1 else "TN") if ok else ("FP" if pred == 1 else "FN")
        color = "green" if ok else "red"
        mark = "OK" if ok else "X"
        ax.set_title(f"[{mark}] {modtag} {outcome}: {truth} -> {guess}\n{name}\nP(tumor)={prob:.3f}",
                     color=color, fontsize=9)
    fig.suptitle("CT/MRI Tumor Classification (MobileNetV3-Large) — 8-case inference gallery\n"
                 "top: correct (TP/TN)   bottom: errors (all FN — 0 FP under Large)   |   real probs from eval_ct_predictions.csv",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "cls_ct_gallery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("[cls] extended CT/MRI gallery -> cls_ct_gallery.png")


ct_gallery_extended()


# --- Segmentation gallery: every available E2E predicted mask (MONAI bundle) ---
def seg_gallery():
    """Grid of all MONAI-bundle E2E segmentation results (FLAIR + GT green + Pred red),
    sorted best -> worst Dice so the spread of quality is visible at a glance."""
    import glob
    import re
    cids = sorted(
        re.sub(r"^.*e2e_(.+)_pred\.nii\.gz$", r"\1", p)
        for p in glob.glob("outputs/e2e_*_pred.nii.gz")
    )
    cids = [c for c in cids if c in cases]
    scored = []
    for cid in cids:
        gt = load(cases[cid]["mask_path"])
        pred = load(f"outputs/e2e_{cid}_pred.nii.gz")
        scored.append((dice(gt, pred), cid))
    scored.sort(reverse=True)

    n = len(scored)
    ncol = 5 if n > 6 else 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (d, cid) in zip(axes, scored):
        rec = cases[cid]
        flair = load(rec["modalities"]["flair"])
        gt = load(rec["mask_path"])
        pred = load(f"outputs/e2e_{cid}_pred.nii.gz")
        z = best_slice(gt)
        ax.imshow(np.rot90(flair[:, :, z]), cmap="gray")
        ax.contour(np.rot90(gt[:, :, z] > 0), colors="lime", linewidths=1.0)
        if (pred[:, :, z] > 0).any():
            ax.contour(np.rot90(pred[:, :, z] > 0), colors="red", linewidths=1.0)
        ax.set_title(f"{cid}\nDice = {d:.3f}", fontsize=9)
    fig.suptitle(f"MONAI Model Zoo bundle segmentation — all {n} test cases (sorted by Dice)\n"
                 "GT (green) vs Pred (red), best tumor slice", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "seg_gallery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[seg] multi-case gallery ({n} cases) -> seg_gallery.png")


seg_gallery()


# --- 3D orbiting GIF of the tumor: GT (green) vs predicted (red) surfaces ---
def seg_orbit_gif(cid="TCGA-DU-7301", n_frames=36, step=2):
    """Render GT and predicted tumor surfaces (marching cubes) and orbit the camera
    360 deg to produce a rotating 3D .gif. Geometry is computed once; only the view
    azimuth changes per frame, so all n_frames share the same mesh."""
    import imageio.v2 as imageio
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rec = cases[cid]
    gt = (load(rec["mask_path"]) > 0).astype(np.uint8)
    pred = (load(f"outputs/e2e_{cid}_pred.nii.gz") > 0).astype(np.uint8)
    d = dice(gt, pred)

    def mesh(vol):
        # pad so marching_cubes closes surfaces touching the volume border
        v = np.pad(vol, 1)
        verts, faces, _, _ = measure.marching_cubes(v, level=0.5, step_size=step)
        return verts, faces

    gv, gf = mesh(gt)
    pv, pf = mesh(pred)

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    # GT as a faint translucent green shell for reference (drawn first / behind).
    ax.add_collection3d(Poly3DCollection(gv[gf], alpha=0.12, facecolor="lime", edgecolor="none"))
    # Predicted tumor as the main shaded surface (depth-lit via plot_trisurf).
    ax.plot_trisurf(
        pv[:, 0], pv[:, 1], pf, pv[:, 2],
        cmap="YlOrRd", linewidth=0, antialiased=True, shade=True, alpha=0.95,
    )
    allv = np.vstack([gv, pv])
    mins, maxs = allv.min(0), allv.max(0)
    ctr = (mins + maxs) / 2
    rad = (maxs - mins).max() / 2 * 1.1
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.suptitle(f"{cid} — 3D tumor: GT (green) vs Pred (red)\nDice = {d:.3f}", fontsize=11)

    frames = []
    for azim in np.linspace(0, 360, n_frames, endpoint=False):
        ax.view_init(elev=18, azim=float(azim))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    gif_path = OUT / "seg_orbit_3d.gif"
    imageio.mimsave(str(gif_path), frames, duration=0.08, loop=0)
    print(f"[seg] 3D orbit gif ({cid}, {n_frames} frames) -> seg_orbit_3d.gif")


seg_orbit_gif()


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


# --- 3D MRI glass-brain orbit: translucent brain shell with the tumor nested inside ---
def mri_orbit_gif(cid="TCGA-HT-8111", n_frames=36, brain_step=3, tumor_step=2,
                  out_name="mri_orbit_3d.gif", subtitle=""):
    """Render the skull-stripped FLAIR brain as a translucent 'glass' shell (marching
    cubes on the non-zero brain region — BraTS volumes are skull-stripped, so >0 is
    brain), then nest the GT (green) and predicted (red) tumor surfaces inside it and
    orbit the camera 360 deg. Unlike seg_orbit_3d.gif (tumor meshes only), this places
    the tumor in its anatomical context so you can see *where in the brain* it sits.

    out_name / subtitle let the same renderer produce both the canonical orbit and a
    failure-case variant (e.g. CS-6669, a WT case the IDH head misreads as mutant)."""
    import imageio.v2 as imageio
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy import ndimage

    rec = cases[cid]
    flair = load(rec["modalities"]["flair"]).astype(np.float32)
    gt = (load(rec["mask_path"]) > 0).astype(np.uint8)
    pred = (load(f"outputs/e2e_{cid}_pred.nii.gz") > 0).astype(np.uint8)
    d = dice(gt, pred)  # full-mask Dice (incl. peripheral FP), reported honestly

    def largest_cc(vol):
        """Keep only the largest 3D connected component — for the orbit we render the
        main tumor body, dropping the scattered peripheral FP specks that would
        otherwise clutter the glass brain (those FP are quantified in seg_error_decomp)."""
        lab, n = ndimage.label(vol)
        if n <= 1:
            return vol
        sizes = ndimage.sum(vol, lab, index=range(1, n + 1))
        return (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)

    gt = largest_cc(gt)
    pred = largest_cc(pred)

    # Brain mask = non-zero FLAIR, lightly cleaned (close holes, drop specks) so the
    # marching-cubes shell is a single smooth surface rather than noisy fragments.
    brain = flair > (flair[flair > 0].mean() * 0.10)
    brain = ndimage.binary_closing(brain, iterations=2)
    brain = ndimage.binary_fill_holes(brain)

    def mesh(vol, step):
        v = np.pad(vol.astype(np.uint8), 1)
        verts, faces, _, _ = measure.marching_cubes(v, level=0.5, step_size=step)
        return verts, faces

    bv, bf = mesh(brain, brain_step)
    gv, gf = mesh(gt, tumor_step)
    pv, pf = mesh(pred, tumor_step)

    fig = plt.figure(figsize=(5.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    # Glass brain: very low alpha so the tumor inside stays visible through the shell.
    ax.add_collection3d(Poly3DCollection(
        bv[bf], alpha=0.06, facecolor="#7fb3ff", edgecolor="none"))
    # GT shell (faint green) for reference, predicted tumor as the solid red body.
    ax.add_collection3d(Poly3DCollection(
        gv[gf], alpha=0.18, facecolor="lime", edgecolor="none"))
    ax.plot_trisurf(
        pv[:, 0], pv[:, 1], pf, pv[:, 2],
        color="#ff2b2b", linewidth=0, antialiased=True, shade=True, alpha=0.95,
    )
    mins, maxs = bv.min(0), bv.max(0)
    ctr = (mins + maxs) / 2
    rad = (maxs - mins).max() / 2 * 1.05
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    head = f"{cid} — 3D MRI glass brain + tumor orbit"
    if subtitle:
        head += f"\n{subtitle}"
    fig.suptitle(f"{head}\n"
                 f"brain shell (blue) · GT (green) · Pred (red)  |  Dice = {d:.3f}",
                 fontsize=11)

    frames = []
    for azim in np.linspace(0, 360, n_frames, endpoint=False):
        ax.view_init(elev=12, azim=float(azim))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

    gif_path = OUT / out_name
    imageio.mimsave(str(gif_path), frames, duration=0.08, loop=0)
    print(f"[seg] 3D MRI glass-brain orbit ({cid}, {n_frames} frames) -> {out_name}")


mri_orbit_gif()
# Failure-case variant: CS-6669 is a true IDH-wildtype case the end-to-end pipeline
# misreads as mutant (P=0.138 > threshold 0.13) DESPITE a high seg Dice — the glass
# brain shows the tumor is cleanly localized, so the failure is in the IDH head, not seg.
mri_orbit_gif(cid="TCGA-CS-6669", out_name="mri_orbit_failure_3d.gif",
              subtitle="FAILURE case: true WT misread as Mut by the IDH head (seg is fine)")


# --- IDH end-to-end gallery: all 10 test cases, real probs from eval_e2e_zoo CSV ---
def idh_e2e_gallery():
    """Grid of every test case's IDH end-to-end result (FLAIR + GT green + Pred-mask
    red), titled with truth -> prediction and the real P(mutant) pulled from
    outputs/eval_e2e_zoo/eval_e2e_zoo_cases.csv. Sorted WT-first then by probability so
    the decision threshold (0.13) reads left-to-right. Green title = correct, red = error."""
    csv_path = ROOT / "outputs" / "eval_e2e_zoo" / "eval_e2e_zoo_cases.csv"
    cfg = json.load(open("artifacts/e2e_idh_config.json"))
    thr = float(cfg["threshold"])
    rows = list(csv.DictReader(open(csv_path)))
    items = []
    for r in rows:
        cid = r["case_id"]
        if cid not in cases:
            continue
        items.append({
            "cid": cid,
            "label": int(r["label"]),
            "pred": int(r["case_pred"]),
            "prob": float(r["case_prob"]),
            "seg_dice": float(r["seg_dice"]),
        })
    # WT (label 0) first, then by ascending P(mutant) so threshold ordering is visible.
    items.sort(key=lambda x: (x["label"], x["prob"]))

    n = len(items)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    name = {1: "Mut", 0: "WT"}
    for ax, it in zip(axes, items):
        cid = it["cid"]
        rec = cases[cid]
        flair = load(rec["modalities"]["flair"])
        gt = load(rec["mask_path"])
        pmask = load(f"outputs/e2e_{cid}_pred.nii.gz")
        z = best_slice(gt)
        ax.imshow(np.rot90(flair[:, :, z]), cmap="gray")
        ax.contour(np.rot90(gt[:, :, z] > 0), colors="lime", linewidths=1.0)
        if (pmask[:, :, z] > 0).any():
            ax.contour(np.rot90(pmask[:, :, z] > 0), colors="red", linewidths=1.0)
        ok = it["label"] == it["pred"]
        outcome = ("TP" if it["label"] == 1 else "TN") if ok else \
                  ("FP" if it["pred"] == 1 else "FN")
        color = "green" if ok else "red"
        mark = "OK" if ok else "X"
        ax.set_title(
            f"[{mark}] {outcome}  {name[it['label']]} -> {name[it['pred']]}\n"
            f"{cid}\nP(mut)={it['prob']:.3f}  seg Dice={it['seg_dice']:.3f}",
            color=color, fontsize=8.5)
    fig.suptitle(
        f"IDH End-to-End — all {n} test cases (MONAI bundle seg + DenseNet121, threshold {thr:.2f})\n"
        "GT (green) vs Pred mask (red) · sorted WT-first then by P(mutant) · real probs from eval CSV",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "idh_e2e_gallery.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[idh] E2E gallery ({n} cases) -> idh_e2e_gallery.png")


idh_e2e_gallery()


# ======================================================================================
# Additional 3D orbiting visualizations (round 2)
# ======================================================================================

def _largest_cc(vol):
    """Keep only the largest 3D connected component of a binary volume."""
    from scipy import ndimage
    lab, n = ndimage.label(vol)
    if n <= 1:
        return vol.astype(np.uint8)
    sizes = ndimage.sum(vol, lab, index=range(1, n + 1))
    return (lab == (int(np.argmax(sizes)) + 1)).astype(np.uint8)


def _mesh(vol, step):
    """Marching-cubes surface, padded so border-touching components close cleanly."""
    from skimage import measure
    v = np.pad(vol.astype(np.uint8), 1)
    verts, faces, _, _ = measure.marching_cubes(v, level=0.5, step_size=step)
    return verts, faces


def _equal_box(ax, verts, pad=1.1):
    mins, maxs = verts.min(0), verts.max(0)
    ctr = (mins + maxs) / 2
    rad = (maxs - mins).max() / 2 * pad
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


# --- 3D error-decomposition orbit: TP / FN / FP as separate orbiting surfaces ---
def seg_error_orbit_3d(cid="TCGA-DU-A6S8", n_frames=36, step=2):
    """The volumetric counterpart of seg_error_decomp.png: split the E2E prediction into
    TP (yellow, correct overlap), FN (green, missed GT) and FP (red, over-segmentation)
    and orbit all three surfaces together. DU-A6S8 (Dice 0.879) has a large TP core with
    both visible FN (missed, green) and FP (extra, red) shells at the periphery — so the
    3D view shows exactly where the Dice loss lives instead of inferring it from a slice."""
    import imageio.v2 as imageio
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rec = cases[cid]
    gt = (load(rec["mask_path"]) > 0).astype(np.uint8)
    pred = (load(f"outputs/e2e_{cid}_pred.nii.gz") > 0).astype(np.uint8)
    d = dice(gt, pred)
    tp = (gt & pred).astype(np.uint8)
    fn = (gt & ~pred).astype(np.uint8)
    fp = (~gt.astype(bool) & pred.astype(bool)).astype(np.uint8)

    fig = plt.figure(figsize=(5.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    allv = []
    # TP as a solid shaded body (the bulk), so it stays visible; FN/FP as translucent
    # colored shells/specks on top. DU-8162 over-segments, so red FP scatters around the
    # yellow core — that scatter IS the peripheral Dice loss the report discusses.
    tv, tf = _mesh(_largest_cc(tp), step)
    ax.plot_trisurf(tv[:, 0], tv[:, 1], tf, tv[:, 2],
                    color="#ffcf33", linewidth=0, antialiased=True, shade=True, alpha=0.95)
    allv.append(tv)
    for vol, col, al in [(fn, "#00d400", 0.4), (fp, "#ff2b2b", 0.45)]:
        if int(vol.sum()) < 8:
            continue
        v, f = _mesh(vol, step)
        ax.add_collection3d(Poly3DCollection(v[f], alpha=al, facecolor=col, edgecolor="none"))
        allv.append(v)
    _equal_box(ax, np.vstack(allv))
    tpn, fnn, fpn = int(tp.sum()), int(fn.sum()), int(fp.sum())
    fig.suptitle(f"{cid} — 3D error decomposition orbit  |  Dice = {d:.3f}\n"
                 f"TP (yellow, {tpn}) · FN (green, {fnn}) · FP (red, {fpn})", fontsize=11)

    frames = []
    for azim in np.linspace(0, 360, n_frames, endpoint=False):
        ax.view_init(elev=16, azim=float(azim))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(str(OUT / "seg_error_orbit_3d.gif"), frames, duration=0.08, loop=0)
    print(f"[seg] 3D error-decomposition orbit ({cid}) -> seg_error_orbit_3d.gif")


seg_error_orbit_3d()


# --- Multi-case tumor-shape orbit: several predicted tumors orbiting side by side ---
def seg_orbit_multicase(cids=("TCGA-DU-7301", "TCGA-HT-8111", "TCGA-DU-8162", "TCGA-CS-6669"),
                        n_frames=36, step=2):
    """Render each case's predicted tumor (largest connected component) in its own 3D
    panel and orbit them in lockstep. Shows the spread of 3D tumor shapes the bundle
    handles — from compact (DU-7301, Dice 0.96) to irregular (CS-6669) — at a glance."""
    import imageio.v2 as imageio
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    panels = []
    for cid in cids:
        rec = cases[cid]
        gt = (load(rec["mask_path"]) > 0).astype(np.uint8)
        pred = (load(f"outputs/e2e_{cid}_pred.nii.gz") > 0).astype(np.uint8)
        d = dice(gt, pred)
        gv, gf = _mesh(_largest_cc(gt), step)
        pv, pf = _mesh(_largest_cc(pred), step)
        panels.append((cid, d, gv, gf, pv, pf))

    ncol = 2
    nrow = (len(panels) + ncol - 1) // ncol
    fig = plt.figure(figsize=(5.0 * ncol, 5.0 * nrow))
    axes = []
    for i, (cid, d, gv, gf, pv, pf) in enumerate(panels):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(gv[gf], alpha=0.15, facecolor="lime", edgecolor="none"))
        ax.plot_trisurf(pv[:, 0], pv[:, 1], pf, pv[:, 2],
                        cmap="YlOrRd", linewidth=0, antialiased=True, shade=True, alpha=0.95)
        _equal_box(ax, np.vstack([gv, pv]))
        ax.set_title(f"{cid}\nDice = {d:.3f}", fontsize=10)
        axes.append(ax)
    fig.suptitle("3D tumor-shape orbit — predicted surface (red) vs GT (green), per test case\n"
                 "largest connected component; orbiting in lockstep", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    frames = []
    for azim in np.linspace(0, 360, n_frames, endpoint=False):
        for ax in axes:
            ax.view_init(elev=18, azim=float(azim))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(str(OUT / "seg_orbit_multicase.gif"), frames, duration=0.08, loop=0)
    print(f"[seg] multi-case tumor-shape orbit ({len(panels)} cases) -> seg_orbit_multicase.gif")


seg_orbit_multicase()


# --- ROI bounding-box orbit: the 3D crop fed to the DenseNet121 IDH classifier ---
def idh_roi_orbit_3d(cid="TCGA-DU-7301", n_frames=36, step=2, margin=4):
    """Draw the predicted tumor surface plus the axis-aligned 3D bounding box (with the
    training margin=4) that the end-to-end pipeline crops and resizes to 96^3 before
    feeding the DenseNet121 IDH head. Orbiting the box makes the seg -> classify ROI
    hand-off explicit: the IDH decision sees only what is inside this cage."""
    import imageio.v2 as imageio
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    rec = cases[cid]
    pred = _largest_cc((load(f"outputs/e2e_{cid}_pred.nii.gz") > 0).astype(np.uint8))
    pv, pf = _mesh(pred, step)  # note: meshed on padded vol, so verts are +1 shifted

    # Bounding box from the predicted mask, expanded by `margin`, clamped to the volume.
    idx = np.argwhere(pred > 0)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin, np.array(pred.shape) - 1)
    lo = lo + 1.0; hi = hi + 1.0  # match the +1 marching-cubes pad shift

    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    edges_idx = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                 (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    segs = [[corners[a], corners[b]] for a, b in edges_idx]

    fig = plt.figure(figsize=(5.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(pv[:, 0], pv[:, 1], pf, pv[:, 2],
                    cmap="YlOrRd", linewidth=0, antialiased=True, shade=True, alpha=0.95)
    ax.add_collection3d(Line3DCollection(segs, colors="#1f77ff", linewidths=1.6))
    # six translucent faces so the cage reads as a box, not just wires
    faces_idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    box_faces = [[corners[i] for i in f] for f in faces_idx]
    ax.add_collection3d(Poly3DCollection(box_faces, alpha=0.05, facecolor="#1f77ff", edgecolor="none"))
    _equal_box(ax, np.vstack([pv, corners]))
    dims = (hi - lo).astype(int)
    fig.suptitle(f"{cid} — IDH ROI bounding box (margin {margin}) around predicted tumor\n"
                 f"box {dims[0]}x{dims[1]}x{dims[2]} -> resized to 96^3 for DenseNet121", fontsize=11)

    frames = []
    for azim in np.linspace(0, 360, n_frames, endpoint=False):
        ax.view_init(elev=16, azim=float(azim))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(str(OUT / "idh_roi_orbit_3d.gif"), frames, duration=0.08, loop=0)
    print(f"[idh] 3D ROI bounding-box orbit ({cid}) -> idh_roi_orbit_3d.gif")


idh_roi_orbit_3d()
