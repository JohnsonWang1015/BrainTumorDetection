"""3D MONAI IDH classification handler for the Gradio app.

Production pipeline: MONAI Model Zoo brats_mri_segmentation bundle (zero-shot)
for tumor segmentation, then 3D DenseNet121 (jitter-trained) on the
predicted-mask 3D ROI for IDH classification. Uses the calibrated decision
threshold persisted in the classifier checkpoint.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.networks.nets import DenseNet121, SegResNet
from monai.transforms import KeepLargestConnectedComponent, NormalizeIntensity

_BUNDLE_CKPT = Path("checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt")
_CLS_CKPT = Path("checkpoints/densenet3d_idh_jitter.pt")

_SEG_MODEL: torch.nn.Module | None = None
_CLS_MODEL: torch.nn.Module | None = None
_CLS_META: dict | None = None
_DEVICE: torch.device | None = None


def _device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                _DEVICE = torch.device("cuda")
            except RuntimeError:
                pass
    return _DEVICE


def _get_seg() -> torch.nn.Module:
    global _SEG_MODEL
    if _SEG_MODEL is not None:
        return _SEG_MODEL
    dev = _device()
    seg = SegResNet(
        spatial_dims=3, in_channels=4, out_channels=3, init_filters=16,
        blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2,
    ).to(dev)
    state = torch.load(_BUNDLE_CKPT, map_location=dev, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    seg.load_state_dict(state)
    seg.train(False)
    _SEG_MODEL = seg
    return _SEG_MODEL


def _get_cls() -> tuple[torch.nn.Module, dict]:
    global _CLS_MODEL, _CLS_META
    if _CLS_MODEL is not None and _CLS_META is not None:
        return _CLS_MODEL, _CLS_META
    dev = _device()
    state = torch.load(_CLS_CKPT, map_location=dev, weights_only=True)
    cls = DenseNet121(spatial_dims=3, in_channels=4, out_channels=1, dropout_prob=0.2).to(dev)
    cls.load_state_dict(state["model"])
    cls.train(False)
    meta = {
        "target_size": tuple(state.get("target_size", (96, 96, 96))),
        "margin": int(state.get("margin", 4)),
        "threshold": float(state.get("threshold", 0.5)),
    }
    _CLS_MODEL = cls
    _CLS_META = meta
    return _CLS_MODEL, _CLS_META


def _bbox3d(mask, margin=4):
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        return None
    mn, mx = nz.min(0), nz.max(0) + 1
    h, w, d = mask.shape
    return (
        max(int(mn[0]) - margin, 0), min(int(mx[0]) + margin, h),
        max(int(mn[1]) - margin, 0), min(int(mx[1]) + margin, w),
        max(int(mn[2]) - margin, 0), min(int(mx[2]) + margin, d),
    )


def _zsc(c):
    return (c - c.mean()) / (c.std() + 1e-6)


def _render_overview(flair, pred_mask, idh_prob, idh_class, threshold):
    sums = pred_mask.sum(axis=(0, 1))
    z_show = int(np.argmax(sums)) if sums.max() > 0 else flair.shape[-1] // 2
    flair_slice = flair[:, :, z_show]
    mask_slice = pred_mask[:, :, z_show]
    fmin, fmax = float(flair_slice.min()), float(flair_slice.max())
    flair_norm = ((flair_slice - fmin) / (fmax - fmin + 1e-6) * 255).astype(np.uint8)
    flair_rgb = np.stack([flair_norm] * 3, axis=-1)
    overlay = flair_rgb.copy()
    overlay[mask_slice > 0] = (overlay[mask_slice > 0] * 0.35 + np.array([255, 60, 60]) * 0.65).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=100)
    axes[0].imshow(np.rot90(flair_norm), cmap="gray")
    axes[0].set_title(f"FLAIR (z={z_show})", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(np.rot90(mask_slice), cmap="gray")
    axes[1].set_title("MONAI bundle mask (zero-shot)", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    axes[2].imshow(np.rot90(overlay))
    axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    label = "IDH Mutant" if idh_class == 1 else ("IDH Wildtype" if idh_class == 0 else "Undetermined")
    color = "#e67e22" if idh_class == 1 else ("#2980b9" if idh_class == 0 else "#7f8c8d")
    fig.suptitle(
        f"{label}  |  prob={idh_prob:.3f}  |  thresh={threshold:.3f}",
        fontsize=15, fontweight="bold", color=color, y=1.01,
    )
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


def _build_diagnosis(idh_prob, idh_class, threshold, n_voxels, latency):
    if np.isnan(idh_prob):
        return (
            "## 3D MONAI IDH Analysis\n\n"
            "**Result**: No tumor detected by the bundle segmentation -- IDH classification skipped.\n\n"
            f"**Inference time**: {latency * 1000:.0f} ms"
        )
    label = "IDH Mutant (1)" if idh_class == 1 else "IDH Wildtype (0)"
    margin = abs(idh_prob - threshold)
    if margin < 0.05:
        note = "⚠ within 0.05 of threshold -- treat as borderline."
    elif margin < 0.15:
        note = "Result is close to the threshold."
    else:
        note = "Result is well separated from the threshold."
    return (
        "## 3D MONAI IDH Analysis\n\n"
        f"**Predicted class**: {label}\n\n"
        f"**Mean tumor-volume probability**: {idh_prob:.4f}\n\n"
        f"**Decision threshold**: {threshold:.4f} (Youden's J on val)\n\n"
        f"**Predicted-tumor voxels**: {n_voxels:,}\n\n"
        f"**Inference time**: {latency * 1000:.0f} ms\n\n"
        f"**Confidence note**: {note}\n\n"
        "---\n"
        "*Pipeline: MONAI Model Zoo brats bundle (zero-shot, Dice 0.926) + "
        "3D DenseNet121 with bbox jitter (E2E AUC 0.75, accuracy 0.80). "
        "For research / decision-support reference only.*"
    )


def predict_idh_monai(flair_path, t1_path, t1gd_path, t2_path):
    paths = {"flair": flair_path, "t1": t1_path, "t1Gd": t1gd_path, "t2": t2_path}
    missing = [k for k, v in paths.items() if not v]
    if missing:
        return {}, None, f"## 3D MONAI IDH Analysis\n\n**Error**: missing modality files: {', '.join(missing)}"

    t0 = time.perf_counter()
    try:
        flair = nib.load(flair_path).get_fdata().astype(np.float32)
        t1 = nib.load(t1_path).get_fdata().astype(np.float32)
        t1gd = nib.load(t1gd_path).get_fdata().astype(np.float32)
        t2 = nib.load(t2_path).get_fdata().astype(np.float32)
    except Exception as exc:
        return {}, None, f"## 3D MONAI IDH Analysis\n\n**Error loading NIfTI**: {exc}"

    if not (flair.shape == t1.shape == t1gd.shape == t2.shape):
        return ({}, None,
                "## 3D MONAI IDH Analysis\n\n**Error**: the four modalities have different shapes; "
                f"flair={flair.shape}, t1={t1.shape}, t1Gd={t1gd.shape}, t2={t2.shape}")

    seg = _get_seg()
    cls, meta = _get_cls()
    threshold = meta["threshold"]
    margin = meta["margin"]
    target_size = meta["target_size"]
    dev = _device()
    use_amp = dev.type == "cuda"
    normalize = NormalizeIntensity(nonzero=True, channel_wise=True)
    keep_largest = KeepLargestConnectedComponent(applied_labels=[1])

    img4 = np.stack([t1gd, t1, t2, flair], axis=0)  # bundle order
    img_t = normalize(torch.from_numpy(img4)).unsqueeze(0).to(dev).float()
    with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        seg_logits = sliding_window_inference(img_t, roi_size=(240, 240, 160), sw_batch_size=1, predictor=seg, overlap=0.5)
    pp = torch.sigmoid(seg_logits).cpu().numpy()[0]
    raw = (pp[1] > 0.5).astype(np.uint8)
    pred_mask = keep_largest(torch.from_numpy(raw).unsqueeze(0)).squeeze(0).numpy().astype(np.uint8)

    bb = _bbox3d(pred_mask, margin=margin)
    if bb is None:
        idh_prob = float("nan")
        idh_class = -1
    else:
        f, t1c, tg, t2c = (_zsc(v[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]) for v in (flair, t1, t1gd, t2))
        crop = np.stack([f, t1c, tg, t2c], axis=0)
        ct = torch.from_numpy(crop).unsqueeze(0)
        ct = F.interpolate(ct, size=target_size, mode="trilinear", align_corners=False).to(dev)
        if dev.type == "cuda":
            ct = ct.contiguous(memory_format=torch.channels_last_3d)
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            logit = cls(ct)
        idh_prob = float(torch.sigmoid(logit.float()).cpu().item())
        idh_class = int(idh_prob >= threshold)

    n_voxels = int(pred_mask.sum())
    latency = time.perf_counter() - t0
    fig_arr = _render_overview(flair, pred_mask, idh_prob, idh_class, threshold)
    diag = _build_diagnosis(idh_prob, idh_class, threshold, n_voxels, latency)
    if np.isnan(idh_prob):
        confidences: dict[str, float] = {}
    else:
        confidences = {"Mutant": float(idh_prob), "Wildtype": float(1.0 - idh_prob)}
    return confidences, fig_arr, diag
