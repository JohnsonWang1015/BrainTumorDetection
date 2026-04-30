"""IDH classification handler for the Gradio app.

Exposes :func:`predict_idh` -- given four NIfTI file paths (flair / t1 / t1Gd
/ t2) it runs the segmentation + ROI-crop + classifier pipeline and returns
Gradio-friendly outputs (label dict, comparison figure, markdown report).
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

from idh_glioma.data.datasets import _crop_roi, _tumor_bbox_2d, zscore
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary
from idh_glioma.models.unet2d import UNet2D

_SEG_CKPT = Path("checkpoints/unet2d_tcga_v1.pt")
_CLS_CKPT = Path("checkpoints/mobilenetv3_idh_best.pt")

_BATCH_SIZE = 16

# Lazy singletons.
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


def _get_seg_model() -> torch.nn.Module:
    global _SEG_MODEL
    if _SEG_MODEL is not None:
        return _SEG_MODEL
    dev = _device()
    model = UNet2D().to(dev)
    state = torch.load(_SEG_CKPT, map_location=dev, weights_only=True)
    model.load_state_dict(state["model"])
    model.train(False)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    _SEG_MODEL = model
    return _SEG_MODEL


def _get_cls_model() -> tuple[torch.nn.Module, dict]:
    global _CLS_MODEL, _CLS_META
    if _CLS_MODEL is not None and _CLS_META is not None:
        return _CLS_MODEL, _CLS_META
    dev = _device()
    state = torch.load(_CLS_CKPT, map_location=dev, weights_only=True)
    meta = {
        "variant": state.get("variant", "small"),
        "use_roi": bool(state.get("use_roi", True)),
        "roi_margin": int(state.get("roi_margin", 10)),
        "img_size": int(state.get("img_size", 224)),
        "threshold": float(state.get("threshold", 0.5)),
    }
    model = build_mobilenetv3_binary(variant=meta["variant"]).to(dev)
    model.load_state_dict(state["model"])
    model.train(False)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    _CLS_MODEL = model
    _CLS_META = meta
    return _CLS_MODEL, _CLS_META


def _load_nifti(path: str | Path) -> np.ndarray:
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32)


@torch.inference_mode()
def _segment_volume(volumes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    flair, t1, t1gd, t2 = volumes
    flair_n, t1_n, t1gd_n, t2_n = (zscore(v) for v in volumes)
    seg_model = _get_seg_model()
    dev = _device()
    pred_mask = np.zeros_like(flair, dtype=np.uint8)
    use_amp = dev.type == "cuda"

    for z_start in range(0, flair.shape[-1], _BATCH_SIZE):
        z_end = min(flair.shape[-1], z_start + _BATCH_SIZE)
        slices = np.stack(
            [
                np.stack(
                    [flair_n[:, :, z], t1_n[:, :, z], t1gd_n[:, :, z], t2_n[:, :, z]],
                    axis=0,
                )
                for z in range(z_start, z_end)
            ],
            axis=0,
        )
        tensor = torch.from_numpy(slices).to(dev, non_blocking=True)
        if dev.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            logits = seg_model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
        for batch_idx, z in enumerate(range(z_start, z_end)):
            pred_mask[:, :, z] = (probs[batch_idx] > 0.5).astype(np.uint8)
    return pred_mask


@torch.inference_mode()
def _classify_tumor_slices(
    volumes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pred_mask: np.ndarray,
    meta: dict,
) -> float:
    """Return mean per-slice IDH probability across tumor-bearing slices."""
    flair, _t1, t1gd, t2 = volumes
    flair_n = zscore(flair)
    t1gd_n = zscore(t1gd)
    t2_n = zscore(t2)
    cls_model, _ = _get_cls_model()
    dev = _device()
    use_amp = dev.type == "cuda"

    tumor_z = np.where(pred_mask.sum(axis=(0, 1)) > 0)[0]
    if len(tumor_z) == 0:
        return float("nan")

    img_size = meta["img_size"]
    use_roi = meta["use_roi"]
    roi_margin = meta["roi_margin"]
    slice_probs: list[float] = []

    for start in range(0, len(tumor_z), _BATCH_SIZE):
        batch_z = tumor_z[start : start + _BATCH_SIZE]
        cropped: list[torch.Tensor] = []
        for z in batch_z:
            stacked = np.stack(
                [flair_n[:, :, z], t1gd_n[:, :, z], t2_n[:, :, z]], axis=0
            )
            if use_roi:
                bbox = _tumor_bbox_2d(pred_mask[:, :, z], roi_margin)
                stacked = _crop_roi(stacked, bbox)
            t = torch.from_numpy(stacked.copy()).unsqueeze(0)
            t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
            cropped.append(t.squeeze(0))
        tensor = torch.stack(cropped, dim=0).to(dev, non_blocking=True)
        if dev.type == "cuda":
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            logits = cls_model(tensor)
        probs = torch.sigmoid(logits.float()).cpu().numpy().reshape(-1)
        slice_probs.extend(probs.tolist())

    return float(np.mean(slice_probs))


def _render_overview(
    flair: np.ndarray, pred_mask: np.ndarray, idh_prob: float, idh_class: int, threshold: float
) -> np.ndarray:
    """3-panel figure: middle tumor slice / mask / overlay."""
    tumor_z = np.where(pred_mask.sum(axis=(0, 1)) > 0)[0]
    if len(tumor_z) == 0:
        z_show = flair.shape[-1] // 2
    else:
        # Pick the slice with the largest tumor area for the most informative view.
        sums = pred_mask.sum(axis=(0, 1))
        z_show = int(np.argmax(sums))

    flair_slice = flair[:, :, z_show]
    mask_slice = pred_mask[:, :, z_show]

    # Normalise FLAIR to [0,255] for display.
    f_min, f_max = float(flair_slice.min()), float(flair_slice.max())
    flair_norm = ((flair_slice - f_min) / (f_max - f_min + 1e-6) * 255).astype(np.uint8)
    flair_rgb = np.stack([flair_norm] * 3, axis=-1)

    overlay = flair_rgb.copy()
    overlay[mask_slice > 0] = (overlay[mask_slice > 0] * 0.35 + np.array([255, 60, 60]) * 0.65).astype(
        np.uint8
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=100)
    axes[0].imshow(np.rot90(flair_norm), cmap="gray")
    axes[0].set_title(f"FLAIR (z={z_show})", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.rot90(mask_slice), cmap="gray")
    axes[1].set_title("Predicted tumor mask", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(np.rot90(overlay))
    axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    label = "IDH Mutant" if idh_class == 1 else "IDH Wildtype"
    color = "#e67e22" if idh_class == 1 else "#2980b9"
    fig.suptitle(
        f"{label}  |  prob={idh_prob:.3f}  |  threshold={threshold:.3f}",
        fontsize=15,
        fontweight="bold",
        color=color,
        y=1.01,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    arr = np.array(Image.open(buf))
    plt.close(fig)
    return arr


def _build_diagnosis(
    idh_prob: float,
    idh_class: int,
    threshold: float,
    n_tumor_slices: int,
    latency: float,
) -> str:
    if np.isnan(idh_prob):
        return (
            "## IDH Analysis\n\n"
            "**Result**: No tumor detected by the segmentation model — IDH classification skipped.\n\n"
            f"**Inference time**: {latency * 1000:.0f} ms"
        )

    label = "IDH Mutant (1)" if idh_class == 1 else "IDH Wildtype (0)"
    margin = abs(idh_prob - threshold)
    if margin < 0.05:
        confidence_note = "⚠ The probability is within 0.05 of the decision threshold — consider this borderline."
    elif margin < 0.15:
        confidence_note = "Result is close to the threshold; treat with caution."
    else:
        confidence_note = "Result is well separated from the threshold."

    return (
        "## IDH Analysis\n\n"
        f"**Predicted class**: {label}\n\n"
        f"**Mean tumor-slice probability**: {idh_prob:.4f}\n\n"
        f"**Decision threshold**: {threshold:.4f} (Youden's J on val split)\n\n"
        f"**Tumor slices evaluated**: {n_tumor_slices}\n\n"
        f"**Inference time**: {latency * 1000:.0f} ms\n\n"
        f"**Confidence note**: {confidence_note}\n\n"
        "---\n"
        "*This result is for reference only and does not constitute a medical diagnosis. "
        "Please consult a physician.*"
    )


def predict_idh(
    flair_path: str | None,
    t1_path: str | None,
    t1gd_path: str | None,
    t2_path: str | None,
) -> tuple[dict[str, float], np.ndarray | None, str]:
    """Gradio entry point. Returns (label dict, figure array, markdown report)."""
    paths = {"flair": flair_path, "t1": t1_path, "t1Gd": t1gd_path, "t2": t2_path}
    missing = [name for name, p in paths.items() if not p]
    if missing:
        return (
            {},
            None,
            f"## IDH Analysis\n\n**Error**: missing modality files: {', '.join(missing)}",
        )

    t0 = time.perf_counter()
    try:
        flair = _load_nifti(flair_path)
        t1 = _load_nifti(t1_path)
        t1gd = _load_nifti(t1gd_path)
        t2 = _load_nifti(t2_path)
    except Exception as exc:
        return ({}, None, f"## IDH Analysis\n\n**Error loading NIfTI**: {exc}")

    if not (flair.shape == t1.shape == t1gd.shape == t2.shape):
        return (
            {},
            None,
            "## IDH Analysis\n\n**Error**: the four modalities have different shapes; "
            f"flair={flair.shape}, t1={t1.shape}, t1Gd={t1gd.shape}, t2={t2.shape}",
        )

    volumes = (flair, t1, t1gd, t2)
    pred_mask = _segment_volume(volumes)
    _, meta = _get_cls_model()
    idh_prob = _classify_tumor_slices(volumes, pred_mask, meta)
    threshold = meta["threshold"]
    if np.isnan(idh_prob):
        idh_class = -1
    else:
        idh_class = int(idh_prob >= threshold)

    n_tumor_slices = int((pred_mask.sum(axis=(0, 1)) > 0).sum())
    latency = time.perf_counter() - t0

    fig_arr = _render_overview(flair, pred_mask, idh_prob, idh_class, threshold)
    diagnosis = _build_diagnosis(idh_prob, idh_class, threshold, n_tumor_slices, latency)

    if np.isnan(idh_prob):
        confidences: dict[str, float] = {}
    else:
        confidences = {"Mutant": float(idh_prob), "Wildtype": float(1.0 - idh_prob)}

    return confidences, fig_arr, diagnosis
