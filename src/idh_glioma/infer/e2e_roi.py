"""Shared helpers for end-to-end ROI extraction and config loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from skimage.measure import label
from skimage.morphology import ball, dilation

from idh_glioma.utils import load_json, save_json

DEFAULT_E2E_CONFIG = {
    "keep_largest": True,
    "dilate_iters": 0,
    "base_margin": 4,
    "view_margins": [0],
    "aggregation": "mean",
    "threshold": 0.5,
    "threshold_objective": "macro_f1",
}


def apply_mask_postprocess(mask: np.ndarray, keep_largest: bool, dilate_iters: int) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    if keep_largest and out.any():
        cc = label(out, connectivity=1)
        component_ids = np.unique(cc)
        component_ids = component_ids[component_ids != 0]
        if component_ids.size > 1:
            sizes = [(int((cc == cid).sum()), int(cid)) for cid in component_ids]
            _, best_id = max(sizes)
            out = (cc == best_id).astype(np.uint8)
    if dilate_iters > 0 and out.any():
        footprint = ball(1)
        dilated = out.astype(bool)
        for _ in range(dilate_iters):
            dilated = dilation(dilated, footprint)
        out = dilated.astype(np.uint8)
    return out


def build_roi_boxes(
    mask: np.ndarray,
    base_margin: int,
    view_margins: list[int],
) -> list[tuple[int, int, int, int, int, int]]:
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        return []
    mn, mx = nz.min(0), nz.max(0) + 1
    h, w, d = mask.shape
    boxes: list[tuple[int, int, int, int, int, int]] = []
    for extra_margin in view_margins:
        margin = base_margin + extra_margin
        boxes.append(
            (
                max(int(mn[0]) - margin, 0),
                min(int(mx[0]) + margin, h),
                max(int(mn[1]) - margin, 0),
                min(int(mx[1]) + margin, w),
                max(int(mn[2]) - margin, 0),
                min(int(mx[2]) + margin, d),
            )
        )
    return boxes


def aggregate_probs(probs: list[float], method: str) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if method == "mean":
        return float(arr.mean())
    if method == "median":
        return float(np.median(arr))
    raise ValueError(f"Unsupported aggregation method: {method}")


def load_e2e_config(path: Path | None) -> dict:
    cfg = dict(DEFAULT_E2E_CONFIG)
    if path is None or not path.exists():
        return cfg
    cfg.update(load_json(path))
    return cfg


def save_e2e_config(path: Path, cfg: dict) -> None:
    merged = dict(DEFAULT_E2E_CONFIG)
    merged.update(cfg)
    save_json(merged, path)


def select_best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
    preferred_threshold: float | None = None,
) -> tuple[float, float]:
    best_threshold = float(thresholds[0])
    best_score = float("-inf")
    for threshold in thresholds:
        score = float(f1_score(labels, probs >= threshold, average="macro"))
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
        elif np.isclose(score, best_score):
            if preferred_threshold is not None:
                current_delta = abs(best_threshold - preferred_threshold)
                candidate_delta = abs(float(threshold) - preferred_threshold)
                if candidate_delta < current_delta or (
                    np.isclose(candidate_delta, current_delta) and float(threshold) > best_threshold
                ):
                    best_threshold = float(threshold)
            elif float(threshold) > best_threshold:
                best_threshold = float(threshold)
    return best_threshold, best_score


def merge_e2e_config(
    path: Path | None,
    *,
    fallback_threshold: float | None = None,
    fallback_base_margin: int | None = None,
) -> dict:
    cfg = load_e2e_config(path)
    if fallback_threshold is not None and (path is None or not path.exists()):
        cfg["threshold"] = float(fallback_threshold)
    if fallback_base_margin is not None:
        cfg["base_margin"] = int(cfg.get("base_margin", fallback_base_margin))
        if path is None or not path.exists():
            cfg["base_margin"] = int(fallback_base_margin)
    return cfg


def zscore_crop(vol: np.ndarray, bbox: tuple[int, int, int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1, z0, z1 = bbox
    crop = vol[y0:y1, x0:x1, z0:z1]
    return (crop - crop.mean()) / (crop.std() + 1e-6)


def predict_multi_view_idh(
    *,
    flair: np.ndarray,
    t1: np.ndarray,
    t1gd: np.ndarray,
    t2: np.ndarray,
    pred_mask: np.ndarray,
    cls_model: torch.nn.Module,
    device: torch.device,
    target_size: tuple[int, int, int],
    cfg: dict,
    use_amp: bool,
) -> tuple[float, list[float], list[tuple[int, int, int, int, int, int]]]:
    boxes = build_roi_boxes(
        pred_mask,
        base_margin=int(cfg["base_margin"]),
        view_margins=list(cfg["view_margins"]),
    )
    view_probs: list[float] = []
    for bbox in boxes:
        crop = np.stack(
            [
                zscore_crop(flair, bbox),
                zscore_crop(t1, bbox),
                zscore_crop(t1gd, bbox),
                zscore_crop(t2, bbox),
            ],
            axis=0,
        )
        ct = torch.from_numpy(crop).unsqueeze(0)
        ct = F.interpolate(ct, size=target_size, mode="trilinear", align_corners=False).to(device)
        if device.type == "cuda":
            ct = ct.contiguous(memory_format=torch.channels_last_3d)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logit = cls_model(ct)
        view_probs.append(float(torch.sigmoid(logit.float()).cpu().item()))
    return aggregate_probs(view_probs, method=str(cfg["aggregation"])), view_probs, boxes
