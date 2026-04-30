"""Tests for IDH classification dataset helpers (ROI crop + balanced sampler).

Run with:
    uv run pytest tests/test_idh_dataset.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from idh_glioma.data import datasets as ds_mod
from idh_glioma.data.datasets import (
    BraTSSliceClassificationDataset,
    _crop_roi,
    _tumor_bbox_2d,
    make_balanced_sampler,
)


# ---------------------------------------------------------------------------
# _tumor_bbox_2d
# ---------------------------------------------------------------------------


def test_tumor_bbox_empty_returns_none():
    mask = np.zeros((50, 50), dtype=np.float32)
    assert _tumor_bbox_2d(mask, margin=5) is None


def test_tumor_bbox_tight_with_margin():
    mask = np.zeros((100, 120), dtype=np.float32)
    mask[20:40, 30:60] = 1.0  # tumor block
    bbox = _tumor_bbox_2d(mask, margin=5)
    assert bbox is not None
    y0, y1, x0, x1 = bbox
    # Tight tumor was rows 20..39 (inclusive), cols 30..59 (inclusive).
    # +1 on the upper bound (np.where range) and ±5 margin.
    assert y0 == 15 and y1 == 45
    assert x0 == 25 and x1 == 65


def test_tumor_bbox_clips_at_image_edges():
    mask = np.zeros((30, 30), dtype=np.float32)
    mask[0:5, 25:30] = 1.0  # tumor in top-right corner
    bbox = _tumor_bbox_2d(mask, margin=10)
    assert bbox is not None
    y0, y1, x0, x1 = bbox
    assert y0 == 0
    assert y1 == 15  # 5 (max+1) + 10
    assert x0 == 15  # 25 - 10
    assert x1 == 30  # clipped at width


def test_crop_roi_passthrough_when_bbox_none():
    image = np.random.rand(3, 40, 40).astype(np.float32)
    out = _crop_roi(image, None)
    assert out is image


def test_crop_roi_shapes():
    image = np.random.rand(3, 40, 40).astype(np.float32)
    cropped = _crop_roi(image, (5, 25, 10, 30))
    assert cropped.shape == (3, 20, 20)


# ---------------------------------------------------------------------------
# make_balanced_sampler
# ---------------------------------------------------------------------------


def test_balanced_sampler_weight_distribution():
    records = [{"idh_label": 1}] * 8 + [{"idh_label": 0}] * 2
    sampler = make_balanced_sampler(records, num_slices_per_case=3)
    weights = np.asarray(sampler.weights, dtype=np.float64)
    assert weights.size == len(records) * 3

    pos_w = weights[: 8 * 3].sum()
    neg_w = weights[8 * 3 :].sum()
    # The two classes carry equal total weight even though records are 8:2.
    assert pytest.approx(neg_w, rel=1e-6) == pos_w
    # Each individual positive slice is lighter than each negative slice.
    assert weights[0] < weights[-1]


def test_balanced_sampler_single_class_falls_back_to_uniform():
    records = [{"idh_label": 1}] * 5
    sampler = make_balanced_sampler(records, num_slices_per_case=2)
    weights = np.asarray(sampler.weights, dtype=np.float64)
    # All weights equal — uniform fallback
    assert np.allclose(weights, weights[0])


# ---------------------------------------------------------------------------
# BraTSSliceClassificationDataset with ROI crop
# ---------------------------------------------------------------------------


def _write_synthetic_manifest(tmp: Path) -> Path:
    """Manifest entry pointing at fake nifti paths; load_nifti is monkey-patched in the test."""
    payload = {
        "train": [
            {
                "case_id": "FAKE-001",
                "modalities": {
                    "flair": "fake_flair.nii.gz",
                    "t1": "fake_t1.nii.gz",
                    "t1Gd": "fake_t1gd.nii.gz",
                    "t2": "fake_t2.nii.gz",
                },
                "mask_path": "fake_mask.nii.gz",
                "idh_label": 1,
            }
        ],
        "val": [],
        "test": [],
    }
    manifest_path = tmp / "manifest.json"
    manifest_path.write_text(json.dumps(payload))
    return manifest_path


def _fake_volume(kind: str) -> np.ndarray:
    rng = np.random.default_rng(42)
    if kind == "mask":
        vol = np.zeros((128, 128, 20), dtype=np.float32)
        vol[40:80, 50:90, 8:14] = 1.0  # tumor block in the middle
        return vol
    return rng.standard_normal((128, 128, 20)).astype(np.float32)


def test_classification_dataset_roi_returns_expected_shape(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = _write_synthetic_manifest(Path(tmp))

        def _fake_load(path):
            return _fake_volume("mask" if "mask" in str(path) else "img")

        monkeypatch.setattr(ds_mod, "load_nifti", _fake_load)

        ds = BraTSSliceClassificationDataset(
            manifest_path,
            split="train",
            num_slices_per_case=2,
            img_size=224,
            use_roi=True,
            roi_margin=5,
            augment=False,
        )
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert label.shape == (1,)
        assert float(label.item()) == 1.0


def test_classification_dataset_no_roi_keeps_full_slice(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = _write_synthetic_manifest(Path(tmp))

        def _fake_load(path):
            return _fake_volume("mask" if "mask" in str(path) else "img")

        monkeypatch.setattr(ds_mod, "load_nifti", _fake_load)

        ds = BraTSSliceClassificationDataset(
            manifest_path,
            split="train",
            num_slices_per_case=1,
            img_size=240,
            use_roi=False,
            augment=False,
        )
        img, _ = ds[0]
        assert img.shape == (3, 240, 240)
