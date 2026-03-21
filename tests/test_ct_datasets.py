"""Unit tests for the CT/MRI data pipeline.

Run with:
    uv run pytest tests/ -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from idh_glioma.data.ct_datasets import BrainImageDataset
from idh_glioma.data.prepare_ct_data import (
    ImageRecord,
    _collect_images,
    build_manifest,
    stratified_split,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_dataset(root: Path) -> None:
    """Create a minimal fake Kaggle-style folder tree with tiny images."""
    for subdir, modality in [
        ("Brain Tumor CT scan Images", "ct"),
        ("Brain Tumor MRI images", "mri"),
    ]:
        for class_name in ("Tumor", "Healthy"):
            folder = root / subdir / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                img = Image.fromarray(
                    (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
                )
                img.save(folder / f"{modality}_{class_name.lower()}_{i:02d}.jpg")


# ---------------------------------------------------------------------------
# prepare_ct_data tests
# ---------------------------------------------------------------------------


def test_collect_images_counts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_dataset(root)
        ct_dir = root / "Brain Tumor CT scan Images"
        records = _collect_images(ct_dir, modality="ct")
        # 10 Tumor + 10 Healthy
        assert len(records) == 20
        assert all(r.modality == "ct" for r in records)
        labels = [r.label for r in records]
        assert labels.count(0) == 10  # Healthy
        assert labels.count(1) == 10  # Tumor


def test_build_manifest_both():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_dataset(root)
        records = build_manifest(root, modality="both")
        assert len(records) == 40  # 20 CT + 20 MRI


def test_build_manifest_ct_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_dataset(root)
        records = build_manifest(root, modality="ct")
        assert all(r.modality == "ct" for r in records)


def test_build_manifest_missing_dir_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            build_manifest(Path(tmp) / "nonexistent", modality="ct")


def test_stratified_split_proportions():
    records = [ImageRecord(path=f"{i}.jpg", label=i % 2, modality="ct") for i in range(100)]
    splits = stratified_split(records, val_ratio=0.15, test_ratio=0.15)
    assert set(splits) == {"train", "val", "test"}
    total = sum(len(v) for v in splits.values())
    assert total == 100
    # train should be the largest split
    assert len(splits["train"]) > len(splits["val"])
    assert len(splits["train"]) > len(splits["test"])


def test_write_and_reload_manifest():
    records = [ImageRecord(path=f"/data/{i}.jpg", label=i % 2, modality="ct") for i in range(20)]
    splits = stratified_split(records, val_ratio=0.2, test_ratio=0.2)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "manifest.json"
        write_manifest(splits, out)
        loaded = json.loads(out.read_text())
        assert set(loaded) == {"train", "val", "test"}
        first = loaded["train"][0]
        assert {"path", "label", "modality"} <= set(first)


# ---------------------------------------------------------------------------
# BrainImageDataset tests
# ---------------------------------------------------------------------------


def test_dataset_len_and_shape():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        _make_fake_dataset(root)
        records = build_manifest(root, modality="ct")
        splits = stratified_split(records, val_ratio=0.2, test_ratio=0.2)

        manifest_path = Path(tmp) / "manifest.json"
        write_manifest(splits, manifest_path)

        ds = BrainImageDataset(manifest_path, split="train", augment=False)
        assert len(ds) > 0

        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert label.shape == (1,)
        assert label.item() in (0.0, 1.0)


def test_dataset_modality_filter():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        _make_fake_dataset(root)
        records = build_manifest(root, modality="both")
        splits = stratified_split(records, val_ratio=0.2, test_ratio=0.2)

        manifest_path = Path(tmp) / "manifest.json"
        write_manifest(splits, manifest_path)

        ds_ct = BrainImageDataset(manifest_path, split="train", modality="ct", augment=False)
        ds_mri = BrainImageDataset(manifest_path, split="train", modality="mri", augment=False)
        ds_both = BrainImageDataset(manifest_path, split="train", modality="both", augment=False)

        assert len(ds_ct) + len(ds_mri) == len(ds_both)


def test_dataset_pixel_range():
    """After normalization values should not be raw [0, 255]."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "data"
        _make_fake_dataset(root)
        records = build_manifest(root, modality="ct")
        splits = stratified_split(records, val_ratio=0.2, test_ratio=0.2)

        manifest_path = Path(tmp) / "manifest.json"
        write_manifest(splits, manifest_path)

        ds = BrainImageDataset(manifest_path, split="train", augment=False)
        img, _ = ds[0]
        # After ImageNet normalization the range should be roughly (-3, 3)
        assert float(img.max()) < 10.0
        assert float(img.min()) > -10.0
