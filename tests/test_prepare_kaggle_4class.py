"""Tests for the Kaggle 4-class manifest builder."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from idh_glioma.data.prepare_kaggle_4class import (
    CLASS_NAMES,
    ImageRecord,
    _collect_split_dir,
    _stratified_train_val_split,
    build_manifest,
    write_manifest,
)


def _make_fake_kaggle4(root: Path, per_class_train: int = 5, per_class_test: int = 2) -> None:
    for split_name, n in (("Training", per_class_train), ("Testing", per_class_test)):
        for class_name in CLASS_NAMES:
            folder = root / split_name / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
                img.save(folder / f"{split_name.lower()}_{class_name}_{i:02d}.jpg")


def test_collect_split_dir_counts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_kaggle4(root, per_class_train=5)
        recs = _collect_split_dir(root / "Training")
        assert len(recs) == 5 * len(CLASS_NAMES)
        for c in CLASS_NAMES:
            assert sum(1 for r in recs if r.class_name == c) == 5


def test_collect_split_dir_missing_class_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "Training"
        (root / "glioma").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            _collect_split_dir(root)


def test_build_manifest_uses_testing_as_held_out():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_kaggle4(root, per_class_train=10, per_class_test=3)
        splits = build_manifest(root, val_ratio=0.2, seed=0)
        assert len(splits["test"]) == 3 * len(CLASS_NAMES)
        assert len(splits["train"]) + len(splits["val"]) == 10 * len(CLASS_NAMES)


def test_stratified_split_balanced_per_class():
    records = [
        ImageRecord(path=f"{i}.jpg", label=i % len(CLASS_NAMES), class_name=CLASS_NAMES[i % len(CLASS_NAMES)])
        for i in range(40)
    ]
    train, val = _stratified_train_val_split(records, val_ratio=0.25, seed=1)
    # each of 4 classes has 10 records → val gets 2 or 3 per class
    for label_idx in range(len(CLASS_NAMES)):
        n_val = sum(1 for r in val if r.label == label_idx)
        assert 1 <= n_val <= 5


def test_write_manifest_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_kaggle4(root, per_class_train=4, per_class_test=2)
        splits = build_manifest(root, val_ratio=0.25, seed=0)
        out = Path(tmp) / "manifest.json"
        write_manifest(splits, out)
        loaded = json.loads(out.read_text())
        assert loaded["classes"] == list(CLASS_NAMES)
        assert set(loaded) >= {"classes", "train", "val", "test"}
        first = loaded["train"][0]
        assert {"path", "label", "class_name"} <= set(first)
