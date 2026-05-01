"""Tests for Phase 2 3D IDH ROI augmentation helpers."""

from __future__ import annotations

from idh_glioma.train.train_idh_monai import _expand_bbox_uniformly


def test_expand_bbox_uniformly_grows_all_sides_and_clips() -> None:
    bbox = (3, 7, 4, 8, 2, 6)
    shape = (10, 12, 9)
    out = _expand_bbox_uniformly(bbox, shape, extra=3)
    assert out == (0, 10, 1, 11, 0, 9)


def test_expand_bbox_uniformly_noop_for_non_positive_extra() -> None:
    bbox = (1, 5, 2, 6, 3, 7)
    shape = (8, 8, 8)
    assert _expand_bbox_uniformly(bbox, shape, extra=0) == bbox
