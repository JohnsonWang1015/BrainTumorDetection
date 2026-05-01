"""Unit tests for end-to-end ROI helpers used by the MONAI pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from idh_glioma.infer.e2e_roi import (
    aggregate_probs,
    apply_mask_postprocess,
    build_roi_boxes,
)


def test_apply_mask_postprocess_keeps_largest_component() -> None:
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    mask[6:7, 6:7, 6:7] = 1
    out = apply_mask_postprocess(mask, keep_largest=True, dilate_iters=0)
    assert int(out.sum()) == 27


def test_build_roi_boxes_clips_expanded_views_to_volume_bounds() -> None:
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:6, 3:8, 4:9] = 1
    boxes = build_roi_boxes(mask, base_margin=2, view_margins=[0, 3])
    assert boxes == [
        (0, 8, 1, 10, 2, 10),
        (0, 10, 0, 10, 0, 10),
    ]


def test_aggregate_probs_supports_mean_and_median() -> None:
    assert aggregate_probs([0.1, 0.7, 0.9], method="mean") == pytest.approx(0.5666666666666667)
    assert aggregate_probs([0.1, 0.7, 0.9], method="median") == pytest.approx(0.7)


def test_eval_pipeline_aggregation_contract_matches_shared_helper() -> None:
    probs = [0.2, 0.4, 0.9]
    assert aggregate_probs(probs, method="median") == pytest.approx(0.4)
