"""Tests for end-to-end calibration config loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from idh_glioma.infer.e2e_roi import load_e2e_config, select_best_threshold


def test_load_e2e_config_returns_default_when_file_missing(tmp_path: Path) -> None:
    cfg = load_e2e_config(tmp_path / "missing.json")
    assert cfg["threshold"] == 0.5
    assert cfg["aggregation"] == "mean"
    assert cfg["view_margins"] == [0]


def test_load_e2e_config_merges_saved_values(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"threshold": 0.23, "aggregation": "median", "view_margins": [0, 4]}')
    cfg = load_e2e_config(path)
    assert cfg["threshold"] == 0.23
    assert cfg["aggregation"] == "median"
    assert cfg["view_margins"] == [0, 4]


def test_macro_f1_threshold_search_prefers_balanced_threshold() -> None:
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.10, 0.45, 0.55, 0.90])
    thresholds = np.array([0.2, 0.5, 0.7])
    best_threshold, best_score = select_best_threshold(labels, probs, thresholds)
    assert best_threshold == 0.5
    assert best_score == f1_score(labels, probs >= 0.5, average="macro")


def test_threshold_search_breaks_macro_f1_ties_toward_preferred_threshold() -> None:
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.04, 0.08, 0.12, 0.90])
    thresholds = np.array([0.05, 0.08])
    best_threshold, best_score = select_best_threshold(
        labels,
        probs,
        thresholds,
        preferred_threshold=0.08,
    )
    assert best_threshold == 0.08
    assert best_score == f1_score(labels, probs >= 0.08, average="macro")
