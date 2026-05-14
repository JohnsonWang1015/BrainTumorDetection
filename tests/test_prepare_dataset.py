"""Unit tests for ``prepare_dataset.parse_case_folder``.

Covers two case-folder layouts:
* TCGA-LGG (legacy) — ``<patient>_<YYYYMMDD>_<suffix>.nii.gz`` with
  GlistrBoost masks.
* BraTS 2021 HF mirror — ``BraTS2021_<num>_<suffix>.nii.gz`` with ``_seg``
  masks and ``_t1ce`` standing in for ``t1Gd``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from idh_glioma.data.prepare_dataset import (
    MODALITIES,
    parse_case_folder,
)


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _make_tcga_case(root: Path, patient: str = "TCGA-CS-4938", date: str = "19960621", manual: bool = True) -> Path:
    case_dir = root / f"{patient}_{date}"
    for mod in ("flair", "t1", "t1Gd", "t2"):
        _touch(case_dir / f"{patient}_{date}_{mod}.nii.gz")
    mask_suffix = "GlistrBoost_ManuallyCorrected" if manual else "GlistrBoost"
    _touch(case_dir / f"{patient}_{date}_{mask_suffix}.nii.gz")
    return case_dir


def _make_brats2021_case(root: Path, case: str = "BraTS2021_00000") -> Path:
    case_dir = root / case
    for mod in ("flair", "t1", "t1ce", "t2", "seg"):
        _touch(case_dir / f"{case}_{mod}.nii.gz")
    return case_dir


# ---------------------------------------------------------------------------
# TCGA-LGG parsing (must remain unchanged)
# ---------------------------------------------------------------------------


def test_tcga_case_id_and_date_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = _make_tcga_case(Path(tmp))
        rec = parse_case_folder(case_dir)
        assert rec is not None
        assert rec.case_id == "TCGA-CS-4938"
        assert rec.date == "19960621"
        assert set(rec.modalities) == set(MODALITIES)


def test_tcga_prefers_manually_corrected_mask():
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = _make_tcga_case(Path(tmp))
        _touch(case_dir / "TCGA-CS-4938_19960621_GlistrBoost.nii.gz")  # add non-manual too
        rec = parse_case_folder(case_dir)
        assert rec is not None
        assert "GlistrBoost_ManuallyCorrected" in rec.mask_path
        assert "GlistrBoost_ManuallyCorrected.nii.gz" in rec.mask_path


def test_tcga_falls_back_to_plain_glistrboost():
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = _make_tcga_case(Path(tmp), manual=False)
        rec = parse_case_folder(case_dir)
        assert rec is not None
        assert rec.mask_path.endswith("_GlistrBoost.nii.gz")


# ---------------------------------------------------------------------------
# BraTS 2021 HF parsing
# ---------------------------------------------------------------------------


def test_brats2021_case_id_is_dir_name():
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = _make_brats2021_case(Path(tmp), case="BraTS2021_00042")
        rec = parse_case_folder(case_dir)
        assert rec is not None
        assert rec.case_id == "BraTS2021_00042"  # full dir name, not "BraTS2021"
        assert rec.date == "unknown"


def test_brats2021_t1ce_is_mapped_to_t1Gd():
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = _make_brats2021_case(Path(tmp))
        rec = parse_case_folder(case_dir)
        assert rec is not None
        assert set(rec.modalities) == set(MODALITIES)
        # the t1Gd entry should point to the *_t1ce.nii.gz file
        assert rec.modalities["t1Gd"].endswith("_t1ce.nii.gz")


def test_brats2021_seg_is_used_as_mask():
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = _make_brats2021_case(Path(tmp))
        rec = parse_case_folder(case_dir)
        assert rec is not None
        assert rec.mask_path.endswith("_seg.nii.gz")


def test_brats2021_missing_modality_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        case_dir = _make_brats2021_case(Path(tmp))
        # delete the t1ce -> t1Gd file to mimic the 4-modality cases on HF
        (case_dir / "BraTS2021_00000_t1ce.nii.gz").unlink()
        rec = parse_case_folder(case_dir)
        assert rec is None


# ---------------------------------------------------------------------------
# Real on-disk sample (skipped if the HF download isn't present yet)
# ---------------------------------------------------------------------------


REAL_BRATS = Path(
    "/mnt/8tb_hdd2/johnson/BrainTumorDetection/datasets/BraTS2021_HF/BraTS2021_00000"
)


@pytest.mark.skipif(not REAL_BRATS.exists(), reason="BraTS 2021 sample not downloaded yet")
def test_parses_real_brats2021_case():
    rec = parse_case_folder(REAL_BRATS)
    assert rec is not None
    assert rec.case_id == "BraTS2021_00000"
    assert set(rec.modalities) == set(MODALITIES)
    assert rec.modalities["t1Gd"].endswith("_t1ce.nii.gz")
    assert rec.mask_path.endswith("_seg.nii.gz")
