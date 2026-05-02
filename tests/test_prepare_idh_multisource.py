from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from idh_glioma.data.prepare_idh_multisource import (
    build_multisource_manifest,
    build_payload,
    classify_modalities,
    infer_source_root,
    load_label_map,
    split_multisource_records,
)


def test_classify_modalities_normalizes_t1ce_alias() -> None:
    files = [
        Path("TCGA-06-0125_flair.nii.gz"),
        Path("TCGA-06-0125_t1.nii.gz"),
        Path("TCGA-06-0125_t1ce.nii.gz"),
        Path("TCGA-06-0125_t2.nii.gz"),
    ]

    out = classify_modalities(files)

    assert out["t1Gd"].endswith("t1ce.nii.gz")
    assert set(out) == {"flair", "t1", "t1Gd", "t2"}


def test_classify_modalities_normalizes_ucsf_pdgm_aliases() -> None:
    files = [
        Path("UCSF-PDGM-0001_T2FLAIR.nii.gz"),
        Path("UCSF-PDGM-0001_T1.nii.gz"),
        Path("UCSF-PDGM-0001_T1c_bias.nii.gz"),
        Path("UCSF-PDGM-0001_T2.nii.gz"),
    ]

    out = classify_modalities(files)

    assert out["flair"].endswith("T2FLAIR.nii.gz")
    assert out["t1Gd"].endswith("T1c_bias.nii.gz")


def test_classify_modalities_normalizes_egd_aliases() -> None:
    files = [
        Path("FLAIR.nii.gz"),
        Path("T1.nii.gz"),
        Path("T1GD.nii.gz"),
        Path("T2.nii.gz"),
    ]

    out = classify_modalities(files)

    assert out["flair"].endswith("FLAIR.nii.gz")
    assert out["t1Gd"].endswith("T1GD.nii.gz")


def test_classify_modalities_requires_all_four_modalities() -> None:
    files = [
        Path("TCGA-06-0125_flair.nii.gz"),
        Path("TCGA-06-0125_t1.nii.gz"),
        Path("TCGA-06-0125_t2.nii.gz"),
    ]

    with pytest.raises(ValueError, match="Missing required modalities"):
        classify_modalities(files)


def test_build_multisource_manifest_allows_maskless_classification_case(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "TCGA-06-0125"
    case_dir.mkdir()
    for name in ("flair", "t1", "t1ce", "t2"):
        (case_dir / f"TCGA-06-0125_{name}.nii.gz").touch()

    records = build_multisource_manifest(
        case_dirs=[case_dir],
        source_dataset="tcga_gbm",
        cohort_id="tcia_preop_pooled",
        acquisition_stage="preop",
        label_map={"TCGA-06-0125": 0},
    )

    assert len(records) == 1
    assert records[0]["mask_path"] is None
    assert records[0]["inclusion_flags"]["eligible_for_segmentation"] is False
    assert records[0]["inclusion_flags"]["eligible_for_3d_idh"] is True


def test_split_multisource_records_keeps_external_cohort_out_of_train() -> None:
    records = [
        {
            "record_id": "a",
            "case_id": "A",
            "cohort_id": "tcia_preop_pooled",
            "idh_label": 0,
        },
        {
            "record_id": "b",
            "case_id": "B",
            "cohort_id": "tcia_preop_pooled",
            "idh_label": 1,
        },
        {
            "record_id": "c",
            "case_id": "C",
            "cohort_id": "ucsf_pdgm_external",
            "idh_label": 1,
        },
    ]

    splits = split_multisource_records(
        records,
        split_mode="external_only",
        holdout_cohorts=["ucsf_pdgm_external"],
        val_ratio=0.2,
        test_ratio=0.2,
    )

    train_ids = {r["record_id"] for r in splits["train"]}
    test_ids = {r["record_id"] for r in splits["test"]}
    assert "c" not in train_ids
    assert "c" in test_ids


def test_split_multisource_records_keeps_unlabeled_cases_in_train() -> None:
    records = [
        {
            "record_id": "a",
            "case_id": "A",
            "cohort_id": "tcga_lgg_only",
            "idh_label": 0,
        },
        {
            "record_id": "b",
            "case_id": "B",
            "cohort_id": "tcga_lgg_only",
            "idh_label": 1,
        },
        {
            "record_id": "c",
            "case_id": "C",
            "cohort_id": "tcga_lgg_only",
            "idh_label": None,
        },
    ]

    splits = split_multisource_records(
        records,
        split_mode="pooled_train_val_test",
        holdout_cohorts=[],
        val_ratio=0.2,
        test_ratio=0.2,
    )

    train_ids = {r["record_id"] for r in splits["train"]}
    assert "c" in train_ids


def test_infer_source_root_finds_default_lgg_and_gbm_locations(tmp_path: Path) -> None:
    lgg = tmp_path / "BraTS-TCGA-LGG" / "Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"
    gbm = tmp_path / "TCGA-GBM"
    egd = tmp_path / "EGD"
    lgg.mkdir(parents=True)
    gbm.mkdir(parents=True)
    egd.mkdir(parents=True)

    assert infer_source_root(tmp_path, "brats_tcga_lgg") == lgg
    assert infer_source_root(tmp_path, "tcga_gbm") == gbm
    assert infer_source_root(tmp_path, "egd") == egd


def test_load_label_map_reads_binary_case_labels(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"case_id": "TCGA-06-0125", "idh_label": 0},
            {"case_id": "TCGA-DU-7015", "idh_label": 1},
        ]
    ).to_csv(csv_path, index=False)

    assert load_label_map(csv_path) == {
        "TCGA-06-0125": 0,
        "TCGA-DU-7015": 1,
    }


def test_load_label_map_reads_ucsf_pdgm_metadata_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "ucsf_metadata.csv"
    pd.DataFrame(
        [
            {"ID": "UCSF-PDGM-0001", "IDH status": "Mutant"},
            {"ID": "UCSF-PDGM-0002", "IDH status": "Wildtype"},
            {"ID": "UCSF-PDGM-0003", "IDH status": "unknown"},
        ]
    ).to_csv(csv_path, index=False)

    assert load_label_map(csv_path) == {
        "UCSF-PDGM-0001": 1,
        "UCSF-PDGM-0002": 0,
    }


def test_load_label_map_reads_tcga_join_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "tcga_join.tsv"
    pd.DataFrame(
        [
            {"submitter_id": "TCGA-06-0125", "paper_IDH_status": "WT"},
            {"submitter_id": "TCGA-06-0126", "paper_IDH_status": "IDH1-mutant"},
            {"submitter_id": "TCGA-06-0127", "paper_IDH_status": "IDH1 and/or IDH2-mutant"},
        ]
    ).to_csv(csv_path, index=False, sep="\t")

    assert load_label_map(csv_path) == {
        "TCGA-06-0125": 0,
        "TCGA-06-0126": 1,
        "TCGA-06-0127": 1,
    }


def test_load_label_map_reads_egd_label_sheet_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "Genetic_and_Histological_labels.csv"
    pd.DataFrame(
        [
            {"subject": "EGD-0001", "IDH mutation status": 1},
            {"subject": "EGD-0002", "IDH mutation status": 0},
            {"subject": "EGD-0003", "IDH mutation status": -1},
        ]
    ).to_csv(csv_path, index=False)

    assert load_label_map(csv_path) == {
        "EGD-0001": 1,
        "EGD-0002": 0,
    }


def test_build_payload_emits_manifest_v2_shape() -> None:
    records = [
        {
            "record_id": "brats_tcga_lgg:TCGA-DU-7015",
            "case_id": "TCGA-DU-7015",
            "source_dataset": "brats_tcga_lgg",
            "cohort_id": "tcga_lgg_only",
            "acquisition_stage": "preop",
            "modalities": {"flair": "a", "t1": "b", "t1Gd": "c", "t2": "d"},
            "mask_path": "m",
            "idh_label": 1,
        }
    ]
    splits = {"train": records, "val": [], "test": []}

    payload = build_payload(records=records, splits=splits, split_mode="pooled_train_val_test")

    assert payload["manifest_version"] == "0.2.0"
    assert payload["splits"]["train"][0]["case_id"] == "TCGA-DU-7015"
    assert payload["cohorts"][0]["cohort_id"] == "tcga_lgg_only"
