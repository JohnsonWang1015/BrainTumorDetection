from __future__ import annotations

from idh_glioma.data.manifest_adapters import to_legacy_manifest_record


def test_to_legacy_manifest_record_keeps_training_fields() -> None:
    record = {
        "case_id": "TCGA-06-0125",
        "date": "unknown",
        "modalities": {
            "flair": "datasets/TCGA-GBM/TCGA-06-0125/flair.nii.gz",
            "t1": "datasets/TCGA-GBM/TCGA-06-0125/t1.nii.gz",
            "t1Gd": "datasets/TCGA-GBM/TCGA-06-0125/t1ce.nii.gz",
            "t2": "datasets/TCGA-GBM/TCGA-06-0125/t2.nii.gz",
        },
        "mask_path": None,
        "idh_label": 0,
        "source_dataset": "tcga_gbm",
        "cohort_id": "tcia_preop_pooled",
    }

    out = to_legacy_manifest_record(record)

    assert out == {
        "case_id": "TCGA-06-0125",
        "date": "unknown",
        "modalities": record["modalities"],
        "mask_path": None,
        "idh_label": 0,
    }
