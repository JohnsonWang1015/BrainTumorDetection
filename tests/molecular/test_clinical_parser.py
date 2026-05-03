from __future__ import annotations

from pathlib import Path

from idh_glioma.molecular.clinical_parser import aggregate_clinical, parse_clinical_xml


def test_parse_clinical_xml_extracts_core_fields(tmp_path: Path) -> None:
    xml_path = tmp_path / "clinical.xml"
    xml_path.write_text(
        """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<gbm:tcga_bcr xmlns:gbm=\"http://tcga.nci/bcr/xml/clinical/gbm/2.7\" xmlns:shared=\"http://tcga.nci/bcr/xml/shared/2.7\" xmlns:clin_shared=\"http://tcga.nci/bcr/xml/clinical/shared/2.7\">
  <gbm:patient>
    <shared:bcr_patient_barcode>TCGA-08-0385</shared:bcr_patient_barcode>
    <shared:gender>MALE</shared:gender>
    <clin_shared:vital_status>Dead</clin_shared:vital_status>
    <clin_shared:days_to_birth>-26234</clin_shared:days_to_birth>
    <clin_shared:days_to_death>82</clin_shared:days_to_death>
    <clin_shared:days_to_last_followup>31</clin_shared:days_to_last_followup>
    <shared:histological_type>Untreated primary (de novo) GBM</shared:histological_type>
    <clin_shared:karnofsky_performance_score>60</clin_shared:karnofsky_performance_score>
  </gbm:patient>
</gbm:tcga_bcr>
""",
        encoding="utf-8",
    )

    row = parse_clinical_xml(xml_path)

    assert row["patient_id"] == "TCGA-08-0385"
    assert row["gender"] == "MALE"
    assert row["vital_status"] == "Dead"
    assert row["days_to_death"] == 82
    assert row["days_to_last_followup"] == 31
    assert row["kps_score"] == 60
    assert row["age_at_diagnosis"] == 71


def test_aggregate_clinical_adds_source(tmp_path: Path) -> None:
    xml_dir = tmp_path / "clinical"
    xml_dir.mkdir(parents=True)
    (xml_dir / "a.xml").write_text(
        """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<tcga xmlns:shared=\"http://tcga.nci/bcr/xml/shared/2.7\"><shared:bcr_patient_barcode>TCGA-XX-0001</shared:bcr_patient_barcode></tcga>
""",
        encoding="utf-8",
    )

    df = aggregate_clinical(xml_dir, source="tcga_lgg")

    assert len(df) == 1
    assert df.iloc[0]["patient_id"] == "TCGA-XX-0001"
    assert df.iloc[0]["source"] == "tcga_lgg"
