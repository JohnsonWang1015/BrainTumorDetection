from __future__ import annotations

import gzip
from pathlib import Path

from idh_glioma.molecular.maf_parser import aggregate_idh_labels, extract_idh_status


def _write_maf(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "#version gdc-1.0.0",
            "Hugo_Symbol\tVariant_Classification\tTumor_Sample_Barcode\tHGVSp_Short\tt_ref_count\tt_alt_count",
            *rows,
        ]
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(body)


def test_extract_idh_status_detects_idh1_idh2_missense(tmp_path: Path) -> None:
    maf_path = tmp_path / "a.maf.gz"
    _write_maf(
        maf_path,
        [
            "IDH1\tMissense_Mutation\tTCGA-06-0125-01A-01D-0000-08\tp.R132H\t80\t20",
            "IDH2\tMissense_Mutation\tTCGA-06-0126-01A-01D-0000-08\tp.R172K\t90\t10",
            "TP53\tMissense_Mutation\tTCGA-06-0127-01A-01D-0000-08\tp.R175H\t70\t30",
            "IDH1\tNonsense_Mutation\tTCGA-06-0128-01A-01D-0000-08\tp.R132*\t10\t90",
        ],
    )

    out = extract_idh_status(maf_path)

    assert len(out) == 2
    assert out[0]["gene"] == "IDH1"
    assert out[0]["mutation_aa"] == "p.R132H"
    assert round(float(out[0]["vaf"]), 4) == 0.2
    assert out[1]["gene"] == "IDH2"


def test_aggregate_idh_labels_marks_wildtype_when_no_idh_mutation(tmp_path: Path) -> None:
    maf_dir = tmp_path / "maf"
    _write_maf(
        maf_dir / "sample_a" / "a.maf.gz",
        [
            "IDH1\tMissense_Mutation\tTCGA-06-0125-01A-01D-0000-08\tp.R132H\t90\t10",
            "TP53\tMissense_Mutation\tTCGA-06-0125-01A-01D-0000-08\tp.R175H\t60\t40",
        ],
    )
    _write_maf(
        maf_dir / "sample_b" / "b.maf.gz",
        [
            "TP53\tMissense_Mutation\tTCGA-06-0126-01A-01D-0000-08\tp.R175H\t75\t25",
        ],
    )

    df = aggregate_idh_labels(maf_dir, source="tcga_lgg")
    by_patient = {row["patient_id"]: row for row in df.to_dict(orient="records")}

    assert by_patient["TCGA-06-0125"]["idh_label"] == 1
    assert by_patient["TCGA-06-0125"]["idh_gene"] == "IDH1"
    assert by_patient["TCGA-06-0126"]["idh_label"] == 0
    assert by_patient["TCGA-06-0126"]["label_source"] == "maf_aggregated"
