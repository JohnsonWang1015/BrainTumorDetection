from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from idh_glioma.molecular.prepare import main


def _write_rnaseq(path: Path, tpm_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "# gene-model: GENCODE v36",
            "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded",
            "N_unmapped\t\t\t100\t100\t100\t\t\t",
            "N_multimapping\t\t\t200\t200\t200\t\t\t",
            "N_noFeature\t\t\t300\t300\t300\t\t\t",
            "N_ambiguous\t\t\t400\t400\t400\t\t\t",
            f"ENSG00000000003.15\tTSPAN6\tprotein_coding\t0\t0\t0\t{tpm_value:.1f}\t0\t0",
            f"ENSG00000000005.6\tTNMD\tprotein_coding\t0\t0\t0\t{(tpm_value/2):.1f}\t0\t0",
            f"ENSG00000000419.13\tDPM1\tprotein_coding\t0\t0\t0\t{(tpm_value/3):.1f}\t0\t0",
        ]
    )
    path.write_text(body, encoding="utf-8")


def _write_maf(path: Path, patient_id: str, mutant: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "Hugo_Symbol\tVariant_Classification\tTumor_Sample_Barcode\tHGVSp_Short\tt_ref_count\tt_alt_count",
    ]
    if mutant:
        rows.append(f"IDH1\tMissense_Mutation\t{patient_id}-01A-01D-0000-08\tp.R132H\t90\t10")
    else:
        rows.append(f"TP53\tMissense_Mutation\t{patient_id}-01A-01D-0000-08\tp.R175H\t90\t10")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(rows))


def _write_methylation(path: Path, beta_shift: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"cg00000001\t{0.2 + beta_shift:.3f}",
        f"cg00000002\t{0.5 + beta_shift:.3f}",
        f"cg00000003\t{0.8 - beta_shift:.3f}",
    ]
    path.write_text("\n".join(rows), encoding="utf-8")


def test_prepare_multimodal_writes_all_artifacts(tmp_path: Path, monkeypatch) -> None:
    gbm_root = tmp_path / "tcga_gbm_downloads"
    data_root = gbm_root / "data"
    rnaseq_root = data_root / "rnaseq_counts"
    maf_root = data_root / "maf"
    meth_root = data_root / "methylation_beta"
    file_to_patient: dict[str, str] = {}
    file_to_patient_meth: dict[str, str] = {}

    for idx in range(3):
        file_uuid = f"00000000-0000-0000-0000-0000000000{idx}"
        patient_id = f"TCGA-XX-00{idx}"
        file_to_patient[file_uuid] = patient_id
        file_to_patient_meth[file_uuid] = patient_id
        _write_rnaseq(
            rnaseq_root / file_uuid / f"{file_uuid}.rna_seq.augmented_star_gene_counts.tsv",
            tpm_value=2.0 + idx,
        )
        _write_maf(
            maf_root / file_uuid / f"{file_uuid}.maf.gz",
            patient_id=patient_id,
            mutant=(idx % 2 == 0),
        )
        _write_methylation(
            meth_root / file_uuid / f"{file_uuid}.sesame.level3betas.txt",
            beta_shift=0.01 * idx,
        )

    (gbm_root / "file_to_patient.json").write_text(json.dumps(file_to_patient), encoding="utf-8")
    (gbm_root / "file_to_patient_methylation.json").write_text(json.dumps(file_to_patient_meth), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare-idh-molecular",
            "--include-sources",
            "tcga_gbm",
            "--gbm-data-root",
            str(data_root),
            "--skip-download",
            "--modalities",
            "rnaseq",
            "methylation",
        ],
    )
    main()

    output_dir = tmp_path / "artifacts" / "molecular_multimodal"
    assert (output_dir / "expression_matrix.parquet").exists()
    assert (output_dir / "methylation_matrix.parquet").exists()
    assert (output_dir / "idh_labels.parquet").exists()
    assert (output_dir / "feature_panel.json").exists()
    assert (output_dir / "cohort_manifest.json").exists()

    expression = pd.read_parquet(output_dir / "expression_matrix.parquet")
    methylation = pd.read_parquet(output_dir / "methylation_matrix.parquet")
    labels = pd.read_parquet(output_dir / "idh_labels.parquet")
    manifest = json.loads((output_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
    assert expression.shape[1] == 3
    assert methylation.shape[1] == 3
    assert labels.shape[0] == 3
    assert manifest["multimodal"]["strict_subset_size"] == 3


def test_prepare_multimodal_missing_methylation_dir_exits(tmp_path: Path, monkeypatch) -> None:
    gbm_root = tmp_path / "tcga_gbm_downloads"
    data_root = gbm_root / "data"
    rnaseq_root = data_root / "rnaseq_counts"
    maf_root = data_root / "maf"
    file_to_patient = {"uuid-1": "TCGA-XX-0001"}
    _write_rnaseq(rnaseq_root / "uuid-1" / "a.rna_seq.augmented_star_gene_counts.tsv", tpm_value=2.0)
    _write_maf(maf_root / "uuid-1" / "a.maf.gz", patient_id="TCGA-XX-0001", mutant=True)
    (gbm_root / "file_to_patient.json").write_text(json.dumps(file_to_patient), encoding="utf-8")
    (gbm_root / "file_to_patient_methylation.json").write_text(json.dumps(file_to_patient), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare-idh-molecular",
            "--include-sources",
            "tcga_gbm",
            "--gbm-data-root",
            str(data_root),
            "--skip-download",
            "--modalities",
            "rnaseq",
            "methylation",
        ],
    )
    with pytest.raises(SystemExit, match="download_lgg_methylation"):
        main()
