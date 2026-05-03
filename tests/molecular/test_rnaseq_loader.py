from __future__ import annotations

from pathlib import Path

from idh_glioma.molecular.rnaseq_loader import build_expression_matrix, parse_rnaseq_tsv


def _write_rnaseq(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "# gene-model: GENCODE v36",
            "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded",
            "N_unmapped\t\t\t100\t100\t100\t\t\t",
            "N_multimapping\t\t\t200\t200\t200\t\t\t",
            "N_noFeature\t\t\t300\t300\t300\t\t\t",
            "N_ambiguous\t\t\t400\t400\t400\t\t\t",
            *rows,
        ]
    )
    path.write_text(body, encoding="utf-8")


def test_parse_rnaseq_tsv_skips_summary_rows(tmp_path: Path) -> None:
    tsv_path = tmp_path / "sample.tsv"
    _write_rnaseq(
        tsv_path,
        [
            "ENSG00000000003.15\tTSPAN6\tprotein_coding\t0\t0\t0\t10.0\t0\t0",
            "ENSG00000000005.6\tTNMD\tprotein_coding\t0\t0\t0\t1.5\t0\t0",
            "ENSG00000000419.13\tDPM1\tprotein_coding\t0\t0\t0\t4.0\t0\t0",
            "ENSG00000000457.14\tSCYL3\tprotein_coding\t0\t0\t0\t7.5\t0\t0",
            "ENSG00000000460.17\tC1orf112\tprotein_coding\t0\t0\t0\t2.0\t0\t0",
        ],
    )

    series = parse_rnaseq_tsv(tsv_path)

    assert len(series) == 5
    assert "N_unmapped" not in series.index
    assert series["ENSG00000000003"] == 10.0


def test_build_expression_matrix_merges_samples_and_logs(tmp_path: Path) -> None:
    rnaseq_dir = tmp_path / "rnaseq_counts"
    uuid_a = "11111111-1111-1111-1111-111111111111"
    uuid_b = "22222222-2222-2222-2222-222222222222"
    _write_rnaseq(
        rnaseq_dir / uuid_a / "a.rna_seq.augmented_star_gene_counts.tsv",
        [
            "ENSG00000000003.15\tTSPAN6\tprotein_coding\t0\t0\t0\t3.0\t0\t0",
            "ENSG00000000005.6\tTNMD\tprotein_coding\t0\t0\t0\t1.0\t0\t0",
            "ENSG00000000419.13\tDPM1\tprotein_coding\t0\t0\t0\t0.0\t0\t0",
            "ENSG00000000457.14\tSCYL3\tprotein_coding\t0\t0\t0\t2.0\t0\t0",
            "ENSG00000000460.17\tC1orf112\tprotein_coding\t0\t0\t0\t4.0\t0\t0",
        ],
    )
    _write_rnaseq(
        rnaseq_dir / uuid_b / "b.rna_seq.augmented_star_gene_counts.tsv",
        [
            "ENSG00000000003.15\tTSPAN6\tprotein_coding\t0\t0\t0\t7.0\t0\t0",
            "ENSG00000000005.6\tTNMD\tprotein_coding\t0\t0\t0\t2.0\t0\t0",
            "ENSG00000000419.13\tDPM1\tprotein_coding\t0\t0\t0\t1.0\t0\t0",
            "ENSG00000000457.14\tSCYL3\tprotein_coding\t0\t0\t0\t0.0\t0\t0",
            "ENSG00000000460.17\tC1orf112\tprotein_coding\t0\t0\t0\t5.0\t0\t0",
        ],
    )

    matrix, gene_metadata, info = build_expression_matrix(
        rnaseq_dir=rnaseq_dir,
        file_to_patient={
            uuid_a: "TCGA-XX-0001",
            uuid_b: "TCGA-XX-0002",
        },
    )

    assert matrix.shape == (5, 2)
    assert gene_metadata.shape[0] == 5
    assert set(matrix.columns) == {"TCGA-XX-0001", "TCGA-XX-0002"}
    assert round(float(matrix.loc["ENSG00000000003", "TCGA-XX-0001"]), 6) == 2.0
    assert round(float(matrix.loc["ENSG00000000003", "TCGA-XX-0002"]), 6) == 3.0
    assert info.selected_file_uuid_by_patient["TCGA-XX-0001"] == uuid_a
