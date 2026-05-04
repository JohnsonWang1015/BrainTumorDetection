from __future__ import annotations

import math
from pathlib import Path

from idh_glioma.molecular.methylation_loader import (
    build_methylation_matrix,
    discover_methylation_files,
    parse_sesame_txt,
)


def _write_sesame(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join([f"{cg_id}\t{beta}" for cg_id, beta in rows])
    path.write_text(body, encoding="utf-8")


def test_parse_sesame_txt_preserves_nan_values(tmp_path: Path) -> None:
    sesame_path = tmp_path / "sample.sesame.level3betas.txt"
    _write_sesame(
        sesame_path,
        [
            ("cg00000001", "0.1"),
            ("cg00000002", "NaN"),
            ("cg00000003", "0.9"),
        ],
    )

    series = parse_sesame_txt(sesame_path)

    assert list(series.index) == ["cg00000001", "cg00000002", "cg00000003"]
    assert float(series.loc["cg00000001"]) == 0.1
    assert math.isnan(float(series.loc["cg00000002"]))


def test_build_methylation_matrix_intersection_transform_and_dedup(tmp_path: Path) -> None:
    meth_dir = tmp_path / "methylation_beta"
    file_to_patient = {
        "a-uuid": "TCGA-XX-0001",
        "z-uuid": "TCGA-XX-0001",
        "b-uuid": "TCGA-XX-0002",
        "c-uuid": "TCGA-XX-0003",
    }

    _write_sesame(
        meth_dir / "a-uuid" / "a.sesame.level3betas.txt",
        [("cg00000001", "0.0"), ("cg00000002", "0.5"), ("cg00000003", "1.0")],
    )
    _write_sesame(
        meth_dir / "z-uuid" / "z.sesame.level3betas.txt",
        [("cg00000001", "0.2"), ("cg00000002", "0.5"), ("cg00000003", "0.8")],
    )
    _write_sesame(
        meth_dir / "b-uuid" / "b.sesame.level3betas.txt",
        [("cg00000001", "0.3"), ("cg00000002", "0.7"), ("cg00000003", "0.1"), ("cg00000004", "0.4")],
    )
    _write_sesame(
        meth_dir / "c-uuid" / "c.sesame.level3betas.txt",
        [("cg00000001", "0.6"), ("cg00000002", "NaN"), ("cg00000003", "0.2"), ("cg00000004", "0.9")],
    )
    (meth_dir / "b-uuid" / "annotations.txt").write_text("ignored", encoding="utf-8")

    discovered = discover_methylation_files(meth_dir)
    assert [uuid for uuid, _ in discovered] == ["a-uuid", "b-uuid", "c-uuid", "z-uuid"]

    matrix, summary = build_methylation_matrix(
        meth_dir=meth_dir,
        file_to_patient=file_to_patient,
        source="tcga_gbm",
        beta_to_mvalue=True,
    )

    assert list(matrix.columns) == ["TCGA-XX-0001", "TCGA-XX-0002", "TCGA-XX-0003"]
    assert list(matrix.index) == ["cg00000001", "cg00000002", "cg00000003"]
    assert summary["intersection_size"] == 3
    assert summary["nan_fill_count"] == 1
    assert summary["multiple_methylation_aliquots"] is True
    assert summary["multiple_aliquot_patient_ids"] == ["TCGA-XX-0001"]

    lower = math.log2(0.001 / 0.999)
    upper = math.log2(0.999 / 0.001)
    assert math.isclose(float(matrix.loc["cg00000001", "TCGA-XX-0001"]), lower, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(float(matrix.loc["cg00000003", "TCGA-XX-0001"]), upper, rel_tol=0, abs_tol=1e-12)
    assert float(matrix.loc["cg00000002", "TCGA-XX-0003"]) == 0.0
