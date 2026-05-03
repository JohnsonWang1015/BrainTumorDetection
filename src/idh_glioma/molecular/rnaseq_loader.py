from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpressionBuildInfo:
    selected_file_uuid_by_patient: dict[str, str]
    multiple_aliquots_by_patient: dict[str, bool]
    missing_ratio_by_patient: dict[str, float]
    missing_mapping_file_uuids: list[str]


def _normalize_gene_id(gene_id: str) -> str:
    return gene_id.split(".", 1)[0]


def _read_rnaseq_table(tsv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        tsv_path,
        sep="\t",
        comment="#",
        dtype=str,
        usecols=["gene_id", "gene_name", "gene_type", "tpm_unstranded"],
        low_memory=False,
    )
    frame = frame[frame["gene_id"].str.startswith("ENSG", na=False)].copy()
    frame["gene_id_base"] = frame["gene_id"].map(_normalize_gene_id)
    frame["gene_name"] = frame["gene_name"].fillna("").astype(str)
    frame["gene_type"] = frame["gene_type"].fillna("").astype(str)
    frame["tpm_unstranded"] = pd.to_numeric(frame["tpm_unstranded"], errors="coerce")
    return frame


def parse_rnaseq_tsv(tsv_path: Path) -> pd.Series:
    frame = _read_rnaseq_table(tsv_path)
    tpm = (
        frame[["gene_id_base", "tpm_unstranded"]]
        .fillna({"tpm_unstranded": 0.0})
        .groupby("gene_id_base", sort=True)["tpm_unstranded"]
        .sum()
        .astype(float)
    )
    tpm.name = tsv_path.stem
    return tpm


def _pick_expression_file(file_dir: Path) -> Path | None:
    candidates = sorted(file_dir.glob("*.rna_seq.*.tsv"))
    if candidates:
        return candidates[0]
    fallback = sorted(path for path in file_dir.glob("*.tsv") if path.name != "annotations.txt")
    return fallback[0] if fallback else None


def _select_patient_files(
    rnaseq_dir: Path,
    file_to_patient: dict[str, str],
) -> tuple[dict[str, tuple[str, Path]], dict[str, bool], list[str]]:
    patient_to_files: dict[str, list[tuple[str, Path]]] = {}
    missing_mapping_file_uuids: list[str] = []

    for entry in sorted(rnaseq_dir.iterdir()):
        if not entry.is_dir():
            continue
        file_uuid = entry.name
        patient_id = file_to_patient.get(file_uuid)
        if patient_id is None:
            missing_mapping_file_uuids.append(file_uuid)
            continue
        tsv_path = _pick_expression_file(entry)
        if tsv_path is None:
            continue
        patient_to_files.setdefault(patient_id, []).append((file_uuid, tsv_path))

    selected: dict[str, tuple[str, Path]] = {}
    multiple_aliquots_by_patient: dict[str, bool] = {}
    for patient_id, files in patient_to_files.items():
        files_sorted = sorted(files, key=lambda item: item[0])
        selected[patient_id] = files_sorted[0]
        multiple_aliquots_by_patient[patient_id] = len(files_sorted) > 1
    return selected, multiple_aliquots_by_patient, sorted(missing_mapping_file_uuids)


def build_expression_matrix(
    rnaseq_dir: Path,
    file_to_patient: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, ExpressionBuildInfo]:
    selected, multiple_aliquots_by_patient, missing_mapping_file_uuids = _select_patient_files(
        rnaseq_dir=rnaseq_dir,
        file_to_patient=file_to_patient,
    )

    expression_by_patient: dict[str, pd.Series] = {}
    missing_ratio_by_patient: dict[str, float] = {}
    selected_file_uuid_by_patient: dict[str, str] = {}
    gene_metadata: pd.DataFrame | None = None

    for patient_id in sorted(selected):
        file_uuid, tsv_path = selected[patient_id]
        frame = _read_rnaseq_table(tsv_path)
        series = (
            frame[["gene_id_base", "tpm_unstranded"]]
            .fillna({"tpm_unstranded": 0.0})
            .groupby("gene_id_base", sort=True)["tpm_unstranded"]
            .sum()
            .astype(float)
        )
        expression_by_patient[patient_id] = np.log2(series + 1.0)
        missing_ratio_by_patient[patient_id] = float(frame["tpm_unstranded"].isna().mean())
        selected_file_uuid_by_patient[patient_id] = file_uuid

        if gene_metadata is None:
            meta = frame[["gene_id_base", "gene_name", "gene_type"]].copy()
            meta = meta.drop_duplicates(subset=["gene_id_base"], keep="first")
            gene_metadata = meta.rename(columns={"gene_id_base": "gene_id", "gene_name": "gene_symbol"})
            gene_metadata = gene_metadata.set_index("gene_id", drop=True).sort_index()

    matrix = pd.DataFrame(expression_by_patient, dtype=float).sort_index()
    matrix = matrix.fillna(0.0)
    if gene_metadata is None:
        gene_metadata = pd.DataFrame(columns=["gene_symbol", "gene_type"], index=pd.Index([], name="gene_id"))
    else:
        gene_metadata = gene_metadata.reindex(matrix.index)
        gene_metadata["gene_symbol"] = gene_metadata["gene_symbol"].fillna("")
        gene_metadata["gene_type"] = gene_metadata["gene_type"].fillna("")

    build_info = ExpressionBuildInfo(
        selected_file_uuid_by_patient=selected_file_uuid_by_patient,
        multiple_aliquots_by_patient=multiple_aliquots_by_patient,
        missing_ratio_by_patient=missing_ratio_by_patient,
        missing_mapping_file_uuids=missing_mapping_file_uuids,
    )
    return matrix, gene_metadata, build_info
