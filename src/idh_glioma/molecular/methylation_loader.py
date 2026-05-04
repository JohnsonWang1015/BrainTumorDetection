from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_sesame_txt(path: Path) -> pd.Series:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["cg_id", "beta"],
        usecols=[0, 1],
        dtype={"cg_id": str, "beta": str},
        low_memory=False,
    )
    frame["cg_id"] = frame["cg_id"].fillna("").astype(str).str.strip()
    frame = frame[frame["cg_id"].str.startswith("cg", na=False)].copy()
    frame["beta"] = pd.to_numeric(frame["beta"], errors="coerce")
    series = frame.set_index("cg_id")["beta"]
    series.name = path.stem
    return series


def discover_methylation_files(meth_dir: Path) -> list[tuple[str, Path]]:
    discovered: list[tuple[str, Path]] = []
    if not meth_dir.exists():
        return discovered
    for file_dir in sorted(meth_dir.iterdir()):
        if not file_dir.is_dir():
            continue
        candidates = sorted(path for path in file_dir.glob("*.txt") if path.name != "annotations.txt")
        if not candidates:
            continue
        discovered.append((file_dir.name, candidates[0]))
    return discovered


def _platform_from_cpg_count(cpg_count: int) -> str:
    if cpg_count >= 100_000:
        return "hm450"
    if cpg_count >= 20_000:
        return "hm27"
    return "unknown"


def build_methylation_matrix(
    meth_dir: Path,
    file_to_patient: dict[str, str],
    source: str,
    beta_to_mvalue: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    discovered = discover_methylation_files(meth_dir)

    patient_to_files: dict[str, list[tuple[str, Path]]] = {}
    missing_mapping_file_uuids: list[str] = []
    for file_uuid, sesame_path in discovered:
        patient_id = file_to_patient.get(file_uuid)
        if patient_id is None:
            missing_mapping_file_uuids.append(file_uuid)
            continue
        patient_to_files.setdefault(patient_id, []).append((file_uuid, sesame_path))

    selected: dict[str, tuple[str, Path]] = {}
    multiple_aliquot_patient_ids: list[str] = []
    for patient_id, entries in patient_to_files.items():
        sorted_entries = sorted(entries, key=lambda item: item[0])
        selected[patient_id] = sorted_entries[0]
        if len(sorted_entries) > 1:
            multiple_aliquot_patient_ids.append(patient_id)

    sample_series: dict[str, pd.Series] = {}
    sample_cpg_counts: dict[str, int] = {}
    sample_nan_ratio: dict[str, float] = {}
    for patient_id in sorted(selected):
        _, sesame_path = selected[patient_id]
        series = parse_sesame_txt(sesame_path)
        sample_series[patient_id] = series
        sample_cpg_counts[patient_id] = len(series)
        sample_nan_ratio[patient_id] = float(series.isna().mean()) if len(series) > 0 else 0.0

    n_patients_per_platform = {"hm27": 0, "hm450": 0, "unknown": 0}
    sample_platforms = {patient_id: _platform_from_cpg_count(count) for patient_id, count in sample_cpg_counts.items()}
    if sample_platforms and all(platform == "unknown" for platform in sample_platforms.values()):
        counts = sorted(set(sample_cpg_counts.values()))
        if len(counts) > 1:
            hm27_count = counts[0]
            hm450_count = counts[-1]
            sample_platforms = {
                patient_id: "hm27"
                if sample_cpg_counts[patient_id] == hm27_count
                else ("hm450" if sample_cpg_counts[patient_id] == hm450_count else "unknown")
                for patient_id in sample_cpg_counts
            }
    for platform in sample_platforms.values():
        n_patients_per_platform[platform] += 1

    if not sample_series:
        empty = pd.DataFrame(dtype=float)
        summary: dict[str, Any] = {
            "source": source,
            "n_patients_per_platform": n_patients_per_platform,
            "intersection_size": 0,
            "nan_fill_count": 0,
            "multiple_methylation_aliquots": False,
            "multiple_aliquot_patient_ids": [],
            "sample_nan_ratio": sample_nan_ratio,
            "missing_mapping_file_uuids": sorted(missing_mapping_file_uuids),
        }
        return empty, summary

    common_cpgs: set[str] | None = None
    for series in sample_series.values():
        cpg_ids = set(series.index.astype(str))
        common_cpgs = cpg_ids if common_cpgs is None else common_cpgs & cpg_ids
    common_index = sorted(common_cpgs or [])

    matrix = pd.DataFrame(index=common_index, dtype=float)
    nan_fill_count = 0
    for patient_id, series in sample_series.items():
        values = series.reindex(common_index)
        nan_fill_count += int(values.isna().sum())
        values = values.fillna(0.5).astype(float)
        if beta_to_mvalue:
            clipped = values.clip(lower=0.001, upper=0.999)
            values = np.log2(clipped / (1.0 - clipped))
        matrix[patient_id] = values.astype(float)
    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    summary = {
        "source": source,
        "n_patients_per_platform": n_patients_per_platform,
        "intersection_size": len(common_index),
        "nan_fill_count": nan_fill_count,
        "multiple_methylation_aliquots": len(multiple_aliquot_patient_ids) > 0,
        "multiple_aliquot_patient_ids": sorted(multiple_aliquot_patient_ids),
        "sample_nan_ratio": sample_nan_ratio,
        "missing_mapping_file_uuids": sorted(missing_mapping_file_uuids),
    }
    return matrix, summary
