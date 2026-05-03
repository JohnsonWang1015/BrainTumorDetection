from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IDH_GENES = {"IDH1", "IDH2"}
IDH_VARIANT_CLASSIFICATION = "Missense_Mutation"


@dataclass(frozen=True)
class IDHMutationRecord:
    tumor_sample_barcode: str
    gene: str
    mutation_aa: str
    vaf: float | None


def _patient_id_from_barcode(barcode: str) -> str:
    parts = barcode.split("-")
    if len(parts) < 3:
        return barcode[:12]
    return "-".join(parts[:3])


def _read_maf_rows(maf_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        maf_path,
        sep="\t",
        comment="#",
        dtype=str,
        usecols=lambda c: c
        in {
            "Hugo_Symbol",
            "Variant_Classification",
            "Tumor_Sample_Barcode",
            "HGVSp_Short",
            "t_ref_count",
            "t_alt_count",
        },
        low_memory=False,
    )


def _to_float(value: str | None) -> float | None:
    if value is None or value == "" or value == ".":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_idh_status(maf_path: Path) -> list[dict]:
    df = _read_maf_rows(maf_path)
    if df.empty:
        return []

    filtered = df[
        (df["Hugo_Symbol"].isin(IDH_GENES))
        & (df["Variant_Classification"] == IDH_VARIANT_CLASSIFICATION)
    ]

    out: list[dict] = []
    for _, row in filtered.iterrows():
        barcode = str(row.get("Tumor_Sample_Barcode", "")).strip()
        if not barcode:
            continue
        ref_count = _to_float(row.get("t_ref_count"))
        alt_count = _to_float(row.get("t_alt_count"))
        total = (ref_count or 0.0) + (alt_count or 0.0)
        vaf = None if total <= 0 else (alt_count or 0.0) / total
        out.append(
            {
                "tumor_sample_barcode": barcode,
                "gene": str(row.get("Hugo_Symbol", "")).strip(),
                "mutation_aa": str(row.get("HGVSp_Short", "")).strip(),
                "vaf": vaf,
            }
        )
    return out


def aggregate_idh_labels(maf_dir: Path, source: str) -> pd.DataFrame:
    patient_to_label: dict[str, int] = {}
    patient_to_gene: dict[str, str | None] = {}
    patient_to_mutation: dict[str, str | None] = {}

    maf_paths = sorted(maf_dir.rglob("*.maf.gz"))
    for maf_path in maf_paths:
        try:
            df = _read_maf_rows(maf_path)
        except pd.errors.EmptyDataError:
            continue

        if df.empty:
            continue

        all_barcodes = df.get("Tumor_Sample_Barcode", pd.Series(dtype=str)).dropna().astype(str)
        for barcode in all_barcodes:
            barcode = barcode.strip()
            if not barcode:
                continue
            patient_id = _patient_id_from_barcode(barcode)
            patient_to_label.setdefault(patient_id, 0)
            patient_to_gene.setdefault(patient_id, None)
            patient_to_mutation.setdefault(patient_id, None)

        for row in extract_idh_status(maf_path):
            patient_id = _patient_id_from_barcode(row["tumor_sample_barcode"])
            patient_to_label[patient_id] = 1
            if not patient_to_gene.get(patient_id):
                patient_to_gene[patient_id] = row["gene"] or None
                patient_to_mutation[patient_id] = row["mutation_aa"] or None

    rows = []
    for patient_id in sorted(patient_to_label):
        rows.append(
            {
                "patient_id": patient_id,
                "source": source,
                "idh_label": int(patient_to_label[patient_id]),
                "idh_gene": patient_to_gene.get(patient_id),
                "idh_mutation_aa": patient_to_mutation.get(patient_id),
                "label_source": "maf_aggregated",
            }
        )

    return pd.DataFrame(rows)
