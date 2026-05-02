from __future__ import annotations


def to_legacy_manifest_record(record: dict) -> dict:
    """Project a manifest-v2-style record into the current training shape."""

    return {
        "case_id": record["case_id"],
        "date": record.get("date", "unknown"),
        "modalities": dict(record["modalities"]),
        "mask_path": record.get("mask_path"),
        "idh_label": record.get("idh_label"),
    }
