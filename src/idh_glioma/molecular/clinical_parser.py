from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _find_first_text(root: ET.Element, *candidate_tags: str) -> str | None:
    for node in root.iter():
        name = _local_name(node.tag)
        if name in candidate_tags and node.text:
            text = node.text.strip()
            if text:
                return text
    return None


def parse_clinical_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()

    patient_id = _find_first_text(root, "bcr_patient_barcode")
    age_at_initial = _safe_int(_find_first_text(root, "age_at_initial_pathologic_diagnosis"))
    days_to_birth = _safe_int(_find_first_text(root, "days_to_birth"))
    age_at_diagnosis = age_at_initial
    if age_at_diagnosis is None and days_to_birth is not None:
        age_at_diagnosis = abs(days_to_birth) // 365

    return {
        "patient_id": patient_id,
        "age_at_diagnosis": age_at_diagnosis,
        "gender": _find_first_text(root, "gender"),
        "vital_status": _find_first_text(root, "vital_status"),
        "days_to_death": _safe_int(_find_first_text(root, "days_to_death")),
        "days_to_last_followup": _safe_int(_find_first_text(root, "days_to_last_followup")),
        "histological_type": _find_first_text(root, "histological_type"),
        "kps_score": _safe_int(_find_first_text(root, "karnofsky_performance_score")),
    }


def aggregate_clinical(clin_dir: Path, source: str) -> pd.DataFrame:
    rows: list[dict] = []
    for xml_path in sorted(clin_dir.rglob("*.xml")):
        parsed = parse_clinical_xml(xml_path)
        if not parsed.get("patient_id"):
            continue
        parsed["source"] = source
        rows.append(parsed)
    return pd.DataFrame(rows)
