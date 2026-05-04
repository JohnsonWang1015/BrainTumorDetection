from __future__ import annotations

import csv
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
DEFAULT_DATA_TYPES = (
    "Gene Expression Quantification",
    "Masked Somatic Mutation",
    "Clinical Supplement",
)
DATA_TYPE_TO_SUBDIR = {
    "Gene Expression Quantification": "rnaseq_counts",
    "Masked Somatic Mutation": "maf",
    "Clinical Supplement": "clinical",
}
METHYLATION_PLATFORMS = (
    "Illumina Human Methylation 450",
    "Illumina Human Methylation 27",
)


@dataclass(frozen=True)
class FileRecord:
    file_uuid: str
    filename: str
    md5: str
    size: int
    case_submitter_id: str
    data_type: str
    state: str


def _build_lgg_filter(data_types: list[str]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": ["TCGA-LGG"],
            },
        },
        {
            "op": "in",
            "content": {
                "field": "data_type",
                "value": data_types,
            },
        },
        {
            "op": "in",
            "content": {
                "field": "access",
                "value": ["open"],
            },
        },
    ]
    if "Methylation Beta Value" in data_types:
        content.append(
            {
                "op": "in",
                "content": {
                    "field": "platform",
                    "value": list(METHYLATION_PLATFORMS),
                },
            }
        )
    return {"op": "and", "content": content}


def query_lgg_files(data_types: list[str]) -> list[FileRecord]:
    params = {
        "filters": _build_lgg_filter(data_types),
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "md5sum",
                "file_size",
                "data_type",
                "state",
                "cases.submitter_id",
            ]
        ),
        "format": "JSON",
        "size": "20000",
    }
    response = requests.post(GDC_FILES_ENDPOINT, json=params, timeout=120)
    response.raise_for_status()

    data = response.json().get("data", {}).get("hits", [])
    records: list[FileRecord] = []
    for hit in data:
        cases = hit.get("cases") or []
        if not cases:
            continue
        case_submitter_id = str(cases[0].get("submitter_id", "")).strip()
        if not case_submitter_id:
            continue
        file_uuid = str(hit.get("file_id", "")).strip()
        filename = str(hit.get("file_name", "")).strip()
        if not file_uuid or not filename:
            continue
        records.append(
            FileRecord(
                file_uuid=file_uuid,
                filename=filename,
                md5=str(hit.get("md5sum", "")),
                size=int(hit.get("file_size") or 0),
                case_submitter_id=case_submitter_id,
                data_type=str(hit.get("data_type", "")).strip(),
                state=str(hit.get("state", "released")),
            )
        )
    return records


def write_manifest(records: list[FileRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "filename", "md5", "size", "state"],
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record.file_uuid,
                    "filename": record.filename,
                    "md5": record.md5,
                    "size": record.size,
                    "state": record.state or "released",
                }
            )


def download_via_gdc_client(
    manifest_path: Path,
    out_dir: Path,
    token_path: Path | None = None,
    gdc_client_path: Path | None = None,
) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = gdc_client_path
    if client is None:
        resolved = shutil.which("gdc-client")
        if resolved is None:
            raise FileNotFoundError("gdc-client not found in PATH")
        client = Path(resolved)

    cmd = [str(client), "download", "-m", str(manifest_path), "-d", str(out_dir)]
    if token_path is not None:
        cmd.extend(["-t", str(token_path)])

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] gdc-client download failed for {manifest_path.name}: {exc}")
        return False


def download_lgg_dataset(
    *,
    base_dir: Path,
    data_types: list[str] | None = None,
    token_path: Path | None = None,
    gdc_client_path: Path | None = None,
) -> dict[str, int]:
    data_types = data_types or list(DEFAULT_DATA_TYPES)
    records = query_lgg_files(data_types)

    manifests_dir = base_dir / "manifests"
    data_dir = base_dir / "data"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    by_type: dict[str, list[FileRecord]] = defaultdict(list)
    for record in records:
        if record.data_type in DATA_TYPE_TO_SUBDIR:
            by_type[record.data_type].append(record)

    summary: dict[str, int] = {}
    for data_type in data_types:
        if data_type not in DATA_TYPE_TO_SUBDIR:
            continue
        group = by_type.get(data_type, [])
        subdir = DATA_TYPE_TO_SUBDIR[data_type]
        manifest_path = manifests_dir / f"{subdir}_manifest.txt"
        write_manifest(group, manifest_path)
        if group:
            download_via_gdc_client(
                manifest_path=manifest_path,
                out_dir=data_dir / subdir,
                token_path=token_path,
                gdc_client_path=gdc_client_path,
            )
        summary[subdir] = len(group)

    return summary


def query_lgg_methylation_files() -> list[FileRecord]:
    return query_lgg_files(["Methylation Beta Value"])


def download_lgg_methylation(
    base_dir: Path,
    gdc_client_path: Path | None = None,
) -> dict[str, int]:
    records = query_lgg_methylation_files()
    manifests_dir = base_dir / "manifests"
    data_dir = base_dir / "data" / "methylation_beta"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifests_dir / "methylation_beta_manifest.txt"
    write_manifest(records, manifest_path)
    if records:
        download_via_gdc_client(
            manifest_path=manifest_path,
            out_dir=data_dir,
            gdc_client_path=gdc_client_path,
        )
    return {"methylation_beta": len(records)}
