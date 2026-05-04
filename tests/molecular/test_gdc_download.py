from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from idh_glioma.molecular.gdc_download import (
    FileRecord,
    download_lgg_methylation,
    query_lgg_methylation_files,
    write_manifest,
)


def test_write_manifest_gdc_format(tmp_path: Path) -> None:
    records = [
        FileRecord(
            file_uuid="uuid-1",
            filename="a.tsv",
            md5="abc",
            size=123,
            case_submitter_id="TCGA-XX-0001",
            data_type="Gene Expression Quantification",
            state="released",
        ),
        FileRecord(
            file_uuid="uuid-2",
            filename="b.maf.gz",
            md5="def",
            size=456,
            case_submitter_id="TCGA-XX-0002",
            data_type="Masked Somatic Mutation",
            state="released",
        ),
    ]

    manifest_path = tmp_path / "maf_manifest.txt"
    write_manifest(records, manifest_path)

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id\tfilename\tmd5\tsize\tstate"
    assert lines[1] == "uuid-1\ta.tsv\tabc\t123\treleased"
    assert lines[2] == "uuid-2\tb.maf.gz\tdef\t456\treleased"


def test_query_lgg_methylation_files_filters_platforms(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_post(url: str, json: dict, timeout: int) -> Mock:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        response = Mock()
        response.raise_for_status = Mock()
        response.json = Mock(
            return_value={
                "data": {
                    "hits": [
                        {
                            "file_id": "m-uuid-1",
                            "file_name": "sample1.sesame.level3betas.txt",
                            "md5sum": "abc",
                            "file_size": 1000,
                            "data_type": "Methylation Beta Value",
                            "state": "released",
                            "cases": [{"submitter_id": "TCGA-XX-0001"}],
                        }
                    ]
                }
            }
        )
        return response

    monkeypatch.setattr("idh_glioma.molecular.gdc_download.requests.post", _fake_post)
    records = query_lgg_methylation_files()

    assert len(records) == 1
    assert records[0].file_uuid == "m-uuid-1"
    payload = captured["json"]
    assert isinstance(payload, dict)
    filters = payload["filters"]["content"]  # type: ignore[index]
    platform_filter = [f for f in filters if f["content"]["field"] == "platform"][0]  # type: ignore[index]
    assert platform_filter["content"]["value"] == [  # type: ignore[index]
        "Illumina Human Methylation 450",
        "Illumina Human Methylation 27",
    ]


def test_download_lgg_methylation_writes_manifest_and_calls_downloader(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "tcga_lgg_downloads"
    records = [
        FileRecord(
            file_uuid="m-uuid-1",
            filename="sample1.sesame.level3betas.txt",
            md5="abc",
            size=1000,
            case_submitter_id="TCGA-XX-0001",
            data_type="Methylation Beta Value",
            state="released",
        ),
        FileRecord(
            file_uuid="m-uuid-2",
            filename="sample2.sesame.level3betas.txt",
            md5="def",
            size=2000,
            case_submitter_id="TCGA-XX-0002",
            data_type="Methylation Beta Value",
            state="released",
        ),
    ]
    calls: list[tuple[Path, Path]] = []

    monkeypatch.setattr("idh_glioma.molecular.gdc_download.query_lgg_methylation_files", lambda: records)

    def _fake_download(*, manifest_path: Path, out_dir: Path, token_path=None, gdc_client_path=None) -> bool:
        del token_path, gdc_client_path
        calls.append((manifest_path, out_dir))
        return True

    monkeypatch.setattr("idh_glioma.molecular.gdc_download.download_via_gdc_client", _fake_download)
    summary = download_lgg_methylation(base_dir=base_dir)

    assert summary["methylation_beta"] == 2
    manifest = base_dir / "manifests" / "methylation_beta_manifest.txt"
    assert manifest.exists()
    assert len(calls) == 1
    assert calls[0][0] == manifest
    assert calls[0][1] == base_dir / "data" / "methylation_beta"
