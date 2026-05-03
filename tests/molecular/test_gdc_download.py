from __future__ import annotations

from pathlib import Path

from idh_glioma.molecular.gdc_download import FileRecord, write_manifest


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
