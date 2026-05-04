from __future__ import annotations

import json
from pathlib import Path


def test_idh_cpg_panel_schema() -> None:
    panel_path = Path("src/idh_glioma/molecular/priors/idh_cpg_panel.json")
    payload = json.loads(panel_path.read_text(encoding="utf-8"))

    assert "_citation" in payload
    assert isinstance(payload["_citation"], list)
    assert payload["_citation"]
    assert all(isinstance(item, str) and item for item in payload["_citation"])

    assert "cpg_ids" in payload
    assert isinstance(payload["cpg_ids"], list)
    cpg_ids = payload["cpg_ids"]
    assert cpg_ids
    assert all(isinstance(cpg_id, str) for cpg_id in cpg_ids)
    assert len(cpg_ids) == len(set(cpg_ids))
    assert all(cpg_id.startswith("cg") for cpg_id in cpg_ids)
