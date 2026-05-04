from __future__ import annotations

import numpy as np
import pandas as pd

from idh_glioma.molecular.feature_select import select_features_modality
from idh_glioma.molecular.multimodal import concat_modalities, select_multimodal_features


def test_concat_modalities_uses_strict_patient_subset_and_prefixes() -> None:
    rnaseq = pd.DataFrame(
        {
            "P1": [1.0, 2.0],
            "P2": [3.0, 4.0],
        },
        index=["ENSG00000000001", "ENSG00000000002"],
    )
    methylation = pd.DataFrame(
        {
            "P2": [0.1, 0.2],
            "P3": [0.3, 0.4],
        },
        index=["cg00000001", "cg00000002"],
    )

    fused = concat_modalities(
        matrices={"rnaseq": rnaseq, "methylation": methylation},
        patient_ids=["P1", "P2", "P3"],
    )

    assert list(fused.index) == ["P2"]
    assert list(fused.columns) == [
        "rnaseq:ENSG00000000001",
        "rnaseq:ENSG00000000002",
        "methylation:cg00000001",
        "methylation:cg00000002",
    ]
    assert fused.shape == (1, 4)
    assert float(fused.loc["P2", "rnaseq:ENSG00000000001"]) == 3.0
    assert float(fused.loc["P2", "methylation:cg00000002"]) == 0.2


def test_select_features_modality_top_k_union_prior_ids() -> None:
    rng = np.random.default_rng(13)
    feature_ids = [f"f{i:03d}" for i in range(30)]
    patient_ids = [f"P{i:03d}" for i in range(20)]
    X_train = pd.DataFrame(rng.normal(0.0, 0.01, size=(30, 20)), index=feature_ids, columns=patient_ids)
    X_train.loc["f001"] = rng.normal(0.0, 2.0, size=20)
    X_train.loc["f002"] = rng.normal(0.0, 1.8, size=20)
    X_train.loc["f029"] = rng.normal(0.0, 0.02, size=20)

    selected = select_features_modality(X_train=X_train, top_k=2, prior_ids={"f029", "f999"})
    assert set(["f001", "f002", "f029"]).issubset(set(selected))
    assert "f999" not in selected


def test_select_multimodal_features_is_per_modality_fold_aware() -> None:
    rnaseq_train = pd.DataFrame(
        {
            "P1": [0.0, 0.0, 10.0],
            "P2": [0.1, 0.0, -10.0],
            "P3": [0.2, 0.0, 10.0],
        },
        index=["ENSG1", "ENSG2", "ENSG3"],
    )
    methyl_train = pd.DataFrame(
        {
            "P1": [0.0, 0.1, 5.0],
            "P2": [0.1, 0.1, -5.0],
            "P3": [0.2, 0.1, 5.0],
        },
        index=["cg1", "cg2", "cg3"],
    )

    selected = select_multimodal_features(
        matrices={"rnaseq": rnaseq_train, "methylation": methyl_train},
        prior_panels={"rnaseq": {"ENSG2"}, "methylation": {"cg2"}},
        top_k_per_modality={"rnaseq": 1, "methylation": 1},
    )

    assert selected["rnaseq"][0] == "ENSG3"
    assert "ENSG2" in selected["rnaseq"]
    assert selected["methylation"][0] == "cg3"
    assert "cg2" in selected["methylation"]
