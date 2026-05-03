from __future__ import annotations

import numpy as np
import pandas as pd

from idh_glioma.molecular.feature_select import select_features


def test_select_features_top_k_and_prior_union_fold_aware() -> None:
    rng = np.random.default_rng(7)
    genes = [f"ENSG{i:011d}" for i in range(100)]
    train_samples = [f"S{i:03d}" for i in range(40)]
    X_train = pd.DataFrame(
        rng.normal(0.0, 0.05, size=(100, 40)),
        index=genes,
        columns=train_samples,
    )

    high_var_train = ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003", "ENSG00000000004", "ENSG00000000005"]
    for idx, gene in enumerate(high_var_train, start=1):
        X_train.loc[gene] = rng.normal(0.0, 2.0 + idx * 0.1, size=len(train_samples))

    gene_metadata = pd.DataFrame(
        {
            "gene_symbol": ["IDH1", "TP53", "PDGFRA"],
            "gene_type": ["protein_coding", "protein_coding", "protein_coding"],
        },
        index=["ENSG00000000090", "ENSG00000000091", "ENSG00000000092"],
    )
    gene_metadata.index.name = "gene_id"

    selected = select_features(
        X_train_log_tpm=X_train,
        top_k=5,
        prior_panel={"IDH1"},
        gene_metadata=gene_metadata,
    )

    assert len(selected) == 6
    assert set(high_var_train).issubset(set(selected))
    assert "ENSG00000000090" in selected
    assert "ENSG00000000091" not in selected
