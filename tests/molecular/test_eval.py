from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idh_glioma.molecular.eval import main


def test_eval_main_all_modes_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(42)
    input_dir = tmp_path / "molecular"
    output_dir = tmp_path / "molecular_eval"
    checkpoint_dir = tmp_path / "ckpt"
    input_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    n_genes = 60
    n_patients = 48
    genes = [f"ENSG{i:011d}" for i in range(n_genes)]
    patients = [f"TCGA-XX-{i:04d}" for i in range(n_patients)]
    y = np.array([0, 1] * (n_patients // 2), dtype=int)
    signal = y.astype(float) * 2.0
    X = rng.normal(0.0, 1.0, size=(n_genes, n_patients))
    X[:5, :] += signal

    expression = pd.DataFrame(X, index=genes, columns=patients)
    labels = pd.DataFrame(
        {
            "patient_id": patients,
            "source": ["tcga_gbm" if i < (n_patients // 2) else "tcga_lgg" for i in range(n_patients)],
            "idh_label": y,
            "idh_gene": [None] * n_patients,
            "idh_mutation_aa": [None] * n_patients,
            "label_source": ["maf_aggregated"] * n_patients,
        }
    )
    gene_metadata = pd.DataFrame(
        {"gene_symbol": [f"GENE{i}" for i in range(n_genes)], "gene_type": ["protein_coding"] * n_genes},
        index=genes,
    )
    gene_metadata.index.name = "gene_id"

    expression.to_parquet(input_dir / "expression_matrix.parquet")
    labels.to_parquet(input_dir / "idh_labels.parquet", index=False)
    gene_metadata.to_parquet(input_dir / "gene_metadata.parquet")

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval-idh-molecular",
            "--input-dir",
            str(input_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--mode",
            "all",
            "--output-dir",
            str(output_dir),
            "--folds",
            "3",
            "--seed",
            "42",
        ],
    )
    main()

    pooled_path = output_dir / "pooled_cv_results.json"
    holdout_path = output_dir / "source_holdout_results.json"
    minority_path = output_dir / "minority_metrics.json"
    assert pooled_path.exists()
    assert holdout_path.exists()
    assert minority_path.exists()
    assert (output_dir / "figures" / "roc_curves.png").exists()
    assert (output_dir / "figures" / "calibration_curve.png").exists()
    assert (output_dir / "figures" / "confusion_matrices.png").exists()

    pooled = json.loads(pooled_path.read_text(encoding="utf-8"))
    for model_name in ("logistic", "lightgbm", "mlp"):
        assert model_name in pooled["models"]
        mean_auc = float(pooled["models"][model_name]["mean_auc"])
        assert 0.0 <= mean_auc <= 1.0
