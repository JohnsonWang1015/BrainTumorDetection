from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idh_glioma.molecular.eval import main as eval_main
from idh_glioma.molecular.train import main as train_main


def test_train_eval_multimodal_end_to_end(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(123)
    input_dir = tmp_path / "molecular_multimodal"
    input_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    n_patients = 30
    patients = [f"TCGA-XX-{i:04d}" for i in range(n_patients)]
    y = np.array([0, 1] * (n_patients // 2), dtype=int)
    signal = y.astype(float) * 2.0

    rnaseq_genes = [f"ENSG{i:011d}" for i in range(10)]
    methylation_cpgs = [f"cg{i:08d}" for i in range(10)]

    X_rna = rng.normal(0.0, 1.0, size=(10, n_patients))
    X_meth = rng.normal(0.0, 1.0, size=(10, n_patients))
    X_rna[:3, :] += signal
    X_meth[:3, :] += signal

    expression = pd.DataFrame(X_rna, index=rnaseq_genes, columns=patients)
    methylation = pd.DataFrame(X_meth, index=methylation_cpgs, columns=patients)
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
        {"gene_symbol": [f"GENE{i}" for i in range(10)], "gene_type": ["protein_coding"] * 10},
        index=rnaseq_genes,
    )
    gene_metadata.index.name = "gene_id"

    expression.to_parquet(input_dir / "expression_matrix.parquet")
    methylation.to_parquet(input_dir / "methylation_matrix.parquet")
    labels.to_parquet(input_dir / "idh_labels.parquet", index=False)
    gene_metadata.to_parquet(input_dir / "gene_metadata.parquet")

    monkeypatch.setattr(
        "sys.argv",
        [
            "train-idh-molecular",
            "--modalities",
            "rnaseq",
            "methylation",
            "--input-dir",
            str(input_dir),
            "--top-k",
            "5",
            "--seed",
            "42",
        ],
    )
    train_main()

    save_dir = tmp_path / "checkpoints" / "molecular_idh_multimodal"
    assert (save_dir / "logistic.joblib").exists()
    assert (save_dir / "lightgbm.txt").exists()
    assert (save_dir / "mlp.pt").exists()
    report = json.loads((save_dir / "training_report.json").read_text(encoding="utf-8"))
    assert report["modalities"] == ["rnaseq", "methylation"]
    assert report["n_features_per_modality"]["rnaseq"] >= 5
    assert report["n_features_per_modality"]["methylation"] >= 5

    monkeypatch.setattr(
        "sys.argv",
        [
            "eval-idh-molecular",
            "--modalities",
            "rnaseq",
            "methylation",
            "--input-dir",
            str(input_dir),
            "--checkpoint-dir",
            str(save_dir),
            "--mode",
            "all",
            "--folds",
            "3",
            "--seed",
            "42",
        ],
    )
    eval_main()

    output_dir = tmp_path / "artifacts" / "molecular_idh_eval_multimodal"
    pooled = output_dir / "pooled_cv_results.json"
    holdout = output_dir / "source_holdout_results.json"
    minority = output_dir / "minority_metrics.json"
    assert pooled.exists()
    assert holdout.exists()
    assert minority.exists()
    pooled_payload = json.loads(pooled.read_text(encoding="utf-8"))
    for model_name in ("logistic", "lightgbm", "mlp"):
        mean_auc = float(pooled_payload["models"][model_name]["mean_auc"])
        assert 0.0 <= mean_auc <= 1.0
