from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from idh_glioma.molecular.feature_select import load_idh_prior_panel, select_features
from idh_glioma.molecular.models import LightGBMIDH, LogisticIDH, MLPIDH
from idh_glioma.utils import save_json

MODEL_CHOICES = ("logistic", "lightgbm", "mlp", "all")


def _load_training_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    expression = pd.read_parquet(input_dir / "expression_matrix.parquet")
    labels = pd.read_parquet(input_dir / "idh_labels.parquet")
    gene_meta_path = input_dir / "gene_metadata.parquet"
    gene_metadata = pd.read_parquet(gene_meta_path) if gene_meta_path.exists() else None
    return expression, labels, gene_metadata


def _aligned_training_set(
    expression: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    labels = labels.copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels = labels.drop_duplicates(subset=["patient_id"], keep="first")
    label_map = labels.set_index("patient_id")["idh_label"].astype(int)
    patient_ids = sorted(set(expression.columns.astype(str)) & set(label_map.index.astype(str)))
    X = expression[patient_ids]
    y = label_map.loc[patient_ids].to_numpy(dtype=int)
    return X, y, patient_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train molecular IDH classifiers from expression matrix")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/molecular"))
    parser.add_argument("--model", choices=MODEL_CHOICES, default="all")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/molecular_idh"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    expression, labels, gene_metadata = _load_training_inputs(args.input_dir)
    X_genes_by_patient, y, patient_ids = _aligned_training_set(expression, labels)
    prior_panel = load_idh_prior_panel()
    features = select_features(
        X_train_log_tpm=X_genes_by_patient,
        top_k=args.top_k,
        prior_panel=prior_panel,
        gene_metadata=gene_metadata,
    )
    X = X_genes_by_patient.loc[features].T.to_numpy(dtype=np.float32)

    selected_models = [args.model] if args.model != "all" else ["logistic", "lightgbm", "mlp"]
    report: dict[str, Any] = {
        "n_samples": int(len(patient_ids)),
        "n_features": int(len(features)),
        "top_k": int(args.top_k),
        "features": features,
        "models": {},
    }

    for model_name in selected_models:
        if model_name == "logistic":
            model = LogisticIDH(random_state=args.seed)
            save_path = args.save_dir / "logistic.joblib"
        elif model_name == "lightgbm":
            model = LightGBMIDH(random_state=args.seed)
            save_path = args.save_dir / "lightgbm.txt"
        else:
            model = MLPIDH(input_dim=X.shape[1])
            save_path = args.save_dir / "mlp.pt"

        t0 = time.perf_counter()
        model.fit(X, y.astype(np.int64))
        fit_time_sec = float(time.perf_counter() - t0)
        probs = model.predict_proba(X)[:, 1]
        train_auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan")
        model.save(save_path)
        report["models"][model_name] = {
            "train_auc": train_auc,
            "fit_time_sec": fit_time_sec,
            "checkpoint": str(save_path),
        }
        print(f"[INFO] {model_name}: train_auc={train_auc:.4f} fit_time_sec={fit_time_sec:.2f}")

    save_json(report, args.save_dir / "training_report.json")
    print(f"[INFO] wrote {args.save_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
