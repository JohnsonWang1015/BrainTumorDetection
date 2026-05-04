from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from idh_glioma.molecular.feature_select import (
    _map_prior_to_gene_ids,
    load_idh_cpg_panel,
    load_idh_prior_panel,
    select_features,
)
from idh_glioma.molecular.models import LightGBMIDH, LogisticIDH, MLPIDH
from idh_glioma.molecular.multimodal import concat_modalities, select_multimodal_features
from idh_glioma.utils import save_json

MODEL_CHOICES = ("logistic", "lightgbm", "mlp", "all")
MODALITY_CHOICES = ("rnaseq", "methylation")
DEFAULT_SAVE_DIR = Path("checkpoints/molecular_idh")


def _normalize_modalities(modalities: list[str]) -> list[str]:
    deduped = list(dict.fromkeys(str(modality).strip().lower() for modality in modalities if str(modality).strip()))
    unknown = sorted(set(deduped) - set(MODALITY_CHOICES))
    if unknown:
        raise SystemExit(f"Unsupported modalities: {unknown}. Allowed values: {list(MODALITY_CHOICES)}")
    if deduped == ["methylation"] or "rnaseq" not in deduped:
        raise SystemExit("RNA-seq modality is required. Use --modalities rnaseq or rnaseq methylation.")
    return deduped


def _resolve_save_dir(save_dir: Path, modalities: list[str]) -> Path:
    if len(modalities) > 1 and save_dir == DEFAULT_SAVE_DIR:
        return Path(f"{save_dir}_multimodal")
    return save_dir


def _load_training_inputs(
    input_dir: Path,
    modalities: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame | None]:
    labels = pd.read_parquet(input_dir / "idh_labels.parquet")
    matrices: dict[str, pd.DataFrame] = {}
    if "rnaseq" in modalities:
        matrices["rnaseq"] = pd.read_parquet(input_dir / "expression_matrix.parquet")
    if "methylation" in modalities:
        matrices["methylation"] = pd.read_parquet(input_dir / "methylation_matrix.parquet")
    gene_meta_path = input_dir / "gene_metadata.parquet"
    gene_metadata = pd.read_parquet(gene_meta_path) if gene_meta_path.exists() else None
    return matrices, labels, gene_metadata


def _aligned_training_set(
    matrices: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], np.ndarray, list[str]]:
    labels = labels.copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels = labels.drop_duplicates(subset=["patient_id"], keep="first")
    label_map = labels.set_index("patient_id")["idh_label"].astype(int)

    common_patients = set(label_map.index.astype(str))
    for matrix in matrices.values():
        common_patients &= set(matrix.columns.astype(str))
    patient_ids = sorted(common_patients)

    aligned = {modality: matrix[patient_ids] for modality, matrix in matrices.items()}
    y = label_map.loc[patient_ids].to_numpy(dtype=int)
    return aligned, y, patient_ids


def _select_features_by_modality(
    matrices: dict[str, pd.DataFrame],
    gene_metadata: pd.DataFrame | None,
    top_k: int,
) -> dict[str, list[str]]:
    if list(matrices.keys()) == ["rnaseq"]:
        return {
            "rnaseq": select_features(
                X_train_log_tpm=matrices["rnaseq"],
                top_k=top_k,
                prior_panel=load_idh_prior_panel(),
                gene_metadata=gene_metadata,
            )
        }

    rnaseq_prior = load_idh_prior_panel()
    mapped_rnaseq_prior = _map_prior_to_gene_ids(rnaseq_prior, matrices["rnaseq"].index, gene_metadata=gene_metadata)
    prior_panels = {
        "rnaseq": mapped_rnaseq_prior,
        "methylation": load_idh_cpg_panel(),
    }
    top_k_per_modality = {modality: top_k for modality in matrices}
    return select_multimodal_features(
        matrices=matrices,
        prior_panels=prior_panels,
        top_k_per_modality=top_k_per_modality,
    )


def _build_feature_matrix(
    matrices: dict[str, pd.DataFrame],
    selected: dict[str, list[str]],
    patient_ids: list[str],
) -> pd.DataFrame:
    selected_matrices = {modality: matrices[modality].loc[features] for modality, features in selected.items()}
    return concat_modalities(selected_matrices, patient_ids=patient_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train molecular IDH classifiers from expression matrix")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/molecular"))
    parser.add_argument("--model", choices=MODEL_CHOICES, default="all")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modalities", nargs="+", default=["rnaseq"], choices=MODALITY_CHOICES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modalities = _normalize_modalities(args.modalities)
    save_dir = _resolve_save_dir(args.save_dir, modalities)
    save_dir.mkdir(parents=True, exist_ok=True)

    matrices, labels, gene_metadata = _load_training_inputs(args.input_dir, modalities)
    aligned_matrices, y, patient_ids = _aligned_training_set(matrices, labels)
    selected = _select_features_by_modality(aligned_matrices, gene_metadata=gene_metadata, top_k=args.top_k)
    fused = _build_feature_matrix(aligned_matrices, selected=selected, patient_ids=patient_ids)
    X = fused.to_numpy(dtype=np.float32)

    selected_models = [args.model] if args.model != "all" else ["logistic", "lightgbm", "mlp"]
    report: dict[str, Any] = {
        "n_samples": int(len(patient_ids)),
        "n_features": int(X.shape[1]),
        "top_k": int(args.top_k),
        "features": list(fused.columns.astype(str)),
        "models": {},
    }
    if len(modalities) > 1:
        report["modalities"] = modalities
        report["n_features_per_modality"] = {modality: int(len(selected.get(modality, []))) for modality in modalities}

    for model_name in selected_models:
        if model_name == "logistic":
            model = LogisticIDH(random_state=args.seed)
            save_path = save_dir / "logistic.joblib"
        elif model_name == "lightgbm":
            model = LightGBMIDH(random_state=args.seed)
            save_path = save_dir / "lightgbm.txt"
        else:
            model = MLPIDH(input_dim=X.shape[1])
            save_path = save_dir / "mlp.pt"

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

    save_json(report, save_dir / "training_report.json")
    print(f"[INFO] wrote {save_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
