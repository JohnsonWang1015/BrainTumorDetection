from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from idh_glioma.molecular.feature_select import load_idh_prior_panel, select_features
from idh_glioma.molecular.models import LightGBMIDH, LogisticIDH, MLPIDH
from idh_glioma.utils import save_json

MODEL_NAMES = ("logistic", "lightgbm", "mlp")


def _load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    expression = pd.read_parquet(input_dir / "expression_matrix.parquet")
    labels = pd.read_parquet(input_dir / "idh_labels.parquet")
    gene_meta_path = input_dir / "gene_metadata.parquet"
    gene_metadata = pd.read_parquet(gene_meta_path) if gene_meta_path.exists() else None
    labels = labels.copy()
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels = labels.drop_duplicates(subset=["patient_id"], keep="first")
    patient_ids = sorted(set(expression.columns.astype(str)) & set(labels["patient_id"]))
    labels = labels.set_index("patient_id").loc[patient_ids].reset_index()
    expression = expression[patient_ids]
    return expression, labels, gene_metadata


def _new_model(model_name: str, seed: int, input_dim: int | None = None):
    if model_name == "logistic":
        return LogisticIDH(random_state=seed)
    if model_name == "lightgbm":
        return LightGBMIDH(random_state=seed)
    if input_dim is None:
        raise ValueError("input_dim is required for mlp")
    return MLPIDH(input_dim=input_dim)


def _safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    if positives < 5:
        return None
    return float(average_precision_score(y_true, y_prob))


def _recall_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, specificity: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    valid = tpr[spec >= specificity]
    if valid.size == 0:
        return 0.0
    return float(np.max(valid))


def _evaluate_pooled_cv(
    expression: pd.DataFrame,
    labels: pd.DataFrame,
    gene_metadata: pd.DataFrame | None,
    folds: int,
    seed: int,
    top_k: int = 2000,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y = labels["idh_label"].to_numpy(dtype=int)
    patient_ids = labels["patient_id"].astype(str).tolist()
    sources = labels["source"].astype(str).tolist()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prior_panel = load_idh_prior_panel()

    fold_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {"folds": folds, "models": {}}
    for model_name in MODEL_NAMES:
        results["models"][model_name] = {"fold_metrics": []}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        print(f"[INFO] pooled_cv fold {fold_idx + 1}/{folds}", flush=True)
        train_patients = [patient_ids[i] for i in train_idx]
        test_patients = [patient_ids[i] for i in test_idx]
        X_train_genes = expression[train_patients]
        selected_features = select_features(
            X_train_log_tpm=X_train_genes,
            top_k=top_k,
            prior_panel=prior_panel,
            gene_metadata=gene_metadata,
        )
        X_train = X_train_genes.loc[selected_features].T.to_numpy(dtype=np.float32)
        X_test = expression[test_patients].loc[selected_features].T.to_numpy(dtype=np.float32)
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name in MODEL_NAMES:
            print(f"[INFO] pooled_cv fold {fold_idx + 1}/{folds} model={model_name}", flush=True)
            model = _new_model(model_name, seed=seed + fold_idx, input_dim=X_train.shape[1])
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            fold_auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else float("nan")
            fold_auprc = float(average_precision_score(y_test, probs))
            results["models"][model_name]["fold_metrics"].append(
                {
                    "fold": fold_idx,
                    "auc": fold_auc,
                    "auprc": fold_auprc,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_features": int(len(selected_features)),
                }
            )
            for local_idx, patient_id in enumerate(test_patients):
                fold_rows.append(
                    {
                        "patient_id": patient_id,
                        "source": sources[test_idx[local_idx]],
                        "idh_label": int(y_test[local_idx]),
                        "fold": fold_idx,
                        "model": model_name,
                        "proba": float(probs[local_idx]),
                    }
                )

    for model_name in MODEL_NAMES:
        fold_metrics = results["models"][model_name]["fold_metrics"]
        aucs = [m["auc"] for m in fold_metrics if not math.isnan(float(m["auc"]))]
        auprcs = [m["auprc"] for m in fold_metrics]
        results["models"][model_name]["mean_auc"] = float(np.mean(aucs))
        results["models"][model_name]["std_auc"] = float(np.std(aucs))
        results["models"][model_name]["mean_auprc"] = float(np.mean(auprcs))
        results["models"][model_name]["std_auprc"] = float(np.std(auprcs))

    return results, pd.DataFrame(fold_rows)


def _evaluate_source_holdout(
    expression: pd.DataFrame,
    labels: pd.DataFrame,
    gene_metadata: pd.DataFrame | None,
    seed: int,
    top_k: int = 2000,
) -> dict[str, Any]:
    prior_panel = load_idh_prior_panel()
    out: dict[str, Any] = {"directions": {}}
    directions = [("tcga_lgg", "tcga_gbm"), ("tcga_gbm", "tcga_lgg")]

    for train_source, test_source in directions:
        print(f"[INFO] source_holdout train={train_source} test={test_source}", flush=True)
        train_ids = labels.loc[labels["source"] == train_source, "patient_id"].astype(str).tolist()
        test_ids = labels.loc[labels["source"] == test_source, "patient_id"].astype(str).tolist()
        y_train = labels.set_index("patient_id").loc[train_ids, "idh_label"].to_numpy(dtype=int)
        y_test = labels.set_index("patient_id").loc[test_ids, "idh_label"].to_numpy(dtype=int)
        X_train_genes = expression[train_ids]
        selected_features = select_features(
            X_train_log_tpm=X_train_genes,
            top_k=top_k,
            prior_panel=prior_panel,
            gene_metadata=gene_metadata,
        )
        X_train = X_train_genes.loc[selected_features].T.to_numpy(dtype=np.float32)
        X_test = expression[test_ids].loc[selected_features].T.to_numpy(dtype=np.float32)

        key = f"train_{train_source}_test_{test_source}"
        out["directions"][key] = {"n_train": len(train_ids), "n_test": len(test_ids), "models": {}}
        for model_name in MODEL_NAMES:
            print(f"[INFO] source_holdout direction={key} model={model_name}", flush=True)
            model = _new_model(model_name, seed=seed, input_dim=X_train.shape[1])
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            auc_score = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else float("nan")
            auprc_score = _safe_auprc(y_test, probs)
            status = "ok" if auprc_score is not None else "insufficient_minority_samples"
            out["directions"][key]["models"][model_name] = {
                "auc": auc_score,
                "auprc": auprc_score,
                "status": status,
            }
    return out


def _evaluate_minority_metrics(cv_predictions: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"source": "tcga_gbm", "models": {}}
    gbm = cv_predictions[cv_predictions["source"] == "tcga_gbm"].copy()
    for model_name in MODEL_NAMES:
        subset = gbm[gbm["model"] == model_name]
        y_true = subset["idh_label"].to_numpy(dtype=int)
        y_prob = subset["proba"].to_numpy(dtype=float)
        if len(y_true) == 0:
            out["models"][model_name] = {
                "n_samples": 0,
                "auprc": None,
                "recall_at_95_specificity": None,
                "brier_score": None,
            }
            continue
        auprc = float(average_precision_score(y_true, y_prob))
        recall95 = _recall_at_specificity(y_true, y_prob, specificity=0.95)
        brier = float(brier_score_loss(y_true, y_prob))
        out["models"][model_name] = {
            "n_samples": int(len(y_true)),
            "auprc": auprc,
            "recall_at_95_specificity": recall95,
            "brier_score": brier,
        }
    return out


def _plot_figures(cv_predictions: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name in MODEL_NAMES:
        subset = cv_predictions[cv_predictions["model"] == model_name]
        y_true = subset["idh_label"].to_numpy(dtype=int)
        y_prob = subset["proba"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled CV ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name in MODEL_NAMES:
        subset = cv_predictions[cv_predictions["model"] == model_name]
        y_true = subset["idh_label"].to_numpy(dtype=int)
        y_prob = subset["proba"].to_numpy(dtype=float)
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Pooled CV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, model_name in enumerate(MODEL_NAMES):
        subset = cv_predictions[cv_predictions["model"] == model_name]
        y_true = subset["idh_label"].to_numpy(dtype=int)
        y_pred = (subset["proba"].to_numpy(dtype=float) >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=["WT", "Mut"])
        ax.set_yticks([0, 1], labels=["WT", "Mut"])
        ax.set_title(model_name)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion Matrices @ 0.5 (Pooled CV)")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate molecular IDH models with pooled CV and source holdout")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/molecular"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/molecular_idh"))
    parser.add_argument("--mode", choices=["pooled_cv", "source_holdout", "minority_metrics", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/molecular_idh_eval"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expression, labels, gene_metadata = _load_inputs(args.input_dir)

    cv_results: dict[str, Any] | None = None
    cv_predictions: pd.DataFrame | None = None

    if args.mode in {"pooled_cv", "all", "minority_metrics"}:
        cv_results, cv_predictions = _evaluate_pooled_cv(
            expression=expression,
            labels=labels,
            gene_metadata=gene_metadata,
            folds=args.folds,
            seed=args.seed,
        )
        save_json(cv_results, args.output_dir / "pooled_cv_results.json")
        if cv_predictions is not None:
            cv_predictions.to_parquet(args.output_dir / "pooled_cv_predictions.parquet", index=False)
        print(f"[INFO] wrote {args.output_dir / 'pooled_cv_results.json'}")

    if args.mode in {"source_holdout", "all"}:
        source_holdout = _evaluate_source_holdout(
            expression=expression,
            labels=labels,
            gene_metadata=gene_metadata,
            seed=args.seed,
        )
        save_json(source_holdout, args.output_dir / "source_holdout_results.json")
        print(f"[INFO] wrote {args.output_dir / 'source_holdout_results.json'}")

    if args.mode in {"minority_metrics", "all"}:
        if cv_predictions is None:
            cv_predictions = pd.read_parquet(args.output_dir / "pooled_cv_predictions.parquet")
        minority = _evaluate_minority_metrics(cv_predictions)
        save_json(minority, args.output_dir / "minority_metrics.json")
        print(f"[INFO] wrote {args.output_dir / 'minority_metrics.json'}")

    if cv_predictions is not None and args.mode in {"pooled_cv", "all", "minority_metrics"}:
        _plot_figures(cv_predictions, args.output_dir)
        print(f"[INFO] wrote figures under {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()
