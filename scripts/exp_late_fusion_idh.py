"""Late-fusion experiment for the multi-omics molecular IDH cohort.

Hypothesis: early concat of RNA-seq and methylation features lets the dominant
modality (methylation, with bounded [0,1] values and ~2000 features) underweight
RNA-seq evidence, which is the GBM minority signal.

This script trains per-modality logistic classifiers, then fuses their
probabilities with (a) arithmetic mean and (b) a small stacking logistic
regressor whose only inputs are the two per-modality probabilities.

Reports pooled 5-fold CV AUC and AUPRC plus GBM-minority AUPRC, recall@95spec,
and Brier, side-by-side with the existing concat numbers from
artifacts/molecular_idh_multimodal_eval/ and artifacts/molecular_idh_eval/.

Usage:
    uv run python scripts/exp_late_fusion_idh.py \\
        --input-dir artifacts/molecular_multimodal \\
        --output artifacts/molecular_idh_multimodal_eval/late_fusion_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from idh_glioma.molecular.feature_select import (
    _map_prior_to_gene_ids,
    load_idh_cpg_panel,
    load_idh_prior_panel,
    select_features_modality,
)
from idh_glioma.molecular.models import LightGBMIDH, LogisticIDH, MLPIDH

BASE_LEARNERS = ("logistic", "lightgbm", "mlp")


def _new_base(name: str, seed: int, input_dim: int):
    if name == "logistic":
        return LogisticIDH(random_state=seed)
    if name == "lightgbm":
        return LightGBMIDH(random_state=seed)
    return MLPIDH(input_dim=input_dim)


def _recall_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, specificity: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    valid = tpr[spec >= specificity]
    if valid.size == 0:
        return 0.0
    return float(np.max(valid))


def _load_inputs(input_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame | None]:
    labels = pd.read_parquet(input_dir / "idh_labels.parquet")
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels = labels.drop_duplicates(subset=["patient_id"], keep="first")

    rna = pd.read_parquet(input_dir / "expression_matrix.parquet")
    meth = pd.read_parquet(input_dir / "methylation_matrix.parquet")
    matrices = {"rnaseq": rna, "methylation": meth}

    common = set(labels["patient_id"])
    for matrix in matrices.values():
        common &= set(matrix.columns.astype(str))
    pids = sorted(common)
    labels = labels.set_index("patient_id").loc[pids].reset_index()
    matrices = {k: v[pids] for k, v in matrices.items()}

    gene_meta_path = input_dir / "gene_metadata.parquet"
    gene_metadata = pd.read_parquet(gene_meta_path) if gene_meta_path.exists() else None
    return matrices, labels, gene_metadata


def _select_per_modality(
    train_matrices: dict[str, pd.DataFrame],
    gene_metadata: pd.DataFrame | None,
    top_k: int,
) -> dict[str, list[str]]:
    rnaseq_prior = load_idh_prior_panel()
    rnaseq_prior_ids = _map_prior_to_gene_ids(
        rnaseq_prior, train_matrices["rnaseq"].index, gene_metadata=gene_metadata
    )
    cpg_prior = load_idh_cpg_panel()
    return {
        "rnaseq": select_features_modality(
            X_train=train_matrices["rnaseq"], top_k=top_k, prior_ids=rnaseq_prior_ids
        ),
        "methylation": select_features_modality(
            X_train=train_matrices["methylation"], top_k=top_k, prior_ids=cpg_prior
        ),
    }


def _fit_per_modality(
    X_train: dict[str, np.ndarray],
    y_train: np.ndarray,
    seed: int,
    base_learner: str,
):
    out: dict[str, Any] = {}
    for modality, X in X_train.items():
        model = _new_base(base_learner, seed=seed, input_dim=X.shape[1])
        model.fit(X, y_train)
        out[modality] = model
    return out


def _per_modality_probs(
    models: dict[str, Any],
    X_test: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {modality: models[modality].predict_proba(X_test[modality])[:, 1] for modality in models}


def _summarize(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "recall_at_95_specificity": _recall_at_specificity(y_true, y_prob, 0.95),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("artifacts/molecular_multimodal"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/molecular_idh_multimodal_eval/late_fusion_results.json"),
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    matrices, labels, gene_metadata = _load_inputs(args.input_dir)

    y = labels["idh_label"].to_numpy(dtype=int)
    sources = labels["source"].astype(str).to_numpy()
    pids = labels["patient_id"].astype(str).tolist()

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        print(f"[fold {fold_idx + 1}/{args.folds}]", flush=True)
        train_pids = [pids[i] for i in train_idx]
        test_pids = [pids[i] for i in test_idx]
        train_matrices = {k: v[train_pids] for k, v in matrices.items()}
        test_matrices = {k: v[test_pids] for k, v in matrices.items()}

        selected = _select_per_modality(train_matrices, gene_metadata, top_k=args.top_k)

        X_train = {
            modality: train_matrices[modality].loc[selected[modality]].T.to_numpy(dtype=np.float32)
            for modality in selected
        }
        X_test = {
            modality: test_matrices[modality].loc[selected[modality]].T.to_numpy(dtype=np.float32)
            for modality in selected
        }
        y_train = y[train_idx]
        y_test = y[test_idx]
        sources_test = sources[test_idx]

        for base in BASE_LEARNERS:
            per_modality_models = _fit_per_modality(X_train, y_train, seed=args.seed + fold_idx, base_learner=base)
            per_modality_train_probs = _per_modality_probs(per_modality_models, X_train)
            per_modality_test_probs = _per_modality_probs(per_modality_models, X_test)

            stack_X_train = np.column_stack([per_modality_train_probs[m] for m in ("rnaseq", "methylation")])
            stack_X_test = np.column_stack([per_modality_test_probs[m] for m in ("rnaseq", "methylation")])
            stack = LogisticIDH(random_state=args.seed + fold_idx)
            stack.fit(stack_X_train, y_train)
            stack_test_prob = stack.predict_proba(stack_X_test)[:, 1]

            mean_test_prob = np.mean(stack_X_test, axis=1)

            entries = (
                (f"{base}__rnaseq_only", per_modality_test_probs["rnaseq"]),
                (f"{base}__methylation_only", per_modality_test_probs["methylation"]),
                (f"{base}__late_mean", mean_test_prob),
                (f"{base}__late_stack_logistic", stack_test_prob),
            )
            for tag, prob in entries:
                for j, pid in enumerate(test_pids):
                    rows.append(
                        {
                            "fold": fold_idx,
                            "model": tag,
                            "patient_id": pid,
                            "source": sources_test[j],
                            "y_true": int(y_test[j]),
                            "y_prob": float(prob[j]),
                        }
                    )

    df = pd.DataFrame(rows)

    summary: dict[str, Any] = {
        "config": {
            "input_dir": str(args.input_dir),
            "folds": int(args.folds),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "n_patients": int(len(y)),
            "label_distribution": {"wildtype": int((y == 0).sum()), "mutant": int((y == 1).sum())},
            "source_distribution": {src: int((sources == src).sum()) for src in sorted(set(sources))},
        },
        "models": {},
    }

    for tag, sub in df.groupby("model"):
        per_fold = []
        for fold_idx, fold_sub in sub.groupby("fold"):
            per_fold.append(_summarize(fold_sub["y_true"].to_numpy(), fold_sub["y_prob"].to_numpy()))
        aucs = np.array([m["auc"] for m in per_fold])
        auprcs = np.array([m["auprc"] for m in per_fold])
        # GBM minority
        gbm = sub[sub["source"] == "tcga_gbm"]
        gbm_metrics = _summarize(gbm["y_true"].to_numpy(), gbm["y_prob"].to_numpy()) if len(gbm) else {}
        summary["models"][tag] = {
            "pooled_cv": {
                "fold_auc": [float(x) for x in aucs.tolist()],
                "mean_auc": float(np.nanmean(aucs)),
                "std_auc": float(np.nanstd(aucs)),
                "mean_auprc": float(np.mean(auprcs)),
                "std_auprc": float(np.std(auprcs)),
            },
            "gbm_minority": gbm_metrics,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    df.to_parquet(args.output.with_name(args.output.stem + "_predictions.parquet"), index=False)

    print()
    print("=" * 72)
    print(f"  late-fusion experiment, n={len(y)} patients")
    print("=" * 72)
    print(f"{'model':<22} {'CV AUC':<14} {'CV AUPRC':<14} {'GBM AUPRC':<10} {'GBM Brier':<10} {'GBM rec95':<10}")
    for tag, info in summary["models"].items():
        cv = info["pooled_cv"]
        gbm = info.get("gbm_minority", {})
        print(
            f"{tag:<22} {cv['mean_auc']:.4f}±{cv['std_auc']:.4f}   "
            f"{cv['mean_auprc']:.4f}±{cv['std_auprc']:.4f}   "
            f"{gbm.get('auprc', float('nan')):.4f}     {gbm.get('brier', float('nan')):.4f}     "
            f"{gbm.get('recall_at_95_specificity', float('nan')):.4f}"
        )
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
