from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_idh_prior_panel(panel_path: Path | None = None) -> set[str]:
    if panel_path is None:
        panel_path = Path(__file__).with_name("priors") / "idh_gene_panel.json"
    payload = json.loads(panel_path.read_text(encoding="utf-8"))
    genes = payload.get("genes", [])
    return {str(gene).strip().upper() for gene in genes if str(gene).strip()}


def _map_prior_to_gene_ids(
    prior_panel: set[str],
    expression_index: pd.Index,
    gene_metadata: pd.DataFrame | None = None,
) -> set[str]:
    expression_ids = set(expression_index.astype(str))
    resolved = {gene for gene in prior_panel if gene in expression_ids}
    unresolved = prior_panel - resolved
    if not unresolved or gene_metadata is None or gene_metadata.empty:
        return resolved

    required = {"gene_symbol"}
    if not required.issubset(gene_metadata.columns):
        return resolved

    symbol_map = (
        gene_metadata.reset_index()
        .rename(columns={gene_metadata.index.name or "index": "gene_id"})
        .assign(gene_symbol=lambda df: df["gene_symbol"].astype(str).str.upper())
    )
    for symbol in unresolved:
        matches = symbol_map[symbol_map["gene_symbol"] == symbol]["gene_id"].astype(str).tolist()
        for gene_id in matches:
            if gene_id in expression_ids:
                resolved.add(gene_id)
    return resolved


def select_features(
    X_train_log_tpm: pd.DataFrame,
    top_k: int = 2000,
    prior_panel: set[str] | None = None,
    gene_metadata: pd.DataFrame | None = None,
) -> list[str]:
    if X_train_log_tpm.empty:
        return []
    if top_k <= 0:
        top_k = 0

    variances = X_train_log_tpm.var(axis=1, ddof=0).sort_values(ascending=False)
    top = list(variances.head(top_k).index.astype(str))
    selected = list(top)
    selected_set = set(selected)

    if prior_panel:
        prior_gene_ids = _map_prior_to_gene_ids(prior_panel, X_train_log_tpm.index, gene_metadata=gene_metadata)
        for gene_id in sorted(prior_gene_ids):
            if gene_id not in selected_set:
                selected.append(gene_id)
                selected_set.add(gene_id)
    return selected
