from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from idh_glioma.molecular.feature_select import load_idh_prior_panel
from idh_glioma.molecular.gdc_download import download_lgg_dataset
from idh_glioma.molecular.maf_parser import aggregate_idh_labels
from idh_glioma.molecular.rnaseq_loader import ExpressionBuildInfo, build_expression_matrix
from idh_glioma.utils import load_json, save_json

SOURCE_CHOICES = ("tcga_gbm", "tcga_lgg")


def _load_file_to_patient_map(data_root: Path) -> dict[str, str]:
    mapping_path = data_root.parent / "file_to_patient.json"
    payload = load_json(mapping_path)
    return {str(key): str(value) for key, value in payload.items()}


def _build_source_bundle(
    *,
    source: str,
    data_root: Path,
    file_to_patient: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ExpressionBuildInfo]:
    labels = aggregate_idh_labels(data_root / "maf", source=source)
    matrix, gene_metadata, build_info = build_expression_matrix(
        rnaseq_dir=data_root / "rnaseq_counts",
        file_to_patient=file_to_patient,
    )
    return labels, matrix, gene_metadata, build_info


def _merge_labels(labels_by_source: list[pd.DataFrame]) -> pd.DataFrame:
    if not labels_by_source:
        return pd.DataFrame(
            columns=["patient_id", "source", "idh_label", "idh_gene", "idh_mutation_aa", "label_source"]
        )
    merged = pd.concat(labels_by_source, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["patient_id"], keep="first")
    merged = merged.sort_values(by=["source", "patient_id"]).reset_index(drop=True)
    return merged


def _build_manifest(
    *,
    include_sources: list[str],
    labels_by_source: dict[str, pd.DataFrame],
    matrices_by_source: dict[str, pd.DataFrame],
    build_info_by_source: dict[str, ExpressionBuildInfo],
    merged_labels: pd.DataFrame,
    merged_matrix: pd.DataFrame,
) -> dict[str, Any]:
    manifest_sources: dict[str, Any] = {}
    manifest_patient_rows: list[dict[str, Any]] = []

    for source in include_sources:
        labels = labels_by_source[source]
        matrix = matrices_by_source[source]
        info = build_info_by_source[source]
        labeled_patient_ids = set(labels["patient_id"].astype(str))
        expression_patient_ids = set(matrix.columns.astype(str))
        overlap = sorted(labeled_patient_ids & expression_patient_ids)
        manifest_sources[source] = {
            "labeled_patients": int(len(labeled_patient_ids)),
            "expression_patients": int(len(expression_patient_ids)),
            "labeled_with_expression": int(len(overlap)),
            "unlabeled_expression_patients": int(len(expression_patient_ids - labeled_patient_ids)),
            "label_distribution": {
                "wildtype": int((labels["idh_label"] == 0).sum()),
                "mutant": int((labels["idh_label"] == 1).sum()),
            },
            "missing_mapping_file_uuids": info.missing_mapping_file_uuids,
            "multiple_primary_aliquots": int(sum(info.multiple_aliquots_by_patient.values())),
        }
        for patient_id, selected_uuid in sorted(info.selected_file_uuid_by_patient.items()):
            manifest_patient_rows.append(
                {
                    "patient_id": patient_id,
                    "source": source,
                    "selected_file_uuid": selected_uuid,
                    "multiple_primary_aliquots": bool(info.multiple_aliquots_by_patient.get(patient_id, False)),
                    "missing_tpm_ratio": float(info.missing_ratio_by_patient.get(patient_id, 0.0)),
                }
            )

    pooled_labels = merged_labels["idh_label"] if not merged_labels.empty else pd.Series(dtype=int)
    return {
        "sources": manifest_sources,
        "pooled": {
            "labeled_patients": int(len(merged_labels)),
            "expression_patients": int(merged_matrix.shape[1]),
            "labeled_with_expression": int(len(set(merged_labels["patient_id"]) & set(merged_matrix.columns))),
            "label_distribution": {
                "wildtype": int((pooled_labels == 0).sum()),
                "mutant": int((pooled_labels == 1).sum()),
            },
            "num_genes": int(merged_matrix.shape[0]),
        },
        "patients": manifest_patient_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TCGA molecular cohort for IDH training")
    parser.add_argument("--include-sources", nargs="+", default=list(SOURCE_CHOICES), choices=SOURCE_CHOICES)
    parser.add_argument(
        "--gbm-data-root",
        type=Path,
        default=Path("datasets/TCGA-GBM/tcga_gbm_downloads/data"),
    )
    parser.add_argument(
        "--lgg-data-root",
        type=Path,
        default=Path("datasets/TCGA-LGG-Molecular/tcga_lgg_downloads/data"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/molecular"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--gdc-token", type=Path, default=None)
    parser.add_argument(
        "--gdc-client-path",
        type=Path,
        default=Path("/mnt/8tb_hdd2/johnson/tools/gdc-client/gdc-client"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_sources = list(dict.fromkeys(args.include_sources))
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if ("tcga_lgg" in include_sources) and (not args.skip_download):
        summary = download_lgg_dataset(
            base_dir=args.lgg_data_root.parent,
            token_path=args.gdc_token,
            gdc_client_path=args.gdc_client_path,
        )
        print(f"[INFO] TCGA-LGG download summary: {summary}")

    source_roots: dict[str, Path] = {
        "tcga_gbm": args.gbm_data_root,
        "tcga_lgg": args.lgg_data_root,
    }

    labels_by_source: dict[str, pd.DataFrame] = {}
    matrices_by_source: dict[str, pd.DataFrame] = {}
    gene_metadata_by_source: dict[str, pd.DataFrame] = {}
    build_info_by_source: dict[str, ExpressionBuildInfo] = {}

    for source in include_sources:
        data_root = source_roots[source]
        file_to_patient = _load_file_to_patient_map(data_root)
        labels, matrix, gene_metadata, build_info = _build_source_bundle(
            source=source,
            data_root=data_root,
            file_to_patient=file_to_patient,
        )
        labels_by_source[source] = labels
        matrices_by_source[source] = matrix
        gene_metadata_by_source[source] = gene_metadata
        build_info_by_source[source] = build_info
        print(
            f"[INFO] {source}: labels={len(labels)} expression_patients={matrix.shape[1]} "
            f"genes={matrix.shape[0]} multiple_aliquots={sum(build_info.multiple_aliquots_by_patient.values())}"
        )

    merged_labels = _merge_labels([labels_by_source[source] for source in include_sources])
    merged_matrix = pd.concat([matrices_by_source[source] for source in include_sources], axis=1).fillna(0.0)
    merged_gene_metadata = gene_metadata_by_source[include_sources[0]]
    for source in include_sources[1:]:
        merged_gene_metadata = merged_gene_metadata.combine_first(gene_metadata_by_source[source])
    merged_gene_metadata = merged_gene_metadata.reindex(merged_matrix.index)

    manifest = _build_manifest(
        include_sources=include_sources,
        labels_by_source=labels_by_source,
        matrices_by_source=matrices_by_source,
        build_info_by_source=build_info_by_source,
        merged_labels=merged_labels,
        merged_matrix=merged_matrix,
    )
    prior_panel = load_idh_prior_panel()
    feature_panel_payload = {
        "strategy": "variance_top_k_union_prior_panel",
        "default_top_k": 2000,
        "prior_gene_symbols": sorted(prior_panel),
        "prior_gene_count": len(prior_panel),
    }

    merged_matrix.to_parquet(output_dir / "expression_matrix.parquet")
    merged_labels.to_parquet(output_dir / "idh_labels.parquet", index=False)
    merged_gene_metadata.to_parquet(output_dir / "gene_metadata.parquet")
    save_json(manifest, output_dir / "cohort_manifest.json")
    save_json(feature_panel_payload, output_dir / "feature_panel.json")

    labeled_with_expression = len(set(merged_labels["patient_id"]) & set(merged_matrix.columns))
    print(f"[INFO] wrote {(output_dir / 'expression_matrix.parquet')}")
    print(f"[INFO] wrote {(output_dir / 'idh_labels.parquet')}")
    print(f"[INFO] wrote {(output_dir / 'cohort_manifest.json')}")
    print(f"[INFO] wrote {(output_dir / 'feature_panel.json')}")
    print(
        f"[INFO] pooled: labels={len(merged_labels)} expression_patients={merged_matrix.shape[1]} "
        f"labeled_with_expression={labeled_with_expression} genes={merged_matrix.shape[0]}"
    )


if __name__ == "__main__":
    main()
