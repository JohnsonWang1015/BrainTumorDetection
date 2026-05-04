from __future__ import annotations

import pandas as pd

from idh_glioma.molecular.feature_select import select_features_modality


def concat_modalities(
    matrices: dict[str, pd.DataFrame],
    patient_ids: list[str],
) -> pd.DataFrame:
    ordered_patients = list(dict.fromkeys(str(patient_id) for patient_id in patient_ids))
    if not matrices:
        return pd.DataFrame(index=ordered_patients)

    common = set(ordered_patients)
    for matrix in matrices.values():
        common &= set(matrix.columns.astype(str))
    strict_patients = [patient_id for patient_id in ordered_patients if patient_id in common]

    fused_blocks: list[pd.DataFrame] = []
    for modality, matrix in matrices.items():
        if not strict_patients:
            block = pd.DataFrame(index=[], columns=[f"{modality}:{f}" for f in matrix.index.astype(str)])
            fused_blocks.append(block)
            continue
        modality_matrix = matrix[strict_patients].copy()
        modality_matrix.index = modality_matrix.index.astype(str)
        block = modality_matrix.T
        block.columns = [f"{modality}:{feature_id}" for feature_id in block.columns.astype(str)]
        fused_blocks.append(block)

    if not fused_blocks:
        return pd.DataFrame(index=strict_patients)
    fused = pd.concat(fused_blocks, axis=1)
    fused = fused.reindex(index=strict_patients)
    return fused


def select_multimodal_features(
    matrices: dict[str, pd.DataFrame],
    prior_panels: dict[str, set[str]],
    top_k_per_modality: dict[str, int],
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for modality, matrix in matrices.items():
        top_k = int(top_k_per_modality.get(modality, 0))
        prior_ids = prior_panels.get(modality, set())
        selected[modality] = select_features_modality(
            X_train=matrix,
            top_k=top_k,
            prior_ids=prior_ids,
        )
    return selected
