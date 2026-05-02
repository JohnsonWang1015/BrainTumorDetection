from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from idh_glioma.utils import save_json

REQUIRED_MODALITIES = ("flair", "t1", "t1Gd", "t2")
MODALITY_ALIASES = {
    "flair": "flair",
    "t2flair": "flair",
    "t1": "t1",
    "t2": "t2",
    "t1gd": "t1Gd",
    "t1ce": "t1Gd",
    "t1c": "t1Gd",
    "t1c_bias": "t1Gd",
    "t1gad_bias": "t1Gd",
}
MASK_SUFFIXES = (
    "GlistrBoost_ManuallyCorrected",
    "GlistrBoost",
    "tumor_segmentation",
)
SOURCE_DEFAULTS = {
    "brats_tcga_lgg": Path("BraTS-TCGA-LGG") / "Pre-operative_TCGA_LGG_NIfTI_and_Segmentations",
    "tcga_gbm": Path("TCGA-GBM"),
    "ucsf_pdgm": Path("UCSF-PDGM"),
    "egd": Path("EGD"),
}
COHORT_DEFAULTS = {
    "brats_tcga_lgg": "tcga_lgg_only",
    "tcga_gbm": "tcia_preop_pooled",
    "ucsf_pdgm": "ucsf_pdgm_external",
    "egd": "tcia_plus_egd",
}
ACQUISITION_STAGE_DEFAULTS = {
    "brats_tcga_lgg": "preop",
    "tcga_gbm": "preop",
    "ucsf_pdgm": "preop",
    "egd": "preop",
}


def _canonical_suffix(path: Path) -> str | None:
    stem = path.name.replace(".nii.gz", "")
    lowered = stem.lower()
    for alias, canonical in MODALITY_ALIASES.items():
        if lowered == alias:
            return canonical
        if lowered.endswith(f"_{alias}"):
            return canonical
    return None


def classify_modalities(files: list[Path]) -> dict[str, str]:
    modalities: dict[str, str] = {}
    for path in files:
        canonical = _canonical_suffix(path)
        if canonical is None:
            continue
        modalities[canonical] = str(path)

    missing = [name for name in REQUIRED_MODALITIES if name not in modalities]
    if missing:
        raise ValueError(f"Missing required modalities: {missing}")
    return modalities


def _select_mask_path(files: list[Path]) -> str | None:
    for suffix in MASK_SUFFIXES:
        for path in files:
            if path.name.replace(".nii.gz", "").endswith(suffix):
                return str(path)
    return None


def _infer_date(files: list[Path]) -> str:
    for path in files:
        stem = path.name.replace(".nii.gz", "")
        parts = stem.split("_")
        if len(parts) >= 3 and "." in parts[1]:
            return parts[1]
    return "unknown"


def infer_source_root(dataset_root: Path, source_dataset: str) -> Path:
    if source_dataset not in SOURCE_DEFAULTS:
        raise ValueError(f"Unsupported source dataset: {source_dataset}")
    root = dataset_root / SOURCE_DEFAULTS[source_dataset]
    if not root.exists():
        raise FileNotFoundError(f"Unable to locate source root for {source_dataset}: {root}")
    return root


def load_label_map(path: Path | None) -> dict[str, int]:
    if path is None or not path.exists():
        return {}
    read_kwargs: dict[str, Any] = {}
    if path.suffix.lower() in {".tsv", ".txt"}:
        read_kwargs["sep"] = "\t"
    labels_df = pd.read_csv(path, **read_kwargs)
    columns = set(labels_df.columns)
    if {"case_id", "idh_label"}.issubset(columns):
        case_col = "case_id"
        label_col = "idh_label"
        mode = "binary"
    elif {"ID", "IDH status"}.issubset(columns):
        case_col = "ID"
        label_col = "IDH status"
        mode = "ucsf_status"
    elif {"submitter_id", "paper_IDH_status"}.issubset(columns):
        case_col = "submitter_id"
        label_col = "paper_IDH_status"
        mode = "text_status"
    elif {"subject", "IDH mutation status"}.issubset(columns):
        case_col = "subject"
        label_col = "IDH mutation status"
        mode = "binary_with_missing"
    else:
        raise ValueError(
            "IDH label CSV must have either columns: case_id, idh_label "
            "or supported metadata columns such as ID/IDH status, "
            "submitter_id/paper_IDH_status, or subject/IDH mutation status"
        )
    out: dict[str, int] = {}
    for _, row in labels_df.iterrows():
        case_id = str(row[case_col]).strip()
        raw_label = row[label_col]
        if pd.isna(raw_label):
            continue
        if mode == "binary":
            idh_label = int(float(str(raw_label).strip()))
            if idh_label not in (0, 1):
                raise ValueError("IDH labels must be binary values: 0 or 1")
        elif mode == "binary_with_missing":
            idh_label = int(float(str(raw_label).strip()))
            if idh_label == -1:
                continue
            if idh_label not in (0, 1):
                raise ValueError("IDH labels must be binary values: 0 or 1")
        else:
            normalized = str(raw_label).strip().lower()
            if normalized in {
                "mutant",
                "idh-mutant",
                "idh mutant",
                "idh1-mutant",
                "idh2-mutant",
                "idh1 and/or idh2-mutant",
            }:
                idh_label = 1
            elif normalized in {
                "wildtype",
                "wt",
                "idh-wildtype",
                "idh wildtype",
                "idh1-wildtype",
            }:
                idh_label = 0
            else:
                continue
        out[case_id] = idh_label
    return out


def collect_case_dirs(root: Path) -> list[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def build_multisource_manifest(
    *,
    case_dirs: list[Path],
    source_dataset: str,
    cohort_id: str,
    acquisition_stage: str,
    label_map: dict[str, int],
) -> list[dict]:
    records: list[dict] = []
    for case_dir in sorted(case_dirs):
        files = sorted(case_dir.glob("*.nii.gz"))
        if not files:
            continue
        modalities = classify_modalities(files)
        case_id = case_dir.name
        mask_path = _select_mask_path(files)
        idh_label = label_map.get(case_id)
        records.append(
            {
                "record_id": f"{source_dataset}:{case_id}",
                "case_id": case_id,
                "source_dataset": source_dataset,
                "source_subject_id": case_id,
                "cohort_id": cohort_id,
                "acquisition_stage": acquisition_stage,
                "date": _infer_date(files),
                "modalities": modalities,
                "mask_path": mask_path,
                "mask_kind": "segmentation_gt" if mask_path else "none",
                "idh_label": idh_label,
                "label_source": "provided_label_map" if idh_label is not None else "missing",
                "label_confidence": "matched_case_id" if idh_label is not None else "missing",
                "inclusion_flags": {
                    "has_all_required_modalities": True,
                    "has_idh_label": idh_label is not None,
                    "eligible_for_segmentation": mask_path is not None,
                    "eligible_for_3d_idh": idh_label is not None,
                },
                "qc_flags": {},
                "provenance": {
                    "source_case_path": str(case_dir),
                },
            }
        )
    return records


def split_multisource_records(
    records: list[dict[str, Any]],
    *,
    split_mode: str,
    holdout_cohorts: list[str],
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    external = [r for r in records if r.get("cohort_id") in set(holdout_cohorts)]
    pooled = [r for r in records if r.get("cohort_id") not in set(holdout_cohorts)]
    labeled = [r for r in pooled if r.get("idh_label") is not None]
    unlabeled = [r for r in pooled if r.get("idh_label") is None]

    if split_mode == "external_only":
        return {"train": pooled, "val": [], "test": external}

    if not labeled:
        train_val, test = train_test_split(
            pooled,
            test_size=test_ratio,
            random_state=42,
        )
        val_relative = val_ratio / max(1.0 - test_ratio, 1e-9)
        train, val = train_test_split(
            train_val,
            test_size=val_relative,
            random_state=42,
        )
        if split_mode == "source_holdout":
            test = test + external
        return {"train": train, "val": val, "test": test}

    labels = [r.get("idh_label") for r in labeled]
    use_stratify = len(set(labels)) > 1
    y = [r.get("idh_label") for r in labeled] if use_stratify else None
    try:
        train_val, test = train_test_split(
            labeled,
            test_size=test_ratio,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        train_val, test = train_test_split(
            labeled,
            test_size=test_ratio,
            random_state=42,
            stratify=None,
        )
    val_relative = val_ratio / max(1.0 - test_ratio, 1e-9)
    if len(train_val) < 2:
        train = list(train_val)
        val: list[dict[str, Any]] = []
        train.extend(unlabeled)
        if split_mode == "source_holdout":
            test = test + external
        return {"train": train, "val": val, "test": test}
    y_train_val = [r.get("idh_label") for r in train_val] if use_stratify else None
    try:
        train, val = train_test_split(
            train_val,
            test_size=val_relative,
            random_state=42,
            stratify=y_train_val,
        )
    except ValueError:
        train, val = train_test_split(
            train_val,
            test_size=val_relative,
            random_state=42,
            stratify=None,
        )
    train.extend(unlabeled)
    if split_mode == "source_holdout":
        test = test + external
    return {"train": train, "val": val, "test": test}


def build_payload(
    *,
    records: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    split_mode: str,
) -> dict[str, Any]:
    cohorts = sorted(
        [
            {
                "cohort_id": cohort_id,
                "source_dataset": source_dataset,
                "acquisition_stage": acquisition_stage,
            }
            for cohort_id, source_dataset, acquisition_stage in {
                (
                    r["cohort_id"],
                    r["source_dataset"],
                    r["acquisition_stage"],
                )
                for r in records
            }
        ],
        key=lambda item: (item["cohort_id"], item["source_dataset"]),
    )
    return {
        "manifest_version": "0.2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "idh_multisource",
        "cohorts": cohorts,
        "split_strategy": {
            "scheme": split_mode,
            "random_state": 42,
        },
        "splits": splits,
    }


def write_multisource_manifest(
    *,
    records: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    split_mode: str,
    output_path: Path,
) -> None:
    payload = build_payload(records=records, splits=splits, split_mode=split_mode)
    save_json(payload, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a multi-source IDH manifest")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--include-sources",
        nargs="+",
        default=["brats_tcga_lgg", "tcga_gbm"],
        choices=sorted(SOURCE_DEFAULTS),
    )
    parser.add_argument(
        "--idh-labels",
        type=Path,
        default=Path("artifacts/idh_labels.csv"),
        help="CSV with columns: case_id, idh_label",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest_v2.json"))
    parser.add_argument(
        "--split-mode",
        choices=["pooled_train_val_test", "source_holdout", "external_only"],
        default="source_holdout",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_map = load_label_map(args.idh_labels)
    all_records: list[dict[str, Any]] = []
    missing_sources: list[str] = []
    for source_dataset in args.include_sources:
        try:
            root = infer_source_root(args.dataset_root, source_dataset)
        except FileNotFoundError:
            missing_sources.append(source_dataset)
            continue
        case_dirs = collect_case_dirs(root)
        records = build_multisource_manifest(
            case_dirs=case_dirs,
            source_dataset=source_dataset,
            cohort_id=COHORT_DEFAULTS[source_dataset],
            acquisition_stage=ACQUISITION_STAGE_DEFAULTS[source_dataset],
            label_map=label_map,
        )
        all_records.extend(records)

    if not all_records:
        raise SystemExit("No source records found. Check dataset roots and selected sources.")

    splits = split_multisource_records(
        all_records,
        split_mode=args.split_mode,
        holdout_cohorts=["ucsf_pdgm_external"],
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    write_multisource_manifest(
        records=all_records,
        splits=splits,
        split_mode=args.split_mode,
        output_path=args.output,
    )
    print(f"Saved multi-source manifest to {args.output}")
    print({name: len(records) for name, records in splits.items()})
    if missing_sources:
        print(f"Skipped missing sources: {missing_sources}")


if __name__ == "__main__":
    main()
