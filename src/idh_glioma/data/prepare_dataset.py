from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

MODALITIES = ("flair", "t1", "t1Gd", "t2")
MASK_PRIORITY = ("GlistrBoost_ManuallyCorrected", "GlistrBoost")


@dataclass
class CaseRecord:
    case_id: str
    date: str
    modalities: dict[str, str]
    mask_path: str
    idh_label: int | None


def parse_case_folder(case_dir: Path) -> CaseRecord | None:
    nii_files = sorted(case_dir.glob("*.nii.gz"))
    if not nii_files:
        return None

    modality_map: dict[str, str] = {}
    mask_path = ""
    case_id = case_dir.name
    date = "unknown"

    for f in nii_files:
        stem = f.name.replace(".nii.gz", "")
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        case_id = parts[0]
        date = parts[1]
        suffix = parts[-1]
        if suffix in MODALITIES:
            modality_map[suffix] = str(f)
        for mname in MASK_PRIORITY:
            if stem.endswith(mname):
                if not mask_path or mname == MASK_PRIORITY[0]:
                    mask_path = str(f)

    if any(k not in modality_map for k in MODALITIES):
        return None
    if not mask_path:
        return None

    return CaseRecord(
        case_id=case_id,
        date=date,
        modalities=modality_map,
        mask_path=mask_path,
        idh_label=None,
    )


def build_manifest(brats_root: Path, idh_labels_csv: Path | None) -> list[CaseRecord]:
    case_dirs = [p for p in brats_root.glob("TCGA-*") if p.is_dir()]
    records: list[CaseRecord] = []
    label_map: dict[str, int] = {}

    if idh_labels_csv and idh_labels_csv.exists():
        labels_df = pd.read_csv(idh_labels_csv)
        required = {"case_id", "idh_label"}
        if not required.issubset(set(labels_df.columns)):
            raise ValueError("IDH label CSV must have columns: case_id, idh_label")
        for _, row in labels_df.iterrows():
            case_id = str(row["case_id"])
            idh_label = int(str(row["idh_label"]))
            label_map[case_id] = idh_label

    for case_dir in sorted(case_dirs):
        rec = parse_case_folder(case_dir)
        if rec is None:
            continue
        if rec.case_id in label_map:
            rec.idh_label = label_map[rec.case_id]
        records.append(rec)

    return records


def stratified_split(
    records: list[CaseRecord], val_ratio: float, test_ratio: float
) -> dict[str, list[CaseRecord]]:
    if not records:
        raise ValueError("No valid cases found for split.")

    idh_available = [r for r in records if r.idh_label is not None]
    idh_missing = [r for r in records if r.idh_label is None]

    if idh_available and len({r.idh_label for r in idh_available}) > 1:
        y = [r.idh_label for r in idh_available]
        train_val, test = train_test_split(
            idh_available,
            test_size=test_ratio,
            random_state=42,
            stratify=y,
        )
        y_train_val = [r.idh_label for r in train_val]
        val_relative = val_ratio / (1.0 - test_ratio)
        train, val = train_test_split(
            train_val,
            test_size=val_relative,
            random_state=42,
            stratify=y_train_val,
        )
    else:
        train_val, test = train_test_split(
            idh_available, test_size=test_ratio, random_state=42
        )
        val_relative = val_ratio / (1.0 - test_ratio)
        train, val = train_test_split(
            train_val, test_size=val_relative, random_state=42
        )

    train.extend(idh_missing)
    return {"train": train, "val": val, "test": test}


def write_manifest(splits: dict[str, list[CaseRecord]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[dict[str, object]]] = {}
    for split_name, records in splits.items():
        payload[split_name] = [
            {
                "case_id": r.case_id,
                "date": r.date,
                "modalities": r.modalities,
                "mask_path": r.mask_path,
                "idh_label": r.idh_label,
            }
            for r in records
        ]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BraTS manifest for segmentation + IDH tasks"
    )
    parser.add_argument(
        "--brats-root",
        type=Path,
        default=Path(
            "datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations"
        ),
    )
    parser.add_argument(
        "--idh-labels",
        type=Path,
        default=None,
        help="CSV with columns: case_id, idh_label",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest.json"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_manifest(args.brats_root, args.idh_labels)
    splits = stratified_split(records, args.val_ratio, args.test_ratio)
    write_manifest(splits, args.output)
    print(f"Saved manifest to {args.output}")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
