"""Build a train/val/test manifest for the Kaggle CT & MRI brain-tumor dataset.

Folder layout expected (Kaggle multimodal dataset):
    <dataset_root>/
        Brain Tumor CT scan Images/
            Tumor/     ← label 1
            Healthy/   ← label 0
        Brain Tumor MRI images/
            Tumor/     ← label 1
            Healthy/   ← label 0

Output manifest format (matches existing BraTS manifest style):
    {
      "train": [{"path": "...", "label": 0|1, "modality": "ct|mri"}, ...],
      "val":   [...],
      "test":  [...]
    }
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CT_SUBDIR = "Brain Tumor CT scan Images"
MRI_SUBDIR = "Brain Tumor MRI images"
LABEL_DIRS = {"Tumor": 1, "Healthy": 0}


@dataclass
class ImageRecord:
    path: str
    label: int
    modality: str  # "ct" | "mri"


def _collect_images(base: Path, modality: str) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for class_name, label in LABEL_DIRS.items():
        class_dir = base / class_name
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                records.append(ImageRecord(path=str(f), label=label, modality=modality))
    return records


def build_manifest(
    dataset_root: Path,
    modality: str,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    if modality in ("ct", "both"):
        ct_dir = dataset_root / CT_SUBDIR
        if not ct_dir.is_dir():
            raise FileNotFoundError(f"CT directory not found: {ct_dir}")
        records.extend(_collect_images(ct_dir, modality="ct"))

    if modality in ("mri", "both"):
        mri_dir = dataset_root / MRI_SUBDIR
        if not mri_dir.is_dir():
            raise FileNotFoundError(f"MRI directory not found: {mri_dir}")
        records.extend(_collect_images(mri_dir, modality="mri"))

    if not records:
        raise ValueError(f"No images found under {dataset_root} for modality={modality!r}")

    return records


def stratified_split(
    records: list[ImageRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    rng = random.Random(seed)

    # Group by (label, modality) to preserve class balance per modality
    groups: dict[tuple[int, str], list[ImageRecord]] = {}
    for r in records:
        key = (r.label, r.modality)
        groups.setdefault(key, []).append(r)

    train_all: list[ImageRecord] = []
    val_all: list[ImageRecord] = []
    test_all: list[ImageRecord] = []

    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            # too few samples — put everything in train
            train_all.extend(group)
            continue
        train_all.extend(group[:n_train])
        val_all.extend(group[n_train : n_train + n_val])
        test_all.extend(group[n_train + n_val :])

    rng.shuffle(train_all)
    return {"train": train_all, "val": val_all, "test": test_all}


def write_manifest(splits: dict[str, list[ImageRecord]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {split: [asdict(r) for r in recs] for split, recs in splits.items()}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CT/MRI brain-tumor manifest (Kaggle multimodal dataset)"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/Kaggle_multimodal/Dataset"),
        help="Root directory containing 'Brain Tumor CT scan Images' and/or 'Brain Tumor MRI images'",
    )
    parser.add_argument(
        "--modality",
        choices=["ct", "mri", "both"],
        default="both",
        help="Which modality to include (default: both)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ct_manifest.json"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_manifest(args.dataset_root, args.modality)
    splits = stratified_split(records, args.val_ratio, args.test_ratio, args.seed)
    write_manifest(splits, args.output)

    counts = {split: len(recs) for split, recs in splits.items()}
    print(f"Saved manifest → {args.output}")
    print(f"Split counts: {counts}")
    total = sum(counts.values())
    tumor = sum(r.label for split in splits.values() for r in split)
    print(f"Total images: {total}  (tumor={tumor}, healthy={total - tumor})")


if __name__ == "__main__":
    main()
