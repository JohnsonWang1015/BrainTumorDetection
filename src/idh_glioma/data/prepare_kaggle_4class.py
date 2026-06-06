"""Build a train/val/test manifest for the Kaggle 4-class brain-tumor dataset.

Source: Kaggle "Brain Tumor MRI Dataset" (masoudnickparvar) — 7,023 MRI slices
across four classes. Already on disk at ``datasets/Kaggle/`` but was not wired
into the pipeline. Kaggle ships a pre-split ``Training/`` (5,600) and
``Testing/`` (1,600) layout, so we preserve ``Testing/`` as the held-out test
split and only split ``Training/`` into train + val.

Folder layout expected::

    <dataset_root>/
        Training/{glioma,meningioma,notumor,pituitary}/*.jpg
        Testing/ {glioma,meningioma,notumor,pituitary}/*.jpg

Output manifest schema (mirrors the project convention)::

    {
      "classes": ["glioma", "meningioma", "notumor", "pituitary"],
      "train": [{"path": "...", "label": 0..3, "class_name": "..."}, ...],
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

CLASS_NAMES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_LABEL: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class ImageRecord:
    path: str
    label: int
    class_name: str


def _collect_split_dir(split_dir: Path) -> list[ImageRecord]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    records: list[ImageRecord] = []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        label = CLASS_TO_LABEL[class_name]
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                records.append(
                    ImageRecord(path=str(f), label=label, class_name=class_name)
                )
    return records


def _stratified_train_val_split(
    records: list[ImageRecord],
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    groups: dict[int, list[ImageRecord]] = {}
    for r in records:
        groups.setdefault(r.label, []).append(r)

    train_all: list[ImageRecord] = []
    val_all: list[ImageRecord] = []
    for group in groups.values():
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_val
        train_all.extend(group[:n_train])
        val_all.extend(group[n_train:])

    rng.shuffle(train_all)
    return train_all, val_all


def build_manifest(
    dataset_root: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ImageRecord]]:
    train_full = _collect_split_dir(dataset_root / "Training")
    test_records = _collect_split_dir(dataset_root / "Testing")

    if not train_full or not test_records:
        raise ValueError(f"No images discovered under {dataset_root}")

    train_records, val_records = _stratified_train_val_split(
        train_full, val_ratio=val_ratio, seed=seed
    )
    return {"train": train_records, "val": val_records, "test": test_records}


def write_manifest(splits: dict[str, list[ImageRecord]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"classes": list(CLASS_NAMES)}
    payload.update({split: [asdict(r) for r in recs] for split, recs in splits.items()})
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _summarize(splits: dict[str, list[ImageRecord]]) -> None:
    print(f"Saved manifest with classes: {list(CLASS_NAMES)}")
    for split, recs in splits.items():
        by_class: dict[str, int] = {c: 0 for c in CLASS_NAMES}
        for r in recs:
            by_class[r.class_name] += 1
        per_class_str = ", ".join(f"{c}={n}" for c, n in by_class.items())
        print(f"  {split}: total={len(recs)}  ({per_class_str})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Kaggle 4-class brain-tumor manifest"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/Kaggle"),
        help="Root directory containing 'Training/' and 'Testing/' subfolders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kaggle4_manifest.json"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = build_manifest(args.dataset_root, val_ratio=args.val_ratio, seed=args.seed)
    write_manifest(splits, args.output)
    print(f"Saved manifest → {args.output}")
    _summarize(splits)


if __name__ == "__main__":
    main()
