"""PyTorch Dataset for the Kaggle CT/MRI brain-tumor classification dataset.

Reads the JSON manifest produced by ``prepare_ct_data.py``.
Each record has keys: ``path``, ``label`` (0=Healthy, 1=Tumor), ``modality``.

Usage::

    from idh_glioma.data.ct_datasets import BrainImageDataset

    train_ds = BrainImageDataset("artifacts/ct_manifest.json", split="train")
    val_ds   = BrainImageDataset("artifacts/ct_manifest.json", split="val", augment=False)
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from idh_glioma.utils import load_json

# ImageNet statistics — reasonable for pretrained torchvision models
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transforms(img_size: int, augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


class BrainImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Binary classification dataset for CT/MRI brain-tumor images.

    Args:
        manifest_path: Path to JSON manifest (from ``prepare-ct`` command).
        split: One of ``"train"``, ``"val"``, ``"test"``.
        modality: Filter to ``"ct"``, ``"mri"``, or ``"both"`` (default).
        img_size: Spatial size to resize images to (default 224).
        augment: Apply random augmentations. Defaults to ``True`` for train
                 split and ``False`` otherwise when not explicitly set.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        split: str = "train",
        modality: str = "both",
        img_size: int = 224,
        augment: bool | None = None,
    ) -> None:
        super().__init__()
        manifest = load_json(Path(manifest_path))
        if split not in manifest:
            raise KeyError(f"Split {split!r} not found in manifest. Available: {list(manifest)}")

        records = manifest[split]
        if modality != "both":
            records = [r for r in records if r["modality"] == modality]

        self.records = records
        self._augment = augment if augment is not None else (split == "train")
        self._tf = _build_transforms(img_size, self._augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]
        img = Image.open(record["path"]).convert("RGB")
        tensor = self._tf(img)
        label = torch.tensor([float(record["label"])], dtype=torch.float32)
        return tensor, label
