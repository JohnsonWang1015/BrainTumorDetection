from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from idh_glioma.utils import load_json

EPS = 1e-6


def zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    return (x - mean) / (std + EPS)


@lru_cache(maxsize=64)
def load_nifti(path: str | Path) -> np.ndarray:
    img = cast(nib.Nifti1Image, nib.load(str(path)))
    return img.get_fdata().astype(np.float32)


class BraTSSliceSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, manifest_path: Path, split: str = "train", num_slices_per_case: int = 12
    ) -> None:
        super().__init__()
        self.num_slices_per_case = num_slices_per_case
        manifest = load_json(manifest_path)
        self.records = manifest[split]

    def __len__(self) -> int:
        return len(self.records) * self.num_slices_per_case

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx // self.num_slices_per_case]
        flair = zscore(load_nifti(record["modalities"]["flair"]))
        t1 = zscore(load_nifti(record["modalities"]["t1"]))
        t1gd = zscore(load_nifti(record["modalities"]["t1Gd"]))
        t2 = zscore(load_nifti(record["modalities"]["t2"]))
        mask = (load_nifti(record["mask_path"]) > 0).astype(np.float32)

        z_max = mask.shape[-1] - 1
        tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
        if len(tumor_slices) > 0:
            z_idx = int(random.choice(tumor_slices))
        else:
            z_idx = random.randint(0, z_max)

        image = np.stack(
            [flair[:, :, z_idx], t1[:, :, z_idx], t1gd[:, :, z_idx], t2[:, :, z_idx]],
            axis=0,
        )
        target = mask[:, :, z_idx][None, ...]
        return torch.from_numpy(image), torch.from_numpy(target)


class BraTSSliceClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, manifest_path: Path, split: str = "train", num_slices_per_case: int = 12
    ) -> None:
        super().__init__()
        manifest = load_json(manifest_path)
        records = [r for r in manifest[split] if r["idh_label"] is not None]
        self.records = records
        self.num_slices_per_case = num_slices_per_case

    def __len__(self) -> int:
        return len(self.records) * self.num_slices_per_case

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx // self.num_slices_per_case]
        flair = zscore(load_nifti(record["modalities"]["flair"]))
        t1gd = zscore(load_nifti(record["modalities"]["t1Gd"]))
        t2 = zscore(load_nifti(record["modalities"]["t2"]))
        mask = (load_nifti(record["mask_path"]) > 0).astype(np.float32)

        z_max = mask.shape[-1] - 1
        tumor_slices = np.where(mask.sum(axis=(0, 1)) > 0)[0]
        if len(tumor_slices) > 0:
            z_idx = int(random.choice(tumor_slices))
        else:
            z_idx = random.randint(0, z_max)

        image = np.stack(
            [flair[:, :, z_idx], t1gd[:, :, z_idx], t2[:, :, z_idx]], axis=0
        )
        label = np.array([record["idh_label"]], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(label)
