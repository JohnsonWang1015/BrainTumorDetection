from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Sampler
import torch.nn.functional as F
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
    # .copy() ensures the array owns its memory (no mmap) so it is safe to
    # use across forked DataLoader worker processes.
    return img.get_fdata().astype(np.float32).copy()


def _resize_slice(t: torch.Tensor, size: int, is_mask: bool = False) -> torch.Tensor:
    """Resize a (C, H, W) tensor to (C, size, size).

    Uses nearest-neighbor for masks to preserve binary values,
    bilinear for images.
    """
    if t.shape[-1] == size and t.shape[-2] == size:
        return t
    mode = "nearest" if is_mask else "bilinear"
    kwargs = {} if is_mask else {"align_corners": False}
    return F.interpolate(t.unsqueeze(0), size=(size, size), mode=mode, **kwargs).squeeze(0)


def _augment_seg(
    image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random spatial augmentations jointly to image (C,H,W) and mask (1,H,W)."""
    # Random horizontal flip
    if random.random() > 0.5:
        image = np.ascontiguousarray(image[:, :, ::-1])
        mask = np.ascontiguousarray(mask[:, :, ::-1])
    # Random vertical flip
    if random.random() > 0.5:
        image = np.ascontiguousarray(image[:, ::-1, :])
        mask = np.ascontiguousarray(mask[:, ::-1, :])
    # Random 90-degree rotation
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()
    # Random intensity jitter (image only)
    if random.random() > 0.5:
        for c in range(image.shape[0]):
            image[c] = image[c] * (1.0 + random.uniform(-0.1, 0.1))
            image[c] = image[c] + random.uniform(-0.1, 0.1)
    return image, mask


def _augment_cls(image: np.ndarray) -> np.ndarray:
    """Random spatial + intensity augmentations for classification (image only, C,H,W)."""
    if random.random() > 0.5:
        image = np.ascontiguousarray(image[:, :, ::-1])
    if random.random() > 0.5:
        image = np.ascontiguousarray(image[:, ::-1, :])
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k, axes=(1, 2)).copy()
    if random.random() > 0.5:
        for c in range(image.shape[0]):
            image[c] = image[c] * (1.0 + random.uniform(-0.1, 0.1))
            image[c] = image[c] + random.uniform(-0.1, 0.1)
    return image


class BraTSSliceSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        num_slices_per_case: int = 12,
        img_size: int = 240,
        augment: bool | None = None,
    ) -> None:
        super().__init__()
        self.num_slices_per_case = num_slices_per_case
        self.img_size = img_size
        manifest = load_json(manifest_path)
        self.records = manifest[split]
        self.augment = augment if augment is not None else (split == "train")

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

        if self.augment:
            image, target = _augment_seg(image, target)

        image_t = _resize_slice(torch.from_numpy(image), self.img_size)
        target_t = _resize_slice(torch.from_numpy(target), self.img_size, is_mask=True)
        return image_t, target_t


class BraTSSliceClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        num_slices_per_case: int = 12,
        img_size: int = 240,
        augment: bool | None = None,
    ) -> None:
        super().__init__()
        manifest = load_json(manifest_path)
        records = [r for r in manifest[split] if r["idh_label"] is not None]
        self.records = records
        self.num_slices_per_case = num_slices_per_case
        self.img_size = img_size
        self.augment = augment if augment is not None else (split == "train")

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
        if self.augment:
            image = _augment_cls(image)
        label = np.array([record["idh_label"]], dtype=np.float32)
        image_t = _resize_slice(torch.from_numpy(image), self.img_size)
        return image_t, torch.from_numpy(label)


class CaseLevelSampler(Sampler[int]):
    """Shuffles at the *case* level so that all slices of a case are yielded
    consecutively.  This maximises ``load_nifti`` LRU-cache hit rate: each
    NIfTI volume is loaded once and reused for all ``num_slices_per_case``
    slices before the next case is accessed.
    """

    def __init__(
        self,
        num_cases: int,
        num_slices_per_case: int,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        self.num_cases = num_cases
        self.num_slices_per_case = num_slices_per_case
        self.shuffle = shuffle
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_cases * self.num_slices_per_case

    def __iter__(self):  # type: ignore[override]
        case_order = list(range(self.num_cases))
        if self.shuffle:
            self._rng.shuffle(case_order)
        for case_idx in case_order:
            base = case_idx * self.num_slices_per_case
            slice_indices = list(range(base, base + self.num_slices_per_case))
            if self.shuffle:
                self._rng.shuffle(slice_indices)
            yield from slice_indices
