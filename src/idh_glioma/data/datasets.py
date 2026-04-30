from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
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


def _tumor_bbox_2d(
    mask_slice: np.ndarray, margin: int = 10
) -> tuple[int, int, int, int] | None:
    """Tight bbox around foreground in a 2-D mask, expanded by ``margin`` and clipped.

    Returns ``(y0, y1, x0, x1)`` such that ``mask_slice[y0:y1, x0:x1]`` covers the
    tumor. Returns ``None`` if the slice has no foreground.
    """
    ys, xs = np.where(mask_slice > 0)
    if ys.size == 0:
        return None
    h, w = mask_slice.shape
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + 1 + margin, h)
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + 1 + margin, w)
    return y0, y1, x0, x1


def _crop_roi(
    image: np.ndarray, bbox: tuple[int, int, int, int] | None
) -> np.ndarray:
    """Crop a (C, H, W) array to ``bbox`` (or return unchanged when bbox is None)."""
    if bbox is None:
        return image
    y0, y1, x0, x1 = bbox
    return image[:, y0:y1, x0:x1]


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
        img_size: int = 224,
        augment: bool | None = None,
        use_roi: bool = True,
        roi_margin: int = 10,
    ) -> None:
        super().__init__()
        manifest = load_json(manifest_path)
        records = [r for r in manifest[split] if r["idh_label"] is not None]
        self.records = records
        self.num_slices_per_case = num_slices_per_case
        self.img_size = img_size
        self.augment = augment if augment is not None else (split == "train")
        self.use_roi = use_roi
        self.roi_margin = roi_margin

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

        if self.use_roi:
            bbox = _tumor_bbox_2d(mask[:, :, z_idx], self.roi_margin)
            if bbox is not None:
                image = _crop_roi(image, bbox)
            else:
                # No tumor on this slice (rare — slice picker prefers tumor slices).
                # Fall back to a centre crop matching the median tumor size so we
                # don't feed a totally different scale than ROI-cropped slices.
                h, w = image.shape[-2], image.shape[-1]
                side = min(h, w) // 2
                y0 = (h - side) // 2
                x0 = (w - side) // 2
                image = image[:, y0 : y0 + side, x0 : x0 + side]

        if self.augment:
            image = _augment_cls(image)
        label = np.array([record["idh_label"]], dtype=np.float32)
        image_t = _resize_slice(torch.from_numpy(image), self.img_size)
        return image_t, torch.from_numpy(label)


def make_balanced_sampler(
    records: list[dict], num_slices_per_case: int
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that balances IDH=0 vs IDH=1 across an epoch.

    Per-record weight is ``0.5 / count_of_that_class`` so that, in expectation,
    each minibatch is class-balanced even when the underlying cohort is skewed.
    The per-record weight is repeated ``num_slices_per_case`` times so every
    slice index in ``BraTSSliceClassificationDataset`` carries a weight.
    """
    labels = [int(r["idh_label"]) for r in records]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        # Degenerate single-class split — fall back to uniform weights.
        record_weights = [1.0] * len(records)
    else:
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        record_weights = [w_pos if y == 1 else w_neg for y in labels]

    sample_weights: list[float] = []
    for w in record_weights:
        sample_weights.extend([w] * num_slices_per_case)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


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
