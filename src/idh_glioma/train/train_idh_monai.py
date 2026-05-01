"""3D IDH classifier with MONAI on the TCGA-LGG cohort.

Crops the 4-channel multimodal volume to a 3D tumor ROI (using the ground
truth mask), resizes to 96x96x96, runs a 3D DenseNet121, and outputs binary
IDH probability. Uses the same balanced-sampling / sqrt-pos-weight /
smoothed-AUC checkpointing recipe as the 2D version, plus 3D-specific
augmentation (flip on each axis).

Run::

    uv run train-idh-monai \
        --manifest artifacts/manifest.json \
        --output checkpoints/densenet3d_idh.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from idh_glioma.utils import load_json


def _zscore(x):
    m, s = x.mean(), x.std()
    return (x - m) / (s + 1e-6)


def _tumor_bbox_3d(mask, margin=4):
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        return None
    mn = nz.min(axis=0)
    mx = nz.max(axis=0) + 1
    h, w, d = mask.shape
    y0 = max(int(mn[0]) - margin, 0)
    y1 = min(int(mx[0]) + margin, h)
    x0 = max(int(mn[1]) - margin, 0)
    x1 = min(int(mx[1]) + margin, w)
    z0 = max(int(mn[2]) - margin, 0)
    z1 = min(int(mx[2]) + margin, d)
    return y0, y1, x0, x1, z0, z1


def _expand_bbox_uniformly(
    bbox: tuple[int, int, int, int, int, int],
    shape: tuple[int, int, int],
    extra: int,
) -> tuple[int, int, int, int, int, int]:
    if extra <= 0:
        return bbox
    y0, y1, x0, x1, z0, z1 = bbox
    h, w, d = shape
    return (
        max(y0 - extra, 0),
        min(y1 + extra, h),
        max(x0 - extra, 0),
        min(x1 + extra, w),
        max(z0 - extra, 0),
        min(z1 + extra, d),
    )


class TumorVolume3DDataset(Dataset):
    def __init__(
        self, records, project_root, target_size=(96, 96, 96), augment=False, margin=4,
        jitter_expand_max=0, jitter_shift_max=0,
        context_view_prob=0.0, context_extra_max=0,
    ):
        self.records = [r for r in records if r.get("idh_label") is not None]
        self.root = project_root
        self.target_size = target_size
        self.augment = augment
        self.margin = margin
        self.jitter_expand_max = jitter_expand_max
        self.jitter_shift_max = jitter_shift_max
        self.context_view_prob = context_view_prob
        self.context_extra_max = context_extra_max

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        mods = r["modalities"]
        flair = nib.load(str(self.root / mods["flair"])).get_fdata().astype(np.float32)
        t1 = nib.load(str(self.root / mods["t1"])).get_fdata().astype(np.float32)
        t1gd = nib.load(str(self.root / mods["t1Gd"])).get_fdata().astype(np.float32)
        t2 = nib.load(str(self.root / mods["t2"])).get_fdata().astype(np.float32)
        mask = (nib.load(str(self.root / r["mask_path"])).get_fdata() > 0).astype(np.float32)

        bbox = _tumor_bbox_3d(mask, margin=self.margin)
        if bbox is None:
            # Fall back to a centre crop if mask is empty (rare).
            h, w, d = mask.shape
            side = min(h, w, d) // 2
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            z0 = (d - side) // 2
            bbox = (y0, y0 + side, x0, x0 + side, z0, z0 + side)
        y0, y1, x0, x1, z0, z1 = bbox

        if self.augment and (self.jitter_expand_max > 0 or self.jitter_shift_max > 0):
            # Simulate the looser/shifted bboxes that come from a noisy
            # predicted mask at inference time. Independent per-side expansion
            # plus a shared bbox shift.
            h, w, d = flair.shape
            ex = self.jitter_expand_max
            if ex > 0:
                y0 = max(y0 - np.random.randint(0, ex + 1), 0)
                y1 = min(y1 + np.random.randint(0, ex + 1), h)
                x0 = max(x0 - np.random.randint(0, ex + 1), 0)
                x1 = min(x1 + np.random.randint(0, ex + 1), w)
                z0 = max(z0 - np.random.randint(0, ex + 1), 0)
                z1 = min(z1 + np.random.randint(0, ex + 1), d)
            sh = self.jitter_shift_max
            if sh > 0:
                sy = np.random.randint(-sh, sh + 1)
                sx = np.random.randint(-sh, sh + 1)
                sz = np.random.randint(-sh, sh + 1)
                y0 = max(y0 + sy, 0); y1 = min(y1 + sy, h)
                x0 = max(x0 + sx, 0); x1 = min(x1 + sx, w)
                z0 = max(z0 + sz, 0); z1 = min(z1 + sz, d)
            # Guard against collapsed bbox after shift
            if y1 <= y0 or x1 <= x0 or z1 <= z0:
                y0, y1, x0, x1, z0, z1 = bbox
            if self.context_view_prob > 0 and np.random.rand() < self.context_view_prob:
                extra = np.random.randint(1, self.context_extra_max + 1) if self.context_extra_max > 0 else 0
                y0, y1, x0, x1, z0, z1 = _expand_bbox_uniformly(
                    (y0, y1, x0, x1, z0, z1),
                    flair.shape,
                    extra,
                )

        flair = _zscore(flair[y0:y1, x0:x1, z0:z1])
        t1 = _zscore(t1[y0:y1, x0:x1, z0:z1])
        t1gd = _zscore(t1gd[y0:y1, x0:x1, z0:z1])
        t2 = _zscore(t2[y0:y1, x0:x1, z0:z1])
        vol = np.stack([flair, t1, t1gd, t2], axis=0)  # (4, h, w, d)

        if self.augment:
            for axis in (1, 2, 3):
                if np.random.rand() > 0.5:
                    vol = np.flip(vol, axis=axis).copy()
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                vol = np.rot90(vol, k, axes=(1, 2)).copy()

        t = torch.from_numpy(vol).unsqueeze(0)  # (1, 4, h, w, d)
        t = F.interpolate(t, size=self.target_size, mode="trilinear", align_corners=False).squeeze(0)
        return t, torch.tensor([float(r["idh_label"])], dtype=torch.float32)


def _make_balanced_sampler(records):
    labels = [int(r["idh_label"]) for r in records if r.get("idh_label") is not None]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    w_pos = 0.5 / n_pos
    w_neg = 0.5 / n_neg
    weights = [w_pos if y == 1 else w_neg for y in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--target-size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--margin", type=int, default=4)
    p.add_argument("--jitter-expand-max", type=int, default=12,
                   help="Max random per-side expansion of training bbox (simulates pred-mask looseness).")
    p.add_argument("--jitter-shift-max", type=int, default=6,
                   help="Max random shift of training bbox (simulates pred-mask offset).")
    p.add_argument("--context-view-prob", type=float, default=0.35,
                   help="Probability of adding an extra uniform context expansion around the jittered bbox.")
    p.add_argument("--context-extra-max", type=int, default=6,
                   help="Max extra voxels per side for the optional context-heavy crop.")
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("checkpoints/densenet3d_idh.pt"))
    return p.parse_args()


def _run_epoch(model, loader, optimizer, device, pos_weight, training):
    model.train(training)
    running = 0.0
    all_probs = []
    all_labels = []
    for x, y in tqdm(loader, desc="train" if training else "val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last_3d)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            probs = torch.sigmoid(logits.float()).detach().cpu().reshape(-1).tolist()
            all_probs.extend(probs)
            all_labels.extend(y.detach().cpu().reshape(-1).int().tolist())
        running += float(loss.item())
    return running / max(len(loader), 1), all_probs, all_labels


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    project_root = Path.cwd()
    manifest = load_json(args.manifest)
    train_records = [r for r in manifest["train"] if r.get("idh_label") is not None]
    val_records = [r for r in manifest["val"] if r.get("idh_label") is not None]
    print(f"train cases: {len(train_records)} | val cases: {len(val_records)}")

    train_lbls = [int(r["idh_label"]) for r in train_records]
    n_pos = sum(train_lbls)
    n_neg = len(train_lbls) - n_pos
    raw_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight_value = math.sqrt(raw_ratio)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"Train labels: WT={n_neg}, Mutant={n_pos}, pos_weight=sqrt({raw_ratio:.3f})={pos_weight_value:.4f}")

    train_ds = TumorVolume3DDataset(
        train_records, project_root, target_size=tuple(args.target_size), augment=True,
        margin=args.margin, jitter_expand_max=args.jitter_expand_max, jitter_shift_max=args.jitter_shift_max,
        context_view_prob=args.context_view_prob, context_extra_max=args.context_extra_max,
    )
    val_ds = TumorVolume3DDataset(
        val_records, project_root, target_size=tuple(args.target_size), augment=False, margin=args.margin
    )
    sampler = _make_balanced_sampler(train_records)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = DenseNet121(spatial_dims=3, in_channels=4, out_channels=1, dropout_prob=0.2).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = args.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_smoothed = -1.0
    auc_history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, _, _ = _run_epoch(model, train_loader, optimizer, device, pos_weight, training=True)
        val_loss, val_probs, val_labels = _run_epoch(model, val_loader, optimizer, device, pos_weight, training=False)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if len(set(val_labels)) >= 2:
            val_auc = float(roc_auc_score(val_labels, val_probs))
        else:
            val_auc = float("nan")
        if not math.isnan(val_auc):
            auc_history.append(val_auc)
        smoothed = (
            float(np.mean(auc_history[-args.smooth_window:]))
            if auc_history
            else float("nan")
        )

        print(
            f"[Epoch {epoch + 1:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
            f"smoothed={smoothed:.4f} lr={current_lr:.6f}"
        )

        if not math.isnan(smoothed) and smoothed > best_smoothed:
            best_smoothed = smoothed
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "smoothed_auc": smoothed,
                    "epoch": epoch + 1,
                    "arch": "densenet121_3d",
                    "target_size": list(args.target_size),
                    "margin": args.margin,
                    "jitter_expand_max": args.jitter_expand_max,
                    "jitter_shift_max": args.jitter_shift_max,
                    "context_view_prob": args.context_view_prob,
                    "context_extra_max": args.context_extra_max,
                    "in_channels": 4,
                },
                args.output,
            )
            print(f"Saved best (smoothed={smoothed:.4f}, auc={val_auc:.4f}): {args.output}")

    print(f"\nTraining complete. Best smoothed val AUC: {best_smoothed:.4f}")


if __name__ == "__main__":
    main()
