from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from idh_glioma.data.datasets import BraTSSliceSegmentationDataset, CaseLevelSampler, load_nifti
from idh_glioma.models.unet2d import UNet2D


def _worker_init(worker_id: int) -> None:
    """Clear the NIfTI LRU cache so each DataLoader worker builds its own."""
    load_nifti.cache_clear()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * target).sum(dim=(1, 2, 3)) + 1e-6
    den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return 1.0 - (num / den).mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for handling class imbalance (tumor pixels << background)."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.sigmoid(logits) * target + (1 - torch.sigmoid(logits)) * (1 - target)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: AdamW,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    running = 0.0
    non_blocking = device.type == "cuda"
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=non_blocking)
        masks = masks.to(device, non_blocking=non_blocking)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = focal_loss(logits, masks) + dice_loss(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item())
    return running / max(len(loader), 1)


@torch.no_grad()
def validation_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    """Returns (loss, dice_score)."""
    model.train(False)
    running_loss = 0.0
    running_dice = 0.0
    n_batches = 0
    non_blocking = device.type == "cuda"
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=non_blocking)
        masks = masks.to(device, non_blocking=non_blocking)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = focal_loss(logits, masks) + dice_loss(logits, masks)

        # Compute dice metric for monitoring
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        intersection = (preds * masks).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)
        running_dice += dice.mean().item()
        running_loss += loss.item()
        n_batches += 1
    n = max(n_batches, 1)
    return running_loss / n, running_dice / n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for glioma segmentation")
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/manifest.json")
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profile", choices=["default", "a6000"], default="default")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/unet2d_best.pt")
    )
    return parser.parse_args()


def apply_profile(args: argparse.Namespace, device: torch.device) -> argparse.Namespace:
    if args.profile != "a6000" or device.type != "cuda":
        return args

    if args.batch_size == 8:
        args.batch_size = 16
    if args.num_workers == min(8, os.cpu_count() or 1):
        args.num_workers = min(12, os.cpu_count() or 1)
    if args.prefetch_factor == 2:
        args.prefetch_factor = 4
    args.amp = True
    args.cudnn_benchmark = True
    args.tf32 = True
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = apply_profile(args, device)
    use_amp = args.amp and device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    train_ds = BraTSSliceSegmentationDataset(args.manifest, split="train")
    val_ds = BraTSSliceSegmentationDataset(args.manifest, split="val")
    use_workers = args.num_workers > 0

    train_sampler = CaseLevelSampler(
        num_cases=len(train_ds.records),
        num_slices_per_case=train_ds.num_slices_per_case,
        shuffle=True,
    )
    val_sampler = CaseLevelSampler(
        num_cases=len(val_ds.records),
        num_slices_per_case=val_ds.num_slices_per_case,
        shuffle=False,
    )

    common: dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if use_workers:
        common["persistent_workers"] = True
        common["prefetch_factor"] = args.prefetch_factor
        common["worker_init_fn"] = _worker_init

    train_loader = DataLoader(train_ds, sampler=train_sampler, **common)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **common)

    model = UNet2D(dropout=args.dropout).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile disabled: {exc}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Cosine annealing with linear warmup
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_dice = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp_enabled=use_amp,
            scaler=scaler,
        )
        val_loss, val_dice = validation_epoch(
            model,
            val_loader,
            device,
            amp_enabled=use_amp,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} lr={current_lr:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                    "epoch": epoch + 1,
                },
                args.output,
            )
            print(f"Saved best checkpoint (dice={val_dice:.4f}): {args.output}")

    print(f"\nTraining complete. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
