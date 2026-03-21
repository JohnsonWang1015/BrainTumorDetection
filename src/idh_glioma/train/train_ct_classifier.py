"""Train MobileNetV3-Small for binary CT/MRI brain-tumor classification.

Reads the JSON manifest produced by ``prepare-ct`` and trains a
MobileNetV3-Small classifier to predict Tumor (1) vs Healthy (0).

Quick start::

    uv run prepare-ct
    uv run train-ct

With GPU profile::

    uv run train-ct --profile a6000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from idh_glioma.data.ct_datasets import BrainImageDataset
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: AdamW | None,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    ctx = torch.inference_mode() if not is_train else torch.enable_grad()
    with ctx:
        for images, labels in tqdm(loader, desc="train" if is_train else "val", leave=False):
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            if is_train:
                if scaler is None:
                    raise RuntimeError("GradScaler must be provided during training")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += float(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds == labels).sum().item())
            total += labels.numel()

    avg_loss = running_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3 binary classifier for CT/MRI brain-tumor detection"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/ct_manifest.json")
    )
    parser.add_argument(
        "--modality",
        choices=["ct", "mri", "both"],
        default="both",
        help="Filter dataset by modality (default: both)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/mobilenetv3_ct_best.pt")
    )
    return parser.parse_args()


def apply_profile(args: argparse.Namespace, device: torch.device) -> argparse.Namespace:
    if args.profile != "a6000" or device.type != "cuda":
        return args
    if args.batch_size == 32:
        args.batch_size = 64
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

    train_ds = BrainImageDataset(
        args.manifest, split="train", modality=args.modality, img_size=args.img_size
    )
    val_ds = BrainImageDataset(
        args.manifest, split="val", modality=args.modality, img_size=args.img_size, augment=False
    )
    print(f"Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    use_workers = args.num_workers > 0
    common: dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if use_workers:
        common["persistent_workers"] = True
        common["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    model = build_mobilenetv3_binary(num_input_channels=3).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile disabled: {exc}")

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, device, amp_enabled=use_amp, scaler=scaler
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, None, device, amp_enabled=use_amp, scaler=None
        )
        print(
            f"[Epoch {epoch + 1:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                    "modality": args.modality,
                },
                args.output,
            )
            print(f"  ✓ Saved best checkpoint → {args.output}")


if __name__ == "__main__":
    main()
