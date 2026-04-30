from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from idh_glioma.data.datasets import (
    BraTSSliceClassificationDataset,
    make_balanced_sampler,
)
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: AdamW | None,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
    pos_weight: torch.Tensor | None = None,
) -> tuple[float, list[float], list[int]]:
    """Returns (mean_loss, all_probs, all_labels)."""
    is_train = optimizer is not None
    model.train(is_train)
    running = 0.0
    non_blocking = device.type == "cuda"
    all_probs: list[float] = []
    all_labels: list[int] = []

    for images, labels in tqdm(
        loader, desc="train" if is_train else "val", leave=False
    ):
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )

        if is_train:
            if scaler is None:
                raise RuntimeError("GradScaler must be provided during training")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            probs = torch.sigmoid(logits.float()).detach().cpu().reshape(-1).tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.detach().cpu().reshape(-1).int().tolist())

        running += float(loss.item())

    return running / max(len(loader), 1), all_probs, all_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MobileNetV3 for IDH mutation classification"
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/manifest.json")
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument(
        "--variant",
        choices=["small", "large"],
        default="large",
        help="MobileNetV3 backbone size. 'large' is preferred for IDH on the small "
        "TCGA-LGG cohort because the bigger pretrained stem transfers better.",
    )
    parser.add_argument(
        "--use-roi", action=argparse.BooleanOptionalAction, default=True,
        help="Crop each slice to the tumor ROI before classification.",
    )
    parser.add_argument("--roi-margin", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--smooth-window", type=int, default=3,
        help="Save the checkpoint when the moving-average val AUC over this many "
        "epochs improves (mitigates noise on a tiny val split).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/mobilenetv3_idh_best.pt")
    )
    return parser.parse_args()


def apply_profile(args: argparse.Namespace, device: torch.device) -> argparse.Namespace:
    if args.profile != "a6000" or device.type != "cuda":
        return args

    if args.batch_size == 16:
        args.batch_size = 32
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

    train_ds = BraTSSliceClassificationDataset(
        args.manifest,
        split="train",
        img_size=args.img_size,
        use_roi=args.use_roi,
        roi_margin=args.roi_margin,
    )
    val_ds = BraTSSliceClassificationDataset(
        args.manifest,
        split="val",
        img_size=args.img_size,
        use_roi=args.use_roi,
        roi_margin=args.roi_margin,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "No IDH labels found. Please provide --idh-labels CSV during dataset preparation."
        )

    # pos_weight = sqrt(num_negative / num_positive) -- a softer rebalancing
    # than the literal ratio, which over-pushes when imbalance is heavy.
    train_labels = [int(r["idh_label"]) for r in train_ds.records]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    raw_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
    pos_weight_value = math.sqrt(raw_ratio)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(
        f"Train labels: WT={n_neg}, Mutant={n_pos}, "
        f"pos_weight=sqrt({raw_ratio:.3f})={pos_weight_value:.4f}"
    )

    use_workers = args.num_workers > 0
    train_sampler = make_balanced_sampler(
        train_ds.records, train_ds.num_slices_per_case
    )
    train_loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "sampler": train_sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    val_loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if use_workers:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    model = build_mobilenetv3_binary(variant=args.variant).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile disabled: {exc}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_smoothed = -1.0
    auc_history: list[float] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp_enabled=use_amp,
            scaler=scaler,
            pos_weight=pos_weight,
        )
        val_loss, val_probs, val_labels = run_epoch(
            model,
            val_loader,
            None,
            device,
            amp_enabled=use_amp,
            scaler=None,
            pos_weight=pos_weight,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if len(set(val_labels)) >= 2:
            val_auc = float(roc_auc_score(val_labels, val_probs))
        else:
            val_auc = float("nan")

        if not math.isnan(val_auc):
            auc_history.append(val_auc)
        smoothed = (
            float(np.mean(auc_history[-args.smooth_window :]))
            if auc_history
            else float("nan")
        )

        print(
            f"[Epoch {epoch + 1:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
            f"smoothed_auc={smoothed:.4f} lr={current_lr:.6f}"
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
                    "variant": args.variant,
                    "use_roi": args.use_roi,
                    "roi_margin": args.roi_margin,
                    "img_size": args.img_size,
                },
                args.output,
            )
            print(
                f"Saved best checkpoint (smoothed_auc={smoothed:.4f}, "
                f"auc={val_auc:.4f}): {args.output}"
            )

    print(f"\nTraining complete. Best smoothed val AUC: {best_smoothed:.4f}")


if __name__ == "__main__":
    main()
