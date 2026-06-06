"""Train MobileNetV3 for binary CT/MRI brain-tumor classification.

Reads the JSON manifest produced by ``prepare-ct`` and trains a MobileNetV3
classifier to predict Tumor (1) vs Healthy (0).

Quick start::

    uv run prepare-ct
    uv run train-ct

With GPU profile::

    uv run train-ct --profile a6000

Recipe notes
------------
The default recipe favours discrimination quality over raw loss:

- **Backbone**: MobileNetV3-Large (richer ImageNet stem transfers better than
  Small; ``--variant small`` keeps the legacy architecture).
- **Schedule**: cosine annealing with linear warmup, mirroring the seg/IDH
  trainers (constant LR under-anneals).
- **EMA**: an exponential-moving-average shadow of the weights is evaluated and
  saved — it generalises better than the raw step weights at no extra cost.
- **Label smoothing**: soft BCE targets curb overconfidence and improve AUC /
  calibration.
- **Checkpoint selection**: best validation **AUC** (direct discrimination
  metric), not val_loss. The checkpoint stores ``variant`` so eval/app load the
  matching architecture automatically.
"""

from __future__ import annotations

import argparse
import copy
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

from idh_glioma.data.ct_datasets import BrainImageDataset
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary


class EMA:
    """Exponential moving average of model weights.

    Tracks every floating-point tensor in ``state_dict`` (parameters *and* BN
    running stats); integer buffers (e.g. ``num_batches_tracked``) are copied
    verbatim. ``copy_to`` swaps the averaged weights into a model for eval/save.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(v)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def _smooth_targets(labels: torch.Tensor, eps: float) -> torch.Tensor:
    """Symmetric label smoothing for binary targets: 1→1-eps/2, 0→eps/2."""
    if eps <= 0.0:
        return labels
    return labels * (1.0 - eps) + 0.5 * eps


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: AdamW | None,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
    *,
    label_smoothing: float = 0.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: EMA | None = None,
) -> tuple[float, float, float]:
    """Returns (avg_loss, accuracy, auc)."""
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    correct = 0
    total = 0
    probs_all: list[float] = []
    labels_all: list[float] = []
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
                target = _smooth_targets(labels, label_smoothing) if is_train else labels
                loss = F.binary_cross_entropy_with_logits(logits, target)

            if is_train:
                if scaler is None:
                    raise RuntimeError("GradScaler must be provided during training")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

            running_loss += float(loss.item())
            sig = torch.sigmoid(logits)
            preds = (sig >= 0.5).float()
            correct += int((preds == labels).sum().item())
            total += labels.numel()
            probs_all.extend(sig.detach().float().cpu().numpy().reshape(-1).tolist())
            labels_all.extend(labels.detach().float().cpu().numpy().reshape(-1).tolist())

    avg_loss = running_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    # AUC needs both classes present in the split.
    auc = (
        float(roc_auc_score(labels_all, probs_all))
        if len(set(labels_all)) > 1
        else float("nan")
    )
    return avg_loss, acc, auc


@torch.inference_mode()
def evaluate_ema(
    ema: EMA,
    template: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    """Evaluate the EMA weights (acc, auc) on ``loader`` without disturbing the
    live training model. ``template`` is a throwaway clone."""
    ema.copy_to(template)
    template.eval()
    _, acc, auc = run_epoch(
        template, loader, None, device, amp_enabled=amp_enabled, scaler=None
    )
    return acc, auc


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
    parser.add_argument(
        "--variant",
        choices=["small", "large"],
        default="large",
        help="MobileNetV3 backbone (default: large; 'small' is the legacy arch)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay; set <=0 to disable EMA and select on raw weights",
    )
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


def _build_scheduler(
    optimizer: AdamW, warmup_epochs: int, total_epochs: int, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Per-step cosine schedule with linear warmup."""
    warmup_steps = max(warmup_epochs * steps_per_epoch, 1)
    total_steps = max(total_epochs * steps_per_epoch, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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

    model = build_mobilenetv3_binary(num_input_channels=3, variant=args.variant).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception as exc:
            print(f"torch.compile disabled: {exc}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(
        optimizer, args.warmup_epochs, args.epochs, len(train_loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    use_ema = args.ema_decay > 0.0
    ema = EMA(model, args.ema_decay) if use_ema else None
    # Throwaway clone used to evaluate EMA weights without touching the live model.
    ema_template = copy.deepcopy(model) if use_ema else None

    best_val_auc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Recipe: variant={args.variant} epochs={args.epochs} warmup={args.warmup_epochs} "
        f"lr={args.lr} wd={args.weight_decay} ls={args.label_smoothing} "
        f"ema={'off' if not use_ema else args.ema_decay} | select on val AUC"
    )

    for epoch in range(args.epochs):
        train_loss, train_acc, train_auc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp_enabled=use_amp,
            scaler=scaler,
            label_smoothing=args.label_smoothing,
            scheduler=scheduler,
            ema=ema,
        )
        # Validate raw weights, and EMA weights if enabled. EMA lags the raw
        # model early (its time constant spans many steps), so the raw weights
        # win in early epochs and EMA takes over once it has caught up — keep
        # whichever discriminates better this epoch.
        _, raw_acc, raw_auc = run_epoch(
            model, val_loader, None, device, amp_enabled=use_amp, scaler=None
        )
        candidates: list[tuple[str, float, float, torch.nn.Module]] = [
            ("raw", raw_auc, raw_acc, model)
        ]
        if use_ema and ema is not None and ema_template is not None:
            ema_acc, ema_auc = evaluate_ema(
                ema, ema_template, val_loader, device, amp_enabled=use_amp
            )
            candidates.append(("ema", ema_auc, ema_acc, ema_template))

        tag, val_auc, val_acc, src_model = max(
            candidates, key=lambda c: (-1.0 if math.isnan(c[1]) else c[1])
        )
        cur_lr = optimizer.param_groups[0]["lr"]
        extra = (
            f" (raw={raw_auc:.4f} ema={candidates[-1][1]:.4f})"
            if len(candidates) > 1
            else ""
        )
        print(
            f"[Epoch {epoch + 1:03d}] lr={cur_lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_auc={train_auc:.4f} | "
            f"val_acc={val_acc:.4f} val_auc={val_auc:.4f} [{tag}]{extra}"
        )

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            state_dict = copy.deepcopy(src_model.state_dict())
            torch.save(
                {
                    "model": state_dict,
                    "variant": args.variant,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                    "modality": args.modality,
                    "ema": tag == "ema",
                },
                args.output,
            )
            print(f"  ✓ Saved best checkpoint ({tag}, val_auc={val_auc:.4f}) → {args.output}")

    print(f"\nBest val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
