"""3D segmentation training with MONAI on the TCGA-LGG cohort.

Uses SegResNet (4-channel BraTS-style multimodal input -> binary whole-tumor
mask) trained on 96x96x96 random crops with DiceCE loss. Sliding-window
inference is used for full-volume validation. Optionally initialises from
the MONAI Model Zoo brats_mri_segmentation bundle weights.

Run::

    uv run train-seg-monai \
        --manifest artifacts/manifest.json \
        --output checkpoints/segresnet_tcga.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
)
from torch.optim import AdamW
from tqdm import tqdm

from idh_glioma.utils import load_json


def _records_to_monai(records, project_root):
    out = []
    for r in records:
        mods = r["modalities"]
        out.append(
            {
                "image": [
                    str(project_root / mods["flair"]),
                    str(project_root / mods["t1"]),
                    str(project_root / mods["t1Gd"]),
                    str(project_root / mods["t2"]),
                ],
                "label": str(project_root / r["mask_path"]),
                "case_id": r.get("case_id", ""),
            }
        )
    return out


def _build_transforms(roi_size, training):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
    if training:
        base.extend(
            [
                RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            ]
        )
    base.append(ToTensord(keys=["image", "label"]))
    return Compose(base)


def _binarize_label(label):
    return (label > 0).float()


def _maybe_load_zoo_init(model, zoo_init):
    if not zoo_init:
        return
    print(f"[train-seg-monai] Loading init weights from {zoo_init}")
    # ``weights_only=False`` so we can also accept our own training checkpoints
    # which carry epoch/val_dice scalars alongside the ``model`` state dict.
    state = torch.load(zoo_init, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    own = model.state_dict()
    loaded = 0
    skipped = []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(own)
    print(f"[train-seg-monai] init: loaded {loaded} tensors, skipped {len(skipped)} (incompat shapes)")


def parse_args():
    p = argparse.ArgumentParser(description="Train MONAI SegResNet for binary brain-tumor segmentation")
    p.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.json"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-rate", type=float, default=1.0)
    p.add_argument("--roi-size", type=int, nargs=3, default=(96, 96, 96))
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--zoo-init", type=str, default=None)
    p.add_argument("--output", type=Path, default=Path("checkpoints/segresnet_tcga.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    project_root = Path.cwd()
    manifest = load_json(args.manifest)
    train_data = _records_to_monai(manifest["train"], project_root)
    val_data = _records_to_monai(manifest["val"], project_root)
    print(f"train cases: {len(train_data)} | val cases: {len(val_data)}")

    train_tf = _build_transforms(tuple(args.roi_size), training=True)
    val_tf = _build_transforms(tuple(args.roi_size), training=False)

    train_ds = CacheDataset(data=train_data, transform=train_tf, cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_ds = CacheDataset(data=val_data, transform=val_tf, cache_rate=args.cache_rate, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        init_filters=args.init_filters,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    ).to(device)
    _maybe_load_zoo_init(model, args.zoo_init)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)

    loss_fn = DiceCELoss(sigmoid=True, to_onehot_y=False, lambda_dice=1.0, lambda_ce=1.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    warmup = args.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    best_dice = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"train e{epoch + 1}", leave=False):
            x = batch["image"].to(device, non_blocking=True).float()
            y = _binarize_label(batch["label"].to(device, non_blocking=True))
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last_3d)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
        train_loss = running / max(len(train_loader), 1)

        model.train(False)
        dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val   e{epoch + 1}", leave=False):
                x = batch["image"].to(device, non_blocking=True).float()
                y = _binarize_label(batch["label"].to(device, non_blocking=True))
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = sliding_window_inference(
                        x, roi_size=tuple(args.roi_size), sw_batch_size=1, predictor=model, overlap=0.5
                    )
                preds = [post_pred(p) for p in decollate_batch(logits)]
                labels = [post_label(t) for t in decollate_batch(y)]
                dice_metric(y_pred=preds, y=labels)
        val_dice = float(dice_metric.aggregate().item())

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch + 1:03d}] train_loss={train_loss:.4f} "
            f"val_dice={val_dice:.4f} lr={current_lr:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "val_dice": val_dice,
                    "epoch": epoch + 1,
                    "init_filters": args.init_filters,
                    "roi_size": list(args.roi_size),
                    "arch": "segresnet",
                },
                args.output,
            )
            print(f"Saved best checkpoint (dice={val_dice:.4f}): {args.output}")

    print(f"\nTraining complete. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
