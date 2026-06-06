"""Temperature-scale the CT/MRI tumor classifier on the val split.

Temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural
Networks") is the simplest, strongest post-hoc calibration method: fit a single
scalar ``T`` that rescales the logits (``p = sigmoid(logit / T)``) by minimising
NLL on a held-out split. Because ``T > 0`` preserves the sign of every logit,
the 0.5 decision — and therefore accuracy/AUC — is **unchanged**; only the
confidences move, lowering ECE.

The learned ``T`` is written back into the checkpoint under ``"temperature"``;
``eval-ct`` and the Gradio app pick it up automatically (default 1.0 = off).

Run::

    uv run python scripts/calibrate_ct_temperature.py \
        --ckpt checkpoints/mobilenetv3_ct_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from idh_glioma.data.ct_datasets import BrainImageDataset
from idh_glioma.eval.eval_ct import expected_calibration_error
from idh_glioma.models.mobilenetv3_classifier import build_mobilenetv3_binary


@torch.inference_mode()
def _collect_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, labels) over the whole loader, on CPU as float32."""
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        logits = model(images).float().cpu().reshape(-1)
        logits_all.append(logits)
        labels_all.append(labels.float().cpu().reshape(-1))
    return torch.cat(logits_all), torch.cat(labels_all)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit T>0 minimising BCE(logits / T, labels) via LBFGS.

    Optimises over log-T so the temperature stays strictly positive regardless
    of the line search.
    """
    # Logits were gathered under inference_mode; clone to plain tensors so they
    # can participate in autograd.
    logits = logits.detach().clone()
    labels = labels.detach().clone()
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(0) = 1.0 at init
    optimizer = torch.optim.LBFGS(
        [log_t], lr=0.05, max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)  # type: ignore[arg-type]
    return float(log_t.exp().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature-scale the CT classifier")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/mobilenetv3_ct_best.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/ct_manifest.json"))
    parser.add_argument("--modality", choices=["ct", "mri", "both"], default="both")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the fitted T and ECE without writing to the checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    variant = state.get("variant", "small")
    print(f"[calibrate-T] ckpt: variant={variant}  prev temperature={state.get('temperature', 1.0)}")

    model = build_mobilenetv3_binary(num_input_channels=3, variant=variant).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    ds = BrainImageDataset(
        args.manifest, split=args.split, modality=args.modality,
        img_size=args.img_size, augment=False,
    )
    if len(ds) == 0:
        raise ValueError(f"{args.split} split is empty (modality={args.modality}).")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    print(f"[calibrate-T] fitting on {len(ds)} {args.split} images...")

    logits, labels = _collect_logits(model, loader, device)
    labels_np = labels.numpy().astype(int)

    # Before / after calibration (val split).
    probs_before = torch.sigmoid(logits).numpy()
    ece_before, _ = expected_calibration_error(probs_before, labels_np)
    nll_before = float(F.binary_cross_entropy_with_logits(logits, labels).item())

    temperature = fit_temperature(logits, labels)

    probs_after = torch.sigmoid(logits / temperature).numpy()
    ece_after, _ = expected_calibration_error(probs_after, labels_np)
    nll_after = float(F.binary_cross_entropy_with_logits(logits / temperature, labels).item())

    print(f"\n[calibrate-T] fitted temperature T = {temperature:.4f}")
    print(f"  val NLL : {nll_before:.4f}  ->  {nll_after:.4f}")
    print(f"  val ECE : {ece_before:.4f}  ->  {ece_after:.4f}")
    # Accuracy is invariant under T>0 (sign of logit preserved); confirm.
    acc_before = float(((probs_before >= 0.5).astype(int) == labels_np).mean())
    acc_after = float(((probs_after >= 0.5).astype(int) == labels_np).mean())
    print(f"  val acc : {acc_before:.4f}  ->  {acc_after:.4f}  (must be identical)")

    if args.dry_run:
        print("\n[calibrate-T] --dry-run: checkpoint NOT modified.")
        return

    state["temperature"] = temperature
    state["temperature_split"] = args.split
    torch.save(state, args.ckpt)
    print(f"\n[calibrate-T] saved temperature={temperature:.4f} → {args.ckpt}")


if __name__ == "__main__":
    main()
