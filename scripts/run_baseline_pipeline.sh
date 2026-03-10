#!/usr/bin/env bash
set -euo pipefail

uv sync --frozen

uv run python -m idh_glioma.data.prepare_dataset \
  --brats-root datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations \
  --idh-labels artifacts/idh_labels.csv \
  --output artifacts/manifest.json

uv run python -m idh_glioma.train.train_segmentation \
  --manifest artifacts/manifest.json \
  --epochs 20 \
  --profile a6000 \
  --batch-size 8 \
  --output checkpoints/unet2d_best.pt

uv run python -m idh_glioma.train.train_idh_classifier \
  --manifest artifacts/manifest.json \
  --epochs 20 \
  --profile a6000 \
  --batch-size 16 \
  --output checkpoints/mobilenetv3_idh_best.pt

echo "Baseline pipeline completed."
