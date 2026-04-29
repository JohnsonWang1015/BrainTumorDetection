# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IDH mutation detection and glioma segmentation pipeline for brain tumor MRI/CT analysis. The main package is `src/idh_glioma/` using PyTorch, with U-Net 2D segmentation and MobileNetV3 classification models.

## Build & Development Commands

```bash
# Install dependencies (uv is the package manager)
uv sync --frozen

# Run all tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_ct_datasets.py::test_collect_images_counts -v

# Type checking
basedpyright src/

# Build wheel
uv build
```

## CLI Entry Points (defined in pyproject.toml)

```bash
uv run prepare-mri    # Scan BraTS dataset → manifest.json
uv run prepare-ct     # Scan Kaggle CT/MRI → ct_manifest.json
uv run train-seg      # Train U-Net 2D segmentation
uv run train-idh      # Train MobileNetV3 IDH classifier
uv run train-ct       # Train CT/MRI classifier
uv run eval-seg       # Evaluate segmentation (Dice/IoU)
uv run eval-ct        # Evaluate classification (AUC/F1)
uv run infer          # End-to-end inference pipeline
uv run train-yolo     # YOLOv11 detection training
uv run tumor-app      # Launch Gradio web interface
```

## Architecture

```
src/idh_glioma/
├── app.py          # Gradio web UI with GradCAM visualization
├── data/           # Dataset preparation and loading
│   ├── prepare_dataset.py    # BraTS folder → manifest.json (train/val/test splits)
│   ├── prepare_ct_data.py    # Kaggle CT/MRI → ct_manifest.json
│   ├── datasets.py           # BraTSSliceSegmentationDataset, BraTSSliceClassificationDataset
│   ├── ct_datasets.py        # BrainImageDataset for CT/MRI images
│   └── export_yolo.py        # Manifest → YOLO format conversion
├── models/         # Neural network architectures
│   ├── unet2d.py                 # 4-channel input → 1-channel segmentation
│   └── mobilenetv3_classifier.py # MobileNetV3-Small binary classifier
├── train/          # Training scripts with CLI argument parsing
├── infer/pipeline.py  # Load volumes → segment → classify → save NIfTI + probability
├── integrations/   # SAM3 and YOLOv11 wrappers
├── eval/           # Metrics computation + visualization output
└── utils.py        # JSON I/O, path utilities
```

## Data Flow

1. **Preparation**: Raw BraTS NIfTI files → `prepare_dataset.py` → `artifacts/manifest.json` with case records (case_id, modalities dict, mask_path, idh_label)
2. **Training**: Manifest → Dataset classes load 4-channel 2D slices (flair/t1/t1Gd/t2) with z-score normalization → model training with Focal+Dice loss
3. **Inference**: Case directory → batch 2D segmentation → 3D mask reconstruction → slice-level classification → NIfTI mask + IDH probability

## Key Conventions

- **Manifest format**: JSON with `train`/`val`/`test` splits, each entry has `case_id`, `modalities` dict, `mask_path`, `idh_label` (0=wildtype, 1=mutant)
- **Data contract**: `configs/pipeline_contract.yaml` defines input modalities, segmentation target priority, preprocessing rules
- **GPU profiles**: `--profile a6000` enables higher batch size, AMP, TF32, cuDNN benchmark
- **NIfTI loading**: Uses `@lru_cache` with per-worker cache clearing in DataLoader `worker_init_fn`
- **Model checkpoints**: Saved as `{"model": state_dict, "epoch": epoch, "val_dice": score}` in `checkpoints/`
- **Python version**: 3.12 (pinned in `.python-version`)
- **Type checking**: basedpyright with lenient settings (see `pyrightconfig.json`)
- **All modules use**: `from __future__ import annotations`

## Best Practices

### Segmentation Training
- **Loss function**: Focal loss + Dice loss. Focal loss handles class imbalance (tumor pixels << background). Never use plain BCE for segmentation.
- **Mask interpolation**: Always use `mode="nearest"` when resizing binary masks. Bilinear creates fractional values that corrupt training.
- **Data augmentation**: Random flips, 90-degree rotations, and intensity jitter are applied during training. Must be applied jointly to image and mask.
- **LR schedule**: Cosine annealing with linear warmup (5 epochs). Prevents both underfitting (cold start) and overfitting (flat LR).
- **Gradient clipping**: `max_norm=1.0` prevents training instability, especially with mixed precision.
- **Checkpoint selection**: Save by best validation Dice score (direct metric), not loss.

### Classification Training
- **ImageNet normalization**: Applied to all CT/MRI images even though they are medical — pretrained features transfer well.
- **Augmentation**: Random horizontal flip, rotation (15°), color jitter for training split only.

### Web Interface (app.py)
- **Model singleton**: `_get_model()` loads once with global caching. Grad is globally disabled; GradCAM re-enables locally via `torch.enable_grad()`.
- **Gradio cache**: Uses `outputs/.gradio_cache` (project-local) instead of `/tmp/gradio` to avoid permission issues.
- **Model pre-warming**: `_get_model()` is called during `build_app()` so first user request is fast.
- **Latency display**: Inference time shown in both the figure title and diagnosis report.

### Performance Optimization
- `torch.set_grad_enabled(False)` globally for inference apps; re-enable locally only for GradCAM.
- `torch.set_float32_matmul_precision("medium")` for faster matmul on supported hardware.
- `pin_memory=True`, `non_blocking=True` for GPU data transfer.
- `channels_last` memory format for convolution-heavy models on CUDA.
- `persistent_workers=True` with `worker_init_fn` that clears NIfTI LRU cache per worker.

### Common Pitfalls
- **JIT tracing breaks MobileNetV3**: Do not use `torch.jit.trace` — squeeze-excitation blocks have data-dependent control flow.
- **Blackwell GPU (sm_120)**: PyTorch <=2.6 doesn't support it. App falls back to CPU automatically.
- **Disk space**: Root partition `/` may be full. Use `TMPDIR=/mnt/8tb_hdd2/tmp` for pip installs.
- **venv permissions**: The `.venv` symlinks to another user's Python. Use system Python with `PYTHONPATH=src` as fallback.

## Docker (GPU)

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm trainer
```

Base image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`

## Web Interface

```bash
uv run tumor-app          # Launch Gradio UI at http://localhost:7860
python -m idh_glioma.app  # Alternative launch
```

Features: Upload CT/MRI images, real-time tumor detection (96.4% accuracy), GradCAM heatmap visualization, diagnostic report with inference latency. Supports 8 example images (CT/MRI tumor/healthy).

## Shell Scripts

- `scripts/run_baseline_pipeline.sh` — Full pipeline: prepare → segment → classify
- `scripts/prepare_idh_data.sh` — Scan BraTS dataset and build manifest
- `scripts/run_end_to_end_inference.sh` — Single-case inference
- `scripts/launch_app.sh` — Launch web detection interface
- `scripts/setup_env.sh` — One-line environment setup

## Current Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| CT/MRI Classification (MobileNetV3) | Accuracy / AUC | 96.4% / 0.993 |
| MRI Segmentation (U-Net 2D, TCGA-LGG) | Dice | 0.7598 ± 0.090 (test, 10 cases); val best 0.8063 |
| YOLOv8 Detection | mAP50 | 0.476 |
