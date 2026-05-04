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
# Data preparation
uv run prepare-mri        # Scan BraTS dataset → manifest.json
uv run prepare-ct         # Scan Kaggle CT/MRI → ct_manifest.json
uv run prepare-idh-molecular  # Build pooled molecular cohort artifacts

# 2D legacy training (TCGA-LGG)
uv run train-seg          # U-Net 2D segmentation
uv run train-idh          # MobileNetV3 IDH classifier
uv run train-ct           # CT/MRI binary tumor classifier
uv run train-yolo         # YOLOv8/v11 detection

# 3D MONAI training (recommended for seg + IDH)
uv run train-seg-monai    # SegResNet 3D
uv run train-idh-monai    # DenseNet121 3D + bbox jitter
uv run train-idh-molecular # Logistic + LightGBM + MLP (RNA-seq molecular IDH)

# Evaluation
uv run eval-seg           # 2D U-Net Dice/IoU
uv run eval-ct            # CT classifier AUC/F1
uv run eval-idh           # 2D IDH case + slice AUC
uv run eval-seg-monai     # 3D SegResNet Dice
uv run eval-seg-zoo       # MONAI Model Zoo bundle (zero-shot)
uv run eval-idh-monai     # 3D IDH (GT-mask ROI)
uv run eval-e2e-zoo       # Full pipeline: bundle seg + jitter cls
uv run eval-idh-molecular # B3 eval: pooled CV + source holdout + GBM minority metrics

# Inference
uv run infer              # 2D end-to-end pipeline
uv run infer-monai        # 3D end-to-end (custom SegResNet)
uv run infer-monai-zoo    # 3D end-to-end (Model Zoo bundle, recommended)

# Web UI
uv run tumor-app          # Gradio (3 tabs: CT/MRI, IDH 2D, IDH 3D MONAI)
```

Helper scripts (not registered as CLI entries):

```bash
# Threshold calibration via Youden's J on val
uv run python scripts/calibrate_idh_threshold.py --ckpt <ckpt>
# 5-fold stratified CV
uv run python scripts/cv_idh.py            # 2D
uv run python scripts/cv_idh_monai.py      # 3D MONAI
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
├── molecular/      # TCGA RNA-seq + MAF molecular IDH pipeline (prepare/train/eval)
└── utils.py        # JSON I/O, path utilities
```

### Molecular IDH (RNA-seq / Multi-omics) Architecture

- Entry points:
  - `prepare-idh-molecular` builds `artifacts/molecular/{expression_matrix.parquet,idh_labels.parquet,cohort_manifest.json,feature_panel.json}` from TCGA-GBM + TCGA-LGG molecular drops.
  - `train-idh-molecular` trains pooled-cohort Logistic / LightGBM / MLP models and writes `checkpoints/molecular_idh/`.
  - `eval-idh-molecular` runs B3 reporting (`pooled_cv`, `source_holdout`, `minority_metrics`) and writes `artifacts/molecular_idh_eval/` plus figures.
  - Multi-omics mode is enabled with `--modalities rnaseq methylation` and writes `artifacts/molecular_multimodal/`, `checkpoints/molecular_idh_multimodal/`, and `artifacts/molecular_idh_multimodal_eval/` when default output directories are used.
- Data contract:
  - Expression matrix index is base ENSG (version stripped), values are `log2(TPM+1)`.
  - Labels come from aggregated public masked MAF (`IDH1/IDH2` missense).
  - Fold-aware feature selection is `top-K variance ∪ curated prior panel`.
- Isolation:
  - `src/idh_glioma/molecular/` is independent from imaging modules and only shares `utils.py` JSON helpers.

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

### Classification Training (CT/MRI, `BrainImageDataset`)
- **ImageNet normalization**: Applied to all CT/MRI images even though they are medical — pretrained features transfer well.
- **Augmentation**: Random horizontal flip, rotation (15°), color jitter for training split only.

### IDH Classification (`BraTSSliceClassificationDataset`)
- **Inputs**: 3-channel slices (flair/t1Gd/t2) with per-volume z-score, NOT ImageNet normalization (intensity ranges differ).
- **Augmentation**: `_augment_cls` mirrors the seg augment (flips + 90-deg rotations + intensity jitter); applied only when `split == "train"`.
- **Pretrained backbone**: `build_mobilenetv3_binary(pretrained=True)` loads ImageNet weights and keeps the 3-channel stem unchanged. Fine-tuning from random init does not work with the small TCGA-LGG cohort (45 train cases).
- **Class imbalance**: BCE-with-logits uses `pos_weight = sqrt(num_neg / num_pos)`. The literal ratio is too aggressive when imbalance is heavy and pushes the model to over-predict the minority class.
- **Checkpointing**: Save by best validation **AUC**, not val_loss. With pos_weight set, val_loss is noisy and a poor signal of discrimination quality.
- **LR / schedule**: Pretrained fine-tuning needs `lr=1e-4` with cosine warmup (3 epochs warmup, then cosine decay). `lr=3e-4` overshoots at epoch 1.

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
- **Disk space**: Root partition `/` may be 99% full. Use `TMPDIR=/mnt/8tb_hdd2/johnson/tmp` for pip installs (the bare `/mnt/8tb_hdd2/tmp` is not writable).
- **`.venv` rebuild**: If `.venv/bin/python3` becomes inaccessible (e.g. created by a different user, paths point to `/home/esl/...`), recover with `rm -rf .venv && uv sync --frozen`. Falling back to system Python won't work because torch is not installed there.
- **Two GPUs on this host**: GPU 0 is RTX PRO 5000 Blackwell (ours), GPU 1 is RTX A6000 typically used by another user (`legal-rag-taide`). Default cuda:0 is correct; don't assume GPU 1 is free.
- **`prepare_dataset.py` int parsing**: `pd.read_csv` infers numeric columns with NaN as float. Always cast labels via `int(float(str(x).strip()))` to handle pandas' float coercion.
- **Manifest cohort mismatch**: `BraTSSliceClassificationDataset` filters out `idh_label is None`, so a mismatched/empty `idh_labels.csv` silently produces 0 training samples and a `RuntimeError`.
- **`weights="DEFAULT"` vs custom stem**: When swapping the first conv of a pretrained backbone, you lose the pretrained stem weights. Only swap when `num_input_channels != 3`.

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

Three tabs:
1. **CT/MRI Tumor Detection** — single-image upload (PNG/JPG), MobileNetV3-Small binary classifier (96.4% acc / AUC 0.993) with GradCAM heatmap.
2. **IDH Mutation Classification (2D)** — 4 BraTS-style NIfTI uploads, U-Net 2D + MobileNetV3-large pipeline with calibrated threshold 0.876. Implementation in `app_idh.py`.
3. **IDH Mutation Classification (3D MONAI, recommended)** — 4 NIfTI uploads, MONAI Model Zoo bundle (zero-shot Dice 0.926) + 3D DenseNet121 (jitter-trained) at threshold 0.0775. E2E AUC 0.75, accuracy 0.80. Implementation in `app_idh_monai.py`.

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
| MRI Segmentation (U-Net 2D, TCGA-LGG) | Dice | 0.7598 ± 0.090 (test); val best 0.8063 |
| MRI Segmentation (MONAI SegResNet 3D, TCGA-LGG) | Dice | **0.9101 ± 0.036** (test, 10 cases); val best 0.9176. 3D + sliding-window + DiceCE; +0.15 over 2D, std halved. |
| IDH Mutation Classifier (MobileNetV3-large 2D, TCGA-LGG, ROI crop) | AUC | Case **0.875** / Slice 0.453 (single-split test); val best 0.8533. 5-fold CV smoothed val AUC **0.764 ± 0.076**. Calibrated threshold 0.876 (Youden's J) → WT recall 50%. |
| IDH Mutation Classifier (MONAI DenseNet121 3D, TCGA-LGG, GT-mask ROI crop) | AUC | Single-split: case **1.000** (test, GT mask). 5-fold CV (`scripts/cv_idh_monai.py`): mean smoothed val AUC **0.916 ± 0.073** (+0.15 vs 2D's 0.764). With predicted-mask E2E (`infer-monai-zoo`, threshold 0.0775): accuracy **0.80**, AUC **0.75**, macro F1 0.69 — jitter recovered the hard case TCGA-CS-6669 to p=0.243 (vs 0.9998 without jitter). |
| MRI Segmentation (MONAI Model Zoo `brats_mri_segmentation`, zero-shot) | Dice | **0.9257 ± 0.031** (test, WT channel). Beats our trained SegResNet 0.910 with no fine-tuning — bundle was pretrained on full BraTS challenge cohort (~500 cases). Recommended production seg model. |
| YOLO Detection (best of yolov8n / yolo11n / yolo11s) | mAP50 | 0.497 (yolo11s); mAP50-95 0.347 (yolov8n best for high-IoU). Dataset (893 train / 223 val) saturates at this ceiling — scaling backbone past yolo11s gives diminishing returns. Breaking 0.55 needs more data, not a bigger model. |
| Molecular IDH Classifier (LightGBM, RNA-seq pooled TCGA-GBM+LGG) | AUC | Pooled 5-fold CV **0.992 ± 0.009** (best of three molecular models); source-holdout AUC: LGG→GBM **0.965**, GBM→LGG **0.956**; GBM-only minority AUPRC **0.947**. |
| Molecular IDH Classifier (RNA-seq + methylation multi-omics) | AUC | Pooled 5-fold CV best **0.993 ± 0.007** (MLP); source-holdout best AUC: LGG→GBM **0.985** (MLP), GBM→LGG **0.977** (Logistic); GBM-only minority AUPRC best **0.942** (LightGBM). |
