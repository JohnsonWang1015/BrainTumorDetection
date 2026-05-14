# Model Card

## 1. Scope

This card documents the trained checkpoints currently present under `checkpoints/` and the training status that can be verified from:

- checkpoint metadata embedded in `.pt` files
- `artifacts/cv/cv_results.json`
- `artifacts/cv_monai/cv_results.json`
- training and evaluation scripts under `src/idh_glioma/train/` and `src/idh_glioma/eval/`
- project notes in `README.md` and `CLAUDE.md`

Where a metric comes from code or artifacts, it is treated as verified from the current repository snapshot. Where a metric appears only in project notes, it should be treated as repository-reported rather than freshly re-run in this session.

## 2. Current Checkpoint Inventory

| Checkpoint | Size | Mtime | Task | Status |
|---|---:|---|---|---|
| `checkpoints/unet2d_tcga_v1.pt` | 30M | 2026-04-28 19:06 | 2D MRI segmentation | Present |
| `checkpoints/mobilenetv3_idh_v3.pt` | 6.0M | 2026-04-29 11:23 | Early 2D IDH classifier | Present |
| `checkpoints/mobilenetv3_idh_best.pt` | 17M | 2026-04-30 12:19 | Main 2D IDH classifier | Present |
| `checkpoints/segresnet_tcga.pt` | 72M | 2026-04-30 16:53 | 3D MRI segmentation (TCGA-LGG, 45 train cases) | Present |
| `checkpoints/segresnet_brats2021.pt` | 75M | 2026-05-14 19:04 | 3D MRI segmentation (BraTS 2021, 872 train cases) | Present |
| `checkpoints/densenet3d_idh.pt` | 44M | 2026-04-30 17:25 | 3D IDH classifier | Present |
| `checkpoints/densenet3d_idh_jitter.pt` | 44M | 2026-04-30 18:26 | 3D IDH classifier with bbox jitter | Present |
| `checkpoints/mobilenetv3_ct_best.pt` | 6.0M | 2026-03-21 14:39 | CT/MRI tumor classifier | Present |
| `checkpoints/yolov8_brain_tumor_best.pt` | 6.0M | 2026-03-28 14:52 | YOLO detector | Present |
| `checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt` | external bundle | vendor asset | MONAI Model Zoo segmentation | Present |
| `checkpoints/molecular_idh/{logistic.joblib, lightgbm.txt, mlp.pt}` | ~6M total | 2026-05-04 | Molecular IDH classifier (RNA-seq) | Present |
| `checkpoints/molecular_idh_multimodal/{logistic.joblib, lightgbm.txt, mlp.pt}` | ~7M total | 2026-05-04 | Molecular IDH classifier (RNA-seq + methylation fusion) | Present |

## 3. Recommended Models

Based on the repository's own reported outcomes and the checkpoints present locally:

- Recommended segmentation model: MONAI Model Zoo `brats_mri_segmentation` bundle (zero-shot Dice 0.9257 on TCGA-LGG test)
- Strongest custom segmentation model: `segresnet_brats2021.pt` — Dice 0.9206 on TCGA-LGG test (beats the older `segresnet_tcga.pt` 0.9101 by +1.05 pp on the same test set) and 0.9101 on the larger 188-case BraTS 2021 test
- Recommended end-to-end IDH pipeline: MONAI Model Zoo segmentation + `densenet3d_idh_jitter.pt` + `artifacts/e2e_idh_config.json`
- `unet2d_tcga_v1.pt` for 2D segmentation baseline
- `mobilenetv3_idh_best.pt` for 2D IDH baseline
- `segresnet_tcga.pt` retained as the small-cohort 3D segmentation baseline (now superseded by `segresnet_brats2021.pt` on every cohort tested)
- `densenet3d_idh.pt` and `densenet3d_idh_jitter.pt` for 3D IDH baselines

## 4. Model Details

### 4.1 `unet2d_tcga_v1.pt`

Task:

- 2D whole-tumor segmentation on BraTS-TCGA-LGG

Architecture:

- custom `UNet2D`
- input channels: 4 (`flair`, `t1`, `t1Gd`, `t2`)
- output channels: 1
- encoder-decoder U-Net with base width 32

Training recipe from code:

- loss: focal loss + Dice loss
- optimizer: `AdamW`
- default epochs: 100
- default lr: `3e-4`
- warmup: 5 epochs
- gradient clipping: `max_norm=1.0`
- checkpoint selection: best validation Dice

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| epoch | 50 |
| val_loss | 0.2521 |
| val_dice | 0.8063 |

Repository-reported evaluation:

- test Dice: `0.7598 ± 0.090`
- best validation Dice: `0.8063`

### 4.2 `mobilenetv3_idh_v3.pt`

Task:

- earlier 2D slice-based IDH mutation classifier on BraTS-TCGA-LGG

Architecture:

- `MobileNetV3` binary classifier
- variant not stored in checkpoint metadata
- input channels: 3 (`flair`, `t1Gd`, `t2`)

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| epoch | 5 |
| val_loss | 0.3166 |
| val_auc | 0.7522 |

Interpretation:

- this looks like an earlier intermediate checkpoint kept for comparison
- `mobilenetv3_idh_best.pt` is the more fully documented 2D IDH checkpoint

### 4.3 `mobilenetv3_idh_best.pt`

Task:

- main 2D slice-based IDH mutation classifier

Architecture:

- `MobileNetV3-large` binary classifier
- pretrained ImageNet backbone
- tumor ROI crop enabled
- input built from `flair`, `t1Gd`, `t2`

Training recipe from code:

- loss: BCE-with-logits
- class weighting: `pos_weight = sqrt(num_neg / num_pos)`
- optimizer: `AdamW`
- default lr: `1e-4`
- warmup: 3 epochs
- checkpoint selection: best smoothed validation AUC
- smoothing window: 3 epochs

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| epoch | 23 |
| val_loss | 0.8731 |
| val_auc | 0.8533 |
| smoothed_auc | 0.8299 |
| variant | `large` |
| use_roi | `True` |
| roi_margin | 10 |
| img_size | 224 |
| threshold | 0.8759 |
| threshold_method | `youden_j` |
| threshold_split | `val` |

Verified supporting artifacts:

- `artifacts/cv/cv_results.json` gives 5-fold smoothed validation AUC mean `0.7639 ± 0.0762`
- raw fold validation AUC mean `0.7467 ± 0.0948`

Repository-reported evaluation:

- single-split case AUC: `0.875`
- single-split slice AUC: `0.453`

### 4.3a End-to-end calibration artifact

`artifacts/e2e_idh_config.json` is a separate runtime artifact for the predicted-mask MONAI pipeline. It stores:

- `threshold` selected on the end-to-end validation split using `macro_f1`
- shared ROI settings such as `view_margins`, `aggregation`, and `dilate_iters`
- bookkeeping fields like `threshold_split`, `macro_f1`, and linked checkpoints

Current verified values from the file are:

- `threshold = 0.13`
- `threshold_objective = macro_f1`
- `aggregation = mean`
- `view_margins = [0]`
- `keep_largest = true`
- `dilate_iters = 0`
- `macro_f1 = 1.0`
- `accuracy = 1.0`
- `wt_recall = 1.0`
- `mutant_recall = 1.0`
- `auc = 1.0`
- `num_valid_cases = 10`

This artifact is consumed by:

- `src/idh_glioma/eval/eval_e2e_monai_zoo.py`
- `src/idh_glioma/infer/pipeline_monai_zoo.py`
- `src/idh_glioma/app_idh_monai.py`

### 4.4 `mobilenetv3_ct_best.pt`

Task:

- 2D binary tumor vs healthy classifier for mixed CT/MRI image dataset

Architecture:

- `MobileNetV3-small` binary classifier
- ImageNet-pretrained backbone
- RGB input

Training recipe from code:

- optimizer: `AdamW`
- default lr: `3e-4`
- default epochs: 20
- checkpoint selection: best validation loss

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| epoch | 19 |
| val_loss | 0.0864 |
| val_acc | 0.9674 |
| modality | `both` |

Repository-reported evaluation:

- accuracy: `96.4%`
- AUC: `0.993`

### 4.5 `segresnet_tcga.pt`

Task:

- custom-trained 3D whole-tumor segmentation on BraTS-TCGA-LGG

Architecture:

- MONAI `SegResNet`
- `spatial_dims=3`
- `in_channels=4`
- `out_channels=1`
- `init_filters=32`
- crop/validation ROI: `96 x 96 x 96`

Training recipe from code:

- loss: `DiceCELoss`
- optimizer: `AdamW`
- default lr: `1e-4`
- warmup: 5 epochs
- validation by sliding-window inference
- checkpoint selection: best validation Dice

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| epoch | 68 |
| val_dice | 0.9176 |
| arch | `segresnet` |
| init_filters | 32 |
| roi_size | `[96, 96, 96]` |

Repository-reported evaluation:

- test Dice: `0.9101 ± 0.036`
- best validation Dice: `0.9176`

### 4.5a `segresnet_brats2021.pt`

Task:

- custom-trained 3D whole-tumor segmentation on the BraTS 2021 cohort (Hugging Face mirror `rocky93/BraTS_segmentation`)
- intended as a drop-in replacement for `segresnet_tcga.pt` with substantially larger training data

Architecture:

- identical to `segresnet_tcga.pt` (MONAI `SegResNet`, `init_filters=32`, 4 in / 1 out, ROI 96³)
- 83 tensors, ≈ 4.7M parameters

Training recipe from code:

- training cohort: 872 cases from `artifacts/brats2021_manifest.json` (val 188 / test 188)
- init weights: `segresnet_tcga.pt` loaded with shape match (83/83 tensors), i.e. fine-tune from the older baseline rather than train from scratch
- loss: `DiceCELoss(lambda_dice=1, lambda_ce=1)`
- optimizer: `AdamW`, lr `1e-4`
- schedule: linear warmup 3 epochs + cosine annealing to 0
- batch size: 4
- augmentation: `RandSpatialCropd(96³)` + 3-axis `RandFlipd(p=0.5)`
- DataLoader workers: 4; CacheDataset `cache_rate=0.0` (multi-worker cache path exhibits a list-of-paths race condition on this cohort — see commit notes)
- gradient clipping: `max_norm=1.0`
- mixed precision: `torch.autocast` fp16
- checkpoint selection: best validation Dice on BraTS 2021 val

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| epoch | 25 |
| val_dice | 0.9289 |
| arch | `segresnet` |
| init_filters | 32 |
| roi_size | `[96, 96, 96]` |

Fresh cross-cohort evaluation (re-run in this session, sliding-window inference):

| Test cohort | n | Dice mean | Dice median | Dice std | IoU mean |
|---|---:|---:|---:|---:|---:|
| TCGA-LGG test | 10 | **0.9206** | 0.9212 | 0.0338 | 0.8546 |
| BraTS 2021 test | 188 | **0.9101** | 0.9449 | 0.1156 | 0.8500 |

Comparison vs `segresnet_tcga.pt` baseline on the **same held-out test splits**:

| Test cohort | baseline Dice | new Dice | Δ |
|---|---:|---:|---:|
| TCGA-LGG test (n=10) | 0.9101 ± 0.036 | **0.9206 ± 0.034** | **+0.0105 pp** |
| BraTS 2021 test (n=188) | 0.8568 ± 0.154 | **0.9101 ± 0.116** | **+0.0533 pp** |

Cross-cohort robustness (same model, different test cohort):

- `segresnet_tcga.pt` drops 5.33 pp moving from TCGA-LGG to BraTS 2021 (severe cohort overfit)
- `segresnet_brats2021.pt` drops only 1.05 pp on the same shift (≈5× more robust)

Individual-case improvement on BraTS 2021 test:

- great-case rate (Dice > 0.9): baseline `101/188` → new `144/188` (+43%)
- failure rate (Dice < 0.5): baseline `7/188` → new `4/188` (−43%)

Limitations:

- TCGA-LGG test has only 10 cases, so the +0.0105 Dice gain on that cohort is at the edge of statistical significance on that split alone; the +0.0533 result on BraTS 2021 test (n=188) is the load-bearing evidence
- the BraTS 2021 mirror used here lacks IDH labels, so this checkpoint contributes to segmentation only and cannot improve the IDH classification path
- not yet evaluated on UCSF-PDGM, EGD, BraTS 2023, or any clinical-routine MRI distribution

Reproduction:

- end-to-end recipe written up in `docs/BraTS2021_SegResNet_Experiment_Report.docx`
- download cohort: `uv run python scripts/download_hf_brats2021.py --workers 8`
- prepare manifest: `uv run prepare-mri --brats-root datasets/BraTS2021_HF --output artifacts/brats2021_manifest.json`
- train: `uv run train-seg-monai --manifest artifacts/brats2021_manifest.json --output checkpoints/segresnet_brats2021.pt --epochs 30 --batch-size 4 --num-workers 4 --cache-rate 0.0 --warmup-epochs 3 --zoo-init checkpoints/segresnet_tcga.pt`

### 4.6 `densenet3d_idh.pt`

Task:

- 3D IDH mutation classifier using ground-truth-mask ROI crops

Architecture:

- MONAI `DenseNet121`
- `spatial_dims=3`
- `in_channels=4`
- `out_channels=1`
- resized crop: `96 x 96 x 96`

Training recipe from code:

- 3D tumor bounding-box crop from mask
- BCE-with-logits with `pos_weight = sqrt(num_neg / num_pos)`
- optimizer: `AdamW`
- default lr: `1e-4`
- warmup: 3 epochs
- optional context-heavy crop augmentation via `--context-view-prob` and `--context-extra-max`
- checkpoint selection: best smoothed validation AUC

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| epoch | 7 |
| val_loss | 0.1427 |
| val_auc | 1.0000 |
| smoothed_auc | 1.0000 |
| arch | `densenet121_3d` |
| target_size | `[96, 96, 96]` |
| margin | 4 |
| in_channels | 4 |

Interpretation:

- this checkpoint reflects the optimistic GT-mask ROI setting
- it is useful as an upper-bound classifier baseline, not a full end-to-end model

### 4.7 `densenet3d_idh_jitter.pt`

Task:

- 3D IDH classifier trained to tolerate looser or shifted tumor boxes from predicted masks

Architecture:

- same base architecture as `densenet3d_idh.pt`
- training ROI perturbation enabled

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| epoch | 6 |
| val_loss | 0.2734 |
| val_auc | 1.0000 |
| smoothed_auc | 1.0000 |
| arch | `densenet121_3d` |
| target_size | `[96, 96, 96]` |
| margin | 4 |
| jitter_expand_max | 12 |
| jitter_shift_max | 6 |
| context_view_prob | 0.35 |
| context_extra_max | 6 |
| in_channels | 4 |
| threshold | `None` |
| threshold_method | `None` |
| threshold_split | `None` |

Verified supporting artifacts:

- `artifacts/cv_monai/cv_results.json` gives 5-fold smoothed validation AUC mean `0.9164 ± 0.0730`
- raw fold validation AUC mean `0.8891 ± 0.1058`

Repository-reported evaluation:

- GT-mask case AUC: `1.000`
- end-to-end with predicted-mask ROI: accuracy `0.80`, AUC `0.75`, macro F1 `0.69`

Interpretation:

- this is the most operationally relevant custom IDH checkpoint in the repo because it models inference-time box noise
- the runtime decision threshold now lives in `artifacts/e2e_idh_config.json`, not inside this checkpoint

### 4.8 `yolov8_brain_tumor_best.pt`

Task:

- object detection for brain-tumor localization in the Ultralytics dataset branch

Architecture and trainer:

- Ultralytics YOLO checkpoint
- embedded `train_args.model = yolov8n.pt`

Checkpoint metadata verified from file:

| Field | Value |
|---|---|
| training date | `2026-03-28T14:51:09.336581` |
| Ultralytics version | `8.4.21` |
| base model | `yolov8n.pt` |
| data | `/mnt/8tb_hdd2/BrainTumorDetection/datasets/Ultralytics/brain-tumor-abs.yaml` |
| epochs | 100 |
| imgsz | 640 |
| batch | 32 |
| patience | 20 |
| lr0 | 0.01 |
| workers | 8 |
| project | `outputs/yolo` |
| name | `yolo_brain_tumor` |

Embedded validation metrics from the checkpoint:

| Metric | Value |
|---|---:|
| precision | 0.4510 |
| recall | 0.8239 |
| mAP50 | 0.4758 |
| mAP50-95 | 0.3467 |

Important note:

- the current training script defaults now point to `yolo11n.pt`, 200 epochs, and different augmentation knobs
- the local checkpoint was actually trained with `yolov8n.pt`, 100 epochs, and `mixup=0.0`, `copy_paste=0.0`, `degrees=0.0`
- repository notes mention broader YOLO comparisons, but the local file here specifically captures the `yolov8n` run above

### 4.9 MONAI Model Zoo bundle

Asset:

- `checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt`

Role:

- zero-shot 3D segmentation backbone used by `eval-seg-zoo` and the end-to-end MONAI pipeline

Architecture assumptions from code:

- `SegResNet`
- `in_channels=4`
- `out_channels=3`
- `init_filters=16`
- input channel order: `t1Gd`, `t1`, `t2`, `flair`
- whole-tumor prediction uses output channel index `1`

Repository-reported evaluation:

- zero-shot test Dice: `0.9257 ± 0.031`

Interpretation:

- according to repository notes, this bundle currently outperforms the custom-trained `segresnet_tcga.pt`
- it is the preferred production segmentation component in this repo snapshot

## 5. Training Status Summary

### 5.1 Verified current status

- BraTS segmentation and IDH pipelines have both 2D and 3D trained checkpoints present locally
- the strongest custom 3D segmentation checkpoint is `segresnet_brats2021.pt` (trained 2026-05-14 on the 1,248-case BraTS 2021 cohort, val Dice 0.9289 on n=188, test Dice 0.9206 on TCGA-LGG vs 0.9101 baseline). End-to-end experiment report: `docs/BraTS2021_SegResNet_Experiment_Report.docx`.
- 3D custom segmentation (legacy `segresnet_tcga.pt`) and both 3D IDH checkpoints were updated on 2026-04-30
- the main 2D IDH checkpoint is calibrated with a stored threshold and paired with 5-fold CV results
- the jitter-trained 3D IDH checkpoint is paired with 5-fold CV results, while the deployed runtime threshold now lives in `artifacts/e2e_idh_config.json`
- the deployed end-to-end decision rule is now anchored by `artifacts/e2e_idh_config.json` with `threshold=0.13` and ROI postprocess settings shared across app/eval/infer
- the molecular IDH checkpoints (`checkpoints/molecular_idh/` for RNA-seq and `checkpoints/molecular_idh_multimodal/` for RNA-seq + methylation fusion) were trained on 2026-05-04 with B3 evaluation artifacts present under `artifacts/molecular_idh_eval/` and `artifacts/molecular_idh_multimodal_eval/`
- the CT/MRI classifier and YOLO detector were trained earlier in March 2026 and remain available

### 5.2 Best available options by task

| Task | Best local choice | Why |
|---|---|---|
| MRI segmentation | MONAI Model Zoo bundle | highest reported Dice |
| 2D segmentation baseline | `unet2d_tcga_v1.pt` | simplest custom baseline |
| 3D segmentation custom (strongest) | `segresnet_brats2021.pt` | TCGA-LGG test Dice 0.9206 (+1.05 pp vs TCGA-trained baseline on the same test split); 5× more cross-cohort robust |
| 3D segmentation baseline (legacy small-cohort) | `segresnet_tcga.pt` | trained on 45 TCGA cases; kept for reference and reproduction |
| 2D IDH baseline | `mobilenetv3_idh_best.pt` | calibrated, metadata-complete, CV-backed |
| 3D IDH upper bound | `densenet3d_idh.pt` | perfect val/test report under GT-mask ROI |
| 3D IDH practical custom model | `densenet3d_idh_jitter.pt` | trained for predicted-mask ROI noise |
| Molecular IDH (RNA-seq, pooled) | `molecular_idh/lightgbm.txt` | best pooled CV AUC and best GBM minority AUPRC |
| Molecular IDH (RNA-seq, cross-cohort GBM→LGG) | `molecular_idh/logistic.joblib` | strongest GBM→LGG transfer AUC |
| Molecular IDH (multi-omics, pooled / LGG→GBM) | `molecular_idh_multimodal/mlp.pt` | best pooled CV and best LGG→GBM transfer |
| Molecular IDH (multi-omics, calibration) | `molecular_idh_multimodal/lightgbm.txt` | best Brier score (~31% better than RNA-seq-only) |
| CT/MRI tumor classification | `mobilenetv3_ct_best.pt` | only active classifier for that branch |
| Tumor detection | `yolov8_brain_tumor_best.pt` | only local YOLO checkpoint present |

## 6. Cross-Model Performance Comparison

A single table covering every trained model in the repo. Columns include the metrics requested (`Accuracy`, `AUC`, `macro F1`) plus a task-specific primary metric where it is the convention for that task (Dice for segmentation, mAP for detection, AUPRC for the heavily-imbalanced minority subgroup). `—` means the metric is not reported for that task or split. **Bold** marks the best value per metric within each task group; ★ marks the recommended production model per task.

| Model / Pipeline | Task | Cohort (n) | Accuracy | AUC | macro F1 | Task-specific primary | Notes |
|---|---|---|---:|---:|---:|---|---|
| **Segmentation** | | | | | | | |
| MONAI Model Zoo `brats_mri_segmentation` (zero-shot) ★ | 3D whole-tumor seg | TCGA-LGG (10 test) | — | — | — | **Dice 0.9257 ± 0.031** | pretrained on ~500 BraTS cases |
| `segresnet_brats2021.pt` ★ (custom, recommended) | 3D whole-tumor seg | TCGA-LGG (10) / BraTS 2021 (188) | — | — | — | Dice **0.9206 ± 0.034** (TCGA test) / **0.9101 ± 0.116** (BraTS 2021 test); val 0.9289 | fine-tuned on 872 BraTS 2021 cases from `segresnet_tcga.pt` init; +1.05 pp on TCGA test, +5.33 pp on BraTS 2021 test vs `segresnet_tcga.pt`; cross-cohort drop only −1.05 pp (vs −5.33 pp for baseline) |
| `segresnet_tcga.pt` | 3D whole-tumor seg | TCGA-LGG (10) / BraTS 2021 (188) | — | — | — | Dice 0.9101 ± 0.036 (TCGA) / 0.8568 ± 0.154 (BraTS 2021) | small-cohort custom baseline (45 train cases) |
| `unet2d_tcga_v1.pt` | 2D whole-tumor seg | TCGA-LGG | — | — | — | Dice 0.7598 ± 0.090 (val 0.8063) | 2D legacy baseline |
| **Tumor binary classification** | | | | | | | |
| `mobilenetv3_ct_best.pt` ★ | 2D CT/MRI tumor vs healthy | Kaggle CT/MRI (val) | **96.4%** | **0.993** | — | — | val_loss 0.0864; only model in this branch |
| **IDH classification — Imaging** | | | | | | | |
| `densenet3d_idh.pt` | 3D IDH (GT-mask ROI, upper bound) | TCGA-LGG (val) | — | val 1.000 (GT-mask only) | — | — | optimistic ceiling, not deployable |
| `densenet3d_idh_jitter.pt` + Zoo bundle ★ | 3D IDH end-to-end (predicted-mask) | TCGA-LGG (10 E2E test) | **0.80** (E2E) | **0.75** (E2E); 0.9164 ± 0.073 (5-fold CV smoothed) | **0.69** (E2E) | GT-mask case AUC 1.000; jitter-trained for box noise | E2E threshold 0.13 from `e2e_idh_config.json` |
| `mobilenetv3_idh_best.pt` | 2D IDH (slice → case) | TCGA-LGG | — | 0.875 (case, single-split); 0.7639 ± 0.076 (5-fold CV smoothed) | — | val AUC 0.8533 | calibrated threshold 0.876 (Youden's J) |
| `mobilenetv3_idh_v3.pt` (early) | 2D IDH (intermediate) | TCGA-LGG (val) | — | val AUC 0.7522 | — | — | kept for comparison |
| **IDH classification — Molecular (RNA-seq pooled TCGA-GBM+LGG, n=759)** | | | | | | | |
| `molecular_idh/lightgbm.txt` ★ | Binary IDH (RNA-seq) | TCGA-GBM+LGG | — | **pooled CV 0.9924 ± 0.009**; LGG→GBM 0.965; GBM→LGG 0.956 | — | **GBM minority AUPRC 0.947**, recall@95spec 0.944, Brier 0.014 | best on pooled CV and GBM minority |
| `molecular_idh/logistic.joblib` | Binary IDH (RNA-seq) | TCGA-GBM+LGG | — | pooled CV 0.9916 ± 0.010; LGG→GBM 0.980; **GBM→LGG 0.972** | — | GBM minority AUPRC 0.933, recall@95spec 0.944, Brier 0.012 | best on cross-cohort GBM→LGG |
| `molecular_idh/mlp.pt` | Binary IDH (RNA-seq) | TCGA-GBM+LGG | — | pooled CV 0.9899 ± 0.009; **LGG→GBM 0.988**; GBM→LGG 0.954 | — | GBM minority AUPRC 0.939, Brier 0.015 | best on cross-cohort LGG→GBM |
| **IDH classification — Molecular multi-omics fusion (RNA-seq + methylation, strict subset n=615)** | | | | | | | |
| **Late fusion (per-modality LightGBM, mean of probs)** ★ | Binary IDH (RNA + methylation) | TCGA-GBM+LGG strict | — | pooled CV 0.9911 ± 0.008 | — | **GBM minority AUPRC 0.9502**, recall@95spec 0.944, Brier 0.012 | new repo best on GBM minority; beats RNA-seq-only LightGBM (0.9469, n=759) and early-concat LightGBM (0.9425). Reproduce: `scripts/exp_late_fusion_idh.py`, see `artifacts/molecular_idh_multimodal_eval/late_fusion_results.json` |
| `molecular_idh_multimodal/mlp.pt` | Binary IDH (RNA + methylation, early concat) | TCGA-GBM+LGG strict | — | **pooled CV 0.9933 ± 0.007**; **LGG→GBM 0.985**; GBM→LGG 0.974 | — | GBM minority AUPRC 0.940, Brier 0.013 | best on pooled CV and LGG→GBM transfer |
| `molecular_idh_multimodal/lightgbm.txt` | Binary IDH (RNA + methylation, early concat) | TCGA-GBM+LGG strict | — | pooled CV 0.9906 ± 0.009; LGG→GBM 0.976; GBM→LGG 0.964 | — | GBM minority AUPRC 0.943, recall@95spec 0.944, **Brier 0.0095** (best calibration) | early-concat baseline; superseded on GBM minority by late fusion above |
| `molecular_idh_multimodal/logistic.joblib` | Binary IDH (RNA + methylation, early concat) | TCGA-GBM+LGG strict | — | pooled CV 0.9904 ± 0.010; LGG→GBM 0.969; **GBM→LGG 0.977** | — | GBM minority AUPRC 0.914, Brier 0.010 | best on GBM→LGG transfer; smallest model |
| **Detection** | | | | | | | |
| `yolov8_brain_tumor_best.pt` ★ | 2D tumor bbox detection | Ultralytics (val) | — | — | — | **mAP50 0.476**, mAP50-95 0.347 | precision 0.451, recall 0.824 |

### Reading the table

- **Accuracy** is reported only when a model has a fixed deployment threshold (`mobilenetv3_ct_best`, end-to-end IDH). Pure classifiers without a chosen threshold report AUC instead, since accuracy depends on the cutoff.
- **macro F1** is reported only for the end-to-end IDH pipeline because that is where the threshold from `e2e_idh_config.json` is applied. Adding a similar number for the standalone classifiers would require re-applying their stored thresholds, which has not been done in this snapshot.
- **AUC** is the standard primary metric for the binary classifiers. For molecular models, three AUC values are reported (pooled 5-fold CV, LGG→GBM source-holdout, GBM→LGG source-holdout) so that cohort confound is visible — see B3 strategy in `docs/MODEL_CARD.md`.
- The molecular multi-omics fusion row gives essentially noise-level AUC change vs the RNA-seq-only baseline (pooled CV +0.0009, GBM minority AUPRC −0.0044 with smaller test n=210), but a real Brier improvement (0.014 → 0.0095, ≈31% better calibration). See `docs/MODEL_CARD.md` for the full B3 side-by-side.

## 7. Caveats

- None of the evaluation scripts were re-run in this session, so test-set metrics are documented from repository artifacts and notes rather than a fresh benchmark pass
- IDH metrics are based on a very small labeled cohort with strong class imbalance
- The local YOLO checkpoint metadata does not match the current training script defaults, so future retraining may not reproduce the same result unless arguments are pinned explicitly
- The MONAI Model Zoo bundle is an external pretrained asset and should be tracked separately from custom-trained checkpoints when reporting provenance
