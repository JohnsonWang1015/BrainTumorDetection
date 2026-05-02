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
| `checkpoints/segresnet_tcga.pt` | 72M | 2026-04-30 16:53 | 3D MRI segmentation | Present |
| `checkpoints/densenet3d_idh.pt` | 44M | 2026-04-30 17:25 | 3D IDH classifier | Present |
| `checkpoints/densenet3d_idh_jitter.pt` | 44M | 2026-04-30 18:26 | 3D IDH classifier with bbox jitter | Present |
| `checkpoints/mobilenetv3_ct_best.pt` | 6.0M | 2026-03-21 14:39 | CT/MRI tumor classifier | Present |
| `checkpoints/yolov8_brain_tumor_best.pt` | 6.0M | 2026-03-28 14:52 | YOLO detector | Present |
| `checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt` | external bundle | vendor asset | MONAI Model Zoo segmentation | Present |

## 3. Recommended Models

Based on the repository's own reported outcomes and the checkpoints present locally:

- Recommended segmentation model: MONAI Model Zoo `brats_mri_segmentation` bundle
- Recommended end-to-end IDH pipeline: MONAI Model Zoo segmentation + `densenet3d_idh_jitter.pt` + `artifacts/e2e_idh_config.json`
- `unet2d_tcga_v1.pt` for 2D segmentation baseline
- `mobilenetv3_idh_best.pt` for 2D IDH baseline
- `segresnet_tcga.pt` for custom-trained 3D segmentation baseline
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
- 3D custom segmentation and both 3D IDH checkpoints were updated on 2026-04-30
- the main 2D IDH checkpoint is calibrated with a stored threshold and paired with 5-fold CV results
- the jitter-trained 3D IDH checkpoint is paired with 5-fold CV results, while the deployed runtime threshold now lives in `artifacts/e2e_idh_config.json`
- the deployed end-to-end decision rule is now anchored by `artifacts/e2e_idh_config.json` with `threshold=0.13` and ROI postprocess settings shared across app/eval/infer
- the CT/MRI classifier and YOLO detector were trained earlier in March 2026 and remain available

### 5.2 Best available options by task

| Task | Best local choice | Why |
|---|---|---|
| MRI segmentation | MONAI Model Zoo bundle | highest reported Dice |
| 2D segmentation baseline | `unet2d_tcga_v1.pt` | simplest custom baseline |
| 3D segmentation baseline | `segresnet_tcga.pt` | strong custom 3D baseline |
| 2D IDH baseline | `mobilenetv3_idh_best.pt` | calibrated, metadata-complete, CV-backed |
| 3D IDH upper bound | `densenet3d_idh.pt` | perfect val/test report under GT-mask ROI |
| 3D IDH practical custom model | `densenet3d_idh_jitter.pt` | trained for predicted-mask ROI noise |
| CT/MRI tumor classification | `mobilenetv3_ct_best.pt` | only active classifier for that branch |
| Tumor detection | `yolov8_brain_tumor_best.pt` | only local YOLO checkpoint present |

## 6. Caveats

- None of the evaluation scripts were re-run in this session, so test-set metrics are documented from repository artifacts and notes rather than a fresh benchmark pass
- IDH metrics are based on a very small labeled cohort with strong class imbalance
- The local YOLO checkpoint metadata does not match the current training script defaults, so future retraining may not reproduce the same result unless arguments are pinned explicitly
- The MONAI Model Zoo bundle is an external pretrained asset and should be tracked separately from custom-trained checkpoints when reporting provenance
