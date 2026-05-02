# Dataset Card

## 1. Scope

This repository currently uses two active datasets plus one legacy dataset that has been retired from the training flow:

| Dataset | Primary use | Local source | Status |
|---|---|---|---|
| BraTS-TCGA-LGG | MRI segmentation and IDH mutation classification | `datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations` | Active, primary research dataset |
| Kaggle multimodal CT/MRI dataset | 2D CT/MRI binary tumor classification | `datasets/Kaggle_multimodal/Dataset` | Active |
| MRIBrainTumor | Earlier segmentation experiments | `datasets/MRIBrainTumor` | Present locally but no longer used in current training pipeline |

This card reflects the repository state inspected on 2026-05-01 from:

- `artifacts/manifest.json`
- `artifacts/ct_manifest.json`
- `artifacts/idh_labels.csv`
- training/data preparation code under `src/idh_glioma/data/`
- project notes in `README.md` and `CLAUDE.md`

## 2. BraTS-TCGA-LGG Dataset

### 2.1 Intended use

- 2D U-Net whole-tumor segmentation
- 3D MONAI SegResNet whole-tumor segmentation
- 2D slice-based IDH mutation classification
- 3D volume-based IDH mutation classification
- End-to-end segmentation + IDH inference

### 2.2 Local file structure

Each valid case folder is expected to contain:

- `*_flair.nii.gz`
- `*_t1.nii.gz`
- `*_t1Gd.nii.gz`
- `*_t2.nii.gz`
- `*_GlistrBoost_ManuallyCorrected.nii.gz`
- `*_GlistrBoost.nii.gz`

Mask selection priority is:

- prefer `*_GlistrBoost_ManuallyCorrected.nii.gz`
- otherwise fall back to `*_GlistrBoost.nii.gz`

The manifest stores relative paths to the four modalities under `modalities` and the selected mask under `mask_path`.

### 2.3 Manifest schema

Each record in `artifacts/manifest.json` contains:

| Field | Meaning |
|---|---|
| `case_id` | TCGA case identifier |
| `date` | scan date parsed from filename |
| `modalities` | relative paths for `flair`, `t1`, `t1Gd`, `t2` |
| `mask_path` | selected tumor mask path |
| `idh_label` | `0 = wildtype`, `1 = mutant`, `null = missing label` |

### 2.4 Dataset size and split status

Observed from the current `artifacts/manifest.json`:

| Split | Cases | IDH-labeled cases | IDH mutant | IDH wildtype |
|---|---:|---:|---:|---:|
| train | 45 | 44 | 37 | 7 |
| val | 10 | 10 | 8 | 2 |
| test | 10 | 10 | 8 | 2 |
| total | 65 | 64 | 53 | 11 |

Additional observations:

- 1 case is missing an IDH label: `TCGA-DU-7014`
- 62/65 cases use `GlistrBoost_ManuallyCorrected`
- 3/65 cases fall back to `GlistrBoost`
- split generation is stratified on available `idh_label` values with `random_state=42`
- unlabeled cases are appended to the train split by design

### 2.5 Label provenance

- `artifacts/idh_labels.csv` is the local join table used during manifest generation
- repository notes state the labels were populated from cBioPortal study `lgggbm_tcga_pub` using `IDH_STATUS`
- the current repository snapshot contains 64 matched labels out of 65 manifest cases

### 2.6 Preprocessing by task

For segmentation:

- 4-channel input: `flair`, `t1`, `t1Gd`, `t2`
- per-volume z-score normalization in the 2D pipeline
- binary mask target using tumor vs background
- 2D training augmentations include flips, 90-degree rotations, and intensity jitter
- 3D MONAI pipeline uses orientation normalization, spacing to `(1.0, 1.0, 1.0)`, channel-wise nonzero intensity normalization, random crop, and 3-axis flips

For IDH classification:

- 2D classifier uses 3-channel slices: `flair`, `t1Gd`, `t2`
- 2D classifier can crop each slice to the tumor ROI, with current trained checkpoints using `roi_margin=10`
- 3D classifier crops a 3D tumor bounding box from the 4 modalities, normalizes each channel with z-score, then resizes to `96 x 96 x 96`
- the jitter-trained 3D classifier expands and shifts the training ROI to simulate noisy predicted masks

### 2.7 Known risks and limitations

- Severe class imbalance for IDH labels: only 11 wildtype cases vs 53 mutant cases
- Very small validation and test splits: only 10 labeled cases each
- One case lacks IDH label and therefore cannot contribute to IDH evaluation
- Three masks are not manually corrected
- Current metrics may be high-variance because the cohort is small

## 3. Kaggle Multimodal CT/MRI Dataset

### 3.1 Intended use

- 2D binary classification: tumor vs healthy
- checkpoint output: `checkpoints/mobilenetv3_ct_best.pt`

### 3.2 Expected local structure

The preparation script expects:

- `datasets/Kaggle_multimodal/Dataset/Brain Tumor CT scan Images/Tumor`
- `datasets/Kaggle_multimodal/Dataset/Brain Tumor CT scan Images/Healthy`
- `datasets/Kaggle_multimodal/Dataset/Brain Tumor MRI images/Tumor`
- `datasets/Kaggle_multimodal/Dataset/Brain Tumor MRI images/Healthy`

Supported image extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

### 3.3 Manifest schema

Each record in `artifacts/ct_manifest.json` contains:

| Field | Meaning |
|---|---|
| `path` | image path |
| `label` | `1 = tumor`, `0 = healthy` |
| `modality` | `ct` or `mri` |

### 3.4 Dataset size and split status

Observed from the current `artifacts/ct_manifest.json`:

| Split | Images | Tumor | Healthy | CT | MRI |
|---|---:|---:|---:|---:|---:|
| train | 6732 | 3722 | 3010 | 3232 | 3500 |
| val | 1443 | 798 | 645 | 693 | 750 |
| test | 1443 | 798 | 645 | 693 | 750 |
| total | 9618 | 5318 | 4300 | 4618 | 5000 |

Split behavior in `prepare_ct_data.py`:

- data is grouped by `(label, modality)`
- each group is shuffled with `seed=42`
- split ratios default to `0.15` validation and `0.15` test

### 3.5 Preprocessing

- RGB input resized to the classifier image size, currently `224 x 224` in training
- ImageNet-pretrained MobileNetV3 preprocessing assumptions are used in the model pipeline
- training augmentation includes horizontal flip, small rotation, and color jitter

### 3.6 Known risks and limitations

- This dataset is for 2D image-level classification only and is not used for volumetric MRI segmentation or IDH prediction
- Mixing CT and MRI in one binary classifier may create domain-shift sensitivity
- Public dataset license terms are not recorded inside this repository and should be checked at the original source before redistribution

## 4. Legacy Dataset Present but Not Active

`datasets/MRIBrainTumor` is still present locally, but repository notes explicitly say it has been removed from the current training path because it is single-modality and does not match the 4-channel BraTS-style pipeline.

Implication:

- do not treat `MRIBrainTumor` as part of the current benchmarked training setup
- if it is reintroduced, it should be handled as a separate 1-channel branch with separate documentation and metrics

## 5. Recommended Interpretation

- Use BraTS-TCGA-LGG as the authoritative dataset for segmentation and IDH work in this repository
- Use the Kaggle multimodal dataset only for the standalone CT/MRI tumor classifier
- Treat all reported IDH metrics as low-sample research results rather than deployment-ready evidence
- Preserve `artifacts/manifest.json` and `artifacts/ct_manifest.json` with the checkpoints they produced, because the local split composition directly affects the reported results
