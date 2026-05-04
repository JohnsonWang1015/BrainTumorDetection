# Dataset Card

## 1. Scope

This repository currently uses three active imaging datasets, two generated molecular cohorts, one multi-source IDH manifest, and one legacy dataset that has been retired from the training flow:

| Dataset | Primary use | Local source | Status |
|---|---|---|---|
| BraTS-TCGA-LGG | MRI segmentation and IDH mutation classification | `datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations` | Active, primary imaging dataset |
| Kaggle multimodal CT/MRI dataset | 2D CT/MRI binary tumor classification | `datasets/Kaggle_multimodal/Dataset` | Active |
| Multi-source IDH manifest v2 | Unified schema for TCGA-LGG / TCGA-GBM / UCSF-PDGM / EGD | `artifacts/manifest_v2.json` | Active schema, currently populated from local TCGA-LGG only |
| TCGA molecular IDH cohort (RNA-seq) | Pooled molecular IDH classification | `artifacts/molecular/` (built from `datasets/TCGA-GBM`, `datasets/TCGA-LGG-Molecular`) | Active |
| TCGA multi-omics IDH cohort (RNA-seq + methylation) | Multi-modal molecular IDH fusion | `artifacts/molecular_multimodal/` | Active |
| MRIBrainTumor | Earlier segmentation experiments | `datasets/MRIBrainTumor` | Present locally but no longer used in current training pipeline |

This card reflects the repository state inspected on 2026-05-04 from:

- `artifacts/manifest.json`
- `artifacts/manifest_v2.json`
- `artifacts/ct_manifest.json`
- `artifacts/idh_labels.csv`
- `artifacts/molecular/{cohort_manifest.json, idh_labels.parquet, expression_matrix.parquet, feature_panel.json}`
- `artifacts/molecular_multimodal/{cohort_manifest.json, idh_labels.parquet, expression_matrix.parquet, methylation_matrix.parquet, feature_panel.json}`
- training/data preparation code under `src/idh_glioma/data/` and `src/idh_glioma/molecular/`
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

## 4. Multi-Source IDH Expansion Status

### 4.1 Intended use

- unify multiple pre-operative glioma cohorts under one manifest contract
- support pooled or source-held-out IDH experiments
- preserve compatibility with the current training code while adding dataset provenance

The active builder is:

- `src/idh_glioma/data/prepare_idh_multisource.py`

The contract is:

- `configs/idh_manifest_v2_contract.yaml`

### 4.2 Supported source datasets

The current importer supports these source identifiers:

| Source | Expected local root | Intended role |
|---|---|---|
| `brats_tcga_lgg` | `datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations` | primary local training cohort |
| `tcga_gbm` | `datasets/TCGA-GBM` | supplement IDH-wildtype coverage |
| `ucsf_pdgm` | `datasets/UCSF-PDGM` | external cohort first, optional later training |
| `egd` | `datasets/EGD` | larger mixed-cohort expansion |

### 4.3 Current generated status

Observed from the committed `artifacts/manifest_v2.json`:

| Field | Value |
|---|---|
| `manifest_version` | `0.2.0` |
| `split_strategy.scheme` | `source_holdout` |
| `cohorts` | `1` |
| populated source datasets | `brats_tcga_lgg` only |
| split sizes | `train=45`, `val=10`, `test=10` |

Interpretation:

- the multi-source importer is implemented and versioned
- the current local workspace did not yet contain `TCGA-GBM`, `UCSF-PDGM`, or `EGD` in importer-readable form when `manifest_v2.json` was generated
- therefore the committed manifest is still effectively a TCGA-LGG-only cohort with richer provenance fields

### 4.4 Added schema fields in manifest v2

Compared with `artifacts/manifest.json`, `manifest_v2` adds:

- `record_id`
- `source_dataset`
- `source_subject_id`
- `cohort_id`
- `acquisition_stage`
- `mask_kind`
- `label_source`
- `label_confidence`
- `inclusion_flags`
- `qc_flags`
- `provenance`

These fields are intended to make future cross-dataset training auditable instead of silently mixing cohorts.

## 5. TCGA Molecular IDH Cohort (RNA-seq)

### 5.1 Intended use

- pooled molecular IDH classification across TCGA-GBM and TCGA-LGG
- runs entirely on tabular omics features and is independent from the imaging pipeline
- consumed by `train-idh-molecular` and `eval-idh-molecular`

### 5.2 Source data

- `datasets/TCGA-GBM/tcga_gbm_downloads/data/` — GDC RNA-seq STAR-counts and public masked MAF for IDH1/IDH2
- `datasets/TCGA-LGG-Molecular/tcga_lgg_downloads/data/` — same modality structure for the LGG cohort
- builder: `src/idh_glioma/molecular/prepare_dataset.py` (entry point `prepare-idh-molecular`)

### 5.3 Generated artifacts

`artifacts/molecular/` holds the prepared cohort:

| File | Meaning |
|---|---|
| `expression_matrix.parquet` | wide gene × patient matrix; values are `log2(TPM+1)`; index is base ENSG (version stripped) |
| `idh_labels.parquet` | per-patient IDH label aggregated from public masked MAF (`1 = IDH1/IDH2 missense mutant`, `0 = wildtype`) |
| `cohort_manifest.json` | per-source and pooled summaries plus per-patient provenance |
| `feature_panel.json` | feature-selection contract: `top-K variance ∪ curated prior gene panel` |
| `gene_metadata.parquet` | gene symbol / biotype lookup table |

### 5.4 Cohort size and label distribution

Observed from the current `artifacts/molecular/cohort_manifest.json`:

| Source | Labeled patients | With expression | Wildtype | Mutant |
|---|---:|---:|---:|---:|
| TCGA-GBM | 371 | 250 | 347 | 24 |
| TCGA-LGG | 509 | 509 | 95 | 414 |
| **Pooled (used for training)** | **880** | **759** | **442** | **438** |

Additional facts:

- expression matrix shape: `60616` genes × `809` expression patients
- pooled labeled-with-expression patients: `759` (training and evaluation are restricted to this intersection)
- the pooled label distribution is roughly balanced (`442` WT vs `438` mutant), but per-source it is highly imbalanced — GBM is `~93%` wildtype and LGG is `~81%` mutant
- 90 GBM patients had multiple primary aliquots that were collapsed during preparation; 14 LGG patients similarly

### 5.5 Feature panel

- strategy: `variance_top_k_union_prior_panel` with `default_top_k = 2000`
- prior panel: ~160 curated glioma-relevant gene symbols (e.g. `IDH1`, `IDH2`, `ATRX`, `TP53`, `EGFR`, `MGMT`, immune and stromal markers)
- selection is fold-aware: for each CV fold the top-K is recomputed on the training half before being unioned with the prior panel

### 5.6 Known risks and limitations

- IDH labels are derived from public masked MAF mutation calls only, so silent or non-coding events are not represented
- the GBM minority subgroup (`~24` mutants) carries most of the cross-cohort signal — small absolute count means CV variance for GBM-dominant metrics
- pooled metrics can hide cohort confounding: the B3 evaluation strategy (`pooled_cv` + `source_holdout` + `minority_metrics`) is the recommended way to read this dataset

## 6. TCGA Multi-omics IDH Cohort (RNA-seq + Methylation)

### 6.1 Intended use

- multi-modal molecular IDH classification combining RNA-seq with DNA methylation
- consumed by `prepare-idh-molecular --modalities rnaseq methylation`, `train-idh-molecular --modalities rnaseq methylation`, and `eval-idh-molecular --modalities rnaseq methylation`

### 6.2 Source data

- same TCGA-GBM and TCGA-LGG molecular drops as Section 5
- adds GDC methylation arrays: HM27 (older 27k probes, mostly TCGA-GBM) and HM450 (450k probes, mostly TCGA-LGG)
- methylation loader at `src/idh_glioma/molecular/methylation.py` builds a unified probe intersection per platform

### 6.3 Generated artifacts

`artifacts/molecular_multimodal/`:

| File | Meaning |
|---|---|
| `expression_matrix.parquet` | RNA-seq matrix restricted to the multimodal intersection cohort |
| `methylation_matrix.parquet` | beta-value matrix on the platform intersection; missing probes filled before downstream use |
| `idh_labels.parquet` | IDH labels for the strict subset |
| `cohort_manifest.json` | per-source RNA-seq and methylation summaries plus the strict multimodal intersection list |
| `feature_panel.json` | per-modality contract; RNA-seq uses gene-symbol prior panel, methylation uses a 50-CpG prior panel |
| `gene_metadata.parquet` | gene metadata shared with the RNA-seq cohort |

### 6.4 Cohort size and label distribution

Observed from the current `artifacts/molecular_multimodal/cohort_manifest.json`:

| Source | Methylation patients | Methylation platforms | Labeled-with-expression |
|---|---:|---|---:|
| TCGA-GBM | 423 | HM27 = 283, HM450 = 140 | 250 |
| TCGA-LGG | 409 | HM27 = 0, HM450 = 409 | 509 |

Strict multimodal intersection (patients with **both** RNA-seq and methylation **and** an IDH label):

| Field | Value |
|---|---:|
| `strict_subset_size` | 615 |
| pooled wildtype | 272 |
| pooled mutant | 343 |
| `rnaseq_only_dropped` | ≈ 144 patients (those in the RNA-seq cohort but missing methylation) |

### 6.5 Feature panel

- strategy: `per_modality_variance_top_k_union_prior_panel` with `default_top_k = {rnaseq: 2000, methylation: 2000}`
- RNA-seq prior: same ~160 gene symbols as the RNA-seq-only cohort
- methylation prior: 50 curated CpG IDs covering G-CIMP / IDH-related methylation markers
- selection is again fold-aware and applied per modality before fusion

### 6.6 Cross-platform handling

- HM27 and HM450 share `~26K` overlapping CpGs (`intersection_size = 25978` for TCGA-GBM)
- TCGA-LGG is HM450-only with `~482K` CpGs available; intersection with GBM keeps only the shared overlap
- patients with multiple methylation aliquots are collapsed the same way as RNA-seq aliquots
- NaN cells from platform mismatch are tracked in the manifest (`nan_fill_count`) and filled before model training

### 6.7 Known risks and limitations

- the multi-omics strict subset (`n=615`) is smaller than the RNA-seq pooled cohort (`n=759`) — direct AUC comparisons must use the same denominator
- HM27 and HM450 differ in probe density and chemistry; treating them as a single feature space introduces platform shift that the prior CpG panel partially mitigates but does not eliminate
- the GBM minority IDH-mutant count drops from `24` (RNA-seq) to a slightly smaller number after intersection, so multi-omics minority metrics have higher variance than RNA-seq-only minority metrics

## 7. Legacy Dataset Present but Not Active

`datasets/MRIBrainTumor` is still present locally, but repository notes explicitly say it has been removed from the current training path because it is single-modality and does not match the 4-channel BraTS-style pipeline.

Implication:

- do not treat `MRIBrainTumor` as part of the current benchmarked training setup
- if it is reintroduced, it should be handled as a separate 1-channel branch with separate documentation and metrics

## 8. Recommended Interpretation

- Use BraTS-TCGA-LGG as the authoritative dataset for imaging segmentation and imaging-based IDH work in this repository
- Use `artifacts/molecular/` (RNA-seq) and `artifacts/molecular_multimodal/` (RNA-seq + methylation) as the molecular IDH cohorts; metrics from the imaging path and the molecular path are not directly comparable because the cohorts and labels differ (BraTS imaging cases vs full TCGA molecular drops)
- Treat `artifacts/manifest_v2.json` as the forward-compatible schema for future imaging IDH expansion, but not yet as evidence that external imaging cohorts are already populated locally
- Use the Kaggle multimodal dataset only for the standalone CT/MRI tumor classifier
- Treat all reported imaging IDH metrics as low-sample research results; molecular IDH metrics are larger-sample but still TCGA-only and inherit any TCGA cohort biases
- Preserve `artifacts/manifest.json`, `artifacts/ct_manifest.json`, and the molecular cohort manifests with the checkpoints they produced, because the local split composition directly affects the reported results
