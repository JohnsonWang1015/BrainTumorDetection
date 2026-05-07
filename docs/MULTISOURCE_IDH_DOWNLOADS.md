# Multi-source IDH imaging cohort downloads

This file is the recipe for populating the imaging cohorts that
`uv run prepare-idh-multisource` expects. Today only `brats_tcga_lgg` is
downloaded locally; the others are schema-supported but missing on disk, which
is why `artifacts/manifest_v2.json` still resolves to a TCGA-LGG-only cohort.

The importer warns about missing sources at the end of every run. Pass
`--strict` to convert those warnings into a non-zero exit code (useful in CI
or in scripted preparation pipelines).

## Cohort table

| Source key | Local path expected by importer | What it is | Access |
|---|---|---|---|
| `brats_tcga_lgg` | `datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations` | TCGA-LGG pre-operative MRI + GlistrBoost masks | Public TCIA collection |
| `tcga_gbm` | `datasets/TCGA-GBM/<case_id>/...nii.gz` | TCGA-GBM **imaging** (BraTS-TCGA-GBM Pre-operative). The local `datasets/TCGA-GBM/` today only has molecular drops, **not imaging** | Public TCIA collection |
| `ucsf_pdgm` | `datasets/UCSF-PDGM/<case_id>/...nii.gz` | UCSF Preoperative Diffuse Glioma MRI | **Credentialed access** required (TCIA agreement) |
| `egd` | `datasets/EGD/<case_id>/...nii.gz` | Erasmus Glioma Database | License agreement (free for research, registration required) |

## 1. `brats_tcga_lgg` (already done)

Already present at `datasets/BraTS-TCGA-LGG/`. No action needed.

## 2. `tcga_gbm` imaging

Important: the existing `datasets/TCGA-GBM/tcga_gbm_downloads/` is the GDC
**molecular** drop (RNA-seq counts, MAF, methylation, clinical) used by the
RNA-seq pipeline. The imaging cohort has to come from TCIA separately.

Source collection: TCIA `BraTS-TCGA-GBM` (a.k.a. `Pre-operative_TCGA_GBM_NIfTI_and_Segmentations`).

```bash
# Option A — TCIA NBIA Data Retriever (GUI)
#   1. Go to https://www.cancerimagingarchive.net/collection/tcga-gbm/
#   2. Filter to "Pre-operative TCGA GBM, NIfTI" subset
#   3. Download the .tcia manifest, open it with NBIA Data Retriever
#   4. Set output dir to datasets/TCGA-GBM-Imaging/
#   5. Symlink or rename to datasets/TCGA-GBM/<case_id>/ to satisfy the importer

# Option B — aspera CLI (faster, ~30 GB)
ascp -QT -l 200M -P 33001 -i <aspera-key> \
  --user=tcia.aspera@cancerimagingarchive.net \
  --host=tcia-aspera.cancerimagingarchive.net \
  /collections/TCGA-GBM/Pre-operative datasets/TCGA-GBM/
```

The importer expects per-case folders named `TCGA-XX-XXXX/` containing
`*_flair.nii.gz`, `*_t1.nii.gz`, `*_t1ce.nii.gz` (treated as `t1Gd`),
`*_t2.nii.gz`, and optionally a segmentation mask matching one of
`MASK_SUFFIXES` in `prepare_idh_multisource.py`.

For IDH labels, reuse the existing molecular MAF labels by exporting them as a
join table:

```bash
uv run python -c "
import pandas as pd
labels = pd.read_parquet('artifacts/molecular/idh_labels.parquet')
labels = labels.rename(columns={'patient_id': 'case_id'})[['case_id', 'idh_label']]
labels.to_csv('artifacts/idh_labels_tcga_combined.csv', index=False)
"
```

Then:

```bash
uv run prepare-idh-multisource \
  --include-sources brats_tcga_lgg tcga_gbm \
  --idh-labels artifacts/idh_labels_tcga_combined.csv \
  --output artifacts/manifest_v2.json \
  --strict
```

## 3. `ucsf_pdgm`

Source collection: TCIA `UCSF-PDGM` — credentialed access. You must:

1. Register on TCIA and request access to UCSF-PDGM (~24 hour turnaround).
2. Sign the data use agreement.
3. Download via NBIA Data Retriever; the bundle is ~80 GB.

Expected layout (importer aliases `T2FLAIR`, `T1c_bias`/`T1gad_bias`,
`tumor_segmentation`):

```text
datasets/UCSF-PDGM/
  UCSF-PDGM-0001/
    UCSF-PDGM-0001_T2FLAIR.nii.gz
    UCSF-PDGM-0001_T1.nii.gz
    UCSF-PDGM-0001_T1c_bias.nii.gz
    UCSF-PDGM-0001_T2.nii.gz
    UCSF-PDGM-0001_tumor_segmentation.nii.gz
```

IDH labels live in the UCSF clinical metadata CSV; the importer accepts
columns `ID, IDH status` (values `Mutant` or `Wildtype`).

## 4. `egd`

Source: <https://xnat.bmia.nl/data/projects/egd> (Erasmus Glioma Database).

1. Register an account.
2. Accept the license.
3. Download the structural MRI bundle (~50 GB).

Expected per-case layout (already final filenames, no aliasing needed):

```text
datasets/EGD/
  EGD-0001/
    FLAIR.nii.gz
    T1.nii.gz
    T1GD.nii.gz
    T2.nii.gz
```

IDH labels are in `Genetic_and_Histological_labels.csv`; the importer accepts
`subject, IDH mutation status` with values in `{0, 1, -1}` (`-1` is treated
as missing and dropped).

## After populating any source

Re-run the importer with `--strict` so missing cohorts no longer fall through
silently:

```bash
uv run prepare-idh-multisource \
  --include-sources brats_tcga_lgg tcga_gbm ucsf_pdgm egd \
  --split-mode source_holdout \
  --idh-labels artifacts/idh_labels_multisource.csv \
  --output artifacts/manifest_v2.json \
  --strict
```

The `source_holdout` split mode currently keeps `ucsf_pdgm_external` as the
held-out test cohort; train/val are drawn from the rest.

## Known gotchas

- **TCGA-GBM imaging vs molecular** — the existing `datasets/TCGA-GBM/`
  contains GDC molecular drops, not imaging. Do not delete it; the molecular
  pipeline depends on it.
- **Patient ID overlap** — TCGA-GBM molecular and TCGA-GBM imaging share
  patient IDs but the imaging cohort has a smaller intersection (~150
  patients). Expect drop-out when joining.
- **UCSF-PDGM modality names** — older releases used `T1c` instead of
  `T1c_bias`; both are aliased to `t1Gd` in the importer.
