# Dataset Card

## TCGA-GBM-Molecular + TCGA-LGG-Molecular

### Sources

- TCGA-GBM project: https://portal.gdc.cancer.gov/projects/TCGA-GBM
- TCGA-LGG project: https://portal.gdc.cancer.gov/projects/TCGA-LGG
- Data types used in v1 molecular pipeline:
  - RNA-seq gene expression quantification (`augmented_star_gene_counts.tsv`)
  - Public masked somatic mutation MAF (`aliquot_ensemble_masked.maf.gz`)
  - Clinical supplement XML (cohort description only, not model features)

### Current Local Counts (real run, 2026-05-04)

| Source | RNA-seq files on disk | MAF files on disk | Clinical files on disk | Expression patients (after aliquot dedup) | MAF-labeled patients | Labeled+Expression |
|---|---:|---:|---:|---:|---:|---:|
| TCGA-GBM-Molecular | 391 | 468 (public masked) | 599 | 293 | 371 | 250 |
| TCGA-LGG-Molecular | 534 | 530 | 537 | 516 | 509 | 509 |
| Pooled | 925 | 998 | 1136 | 809 | 880 | 759 |

### IDH Label Distribution (from MAF aggregation)

| Source | IDH-mutant (`1`) | IDH-wildtype (`0`) | Mutant fraction |
|---|---:|---:|---:|
| TCGA-GBM-Molecular | 24 | 347 | 6.5% |
| TCGA-LGG-Molecular | 414 | 95 | 81.3% |
| Pooled labels | 438 | 442 | 49.8% |

### Notes

- Labels are derived from any `IDH1/IDH2` `Missense_Mutation` found per patient across all available public masked MAF files.
- Expression matrix stores `log2(TPM+1)` with base ENSG IDs (version suffix stripped).
- Multiple RNA-seq aliquots per patient are deduplicated by deterministic UUID ordering in v1 and flagged in `cohort_manifest.json`.
