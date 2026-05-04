# Model Card

## Molecular IDH classifier

### Overview

- Task: binary IDH mutation status classification (`0 = wildtype`, `1 = mutant`) from RNA-seq expression.
- Features: `log2(TPM+1)` expression with fold-aware selection (`top-K variance ∪ prior panel`).
- Cohort used for this run: pooled TCGA-GBM + TCGA-LGG molecular data prepared under `artifacts/molecular/`.
- Models: `LogisticIDH`, `LightGBMIDH`, `MLPIDH`.

### B3 metrics (real eval run, `--mode all --folds 5 --seed 42`)

#### 1) Pooled 5-fold stratified CV

| Model | Mean AUC | AUC std | Mean AUPRC | AUPRC std |
|---|---:|---:|---:|---:|
| Logistic | 0.9916 | 0.0098 | 0.9899 | 0.0135 |
| LightGBM | 0.9924 | 0.0089 | 0.9896 | 0.0152 |
| MLP | 0.9899 | 0.0085 | 0.9876 | 0.0118 |

#### 2) Source-holdout (bidirectional)

| Direction | Model | AUC | AUPRC |
|---|---|---:|---:|
| Train LGG -> Test GBM | Logistic | 0.9801 | 0.9514 |
| Train LGG -> Test GBM | LightGBM | 0.9650 | 0.9290 |
| Train LGG -> Test GBM | MLP | 0.9880 | 0.9458 |
| Train GBM -> Test LGG | Logistic | 0.9722 | 0.9891 |
| Train GBM -> Test LGG | LightGBM | 0.9555 | 0.9868 |
| Train GBM -> Test LGG | MLP | 0.9539 | 0.9846 |

#### 3) Minority metrics on GBM subset (from pooled CV predictions)

| Model | n (GBM test predictions) | AUPRC | Recall @ 95% specificity | Brier score |
|---|---:|---:|---:|---:|
| Logistic | 250 | 0.9326 | 0.9444 | 0.0121 |
| LightGBM | 250 | 0.9469 | 0.9444 | 0.0138 |
| MLP | 250 | 0.9386 | 0.9444 | 0.0151 |

### Intended use

- Research-only molecular baseline for IDH discrimination from TCGA transcriptomics.
- Cross-modality reference signal to compare with the imaging IDH pipeline when paired data exists.
- Offline analysis when MRI is unavailable.

### Out-of-scope use

- Any clinical diagnosis or treatment decision support.
- Prospective deployment on non-TCGA cohorts without external validation.
- Use as a replacement for pathology/genomics workflows in patient care.
- CNV/survival/subtype claims, which are not part of this release.

### Multi-omics fusion (RNA-seq + methylation)

#### B3 side-by-side comparison (real eval run, `--mode all --folds 5 --seed 42`)

| Report | RNA-seq-only baseline | Multi-omics fusion | Lift (multi - baseline) |
|---|---:|---:|---:|
| Pooled 5-fold CV best AUC | 0.9924 (LightGBM) | 0.9933 (MLP) | +0.0009 |
| Source holdout AUC: LGG -> GBM (best model) | 0.9880 (MLP) | 0.9847 (MLP) | -0.0034 |
| Source holdout AUC: GBM -> LGG (best model) | 0.9722 (Logistic) | 0.9772 (Logistic) | +0.0050 |
| GBM minority AUPRC (best model) | 0.9469 (LightGBM, n=250) | 0.9425 (LightGBM, n=210) | -0.0044 |
| GBM minority recall @95% specificity (best model) | 0.9444 | 0.9444 | +0.0000 |
| GBM minority Brier (best model) | 0.0138 | 0.0095 | -0.0043 |

Observed outcome on this cohort: pooled CV improved slightly, GBM->LGG transfer improved, but LGG->GBM and GBM minority AUPRC did not improve.

### Risks and limitations (verbatim from spec)

1. **Smaller training set in strict mode** — multi-modal subset is ~600 vs RNA-seq-only 759, increasing variance.
2. **HM27 platform ceiling** — intersecting to HM27 caps methylation resolution at ~27K CpGs vs HM450's ~485K.
3. **Batch effects across aliquots** — methylation and RNA-seq may come from different sample/aliquot for the same patient.
4. **M-value clipping** — extreme beta values (~0 or ~1) lose information after the `[0.001, 0.999]` clip.
5. **Cohort confound persists** — same diagnosis caveat as RNA-seq-only B3; pooled CV remains LGG-vs-GBM susceptible.
