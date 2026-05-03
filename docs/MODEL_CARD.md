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
- Multi-omics fusion claims (methylation/CNV/survival/subtype), which are not part of v1.

### Risks and limitations (verbatim from spec)

1. **Cohort confound** — pooled training mixes LGG (mostly IDH-mutant) and GBM (mostly IDH-wildtype). The model may learn cohort identity rather than true IDH biology. The source-holdout metric is the key diagnostic.
2. **Retrospective TCGA cohort** — not prospectively collected; subject to selection bias; not validated for clinical decision-making.
3. **Population specificity** — IDH-mutant cases concentrate in LGG. Performance in IDH-wildtype-only populations is untested and likely degraded.
4. **Aliquot selection sensitivity** — the primary-tumor + alphabetically-first rule is an arbitrary tiebreaker; not ablated in v1.
5. **Public MAF only** — controlled-access MAFs would add validation power but require GDC token; v1 uses only `aliquot_ensemble_masked.maf.gz` public files.
