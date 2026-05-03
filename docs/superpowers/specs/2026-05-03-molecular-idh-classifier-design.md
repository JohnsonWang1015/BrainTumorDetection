# Molecular IDH Classifier Design

## Goal

Train a molecular-layer IDH mutation classifier from RNA-seq gene expression that runs in parallel to the existing imaging-based IDH pipeline. The classifier predicts IDH mutant vs IDH wildtype from `log2(TPM+1)` expression of a curated ~2K gene panel, trained on a pooled TCGA-LGG + TCGA-GBM cohort.

The classifier is not a replacement for the imaging pipeline. It is a second, independent modality that can:

- validate IDH labels coming from imaging-only patients when paired molecular data exists
- serve as an offline reference signal when MRI is unavailable
- expose how much of the IDH discrimination signal lives in transcriptomics vs imaging

## Context

The repository already runs an imaging IDH pipeline (2D MobileNetV3 + 3D MONAI DenseNet) on the local BraTS-TCGA-LGG cohort. A new dataset has landed under `datasets/TCGA-GBM/`, but the contents are molecular (clinical XML, biospecimen XML, MAF somatic mutation calls, RNA-seq gene counts), not MRI volumes. There are no GBM imaging files in this drop, so the existing imaging pipeline cannot consume it.

The recently merged multi-source manifest contract (`configs/idh_manifest_v2_contract.yaml`) anticipated `datasets/TCGA-GBM/` as a 4-modality MRI source. That assumption does not hold for this drop. The molecular pipeline lives outside that contract on purpose, in a separate subpackage with its own artifacts directory, so the imaging pipeline is unaffected.

## Cohort

| Source | Patients | RNA-seq | MAF (IDH-callable) | Expected IDH-mutant |
|---|---|---|---|---|
| TCGA-GBM (already downloaded) | 599 clinical | 391 | 468 public masked MAF | ~5% (~20) |
| TCGA-LGG (to be downloaded via GDC API) | ~510 expected | ~510 expected | ~510 expected | ~70% (~360) |
| Pooled | ~900 | ~900 | ~900 | ~42% (~380) |

Pooling LGG with GBM mitigates the IDH class imbalance that GBM alone would impose. The cohort confound this introduces (LGG dominantly IDH-mutant, GBM dominantly IDH-wildtype) is addressed by the B3 evaluation strategy below, not by ignoring it.

## Approach

### Data flow

```
TCGA-GBM (downloaded)              TCGA-LGG (download via GDC API)
  ├─ MAF (Masked Somatic)            ├─ MAF
  ├─ RNA-seq (TPM)                   ├─ RNA-seq (TPM)
  └─ Clinical Supplement             └─ Clinical Supplement
         │                                  │
         └─────────────┬────────────────────┘
                       ▼
         prepare-idh-molecular CLI
                       │
                       ▼
       artifacts/molecular/
         ├─ expression_matrix.parquet   (~900 patients × ~60K genes, log2 TPM+1)
         ├─ idh_labels.parquet          (patient_id, source, idh_label, label_source)
         ├─ feature_panel.json          (final ~2K gene IDs after fold-aware selection)
         └─ cohort_manifest.json        (per-source counts, labeled vs unlabeled)
                       │
                       ▼
         train-idh-molecular CLI
                       │
                       ▼
       checkpoints/molecular_idh/
         ├─ logistic.joblib
         ├─ lightgbm.txt
         └─ mlp.pt
                       │
                       ▼
         eval-idh-molecular CLI (B3 strategy)
                       │
                       ▼
       artifacts/molecular_idh_eval/
         ├─ pooled_cv_results.json
         ├─ source_holdout_results.json
         ├─ minority_metrics.json
         └─ figures/{roc_curves.png, calibration_curve.png, confusion_matrices.png}
```

### Subpackage layout

```
src/idh_glioma/molecular/
  ├─ __init__.py
  ├─ gdc_download.py              # GDC API query + manifest builder for TCGA-LGG
  ├─ maf_parser.py                # MAF.gz → IDH1/IDH2 mutation status per patient
  ├─ rnaseq_loader.py             # GDC RNA-seq TSV → log2(TPM+1) expression matrix
  ├─ clinical_parser.py           # Clinical XML → demographics / survival table
  ├─ feature_select.py            # Fold-aware top-K variance ∪ prior IDH gene panel
  ├─ models.py                    # LogisticIDH / LightGBMIDH / MLPIDH wrappers
  ├─ train.py                     # CLI: train-idh-molecular
  ├─ eval.py                      # CLI: eval-idh-molecular (B3 evaluation)
  ├─ prepare.py                   # CLI: prepare-idh-molecular
  └─ priors/idh_gene_panel.json   # ~150 curated IDH/G-CIMP genes with citation
```

The molecular subpackage does not import from `src/idh_glioma/data/` or `src/idh_glioma/models/`. Its only shared dependency with the imaging pipeline is `src/idh_glioma/utils.py` (JSON I/O helpers).

### Module contracts

**`gdc_download.py`**

- `query_lgg_files(data_types: list[str]) -> list[FileRecord]` — POST to `https://api.gdc.cancer.gov/files` with filter `cases.project.project_id = TCGA-LGG` and `data_type ∈ {"Gene Expression Quantification", "Masked Somatic Mutation", "Clinical Supplement"}`. Returns dicts with `file_uuid, filename, md5, size, case_submitter_id`.
- `write_manifest(records, path)` — writes a GDC-compliant TSV (columns: `id, filename, md5, size, state`).
- `download_via_gdc_client(manifest_path, out_dir, token_path=None)` — subprocess call; catches `CalledProcessError` and logs partial failures.
- Output: `datasets/TCGA-LGG-Molecular/tcga_lgg_downloads/{manifests,data}/...` mirroring the GBM layout.

**`maf_parser.py`**

- `extract_idh_status(maf_path: Path) -> list[dict]` — parses `*.maf.gz`, scans for `Hugo_Symbol ∈ {IDH1, IDH2}` with `Variant_Classification = Missense_Mutation`. Returns `{tumor_sample_barcode, gene, mutation_aa, vaf}` per mutation.
- `aggregate_idh_labels(maf_dir: Path, source: str) -> pd.DataFrame` — walks all MAFs in directory; per `patient_id` (barcode chars 0..12, e.g. `TCGA-06-0125`), assigns `idh_label = 1` if any IDH1/IDH2 missense found, else `0`.
- Output schema: `patient_id, source, idh_label, idh_gene, idh_mutation_aa, label_source = "maf_aggregated"`.

**`rnaseq_loader.py`**

- `parse_rnaseq_tsv(tsv_path: Path) -> pd.Series` — reads GDC RNA-seq TSV, skips the header rows (`N_unmapped`, `N_multimapping`, `N_noFeature`, `N_ambiguous`), keeps `gene_id` and `tpm_unstranded`.
- `build_expression_matrix(rnaseq_dir: Path, file_to_patient: dict) -> pd.DataFrame` — one column per patient; index is base ENSG (version stripped); applies `log2(TPM + 1)`. NaN gene quantifications filled with 0.
- `file_to_patient` is constructed by joining the GDC sample sheet (downloaded with the manifest) on file UUID.
- Output: parquet with shape `(~60K genes, ~900 patients)` and a sidecar `gene_metadata.parquet` (gene_symbol, gene_type).

**`clinical_parser.py`**

- `parse_clinical_xml(xml_path: Path) -> dict` — extracts `patient_id, age_at_diagnosis, gender, vital_status, days_to_death, days_to_last_followup, histological_type, kps_score`.
- `aggregate_clinical(clin_dir: Path, source: str) -> pd.DataFrame` — for cohort description tables; not used as training features in v1.

**`feature_select.py`**

- `load_idh_prior_panel() -> set[str]` — reads `priors/idh_gene_panel.json`. Initial panel: IDH1, IDH2, plus G-CIMP signature genes from Noushmehr et al. 2010 and Verhaak et al. 2010 glioma subtype markers (~150 genes). Citations stored in the JSON.
- `select_features(X_train_log_tpm: pd.DataFrame, top_k: int = 2000, prior_panel: set[str] | None = None) -> list[str]` — computes per-gene variance on the train fold only, takes top-K, unions with the prior panel mapped to ENSG IDs, returns final feature list.
- Contract: this function never sees test fold data. The eval pipeline calls it once per fold inside the CV loop.

**`models.py`**

All three models implement the same protocol:

```python
class MolecularIDHModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...  # shape (n, 2)
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "MolecularIDHModel": ...
```

- `LogisticIDH` — `sklearn.linear_model.LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000)`.
- `LightGBMIDH` — `lightgbm.LGBMClassifier(is_unbalance=True, n_estimators=500, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20)`.
- `MLPIDH` — PyTorch `Linear(2000 → 256) → ReLU → Dropout(0.3) → Linear(256 → 1)`, BCEWithLogitsLoss with `pos_weight = sqrt(num_neg / num_pos)`, AdamW lr=1e-3, 50 epochs, early stopping on val AUC.

**`prepare.py`** — CLI: `prepare-idh-molecular`

```
--include-sources tcga_gbm tcga_lgg          # default: both
--gdc-token PATH                              # optional, only needed for controlled-access
--gbm-data-root datasets/TCGA-GBM/tcga_gbm_downloads/data
--lgg-data-root datasets/TCGA-LGG-Molecular/tcga_lgg_downloads/data
--output-dir artifacts/molecular
--skip-download                               # use existing on-disk data only
```

Steps: (1) optionally download LGG via `gdc_download.py`, (2) run MAF parser per source, (3) build expression matrix, (4) join on patient_id, (5) write four artifacts.

**`train.py`** — CLI: `train-idh-molecular`

```
--input-dir artifacts/molecular
--model {logistic,lightgbm,mlp,all}           # default: all
--top-k 2000
--save-dir checkpoints/molecular_idh
--seed 42
```

Trains on the full pooled labeled set with feature selection. Saves three checkpoints plus a `training_report.json` with per-model train AUC, fit time, feature count.

**`eval.py`** — CLI: `eval-idh-molecular`

```
--input-dir artifacts/molecular
--checkpoint-dir checkpoints/molecular_idh
--mode {pooled_cv,source_holdout,minority_metrics,all}
--output-dir artifacts/molecular_idh_eval
--folds 5
--seed 42
```

- `pooled_cv` — 5-fold stratified CV on pooled labeled set; reports mean AUC ± std, ROC, calibration.
- `source_holdout` — bidirectional: train(LGG) → test(GBM) and train(GBM) → test(LGG); reports AUC and AUPRC per direction.
- `minority_metrics` — extracts GBM-only test fold from pooled CV; reports AUPRC, recall@95% specificity, calibration on minority class.

## Evaluation strategy (B3, three reports)

The B3 strategy reports all three metric sets so future readers cannot accuse the result of being cherry-picked:

1. **Pooled 5-fold stratified CV** — mean AUC ± std on the full ~900-patient pool.
2. **Source-holdout (bidirectional)** — train on one cohort, test on the other; isolates how much signal generalizes across cohorts vs how much is cohort-specific.
3. **Minority metrics on GBM** — AUPRC and recall@95% specificity in the GBM subset alone, since IDH-mutant in GBM is the rarest and clinically most-interesting subgroup.

The pooled CV will likely show the highest AUC but is the most cohort-confounded. Source-holdout will likely be lower but more honest about generalization. The MODEL_CARD must report all three.

## Edge cases

| Situation | Handling |
|---|---|
| Patient has RNA-seq but no MAF | Listed as `unlabeled` in cohort_manifest; excluded from training |
| Patient has MAF but no RNA-seq | Listed in cohort table but excluded from training (no features) |
| Patient has multiple RNA-seq aliquots | Take primary tumor (barcode segment 4 starts with `01`); if multiple primary, take alphabetically first; flag `multiple_primary_aliquots = true` in cohort_manifest |
| GBM controlled-access MAF download fails | Already handled in `download_tcga_gbm.py` patch; `prepare-idh-molecular` warns if public masked MAF count < 80% of RNA-seq cohort |
| Test fold positive class < 5 (source_holdout) | Return `auprc = null` and tag `insufficient_minority_samples` in result JSON |
| NaN gene quantifications | Filled with 0 (TPM=0 = not expressed is a reasonable prior); record per-sample missing ratio in `cohort_manifest.json` |
| ENSG version drift across files (`ENSG00000001036.14` vs `ENSG00000001036.13`) | Strip version suffix (`.split('.')[0]`) before joining |
| Manifest version conflict with imaging pipeline | Molecular pipeline has its own `cohort_manifest.json`; never written to the v2 imaging manifest |

## Testing

| Test file | Focus |
|---|---|
| `tests/molecular/test_maf_parser.py` | Fixtures: 3 small MAF.gz (IDH1 mutant, IDH2 mutant, no IDH); verify labels and patient_id extraction |
| `tests/molecular/test_rnaseq_loader.py` | Mock TSV with 5 genes + 4 N_xxx header rows; verify header skip, log transform, column merge |
| `tests/molecular/test_feature_select.py` | Synthetic 100-gene × 50-sample matrix with 5 known high-variance genes; verify top-K selection, prior panel union, no test-fold leakage |
| `tests/molecular/test_models.py` | Toy 200-sample × 50-feature data; verify `predict_proba` shape, save/load roundtrip for all three models |
| `tests/molecular/test_eval.py` | Fixed-seed toy data; verify `pooled_cv` and `source_holdout` JSON schema and AUC range |
| `tests/molecular/test_prepare_integration.py` | Mock directory with 5 MAF + 5 RNA-seq + matching IDs; verify end-to-end parquet shape |

`gdc_download.py` is not tested against the live GDC API. Unit tests verify manifest TSV format only, with stub records.

## Dependencies

New dependencies (added to `pyproject.toml`):

- `lightgbm>=4.3` — primary boosted-tree model
- `pyarrow>=15` — parquet I/O (called implicitly by pandas)

Existing dependencies used: `scikit-learn>=1.5`, `pandas`, `numpy`, `torch`, `requests` (for GDC API).

## Risks and limitations (must surface in MODEL_CARD)

1. **Cohort confound** — pooled training mixes LGG (mostly IDH-mutant) and GBM (mostly IDH-wildtype). The model may learn cohort identity rather than true IDH biology. The source-holdout metric is the key diagnostic.
2. **Retrospective TCGA cohort** — not prospectively collected; subject to selection bias; not validated for clinical decision-making.
3. **Population specificity** — IDH-mutant cases concentrate in LGG. Performance in IDH-wildtype-only populations is untested and likely degraded.
4. **Aliquot selection sensitivity** — the primary-tumor + alphabetically-first rule is an arbitrary tiebreaker; not ablated in v1.
5. **Public MAF only** — controlled-access MAFs would add validation power but require GDC token; v1 uses only `aliquot_ensemble_masked.maf.gz` public files.

## Out of scope (deferred)

- Methylation beta and CNV gene-level features (data is downloaded, but not used in v1)
- Survival prediction (DeepSurv, Cox)
- Verhaak molecular subtype classification
- Multi-omics fusion (RNA-seq + methylation + CNV jointly)
- Bridging molecular IDH to imaging IDH (requires patients with both modalities)

## Implementation milestones

1. Data layer: `gdc_download.py` + `maf_parser.py` + `clinical_parser.py` + tests; produce LGG `idh_labels.csv` and demo cohort table.
2. RNA-seq matrix + feature select: `rnaseq_loader.py` + `feature_select.py` + tests; produce `expression_matrix.parquet`.
3. Prepare CLI: `prepare.py` + integration tests; produce full `artifacts/molecular/`.
4. Models + training: `models.py` + `train.py` + tests; checkpoint three models.
5. B3 evaluation + visualization + docs: `eval.py` + updates to `CLAUDE.md`, `README.md`, `docs/DATASET_CARD.md`, `docs/MODEL_CARD.md`.
