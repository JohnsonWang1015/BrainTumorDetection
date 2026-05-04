# Multi-omics IDH Fusion Design (RNA-seq + Methylation)

## Goal

Extend the existing molecular IDH classifier (`src/idh_glioma/molecular/`) with a second modality: DNA methylation beta values from TCGA-GBM and TCGA-LGG. The classifier predicts IDH mutant vs wildtype from a fold-aware concatenation of `log2(TPM+1)` RNA-seq features and methylation `M-values` (`log2(beta / (1 - beta))`).

The multi-omics path runs alongside the RNA-seq-only path. It does not replace it. Both produce B3 evaluation reports so the lift from adding methylation is visible.

## Why methylation

Noushmehr et al. 2010 (PMID 20399149) showed that IDH-mutant gliomas exhibit a CpG island methylator phenotype (G-CIMP) that distinguishes them at ~95–99% accuracy from a small CpG signature. RNA-seq gives transcriptional evidence; methylation gives the underlying epigenetic state and is the literature-canonical IDH biomarker. Adding methylation should:

- raise minority-class accuracy on GBM IDH-mutant patients (currently AUPRC 0.947, recall@95spec 0.944)
- increase robustness on cross-cohort source-holdout (currently 0.95–0.99 AUC)
- expose any RNA-seq overfitting that current 0.99 pooled CV may be hiding

## Cohort

| Source | RNA-seq files | Methylation files (HM27 + HM450) | Multi-modal subset (rough estimate) |
|---|---:|---:|---:|
| TCGA-GBM (already on disk) | 391 | 526 (371 HM27 + 155 HM450) | ~250 |
| TCGA-LGG (RNA-seq on disk; methylation to be downloaded) | 534 | ~516 expected from GDC API | ~500 |
| Pooled strict subset | — | — | ~600–700 |

The strict multi-modal subset (patients with both RNA-seq and methylation) is smaller than the RNA-seq-only labeled+expression set of 759. This is an explicit trade-off: stricter subset, richer features per patient. No imputation in v1.

## Approach

### Data flow

```
TCGA-GBM (downloaded)            TCGA-LGG (RNA-seq downloaded; methylation to download)
  ├─ rnaseq_counts (391)           ├─ rnaseq_counts (534)
  └─ methylation_beta (526)        └─ methylation_beta (~516 expected)
         │                                │
         └────────────┬───────────────────┘
                      ▼
       prepare-idh-molecular --modalities rnaseq methylation
                      │
                      ▼
       artifacts/molecular_multimodal/
         ├─ expression_matrix.parquet      (existing format, log2(TPM+1))
         ├─ methylation_matrix.parquet     (NEW, M-values, HM27-intersected ~27K CpGs)
         ├─ idh_labels.parquet             (filtered to multi-modal subset)
         ├─ feature_panel.json             (per-modality)
         └─ cohort_manifest.json           (multimodal section + per-modality dropouts)
                      │
                      ▼
       train-idh-molecular --modalities rnaseq methylation \
                           --input-dir artifacts/molecular_multimodal \
                           --save-dir checkpoints/molecular_idh
                      │
                      ▼
       checkpoints/molecular_idh_multimodal/{logistic.joblib, lightgbm.txt, mlp.pt, training_report.json}
                      │
                      ▼
       eval-idh-molecular --modalities rnaseq methylation \
                          --input-dir artifacts/molecular_multimodal \
                          --output-dir artifacts/molecular_idh_multimodal_eval
                      │
                      ▼
       artifacts/molecular_idh_multimodal_eval/{pooled_cv,source_holdout,minority_metrics}.json + figures
```

### Subpackage layout

Existing `src/idh_glioma/molecular/` keeps its layout. Three new files; four existing files get small extensions:

```
src/idh_glioma/molecular/
  ├─ __init__.py                  (existing)
  ├─ gdc_download.py              (extend: methylation queries + downloader)
  ├─ maf_parser.py                (existing, untouched)
  ├─ rnaseq_loader.py             (existing, untouched)
  ├─ clinical_parser.py           (existing, untouched)
  ├─ feature_select.py            (extend: generic select_features_modality)
  ├─ models.py                    (existing, untouched)
  ├─ train.py                     (extend: --modalities flag)
  ├─ eval.py                      (extend: --modalities flag)
  ├─ prepare.py                   (extend: --modalities flag)
  ├─ methylation_loader.py        (NEW)
  ├─ multimodal.py                (NEW)
  └─ priors/
      ├─ idh_gene_panel.json      (existing)
      └─ idh_cpg_panel.json       (NEW, ~50 G-CIMP CpGs from Noushmehr 2010)
```

### Module contracts

**`methylation_loader.py`** (new)

- `parse_sesame_txt(path: Path) -> pd.Series` — reads `<uuid>.methylation_array.sesame.level3betas.txt` (no header TSV with two columns: `cg_id`, `beta`). Returns Series indexed by `cg_id`. NaN beta values are preserved.
- `discover_methylation_files(meth_dir: Path) -> list[tuple[str, Path]]` — scans `<source>/methylation_beta/<file_uuid>/<sample>.sesame.level3betas.txt`; returns `(file_uuid, path)` pairs. Skips `annotations.txt`.
- `build_methylation_matrix(meth_dir: Path, file_to_patient: dict[str, str], source: str, beta_to_mvalue: bool = True) -> tuple[pd.DataFrame, dict]`:
  - Parses each file via `parse_sesame_txt`, builds `(cg_id × patient_id)` DataFrame.
  - Cross-platform handling: computes intersection of `cg_id` across all files (HM27 ⊂ HM450 ⇒ intersection ≈ HM27 ~27K CpGs).
  - Multiple aliquots per patient: deterministic dedup (alphabetically first file UUID, mirroring RNA-seq); flagged in summary dict as `multiple_methylation_aliquots`.
  - NaN beta filling: NaN → 0.5 (unknown methylation prior); records per-sample NaN ratio.
  - Beta → M-value: `M = log2(beta / (1 - beta))`, with beta clipped to `[0.001, 0.999]`. Only applied when `beta_to_mvalue=True`.
  - Returns `(matrix, summary)` where summary contains `n_patients_per_platform`, `intersection_size`, `nan_fill_count`, `multiple_aliquot_patient_ids`.

**`multimodal.py`** (new, fusion coordinator)

- `concat_modalities(matrices: dict[str, pd.DataFrame], patient_ids: list[str]) -> pd.DataFrame`:
  - Input: `{"rnaseq": df_genes, "methylation": df_cpgs}` with patient_id columns.
  - Strict subset: keeps only `patient_ids` present in all matrices.
  - Output: `(n_patients, n_features_total)` ndarray-ready DataFrame; feature names are prefixed (`rnaseq:ENSG...`, `methylation:cg...`) for traceability.
- `select_multimodal_features(matrices: dict[str, pd.DataFrame], prior_panels: dict[str, set[str]], top_k_per_modality: dict[str, int]) -> dict[str, list[str]]`:
  - Per modality, calls `feature_select.select_features_modality(...)` to compute top-K variance ∪ prior panel.
  - Fold-aware: caller passes train-fold-only matrices.
  - Returns `{"rnaseq": [...], "methylation": [...]}`.

**`priors/idh_cpg_panel.json`** (new)

- Schema: `{"_citation": ["Noushmehr 2010 PMID:20399149", ...], "cpg_ids": ["cg00000292", ...]}`.
- ~50 G-CIMP signature CpGs from Noushmehr 2010 (TCGA glioma cohort). All representable on HM27 to ensure intersection compatibility.

**`feature_select.py`** (extend)

- Existing `select_features(X_train_log_tpm, top_k, prior_panel, gene_metadata)` is preserved unchanged for backward compatibility.
- Add `select_features_modality(X_train: pd.DataFrame, top_k: int, prior_ids: set[str]) -> list[str]`:
  - Generic: does not assume ENSG vs CpG identifier.
  - Returns features sorted by variance descending, unioned with `prior_ids` that exist in `X_train.index`.

**`gdc_download.py`** (extend)

- Add `query_lgg_methylation_files() -> list[FileRecord]`:
  - GDC filters: `cases.project.project_id = TCGA-LGG`, `data_type = "Methylation Beta Value"`, `platform ∈ {"Illumina Human Methylation 450", "Illumina Human Methylation 27"}`, `access = "open"`.
- Add `download_lgg_methylation(base_dir: Path, gdc_client_path: Path | None = None) -> dict[str, int]`:
  - Mirrors `download_lgg_dataset` structure: writes manifest under `manifests/methylation_beta_manifest.txt`, downloads to `data/methylation_beta/`.
- Existing `query_lgg_files` and `download_lgg_dataset` are unchanged.

**`prepare.py`** (extend)

- New CLI flag: `--modalities` (default `["rnaseq"]`, can pass multiple e.g. `--modalities rnaseq methylation`).
- New CLI flag: `--methylation-transform {beta,mvalue}` (default `mvalue`).
- New CLI flag: `--methylation-data-root <path>` if methylation lives outside default GBM/LGG molecular paths.
- When methylation is requested:
  - Verifies `<source>/methylation_beta/` exists per source. If missing, prints actionable error suggesting `download_lgg_methylation` and exits non-zero.
  - Calls `build_methylation_matrix` per source, concatenates patient columns, writes `methylation_matrix.parquet`.
  - Adds `multimodal` block to `cohort_manifest.json`: `strict_subset_size`, `rnaseq_only_dropped` (patient list), `methylation_only_dropped` (patient list), per-modality summaries from loaders.
- Output dir suffix: when `len(modalities) > 1`, write to `artifacts/molecular_multimodal/` regardless of `--output-dir` default; explicit `--output-dir` is honored.

**`train.py`** (extend)

- New CLI flag: `--modalities` (default `["rnaseq"]`).
- Internal: loads each requested modality matrix, calls `multimodal.concat_modalities` for the strict patient subset, applies `select_multimodal_features` per modality on the full training set, concatenates selected features, fits all three models (Logistic, LightGBM, MLP).
- Save dir suffix: when `len(modalities) > 1` and `--save-dir` is the default, append `_multimodal`; explicit `--save-dir` is honored.
- Reports per-model train AUC + fit time to `training_report.json`; adds `modalities` and `n_features_per_modality` to the report.

**`eval.py`** (extend)

- New CLI flag: `--modalities` (default `["rnaseq"]`).
- Loads matrices per modality, runs B3 (`pooled_cv`, `source_holdout`, `minority_metrics`) on the strict subset, generates the same figure set.
- Output dir suffix: same `_multimodal` rule as train.

### Evaluation strategy

Identical to the RNA-seq-only B3:

1. **Pooled 5-fold stratified CV** — mean AUC ± std on the strict subset.
2. **Source-holdout (bidirectional)** — train(LGG) → test(GBM) and reverse.
3. **Minority metrics on GBM** — AUPRC, recall@95% specificity, Brier on the GBM subset of pooled CV predictions.

The MODEL_CARD reports both RNA-seq-only baseline and multi-modal fusion side by side so the lift is unambiguous.

## Edge cases

| Situation | Handling |
|---|---|
| LGG methylation not yet downloaded | `prepare-idh-molecular --modalities rnaseq methylation` exits non-zero with an actionable error (suggest running `download_lgg_methylation`) |
| Patient has RNA-seq but no methylation | Listed in `cohort_manifest.json:multimodal.methylation_only_dropped`; excluded from training in strict mode (no imputation in v1) |
| Patient has methylation but no RNA-seq | Symmetric to above; listed in `rnaseq_only_dropped` |
| Cross-platform HM27 vs HM450 | Intersect to common CpGs; record `n_hm27`, `n_hm450`, `intersection_size` in summary |
| Multiple methylation aliquots per patient | Take alphabetically first file UUID; flag `multiple_methylation_aliquots: true` |
| NaN beta values per CpG | Fill with 0.5 (midpoint prior) → M-value = 0; record per-sample NaN ratio in cohort_manifest |
| Beta exactly 0 or 1 | Clip to `[0.001, 0.999]` before M-value transform to avoid `±inf` |
| Strict subset < 200 patients | `prepare.py` prints warning; pipeline continues; MODEL_CARD will note potential underpowering |
| Multiple modalities but only one prior panel found | That modality degrades to top-K-only selection; warn but continue |
| `--modalities rnaseq rnaseq` (user typo) | Argparse-level dedup; warn |
| Feature name collision (cg_id matches an ENSG identifier by chance) | Independent namespaces enforced via `rnaseq:` / `methylation:` prefix on all feature names |

## Testing

| Test file | Focus |
|---|---|
| `tests/molecular/test_methylation_loader.py` | parse_sesame_txt with NaN beta + headerless TSV; build_methylation_matrix mixed HM27/HM450 → intersection size matches smaller platform; M-value boundary (beta=0/1/0.5 → clipped); NaN→0.5→M-value=0 fill |
| `tests/molecular/test_multimodal.py` | concat_modalities strict subset = patient intersection; select_multimodal_features fold-awareness per modality with no leakage; feature name prefixing |
| `tests/molecular/test_idh_cpg_panel.py` | priors/idh_cpg_panel.json schema (`_citation` list, `cpg_ids` list[str], no duplicates) |
| `tests/molecular/test_prepare_multimodal_integration.py` | Mock 3 RNA-seq + 3 methylation per source with matching IDs; verify multimodal artifacts exist with strict subset = 3 |
| `tests/molecular/test_train_eval_multimodal.py` | Toy multi-modal data (30 patients × (10 RNA + 10 meth)); train + eval end-to-end with `--modalities rnaseq methylation`; JSON schema matches RNA-seq-only |

Existing tests must continue to pass unchanged; default `--modalities rnaseq` behavior must produce bit-identical artifacts to the current pipeline.

## Dependencies

No new dependencies. Methylation parsing uses pandas; M-value transform uses numpy; GDC API already uses requests.

## Hypothesized result (write into MODEL_CARD whether confirmed or not)

| Metric | RNA-seq-only baseline | Multi-modal expectation |
|---|---|---|
| Pooled CV AUC (best model) | 0.992 | ~0.995 (small lift; RNA-seq near saturation) |
| LGG → GBM AUC | 0.965–0.988 | ~0.98 (small lift) |
| GBM → LGG AUC | 0.954–0.972 | ~0.97 (small lift) |
| **GBM minority AUPRC** | 0.933–0.947 | **~0.96 (largest expected lift)** |
| GBM minority recall @ 95% spec | 0.944 | ~0.96 |

If observed lift is below 0.005 across all metrics, the MODEL_CARD will state plainly that fusion gives no significant benefit on this cohort and that RNA-seq alone is already saturated.

## Risks and limitations (write into MODEL_CARD verbatim)

1. **Smaller training set in strict mode** — multi-modal subset is ~600 vs RNA-seq-only 759, increasing variance.
2. **HM27 platform ceiling** — intersecting to HM27 caps methylation resolution at ~27K CpGs vs HM450's ~485K.
3. **Batch effects across aliquots** — methylation and RNA-seq may come from different sample/aliquot for the same patient.
4. **M-value clipping** — extreme beta values (~0 or ~1) lose information after the `[0.001, 0.999]` clip.
5. **Cohort confound persists** — same diagnosis caveat as RNA-seq-only B3; pooled CV remains LGG-vs-GBM susceptible.

## Out of scope (deferred)

- CNV gene-level features (data is on disk; A2 path)
- Methylation-only ablation comparison (A3 path)
- Imputation of missing modalities (F2 late fusion)
- Deep-learning fusion encoders (F3 intermediate fusion)
- External validation cohort (UCSF-PDGM, EGD)
- Calibration / Youden's threshold tuning
- Verhaak molecular subtype prediction
- Survival prediction (DeepSurv / Cox)

## Implementation milestones

1. Methylation data layer: `gdc_download.py` methylation helpers + `methylation_loader.py` + tests + actual LGG methylation download from GDC.
2. Multimodal core: `multimodal.py` + `priors/idh_cpg_panel.json` + `feature_select.py` extension + tests.
3. Prepare CLI: `prepare.py` `--modalities` flag + integration test + end-to-end multimodal artifacts.
4. Train + eval CLI: `train.py` and `eval.py` `--modalities` flag + checkpoints + tests.
5. Multi-modal B3 eval + docs: real eval run + updates to CLAUDE.md, README.md, docs/DATASET_CARD.md, docs/MODEL_CARD.md with actual lift numbers.
