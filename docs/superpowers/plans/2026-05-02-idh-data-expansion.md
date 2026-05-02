# IDH Data Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a multi-source data contract and staged implementation path for expanding IDH training beyond the current local `BraTS-TCGA-LGG` subset.

**Architecture:** Introduce a manifest v2 contract that preserves the current per-record training fields while adding source provenance, cohort identity, task-eligibility flags, and source-aware split policy. Implement the first importer for `TCGA-LGG + TCGA-GBM`, keep `UCSF-PDGM` external by default, and add `EGD` only after the adapter path is stable.

**Tech Stack:** Python 3.12, JSON, YAML, pandas, pytest

---

## File Structure

- Added: `configs/idh_manifest_v2_contract.yaml`
  - canonical multi-source manifest contract
- Added: `docs/superpowers/specs/2026-05-02-idh-data-expansion-design.md`
  - design and cohort policy
- Create next: `src/idh_glioma/data/prepare_idh_multisource.py`
  - builder for multi-source manifest v2
- Create next: `src/idh_glioma/data/manifest_adapters.py`
  - adapter from manifest v2 record to current v1 training view
- Create next: `tests/test_prepare_idh_multisource.py`
- Create next: `tests/test_manifest_adapters.py`

## Task 1: Lock the manifest contract

**Files:**
- Added: `configs/idh_manifest_v2_contract.yaml`

- [ ] Confirm the contract covers:
  - current v1 fields
  - cohort registry
  - per-case provenance
  - task eligibility
  - split strategy
- [ ] Keep the compatibility guarantee explicit:
  - `case_id`
  - `date`
  - `modalities`
  - `mask_path`
  - `idh_label`

## Task 2: Implement a multi-source builder for TCGA cohorts first

**Files:**
- Create: `src/idh_glioma/data/prepare_idh_multisource.py`
- Create: `tests/test_prepare_idh_multisource.py`

- [ ] Add source parsers for:
  - `BraTS-TCGA-LGG`
  - `TCGA-GBM`
- [ ] Normalize modality aliases:
  - `t1ce` -> `t1Gd`
  - `t1c` -> `t1Gd`
- [ ] Allow maskless classification-only cases.
- [ ] Emit cohort-aware records with:
  - `source_dataset`
  - `source_subject_id`
  - `cohort_id`
  - `acquisition_stage`
  - `inclusion_flags`
  - `provenance`
- [ ] Write a failing test first for:
  - modality alias normalization
  - maskless classification-only case inclusion
  - hard fail on missing required modalities

## Task 3: Add the v2-to-v1 compatibility adapter

**Files:**
- Create: `src/idh_glioma/data/manifest_adapters.py`
- Create: `tests/test_manifest_adapters.py`

- [ ] Add a function that maps a v2 record into the current training/eval shape.
- [ ] Make `mask_path` nullable in the adapter output for classification-only pipelines.
- [ ] Keep path strings untouched so current loaders continue to work.
- [ ] Add tests that verify exact adapter output for:
  - segmentation-eligible records
  - classification-only records

## Task 4: Add source-aware split policy

**Files:**
- Modify: `src/idh_glioma/data/prepare_idh_multisource.py`
- Test: `tests/test_prepare_idh_multisource.py`

- [ ] Implement split modes:
  - `pooled_train_val_test`
  - `source_holdout`
  - `external_only`
- [ ] Default first production mode to:
  - pooled `TCGA-LGG + TCGA-GBM`
  - `UCSF-PDGM` held out
- [ ] Ensure metrics can later be grouped by `source_dataset` and `cohort_id`.

## Task 5: Add UCSF-PDGM ingestion in external-validation mode

**Files:**
- Modify: `src/idh_glioma/data/prepare_idh_multisource.py`
- Modify: tests

- [ ] Parse `UCSF-PDGM` into manifest v2 records.
- [ ] Mark records with:
  - `source_dataset = ucsf_pdgm`
  - `cohort_id = ucsf_pdgm_external`
  - `acquisition_stage = preop` only when confirmed by dataset metadata
- [ ] Do not mix into default training splits yet.

## Task 6: Add EGD ingestion after the adapter path is stable

**Files:**
- Modify: `src/idh_glioma/data/prepare_idh_multisource.py`
- Modify: tests

- [ ] Parse `EGD` into manifest v2 records.
- [ ] Normalize modality naming and segmentation availability.
- [ ] Explicitly tag QC concerns where metadata is incomplete.
- [ ] Add a `tcia_plus_egd` pooled cohort option, but keep it opt-in.

## Task 7: Expose reporting utilities

**Files:**
- Create: `scripts/report_idh_manifest_stats.py`
- Create: tests if scope permits

- [ ] Print:
  - case counts by source
  - IDH counts by source
  - segmentation-eligible counts by source
  - preop vs postop counts
- [ ] Use this report as the first gate before any new training run.

## Validation Gates

- [ ] Unit tests pass for builder and adapters.
- [ ] A generated v2 manifest can be reduced into the current v1 training shape.
- [ ] The first merged cohort report shows expected class-balance changes:
  - more `IDH-wildtype` after `TCGA-GBM`
- [ ] `UCSF-PDGM` remains external-only unless explicitly requested.
- [ ] `EGD` is excluded by default until source-aware validation is in place.

## Recommended Execution Order

1. Contract is already written.
2. Implement `TCGA-LGG + TCGA-GBM` builder.
3. Add v2-to-v1 adapter.
4. Run pooled TCGA experiments.
5. Add `UCSF-PDGM` as external validation.
6. Add `EGD` as opt-in expansion.
