# Graph Report - .  (2026-06-04)

## Corpus Check
- 82 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 545 nodes · 778 edges · 65 communities detected
- Extraction: 61% EXTRACTED · 39% INFERRED · 0% AMBIGUOUS · INFERRED: 307 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `UNet2D` - 12 edges
2. `BrainImageDataset` - 11 edges
3. `main()` - 10 edges
4. `main()` - 9 edges
5. `_make_fake_dataset()` - 8 edges
6. `predict_idh()` - 8 edges
7. `main()` - 8 edges
8. `main()` - 8 edges
9. `CaseLevelSampler` - 8 edges
10. `main()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Tests for IDH classification dataset helpers (ROI crop + balanced sampler).  Run` --uses--> `BraTSSliceClassificationDataset`  [INFERRED]
  tests/test_idh_dataset.py → src/idh_glioma/data/datasets.py
- `Manifest entry pointing at fake nifti paths; load_nifti is monkey-patched in the` --uses--> `BraTSSliceClassificationDataset`  [INFERRED]
  tests/test_idh_dataset.py → src/idh_glioma/data/datasets.py
- `Unit tests for the CT/MRI data pipeline.  Run with:     uv run pytest tests/ -v` --uses--> `BrainImageDataset`  [INFERRED]
  tests/test_ct_datasets.py → src/idh_glioma/data/ct_datasets.py
- `Unit tests for the CT/MRI data pipeline.  Run with:     uv run pytest tests/ -v` --uses--> `ImageRecord`  [INFERRED]
  tests/test_ct_datasets.py → src/idh_glioma/data/prepare_ct_data.py
- `Create a minimal fake Kaggle-style folder tree with tiny images.` --uses--> `BrainImageDataset`  [INFERRED]
  tests/test_ct_datasets.py → src/idh_glioma/data/ct_datasets.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (32): BrainImageDataset, _build_transforms(), PyTorch Dataset for the Kaggle CT/MRI brain-tumor classification dataset.  Reads, Binary classification dataset for CT/MRI brain-tumor images.      Args:, evaluate(), main(), parse_args(), Evaluate the CT/MRI binary tumor classifier on the held-out test split.  Outputs (+24 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (18): _fit_per_modality(), _load_inputs(), main(), _new_base(), parse_args(), _per_modality_probs(), Late-fusion experiment for the multi-omics molecular IDH cohort.  Hypothesis: ea, _recall_at_specificity() (+10 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (28): _augment_cls(), _augment_seg(), BraTSSliceSegmentationDataset, CaseLevelSampler, _crop_roi(), load_nifti(), make_balanced_sampler(), Random spatial + intensity augmentations for classification (image only, C,H,W). (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (24): _build_diagnosis(), _classify_tumor_slices(), _device(), _get_cls_model(), _get_seg_model(), _load_nifti(), predict_idh(), IDH classification handler for the Gradio app.  Exposes :func:`predict_idh` -- g (+16 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (11): BraTSSliceClassificationDataset, Tests for IDH classification dataset helpers (ROI crop + balanced sampler).  Run, Manifest entry pointing at fake nifti paths; load_nifti is monkey-patched in the, test_classification_dataset_no_roi_keeps_full_slice(), test_classification_dataset_roi_returns_expected_shape(), _write_synthetic_manifest(), apply_profile(), main() (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.24
Nodes (16): best_slice(), ct_gallery_extended(), dice(), idh_montage(), load(), Generate figures for the model report:   1. Segmentation overlays (FLAIR + GT gr, 8-panel gallery (CT + MRI x TP/TN/FP/FN) with REAL probabilities pulled     from, Row of 3 MONAI-bundle segmentation results (FLAIR + GT green + Pred red). (+8 more)

### Community 6 - "Community 6"
Cohesion: 0.21
Nodes (14): build_app(), _build_diagnosis(), _collect_examples(), _fig_to_array(), _get_model(), _gradcam(), main(), _overlay_heatmap() (+6 more)

### Community 7 - "Community 7"
Cohesion: 0.29
Nodes (14): _build_feature_matrix(), _evaluate_minority_metrics(), _evaluate_pooled_cv(), _evaluate_source_holdout(), _load_inputs(), main(), _new_model(), _normalize_modalities() (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (0): 

### Community 9 - "Community 9"
Cohesion: 0.27
Nodes (13): build_multisource_manifest(), build_payload(), _canonical_suffix(), classify_modalities(), collect_case_dirs(), _infer_date(), infer_source_root(), load_label_map() (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.22
Nodes (10): Dataset, _expand_bbox_uniformly(), main(), _make_balanced_sampler(), parse_args(), 3D IDH classifier with MONAI on the TCGA-LGG cohort.  Crops the 4-channel multim, _run_epoch(), _tumor_bbox_3d() (+2 more)

### Community 11 - "Community 11"
Cohesion: 0.35
Nodes (10): _build_manifest(), _build_source_bundle(), _load_file_to_patient_map(), _load_methylation_file_map(), main(), _merge_labels(), _normalize_modalities(), parse_args() (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.25
Nodes (7): aggregate_probs(), build_roi_boxes(), load_e2e_config(), merge_e2e_config(), predict_multi_view_idh(), Shared helpers for end-to-end ROI extraction and config loading., zscore_crop()

### Community 13 - "Community 13"
Cohesion: 0.42
Nodes (8): collect_cases(), find_modality(), infer_case(), load_case(), main(), parse_args(), save_outputs(), zscore()

### Community 14 - "Community 14"
Cohesion: 0.42
Nodes (8): _aligned_training_set(), _build_feature_matrix(), _load_training_inputs(), main(), _normalize_modalities(), parse_args(), _resolve_save_dir(), _select_features_by_modality()

### Community 15 - "Community 15"
Cohesion: 0.47
Nodes (8): _build_lgg_filter(), download_lgg_dataset(), download_lgg_methylation(), download_via_gdc_client(), FileRecord, query_lgg_files(), query_lgg_methylation_files(), write_manifest()

### Community 16 - "Community 16"
Cohesion: 0.42
Nodes (8): build_manifest(), CaseRecord, infer_brats_root(), main(), parse_args(), parse_case_folder(), stratified_split(), write_manifest()

### Community 17 - "Community 17"
Cohesion: 0.5
Nodes (7): _build_diagnosis(), _device(), _get_cls(), _get_seg(), predict_idh_monai(), 3D MONAI IDH classification handler for the Gradio app.  Production pipeline: MO, _render_overview()

### Community 18 - "Community 18"
Cohesion: 0.43
Nodes (6): build_expression_matrix(), ExpressionBuildInfo, parse_rnaseq_tsv(), _pick_expression_file(), _read_rnaseq_table(), _select_patient_files()

### Community 19 - "Community 19"
Cohesion: 0.43
Nodes (7): _binarize_label(), _build_transforms(), main(), _maybe_load_zoo_init(), parse_args(), 3D segmentation training with MONAI on the TCGA-LGG cohort.  Uses SegResNet (4-c, _records_to_monai()

### Community 20 - "Community 20"
Cohesion: 0.48
Nodes (6): _load_models(), main(), parse_args(), _parse_view_margin_set(), _predict_seg_mask(), Tune end-to-end IDH settings on a manifest split using macro F1.  Runs the full

### Community 21 - "Community 21"
Cohesion: 0.52
Nodes (6): check_gdc_client(), download_manifest(), main(), match_file(), read_manifest(), write_manifest()

### Community 22 - "Community 22"
Cohesion: 0.52
Nodes (6): aggregate_idh_labels(), extract_idh_status(), IDHMutationRecord, _patient_id_from_barcode(), _read_maf_rows(), _to_float()

### Community 23 - "Community 23"
Cohesion: 0.48
Nodes (6): main(), parse_args(), _predict_case(), Evaluate the MONAI 3D IDH classifier on the held-out test split.  Mirrors eval_i, _tumor_bbox_3d(), _zscore()

### Community 24 - "Community 24"
Cohesion: 0.43
Nodes (6): main(), parse_args(), _predict_case(), Evaluate the MobileNetV3 IDH classifier on the held-out test split.  Slice-level, Returns (case_prob, slice_probs) -- mean probability across tumor slices., score_test_split()

### Community 25 - "Community 25"
Cohesion: 0.52
Nodes (6): apply_profile(), _load_case(), main(), parse_args(), predict(), _zscore()

### Community 26 - "Community 26"
Cohesion: 0.33
Nodes (1): Unit tests for end-to-end ROI helpers used by the MONAI pipeline.

### Community 27 - "Community 27"
Cohesion: 0.33
Nodes (1): Tests for end-to-end calibration config loading.

### Community 28 - "Community 28"
Cohesion: 0.47
Nodes (5): main(), produce_datalist(), produce_sample_dict(), This function is used to split the dataset.     It will produce "train_size" num, split the dataset and output the data list into a json file.

### Community 29 - "Community 29"
Cohesion: 0.6
Nodes (5): _load_sam3_model(), main(), _normalize_to_uint8(), parse_args(), run_sam3_inference()

### Community 30 - "Community 30"
Cohesion: 0.6
Nodes (5): main(), parse_args(), predict_yolov11(), train_yolov11(), validate_yolov11()

### Community 31 - "Community 31"
Cohesion: 0.6
Nodes (5): aggregate_clinical(), _find_first_text(), _local_name(), parse_clinical_xml(), _safe_int()

### Community 32 - "Community 32"
Cohesion: 0.4
Nodes (2): _map_prior_to_gene_ids(), select_features()

### Community 33 - "Community 33"
Cohesion: 0.53
Nodes (5): _build_transforms(), main(), parse_args(), Evaluate the MONAI SegResNet checkpoint on the held-out test split.  Mirrors eva, _records_to_monai()

### Community 34 - "Community 34"
Cohesion: 0.6
Nodes (5): export_split(), main(), mask_to_bbox(), normalize_to_uint8(), parse_args()

### Community 35 - "Community 35"
Cohesion: 0.53
Nodes (5): _bbox3d(), main(), parse_args(), predict(), End-to-end inference with MONAI 3D models (SegResNet + DenseNet121).  Mirrors in

### Community 36 - "Community 36"
Cohesion: 0.67
Nodes (5): test_prepare_multimodal_missing_methylation_dir_exits(), test_prepare_multimodal_writes_all_artifacts(), _write_maf(), _write_methylation(), _write_rnaseq()

### Community 37 - "Community 37"
Cohesion: 0.7
Nodes (4): build_methylation_matrix(), discover_methylation_files(), parse_sesame_txt(), _platform_from_cpg_count()

### Community 38 - "Community 38"
Cohesion: 0.6
Nodes (4): main(), parse_args(), predict(), End-to-end inference combining the MONAI Model Zoo brats bundle (seg) with our 3

### Community 39 - "Community 39"
Cohesion: 0.7
Nodes (4): test_lightgbm_predict_proba_and_roundtrip(), test_logistic_predict_proba_and_roundtrip(), test_mlp_predict_proba_and_roundtrip(), _toy_data()

### Community 40 - "Community 40"
Cohesion: 0.5
Nodes (1): Tests for Phase 2 3D IDH ROI augmentation helpers.

### Community 41 - "Community 41"
Cohesion: 0.67
Nodes (3): _case_prob(), main(), Pick the optimal IDH classification threshold on the val split.  Uses case-level

### Community 42 - "Community 42"
Cohesion: 0.67
Nodes (2): ensure_dir(), save_json()

### Community 43 - "Community 43"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Evaluate the MONAI Model Zoo `brats_mri_segmentation` bundle on TCGA-LGG.  Zero-

### Community 44 - "Community 44"
Cohesion: 0.67
Nodes (3): main(), parse_args(), End-to-end evaluation: MONAI Model Zoo bundle (seg) + 3D DenseNet (cls).  Comput

### Community 45 - "Community 45"
Cohesion: 0.67
Nodes (3): main(), parse_args(), Train YOLOv8 brain-tumor detection model on Ultralytics brain-tumor dataset.

### Community 46 - "Community 46"
Cohesion: 0.5
Nodes (0): 

### Community 47 - "Community 47"
Cohesion: 0.83
Nodes (3): test_build_expression_matrix_merges_samples_and_logs(), test_parse_rnaseq_tsv_skips_summary_rows(), _write_rnaseq()

### Community 48 - "Community 48"
Cohesion: 0.5
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 0.83
Nodes (3): test_build_methylation_matrix_intersection_transform_and_dedup(), test_parse_sesame_txt_preserves_nan_values(), _write_sesame()

### Community 50 - "Community 50"
Cohesion: 0.83
Nodes (3): test_prepare_main_end_to_end_with_mock_data(), _write_maf(), _write_rnaseq()

### Community 51 - "Community 51"
Cohesion: 0.83
Nodes (3): test_aggregate_idh_labels_marks_wildtype_when_no_idh_mutation(), test_extract_idh_status_detects_idh1_idh2_missense(), _write_maf()

### Community 52 - "Community 52"
Cohesion: 0.67
Nodes (1): 5-fold stratified CV for the 3D MONAI IDH classifier.  Pools every labelled case

### Community 53 - "Community 53"
Cohesion: 0.67
Nodes (1): 5-fold stratified cross-validation for the IDH classifier.  Pools every labelled

### Community 54 - "Community 54"
Cohesion: 0.67
Nodes (0): 

### Community 55 - "Community 55"
Cohesion: 0.67
Nodes (2): Project a manifest-v2-style record into the current training shape., to_legacy_manifest_record()

### Community 56 - "Community 56"
Cohesion: 0.67
Nodes (2): build_mobilenetv3_binary(), Build a MobileNetV3 binary classifier.      ``variant="small"`` keeps backwards

### Community 57 - "Community 57"
Cohesion: 0.67
Nodes (0): 

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (0): 

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Training entry points.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (0): 

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (0): 

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (0): 

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (0): 

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **47 isolated node(s):** `Unit tests for end-to-end ROI helpers used by the MONAI pipeline.`, `Tests for Phase 2 3D IDH ROI augmentation helpers.`, `Tests for end-to-end calibration config loading.`, `Pick the optimal IDH classification threshold on the val split.  Uses case-level`, `5-fold stratified CV for the 3D MONAI IDH classifier.  Pools every labelled case` (+42 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 58`** (2 nodes): `test_manifest_adapters.py`, `test_to_legacy_manifest_record_keeps_training_fields()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (2 nodes): `__init__.py`, `Training entry points.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (2 nodes): `test_eval.py`, `test_eval_main_all_modes_writes_outputs()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (2 nodes): `test_feature_select.py`, `test_select_features_top_k_and_prior_union_fold_aware()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (2 nodes): `test_train_eval_multimodal.py`, `test_train_eval_multimodal_end_to_end()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (2 nodes): `test_idh_cpg_panel.py`, `test_idh_cpg_panel_schema()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `model.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `UNet2D` connect `Community 3` to `Community 2`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `BraTSSliceClassificationDataset` connect `Community 4` to `Community 2`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `UNet2D` (e.g. with `IDH classification handler for the Gradio app.  Exposes :func:`predict_idh` -- g` and `Return mean per-slice IDH probability across tumor-bearing slices.`) actually correct?**
  _`UNet2D` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `BrainImageDataset` (e.g. with `Unit tests for the CT/MRI data pipeline.  Run with:     uv run pytest tests/ -v` and `Create a minimal fake Kaggle-style folder tree with tiny images.`) actually correct?**
  _`BrainImageDataset` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `main()` (e.g. with `parse_args()` and `_normalize_modalities()`) actually correct?**
  _`main()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `main()` (e.g. with `parse_args()` and `_normalize_modalities()`) actually correct?**
  _`main()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `_make_fake_dataset()` (e.g. with `test_collect_images_counts()` and `test_build_manifest_both()`) actually correct?**
  _`_make_fake_dataset()` has 6 INFERRED edges - model-reasoned connections that need verification._