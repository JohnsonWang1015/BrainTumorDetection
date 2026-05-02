# End-to-End IDH Macro F1 Improvement Design

## Goal

Improve end-to-end IDH mutation classification under the predicted-mask pipeline, with `macro F1` as the primary optimization target and `WT recall`, `Mutant recall`, `AUC`, and `accuracy` as secondary guardrails.

## Current State

The current recommended MRI pipeline is:

1. MONAI Model Zoo `brats_mri_segmentation` bundle for tumor segmentation
2. `densenet3d_idh_jitter.pt` for 3D IDH classification
3. A classifier threshold persisted in checkpoint metadata and reused by:
   - `src/idh_glioma/eval/eval_e2e_monai_zoo.py`
   - `src/idh_glioma/infer/pipeline_monai_zoo.py`
   - `src/idh_glioma/app_idh_monai.py`

Observed repository-reported behavior:

- Segmentation is strong: end-to-end test Dice is around `0.926`
- GT-mask 3D IDH classification is strong: CV AUC is around `0.916`
- End-to-end classification is weaker: reported `AUC 0.75`, `accuracy 0.80`, `macro F1 0.69`

The mismatch indicates the main failure mode is not the base classifier in isolation, but the transition from predicted segmentation mask to classifier-ready ROI plus an AUC-oriented threshold that is not optimized for class balance.

## Problem Statement

The current end-to-end path has three coupled weaknesses:

1. `pred_mask -> bbox` is brittle because it uses one fixed postprocess shape and one fixed margin.
2. `bbox -> crop -> classifier` is brittle because it relies on a single ROI view.
3. Threshold calibration is mismatched because it uses `Youden's J` rather than end-to-end `macro F1`.

On this cohort, class imbalance is severe. Any solution that improves overall accuracy while sacrificing WT performance is not acceptable.

## Non-Goals

- Replacing the MONAI Model Zoo bundle as the default segmentation backbone in Phase 1
- Reworking the 2D IDH or CT/MRI classification pipelines
- Broad dataset changes or adding new external datasets in this iteration
- Optimizing for deployment latency ahead of classification balance

## Success Criteria

### Primary

- Validation end-to-end `macro F1` improves over the current default pipeline using the same segmentation and classifier checkpoints.

### Secondary

- Validation `WT recall` does not regress materially while chasing mutant recall.
- Validation `AUC` remains at least directionally stable.
- The chosen configuration can be replayed consistently by evaluation CLI, inference CLI, and Gradio app.

### Decision Rule

- Tune on `val`
- Freeze the chosen config
- Report once on `test`

The `test` split is not used to search thresholds or ROI parameters.

## Phase 1: ROI Robustness and E2E Calibration

Phase 1 focuses on inference-time robustness without changing the segmentation backbone or forcing a classifier retrain.

### 1. Shared E2E ROI Config

Add a shared representation for end-to-end IDH inference settings that can be loaded by:

- `eval_e2e_monai_zoo.py`
- `pipeline_monai_zoo.py`
- `app_idh_monai.py`

The config must cover:

- mask connected-component handling
- optional binary dilation / ROI expansion
- base bbox margin
- one or more ROI extraction views
- aggregation method across multi-view probabilities
- threshold
- threshold objective metadata

This config should live outside the current checkpoint-only metadata path so that end-to-end calibration can evolve independently from GT-mask classifier training.

### 2. Mask Postprocess Options

Generalize current mask cleanup from a single hard-coded `KeepLargestConnectedComponent` path into configurable postprocess steps. Phase 1 should support:

- keeping the largest connected component
- optional morphological expansion before bbox extraction
- a safe fallback when no positive mask survives postprocessing

The objective is not better segmentation Dice by itself. The objective is more stable ROI generation for IDH classification.

### 3. Multi-View ROI Extraction

Replace the single `bbox + margin` inference path with a small set of ROI views derived from the same predicted mask. Example views:

- base bbox with stored classifier margin
- expanded bbox
- slightly more context-preserving bbox

Each view produces one classifier probability. These are aggregated into one case probability by a simple deterministic reducer such as `mean` or `median`.

This reduces sensitivity to small predicted-mask shifts that currently flip borderline cases.

### 4. End-to-End Macro F1 Calibration

Add a calibration/sweep workflow that runs the full predicted-mask pipeline on the validation split and searches over:

- ROI postprocess settings
- ROI view recipes
- aggregation method
- decision threshold

The optimization target is validation `macro F1`, not `AUC` and not `Youden's J`.

The sweep output should be saved as a reusable artifact so production inference can consume the chosen configuration without code edits.

### 5. Unified Consumers

After calibration, the same chosen config must drive:

- CLI evaluation
- single-case inference
- Gradio 3D IDH tab

This removes the current drift risk where the app and evaluation may describe the same model but not the same thresholding objective.

## Phase 2: Retraining Only If Phase 1 Plateaus

Phase 2 is conditional, not automatic.

Trigger Phase 2 only if one of these is true after Phase 1:

- validation macro F1 remains unsatisfactory
- WT recall remains fragile
- the best configuration depends on aggressive ROI tricks that indicate the classifier is still poorly matched to noisy predicted masks

### Phase 2 Changes

Upgrade `train_idh_monai.py` so training crops better approximate predicted-mask noise rather than only GT-mask bboxes with simple random jitter. Candidate extensions:

- stronger structured bbox expansion policies
- structured bbox shifts
- optional context-heavy crop variants sampled during training
- metadata persistence for the new ROI-robust training recipe

Phase 2 still uses the same validation-time end-to-end macro-F1 calibration flow from Phase 1.

## Data Flow After Phase 1

1. Load MRI modalities
2. Run MONAI bundle segmentation
3. Apply configurable mask postprocess
4. Generate one or more ROI views from predicted mask
5. Run classifier on each ROI view
6. Aggregate probabilities
7. Apply end-to-end calibrated threshold
8. Report one case prediction with shared metadata

## Error Handling

The pipeline must handle these cases explicitly:

- no tumor predicted after postprocess
- ROI collapse after expansion or bbox clipping
- incompatible or missing E2E config artifact
- NaN or invalid probability aggregation inputs

Behavior should be deterministic and surfaced consistently in eval CLI, infer CLI, and app text output.

## Testing Strategy

### Unit tests

- mask postprocess produces deterministic outputs on synthetic 3D masks
- ROI view generation returns valid clipped boxes
- probability aggregation behaves correctly for multi-view inputs
- E2E config loading resolves defaults and override metadata correctly

### Small integration tests

- end-to-end evaluation helpers can consume a saved config artifact
- inference path and app helper both use the same loaded threshold and aggregation metadata

### Manual verification

- run calibration on validation split
- run end-to-end evaluation with baseline config
- run end-to-end evaluation with tuned config
- compare `macro F1`, recalls, `AUC`, and `accuracy`

## Files Likely Affected

### New

- `src/idh_glioma/infer/e2e_roi.py`
- `scripts/calibrate_idh_e2e.py`
- `tests/test_e2e_roi.py`
- `tests/test_e2e_calibration_config.py`

### Modify

- `src/idh_glioma/eval/eval_e2e_monai_zoo.py`
- `src/idh_glioma/infer/pipeline_monai_zoo.py`
- `src/idh_glioma/app_idh_monai.py`
- `src/idh_glioma/train/train_idh_monai.py`
- `README.md`
- `MODEL_CARD.md`

## Trade-Offs Considered

### Option A: Threshold-only fix

Rejected as the main strategy because current wrong cases are not cleanly separable by one threshold alone.

### Option B: ROI robustness first

Recommended because it directly targets the predicted-mask mismatch while preserving the strong current segmentation and classifier baselines.

### Option C: Immediate full retrain

Deferred because it mixes distribution-robustness gains with inference changes and makes attribution harder.

## Recommended Sequence

1. Implement Phase 1 shared ROI config and reusable helpers
2. Add end-to-end calibration sweep on validation macro F1
3. Wire tuned config into eval, infer, and app
4. Compare baseline vs tuned end-to-end metrics
5. Only then decide whether Phase 2 retraining is required
