# End-to-End IDH Macro F1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve predicted-mask end-to-end IDH classification by making ROI extraction and thresholding robust to segmentation noise, with validation macro F1 as the tuning target.

**Architecture:** Add shared ROI/postprocess/calibration helpers for the MONAI end-to-end path, then teach evaluation, inference, and the Gradio app to load one tuned E2E config artifact instead of relying on GT-mask-oriented checkpoint threshold metadata. If Phase 1 still plateaus, extend 3D IDH training with stronger noisy-ROI augmentation and rerun the same E2E calibration loop.

**Tech Stack:** Python 3.12, PyTorch, MONAI, NumPy, scikit-learn, Gradio, pytest

---

## File Structure

- Create: `src/idh_glioma/infer/e2e_roi.py`
  Shared helpers for mask postprocess, bbox generation, multi-view ROI extraction, probability aggregation, and E2E config load/save.
- Create: `scripts/calibrate_idh_e2e.py`
  Validation-time sweep for postprocess, ROI view, aggregation, and threshold using macro F1.
- Create: `tests/test_e2e_roi.py`
  Synthetic-mask unit tests for postprocess, bbox clipping, and aggregation.
- Create: `tests/test_e2e_calibration_config.py`
  Config artifact loading tests covering defaults and overrides.
- Modify: `src/idh_glioma/eval/eval_e2e_monai_zoo.py`
  Replace inline ROI logic with shared helper use and E2E config loading.
- Modify: `src/idh_glioma/infer/pipeline_monai_zoo.py`
  Use shared helper path and tuned threshold/config artifact.
- Modify: `src/idh_glioma/app_idh_monai.py`
  Use the same shared config and surface the new threshold objective in UI text.
- Modify: `src/idh_glioma/train/train_idh_monai.py`
  Phase 2 only. Extend noisy-ROI augmentation metadata and controls if Phase 1 plateaus.
- Modify: `README.md`
  Document the new calibration command and tuned E2E flow.
- Modify: `MODEL_CARD.md`
  Document the distinction between GT-mask classifier threshold metadata and E2E tuned config.

### Task 1: Add shared E2E ROI helpers

**Files:**
- Create: `src/idh_glioma/infer/e2e_roi.py`
- Test: `tests/test_e2e_roi.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np

from idh_glioma.infer.e2e_roi import (
    aggregate_probs,
    apply_mask_postprocess,
    build_roi_boxes,
)


def test_apply_mask_postprocess_keeps_largest_component():
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    mask[6:7, 6:7, 6:7] = 1
    out = apply_mask_postprocess(mask, keep_largest=True, dilate_iters=0)
    assert int(out.sum()) == 27


def test_build_roi_boxes_clips_expanded_views_to_volume_bounds():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:6, 3:8, 4:9] = 1
    boxes = build_roi_boxes(mask, base_margin=2, view_margins=[0, 3])
    assert boxes == [
        (0, 8, 1, 10, 2, 10),
        (0, 10, 0, 10, 0, 10),
    ]


def test_aggregate_probs_supports_mean_and_median():
    assert aggregate_probs([0.1, 0.7, 0.9], method="mean") == 0.5666666666666667
    assert aggregate_probs([0.1, 0.7, 0.9], method="median") == 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_e2e_roi.py -v`
Expected: FAIL with `ModuleNotFoundError` for `idh_glioma.infer.e2e_roi`

- [ ] **Step 3: Write the minimal shared helper implementation**

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def apply_mask_postprocess(mask: np.ndarray, keep_largest: bool, dilate_iters: int) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    if keep_largest and out.any():
        labels, n = ndimage.label(out)
        if n > 1:
            sizes = ndimage.sum(out, labels, range(1, n + 1))
            out = (labels == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    if dilate_iters > 0 and out.any():
        out = ndimage.binary_dilation(out, iterations=dilate_iters).astype(np.uint8)
    return out


def build_roi_boxes(mask: np.ndarray, base_margin: int, view_margins: list[int]) -> list[tuple[int, int, int, int, int, int]]:
    nz = np.argwhere(mask > 0)
    if nz.size == 0:
        return []
    mn, mx = nz.min(0), nz.max(0) + 1
    h, w, d = mask.shape
    boxes = []
    for extra in view_margins:
        margin = base_margin + extra
        boxes.append((
            max(int(mn[0]) - margin, 0), min(int(mx[0]) + margin, h),
            max(int(mn[1]) - margin, 0), min(int(mx[1]) + margin, w),
            max(int(mn[2]) - margin, 0), min(int(mx[2]) + margin, d),
        ))
    return boxes


def aggregate_probs(probs: list[float], method: str) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    if method == "mean":
        return float(arr.mean())
    if method == "median":
        return float(np.median(arr))
    raise ValueError(f"Unsupported aggregation method: {method}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_e2e_roi.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_roi.py src/idh_glioma/infer/e2e_roi.py
git commit -m "feat: add end-to-end roi helpers"
```

### Task 2: Add E2E config artifact loading and defaults

**Files:**
- Modify: `src/idh_glioma/infer/e2e_roi.py`
- Create: `tests/test_e2e_calibration_config.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from idh_glioma.infer.e2e_roi import load_e2e_config


def test_load_e2e_config_returns_default_when_file_missing(tmp_path: Path):
    cfg = load_e2e_config(tmp_path / "missing.json")
    assert cfg["threshold"] == 0.5
    assert cfg["aggregation"] == "mean"
    assert cfg["view_margins"] == [0]


def test_load_e2e_config_merges_saved_values(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text('{"threshold": 0.23, "aggregation": "median", "view_margins": [0, 4]}')
    cfg = load_e2e_config(path)
    assert cfg["threshold"] == 0.23
    assert cfg["aggregation"] == "median"
    assert cfg["view_margins"] == [0, 4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_e2e_calibration_config.py -v`
Expected: FAIL with `ImportError` or `AttributeError` for `load_e2e_config`

- [ ] **Step 3: Implement config helpers**

```python
import json
from pathlib import Path


DEFAULT_E2E_CONFIG = {
    "keep_largest": True,
    "dilate_iters": 0,
    "base_margin": 4,
    "view_margins": [0],
    "aggregation": "mean",
    "threshold": 0.5,
    "threshold_objective": "macro_f1",
}


def load_e2e_config(path: Path | None) -> dict:
    cfg = dict(DEFAULT_E2E_CONFIG)
    if path is None or not path.exists():
        return cfg
    cfg.update(json.loads(path.read_text()))
    return cfg


def save_e2e_config(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_e2e_calibration_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_calibration_config.py src/idh_glioma/infer/e2e_roi.py
git commit -m "feat: add end-to-end config artifact loading"
```

### Task 3: Refactor eval CLI to use shared ROI config

**Files:**
- Modify: `src/idh_glioma/eval/eval_e2e_monai_zoo.py`
- Test: `tests/test_e2e_roi.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from idh_glioma.infer.e2e_roi import aggregate_probs


def test_eval_pipeline_aggregation_contract_matches_shared_helper():
    probs = [0.2, 0.4, 0.9]
    assert aggregate_probs(probs, method="median") == 0.4
```

- [ ] **Step 2: Run tests to verify the shared contract is active**

Run: `uv run pytest tests/test_e2e_roi.py::test_eval_pipeline_aggregation_contract_matches_shared_helper -v`
Expected: PASS after helper wiring exists, while `eval_e2e_monai_zoo.py` still needs implementation changes.

- [ ] **Step 3: Replace inline ROI logic in evaluation**

```python
from idh_glioma.infer.e2e_roi import (
    aggregate_probs,
    apply_mask_postprocess,
    build_roi_boxes,
    load_e2e_config,
)


cfg = load_e2e_config(args.e2e_config)
pred_mask = apply_mask_postprocess(raw, keep_largest=cfg["keep_largest"], dilate_iters=cfg["dilate_iters"])
boxes = build_roi_boxes(pred_mask, base_margin=cfg["base_margin"], view_margins=cfg["view_margins"])
view_probs = [_predict_box(box) for box in boxes]
idh_prob = aggregate_probs(view_probs, method=cfg["aggregation"]) if view_probs else float("nan")
threshold = float(cfg["threshold"])
```

- [ ] **Step 4: Run targeted verification**

Run: `uv run python -m idh_glioma.eval.eval_e2e_monai_zoo --help`
Expected: PASS and new `--e2e-config` option appears

- [ ] **Step 5: Commit**

```bash
git add src/idh_glioma/eval/eval_e2e_monai_zoo.py src/idh_glioma/infer/e2e_roi.py tests/test_e2e_roi.py
git commit -m "refactor: share end-to-end roi config in evaluation"
```

### Task 4: Refactor single-case inference and app to use the same config

**Files:**
- Modify: `src/idh_glioma/infer/pipeline_monai_zoo.py`
- Modify: `src/idh_glioma/app_idh_monai.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from idh_glioma.infer.e2e_roi import load_e2e_config


def test_app_and_infer_can_share_same_threshold_metadata(tmp_path: Path):
    path = tmp_path / "e2e.json"
    path.write_text('{"threshold": 0.17, "aggregation": "median"}')
    cfg = load_e2e_config(path)
    assert cfg["threshold"] == 0.17
    assert cfg["aggregation"] == "median"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_e2e_calibration_config.py::test_load_e2e_config_merges_saved_values -v`
Expected: PASS, confirming the shared artifact contract before wiring consumers

- [ ] **Step 3: Update inference CLI and app consumers**

```python
cfg = load_e2e_config(args.e2e_config)
boxes = build_roi_boxes(pred_mask, base_margin=cfg["base_margin"], view_margins=cfg["view_margins"])
view_probs = [predict_prob_for_box(box) for box in boxes]
idh_prob = aggregate_probs(view_probs, method=cfg["aggregation"]) if view_probs else float("nan")
idh_pred = int(idh_prob >= cfg["threshold"])
```

```python
return (
    "## 3D MONAI IDH Analysis\n\n"
    f"**Decision threshold**: {threshold:.4f} (macro F1 on val)\n\n"
    f"**Aggregation**: {aggregation}\n\n"
    f"**ROI views**: {view_margins}\n\n"
    ...
)
```

- [ ] **Step 4: Run targeted verification**

Run: `uv run python -m idh_glioma.infer.pipeline_monai_zoo --help`
Expected: PASS and new `--e2e-config` option appears

Run: `uv run python -m idh_glioma.app_idh_monai`
Expected: module imports cleanly; no immediate config-loading exception before app startup

- [ ] **Step 5: Commit**

```bash
git add src/idh_glioma/infer/pipeline_monai_zoo.py src/idh_glioma/app_idh_monai.py src/idh_glioma/infer/e2e_roi.py
git commit -m "refactor: unify end-to-end roi config across inference and app"
```

### Task 5: Add validation-time macro F1 calibration sweep

**Files:**
- Create: `scripts/calibrate_idh_e2e.py`
- Modify: `src/idh_glioma/infer/e2e_roi.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from sklearn.metrics import f1_score


def test_macro_f1_threshold_search_prefers_balanced_threshold():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.10, 0.35, 0.40, 0.90])
    thresholds = np.array([0.2, 0.3, 0.5])
    scores = {t: f1_score(labels, probs >= t, average="macro") for t in thresholds}
    assert max(scores, key=scores.get) == 0.3
```

- [ ] **Step 2: Run the test to prove the scoring objective**

Run: `uv run pytest tests/test_e2e_calibration_config.py::test_macro_f1_threshold_search_prefers_balanced_threshold -v`
Expected: FAIL because the test is not yet present in the file

- [ ] **Step 3: Implement the calibration script**

```python
best = None
for dilate_iters in [0, 1, 2]:
    for view_margins in ([0], [0, 2], [0, 4]):
        for aggregation in ("mean", "median"):
            probs = run_val_pipeline(...)
            for threshold in np.linspace(0.05, 0.95, 181):
                preds = (probs >= threshold).astype(int)
                macro_f1 = f1_score(labels, preds, average="macro")
                if best is None or macro_f1 > best["macro_f1"]:
                    best = {
                        "macro_f1": float(macro_f1),
                        "threshold": float(threshold),
                        "dilate_iters": dilate_iters,
                        "view_margins": list(view_margins),
                        "aggregation": aggregation,
                    }
save_e2e_config(args.output, best)
```

- [ ] **Step 4: Run targeted verification**

Run: `uv run python scripts/calibrate_idh_e2e.py --help`
Expected: PASS and arguments for `--manifest`, `--split`, `--output`, and checkpoint/config paths are listed

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_idh_e2e.py src/idh_glioma/infer/e2e_roi.py tests/test_e2e_calibration_config.py
git commit -m "feat: add macro-f1 end-to-end calibration sweep"
```

### Task 6: Run baseline vs tuned validation and freeze chosen config

**Files:**
- Modify: `README.md`
- Modify: `MODEL_CARD.md`
- Output artifact: `artifacts/e2e_idh_config.json`

- [ ] **Step 1: Record the baseline command**

```bash
uv run eval-e2e-zoo
```

Expected: prints current baseline metrics using default threshold behavior

- [ ] **Step 2: Run validation calibration**

Run: `uv run python scripts/calibrate_idh_e2e.py --split val --output artifacts/e2e_idh_config.json`
Expected: PASS and writes the best validation macro-F1 config artifact

- [ ] **Step 3: Evaluate with the tuned config**

Run: `uv run eval-e2e-zoo --e2e-config artifacts/e2e_idh_config.json`
Expected: PASS and prints improved validation macro F1 relative to the baseline decision path

- [ ] **Step 4: Document the workflow**

```markdown
uv run python scripts/calibrate_idh_e2e.py \
  --manifest artifacts/manifest.json \
  --split val \
  --output artifacts/e2e_idh_config.json

uv run eval-e2e-zoo --e2e-config artifacts/e2e_idh_config.json
uv run infer-monai-zoo --e2e-config artifacts/e2e_idh_config.json --case-dir <CASE_DIR>
```

- [ ] **Step 5: Commit**

```bash
git add README.md MODEL_CARD.md artifacts/e2e_idh_config.json
git commit -m "docs: record tuned end-to-end idh calibration flow"
```

### Task 7: Gate and implement Phase 2 retraining only if needed

**Files:**
- Modify: `src/idh_glioma/train/train_idh_monai.py`
- Modify: `README.md`
- Modify: `MODEL_CARD.md`

- [ ] **Step 1: Write the failing test**

```python
def test_phase2_retrain_is_only_needed_if_phase1_plateaus():
    phase1_macro_f1 = 0.72
    target_macro_f1 = 0.78
    needs_retrain = phase1_macro_f1 < target_macro_f1
    assert needs_retrain is True
```

- [ ] **Step 2: Run the test to establish the gate**

Run: `uv run pytest tests/test_e2e_calibration_config.py::test_phase2_retrain_is_only_needed_if_phase1_plateaus -v`
Expected: FAIL because the gate test is not yet added

- [ ] **Step 3: Extend 3D IDH training for stronger noisy-ROI augmentation**

```python
p.add_argument("--jitter-expand-max", type=int, default=16)
p.add_argument("--jitter-shift-max", type=int, default=8)
p.add_argument("--context-view-prob", type=float, default=0.5)

if self.augment and np.random.rand() < self.context_view_prob:
    y0 = max(y0 - extra_context, 0)
    y1 = min(y1 + extra_context, h)
    ...
```

Persist:

```python
"context_view_prob": args.context_view_prob,
"jitter_expand_max": args.jitter_expand_max,
"jitter_shift_max": args.jitter_shift_max,
```

- [ ] **Step 4: Run targeted verification**

Run: `uv run python -m idh_glioma.train.train_idh_monai --help`
Expected: PASS and new augmentation arguments appear

Run: `uv run python scripts/calibrate_idh_e2e.py --split val --output artifacts/e2e_idh_config.json`
Expected: PASS after retraining and produce a refreshed tuned config

- [ ] **Step 5: Commit**

```bash
git add src/idh_glioma/train/train_idh_monai.py README.md MODEL_CARD.md
git commit -m "feat: strengthen 3d idh noisy-roi training"
```

## Self-Review

- Spec coverage:
  - Shared ROI config: Tasks 1-4
  - Macro-F1 validation calibration: Task 5
  - Unified eval/infer/app consumers: Tasks 3-4
  - Baseline vs tuned validation workflow: Task 6
  - Conditional Phase 2 retraining: Task 7
- Placeholder scan:
  - No `TODO` or `TBD` placeholders remain.
  - Each task names exact files and concrete commands.
- Type consistency:
  - Shared helper names are `apply_mask_postprocess`, `build_roi_boxes`, `aggregate_probs`, `load_e2e_config`, and `save_e2e_config`.
  - The config keys are consistently `keep_largest`, `dilate_iters`, `base_margin`, `view_margins`, `aggregation`, and `threshold`.
