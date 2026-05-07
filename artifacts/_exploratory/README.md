# Exploratory artifacts — DO NOT use as production

These configs and the paired checkpoint under `checkpoints/_exploratory/` were
side experiments run on 2026-05-03 to try to push the imaging end-to-end IDH
pipeline above 90% test accuracy. They are kept for reference only.

## Why each is quarantined

### `e2e_idh_config_acc90.json`
- `threshold = 0.265`, `accuracy = 0.90`, `macro_f1 = 0.867`
- `threshold_split = "test"` — threshold was selected on the **test** split,
  so the reported accuracy is not a generalization estimate.

### `e2e_idh_config_val_retrain_v2.json`
- Pairs with `checkpoints/_exploratory/densenet3d_idh_retrain_v2.pt`.
- Threshold tuned on val (`val acc 0.90`, `val AUC 0.94`).
- When evaluated on the held-out test split with the same val-tuned threshold,
  the v2 retrain pipeline **collapses to acc 0.50, macro F1 0.45, AUC 0.75**
  (see `outputs/eval_e2e_zoo/eval_e2e_zoo_cases.csv`, 2026-05-03 01:39).
  The aggressive jitter knobs (`expand_max=16`, `shift_max=8`,
  `context_view_prob=0.5`, `context_extra_max=10`) overfit val without
  improving test.

### `e2e_idh_config_train.json` / `e2e_idh_config_val_strict.json`
- Threshold sweeps recorded during the calibration run; superseded by the
  production `artifacts/e2e_idh_config.json`.

### `manifest_val_as_test.json`
- Diagnostic manifest where the val split was duplicated into the test slot
  to debug the calibration loop. Not a real evaluation artifact.

## Conclusion

The 10-case test split is too small for `>90%` accuracy claims to be stable
under a different val-tuned threshold. The methodologically correct path to
higher imaging E2E accuracy is to enlarge the test cohort (TCGA-GBM imaging,
UCSF-PDGM, EGD), not to keep retraining the classifier with stronger
augmentation on the same TCGA-LGG split.

See the §1.2 work in `docs/`.
