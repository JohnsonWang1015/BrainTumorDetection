#!/usr/bin/env bash
# Download high-value brain-tumor datasets from Kaggle.
#
# Prerequisites
# -------------
#   1. Install the Kaggle CLI (one-time, isolated from project deps):
#         uv tool install kaggle
#      Or via pip:
#         pip install --user kaggle
#
#   2. Create an API token at https://www.kaggle.com/<user>/account → "Create New API Token"
#      which downloads ``kaggle.json``. Place it at:
#         mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
#         chmod 600 ~/.kaggle/kaggle.json
#
#   3. Confirm auth:
#         kaggle datasets list -s brain --max-size 100000000 | head
#
# Targets selected for this project
# ---------------------------------
#   Each entry is annotated with: (a) size, (b) what it adds vs. the data we
#   already have on disk, (c) which pipeline can consume it.
#
# Usage
# -----
#   bash scripts/download_kaggle_datasets.sh           # download every target
#   bash scripts/download_kaggle_datasets.sh brats20   # download a single target by tag
#
set -euo pipefail

DATASETS_DIR="${DATASETS_DIR:-datasets}"
mkdir -p "${DATASETS_DIR}"

run_target() {
  local tag="$1"
  local kaggle_slug="$2"
  local out_subdir="$3"
  local note="$4"
  echo "──────────────────────────────────────────────────────────────"
  echo "[$tag] $kaggle_slug → ${DATASETS_DIR}/${out_subdir}"
  echo "  ${note}"
  echo "──────────────────────────────────────────────────────────────"
  if [[ -d "${DATASETS_DIR}/${out_subdir}" && -n "$(ls -A "${DATASETS_DIR}/${out_subdir}" 2>/dev/null || true)" ]]; then
    echo "  already present, skipping"
    return 0
  fi
  mkdir -p "${DATASETS_DIR}/${out_subdir}"
  kaggle datasets download -d "${kaggle_slug}" -p "${DATASETS_DIR}/${out_subdir}" --unzip
}

declare -A TARGETS=(
  # tag           slug → out_subdir | description
  [brats20]="awsaf49/brats20-dataset-training-validation|BraTS2020|BraTS 2020 (~6 GB). 369 training cases, 4 modalities + segmentation. Lets us re-train SegResNet on a larger labelled cohort than TCGA-LGG (currently 65)."
  [brats21]="dschettler8845/brats-2021-task1|BraTS2021|BraTS 2021 task 1 (~50 GB). 1,251 multimodal MRI volumes with seg masks. Largest open glioma cohort on Kaggle."
  [brats23_men]="dschettler8845/brats-2023-meningioma|BraTS2023-MEN|BraTS 2023 Meningioma (~20 GB). 1,000+ cases — complements glioma data and matches the 'meningioma' class in our 4-class classifier."
  [figshare3064]="ahmedhamada0/brain-tumor-detection|Brain-Tumor-Figshare|Cheng et al. figshare 3,064-slice T1-weighted MRI dataset (~700 MB). Has pixel-level tumor masks for 233 patients across 3 classes; useful for cross-source 4-class validation."
  [rsna2021]="rsna-miccai-brain-tumor-radiogenomic-classification|RSNA-MICCAI-2021|RSNA-MICCAI 2021 MGMT methylation challenge (~140 GB). Multimodal MRI w/ MGMT labels — relevant as a second molecular endpoint. WARNING: very large."
  [navoneel]="navoneel/brain-mri-images-for-brain-tumor-detection|Brain-MRI-Navoneel|Small 253-image binary tumor classification dataset (~15 MB). Useful as a tiny out-of-distribution holdout for the CT/MRI classifier."
)

ORDER=(brats20 figshare3064 navoneel brats23_men brats21 rsna2021)

if [[ $# -gt 0 ]]; then
  ORDER=("$@")
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found. Install with: uv tool install kaggle" >&2
  echo "       then place credentials at ~/.kaggle/kaggle.json (chmod 600)." >&2
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" && -z "${KAGGLE_USERNAME:-}" ]]; then
  echo "ERROR: no Kaggle credentials. Either:" >&2
  echo "  (a) place kaggle.json at ~/.kaggle/kaggle.json (chmod 600), or" >&2
  echo "  (b) export KAGGLE_USERNAME=... KAGGLE_KEY=..." >&2
  exit 1
fi

for tag in "${ORDER[@]}"; do
  spec="${TARGETS[$tag]:-}"
  if [[ -z "${spec}" ]]; then
    echo "Unknown tag: ${tag}" >&2
    continue
  fi
  IFS='|' read -r slug subdir note <<< "${spec}"
  run_target "${tag}" "${slug}" "${subdir}" "${note}"
done

echo "──────────────────────────────────────────────────────────────"
echo "Done. Inspect: ls ${DATASETS_DIR}"
