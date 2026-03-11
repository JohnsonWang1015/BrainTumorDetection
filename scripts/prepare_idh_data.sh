#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="datasets"
OUTPUT_MANIFEST="artifacts/manifest.json"
IDH_LABELS="artifacts/idh_labels.csv"
VAL_RATIO="0.15"
TEST_RATIO="0.15"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_MANIFEST="$2"
      shift 2
      ;;
    --idh-labels)
      IDH_LABELS="$2"
      shift 2
      ;;
    --val-ratio)
      VAL_RATIO="$2"
      shift 2
      ;;
    --test-ratio)
      TEST_RATIO="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "$OUTPUT_MANIFEST")"
mkdir -p "$(dirname "$IDH_LABELS")"

if [[ ! -f "$IDH_LABELS" ]]; then
  cp "artifacts/idh_labels.template.csv" "$IDH_LABELS"
  echo "Created template label file: $IDH_LABELS"
  echo "Please fill IDH labels (0=wildtype, 1=mutant), then run this script again."
  exit 0
fi

PREPARE_ARGS=(
  -m idh_glioma.data.prepare_dataset
  --output "$OUTPUT_MANIFEST"
  --idh-labels "$IDH_LABELS"
  --val-ratio "$VAL_RATIO"
  --test-ratio "$TEST_RATIO"
)

if [[ -d "$DATASET_PATH/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations" ]]; then
  PREPARE_ARGS+=(--brats-root "$DATASET_PATH/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations")
elif [[ -d "$DATASET_PATH/BraTS-TCGA-LGG" || "$DATASET_PATH" == "datasets" ]]; then
  PREPARE_ARGS+=(--dataset-root "$DATASET_PATH")
else
  PREPARE_ARGS+=(--brats-root "$DATASET_PATH")
fi

uv run python "${PREPARE_ARGS[@]}"

echo "Manifest ready: $OUTPUT_MANIFEST"
echo "Next step:"
echo "  uv run python -m idh_glioma.train.train_idh_classifier --manifest $OUTPUT_MANIFEST --profile a6000 --epochs 50 --output checkpoints/mobilenetv3_idh_best.pt"
