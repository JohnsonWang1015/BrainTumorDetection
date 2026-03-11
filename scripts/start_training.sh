#!/usr/bin/env bash
set -euo pipefail

MODE="auto"
DATASET_PATH="datasets"
IDH_LABELS=""
PROFILE="a6000"
EPOCHS="20"
MANIFEST="artifacts/manifest.json"
YOLO_EPOCHS="100"
YOLO_TASK="detect"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --idh-labels)
      IDH_LABELS="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --yolo-epochs)
      YOLO_EPOCHS="$2"
      shift 2
      ;;
    --yolo-task)
      YOLO_TASK="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

detect_mode() {
  local p="$1"
  if [[ -d "$p/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations" ]]; then
    echo "brats"
    return
  fi

  if compgen -G "$p/TCGA-*" >/dev/null; then
    echo "brats"
    return
  fi

  if [[ -d "$p/BraTS-TCGA-LGG" ]]; then
    echo "brats"
    return
  fi

  if [[ -f "$p/brain-tumor.yaml" || -f "$p/dataset.yaml" ]]; then
    echo "ultralytics"
    return
  fi

  if [[ -d "$p/Training" || -d "$p/training" ]]; then
    echo "kaggle"
    return
  fi

  if [[ -d "$p/Dataset" ]]; then
    echo "kaggle"
    return
  fi

  echo "unknown"
}

resolve_kaggle_train_dir() {
  local p="$1"
  if [[ -d "$p/Training" ]]; then
    echo "$p/Training"
    return
  fi
  if [[ -d "$p/training" ]]; then
    echo "$p/training"
    return
  fi
  if [[ -d "$p/train" ]]; then
    echo "$p/train"
    return
  fi
  if [[ -d "$p/Dataset" ]]; then
    echo "$p/Dataset"
    return
  fi
  echo "$p"
}

if [[ "$MODE" == "auto" ]]; then
  MODE="$(detect_mode "$DATASET_PATH")"
fi

if [[ "$MODE" == "brats" ]]; then
  echo "[start_training] Mode: BraTS"

  PREPARE_ARGS=(
    -m idh_glioma.data.prepare_dataset
    --output "$MANIFEST"
  )

  if [[ -n "$IDH_LABELS" ]]; then
    PREPARE_ARGS+=(--idh-labels "$IDH_LABELS")
  fi

  if [[ -d "$DATASET_PATH/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations" ]]; then
    PREPARE_ARGS+=(--brats-root "$DATASET_PATH/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations")
  elif [[ -d "$DATASET_PATH/BraTS-TCGA-LGG" || "$DATASET_PATH" == "datasets" ]]; then
    PREPARE_ARGS+=(--dataset-root "$DATASET_PATH")
  else
    PREPARE_ARGS+=(--brats-root "$DATASET_PATH")
  fi

  uv run python "${PREPARE_ARGS[@]}"

  uv run python -m idh_glioma.train.train_segmentation \
    --manifest "$MANIFEST" \
    --profile "$PROFILE" \
    --epochs "$EPOCHS" \
    --output checkpoints/unet2d_best.pt

  if [[ -n "$IDH_LABELS" && -f "$IDH_LABELS" ]]; then
    uv run python -m idh_glioma.train.train_idh_classifier \
      --manifest "$MANIFEST" \
      --profile "$PROFILE" \
      --epochs "$EPOCHS" \
      --output checkpoints/mobilenetv3_idh_best.pt
  else
    echo "[start_training] Skip IDH classifier (no --idh-labels provided)."
  fi
elif [[ "$MODE" == "ultralytics" ]]; then
  echo "[start_training] Mode: Ultralytics"
  DATA_YAML="$DATASET_PATH/brain-tumor.yaml"
  if [[ ! -f "$DATA_YAML" ]]; then
    DATA_YAML="$DATASET_PATH/dataset.yaml"
  fi
  if [[ ! -f "$DATA_YAML" ]]; then
    echo "No dataset yaml found under $DATASET_PATH (need brain-tumor.yaml or dataset.yaml)."
    exit 1
  fi

  uv run python -m idh_glioma.integrations.yolov11_runner \
    --mode train \
    --task "$YOLO_TASK" \
    --data "$DATA_YAML" \
    --model yolo11n.pt \
    --epochs "$YOLO_EPOCHS" \
    --imgsz 640
elif [[ "$MODE" == "kaggle" ]]; then
  echo "[start_training] Mode: Kaggle image classification"
  CLS_DATA_DIR="$(resolve_kaggle_train_dir "$DATASET_PATH")"
  if [[ ! -d "$CLS_DATA_DIR" ]]; then
    echo "No train directory found for Kaggle dataset under $DATASET_PATH"
    exit 1
  fi

  uv run python -m idh_glioma.integrations.yolov11_runner \
    --mode train \
    --task classify \
    --data "$CLS_DATA_DIR" \
    --model yolo11n-cls.pt \
    --epochs "$YOLO_EPOCHS" \
    --imgsz 224
else
  echo "Unable to detect training mode from dataset path: $DATASET_PATH"
  echo "Please set --mode brats, --mode ultralytics, or --mode kaggle"
  exit 1
fi
