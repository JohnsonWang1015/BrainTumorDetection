#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <CASE_DIR> [SEG_CKPT] [CLS_CKPT] [OUTPUT_MASK]"
  exit 1
fi

CASE_DIR="$1"
SEG_CKPT="${2:-checkpoints/unet2d_best.pt}"
CLS_CKPT="${3:-checkpoints/mobilenetv3_idh_best.pt}"
OUTPUT_MASK="${4:-outputs/pred_mask.nii.gz}"

uv sync --frozen
uv run python -m idh_glioma.infer.pipeline \
  --case-dir "${CASE_DIR}" \
  --seg-ckpt "${SEG_CKPT}" \
  --cls-ckpt "${CLS_CKPT}" \
  --profile a6000 \
  --output-mask "${OUTPUT_MASK}"
