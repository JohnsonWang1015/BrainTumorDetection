#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="datasets/MRIBrainTumor"
OUT_DIR="${ROOT_DIR}/extracted"
mkdir -p "${OUT_DIR}"

for z in "${ROOT_DIR}"/*.zip; do
  unzip -o "${z}" -d "${OUT_DIR}"
done

echo "Extracted archives to ${OUT_DIR}"
