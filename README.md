# Detect and Segment IDH Mutation Status in Brain Gliomas

本專案提供一個可直接落地的 End-to-End 流程，整合：

- 腦瘤分割（預設 U-Net 2D，並提供 SAM3 介接）
- IDH mutation 狀態分類（MobileNetV3）
- YOLOv11 資料匯出與訓練介接（detect/segment/classify 任務）
- 針對你目前 `datasets/` 目錄下資料集的自動掃描與 manifest 建置

---

## 1. 專案結構

```text
brain-tumor-detection/
├── datasets/
├── src/idh_glioma/
│   ├── data/
│   │   ├── prepare_dataset.py
│   │   ├── datasets.py
│   │   └── export_yolo.py
│   ├── models/
│   │   ├── unet2d.py
│   │   └── mobilenetv3_classifier.py
│   ├── train/
│   │   ├── train_segmentation.py
│   │   └── train_idh_classifier.py
│   ├── integrations/
│   │   ├── sam3_runner.py
│   │   └── yolov11_runner.py
│   └── infer/pipeline.py
├── pyproject.toml
├── uv.lock
├── configs/pipeline_contract.yaml
└── README.md
```

`configs/pipeline_contract.yaml` 定義統一資料契約（輸入模態、標註優先順序、split 規則、輸出格式），讓 U-Net / MobileNetV3 / SAM3 / YOLOv11 可以共用同一個資料語意。

---

## 2. 已對應目前資料集狀況

### 2.1 BraTS-TCGA-LGG

已偵測到結構為：

- `TCGA-xxxx/TCGA-xxxx_date_flair.nii.gz`
- `TCGA-xxxx/TCGA-xxxx_date_t1.nii.gz`
- `TCGA-xxxx/TCGA-xxxx_date_t1Gd.nii.gz`
- `TCGA-xxxx/TCGA-xxxx_date_t2.nii.gz`
- `TCGA-xxxx/TCGA-xxxx_date_GlistrBoost_ManuallyCorrected.nii.gz`（優先）
- `TCGA-xxxx/TCGA-xxxx_date_GlistrBoost.nii.gz`

`prepare_dataset.py` 會自動找出以上影像與 mask，建立 `artifacts/manifest.json`。

### 2.2 MRIBrainTumor

目前為多個 zip 分片，可先在本地或 A6000 主機解壓後納入資料流程。

```bash
./scripts/unzip_mri_dataset.sh
```

---

## 3. 環境安裝

```bash
uv sync --frozen
```

或一鍵執行：

```bash
./scripts/setup_env.sh
```

如果你的 A6000 主機要用 CUDA 版 PyTorch，請改用官方 CUDA wheel 安裝指令。

### 3.1 Docker + GPU（A6000）

專案已補上 GPU 版容器環境：

- `Dockerfile`
- `docker-compose.gpu.yml`
- `scripts/run_baseline_docker.sh`

直接執行 baseline：

```bash
./scripts/run_baseline_docker.sh
```

手動執行方式：

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm trainer
```

如果要跑推論，也可改 command：

```bash
docker compose -f docker-compose.gpu.yml run --rm \
  trainer bash scripts/run_end_to_end_inference.sh <CASE_DIR>
```

建議在 A6000 上先確認：

- `python --version`
- `nvidia-smi`
- `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

---

## 4. 準備 IDH 標籤（重要）

你目前資料夾內可直接找到 segmentation 標註，但未發現可直接訓練分類用的 IDH 標籤表。

請準備一個 CSV（例如 `artifacts/idh_labels.csv`）：

```csv
case_id,idh_label
TCGA-CS-4942,1
TCGA-CS-4944,0
```

- `idh_label`: `0=IDH wildtype`, `1=IDH mutant`
- `case_id` 必須對應病例資料夾名稱（例如 `TCGA-CS-4942`）

可直接複製模板：`artifacts/idh_labels.template.csv`

---

## 5. 建立 manifest 與資料切分

```bash
uv run python -m idh_glioma.data.prepare_dataset \
  --brats-root datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations \
  --idh-labels artifacts/idh_labels.csv \
  --output artifacts/manifest.json
```

若你暫時沒有 `idh_labels.csv`，也可先不帶 `--idh-labels`，先跑 segmentation。

也可讓程式自動從 `datasets/` 掃描 BraTS 根目錄：

```bash
uv run python -m idh_glioma.data.prepare_dataset \
  --dataset-root datasets \
  --output artifacts/manifest.json
```

---

## 6. 訓練流程

### 6.1 U-Net 分割訓練

```bash
uv run python -m idh_glioma.train.train_segmentation \
  --manifest artifacts/manifest.json \
  --profile a6000 \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --amp \
  --cudnn-benchmark \
  --tf32 \
  --lr 1e-4 \
  --output checkpoints/unet2d_best.pt
```

### 6.2 MobileNetV3 IDH 分類訓練

```bash
uv run python -m idh_glioma.train.train_idh_classifier \
  --manifest artifacts/manifest.json \
  --profile a6000 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --amp \
  --cudnn-benchmark \
  --tf32 \
  --lr 3e-4 \
  --output checkpoints/mobilenetv3_idh_best.pt
```

若出現 `No IDH labels found`，代表你尚未提供 `idh_labels.csv` 或欄位不符。

---

## 7. 端到端推論

```bash
uv run python -m idh_glioma.infer.pipeline \
  --case-dir datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-CS-4942 \
  --seg-ckpt checkpoints/unet2d_best.pt \
  --cls-ckpt checkpoints/mobilenetv3_idh_best.pt \
  --profile a6000 \
  --batch-size 16 \
  --amp \
  --cudnn-benchmark \
  --tf32 \
  --output-mask outputs/TCGA-CS-4942_pred_mask.nii.gz
```

輸出：

- 預測 segmentation mask (`.nii.gz`)
- 預測 IDH mutation 機率與類別

也可用 golden-path 一鍵腳本：

```bash
./scripts/run_end_to_end_inference.sh \
  datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-CS-4942
```

---

## 8. SAM3 介接（Meta）

你要求 segmentation 可使用 SAM3，已提供 `sam3_runner.py`。

```bash
uv run python -m idh_glioma.integrations.sam3_runner \
  --image datasets/.../TCGA-XXX_flair.nii.gz \
  --checkpoint /path/to/sam3_checkpoint.pt \
  --model-type vit_h \
  --output outputs/sam3_mask.nii.gz
```

說明：

- 需在目標主機安裝官方 SAM3 套件與相容權重
- 本專案用切片式推論（3D volume 逐 slice）
- 預設使用中心框 prompt，可自行擴充成點、框或自動 prompt

---

## 9. YOLOv11 介接

### 9.1 匯出 YOLO 資料格式

```bash
uv run python -m idh_glioma.data.export_yolo \
  --manifest artifacts/manifest.json \
  --out artifacts/yolo
```

### 9.2 用 YOLOv11 訓練/驗證/推論

```bash
# Train (segment/detect/classify 任一)
uv run python -m idh_glioma.integrations.yolov11_runner \
  --mode train \
  --task detect \
  --data artifacts/yolo/dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640

# Validate
uv run python -m idh_glioma.integrations.yolov11_runner \
  --mode val \
  --task detect \
  --model-path runs/yolo11/detect/train/weights/best.pt \
  --data artifacts/yolo/dataset.yaml

# Predict
uv run python -m idh_glioma.integrations.yolov11_runner \
  --mode predict \
  --task detect \
  --model-path runs/yolo11/detect/train/weights/best.pt \
  --source artifacts/yolo/images/test
```

---

## 10. 建議訓練策略（A6000）

- Segmentation：先用 U-Net baseline 收斂，再嘗試 SAM3 fine-tuning 或 prompt-based hybrid
- Classification：先做 slice-level MobileNetV3，後續可升級到 case-level MIL/3D backbone
- 多任務整合：可在後續階段將 segmentation mask 特徵與分類特徵融合

建議監控指標：

- Segmentation：Dice, IoU, Hausdorff (可後續擴充)
- IDH Classification：AUC, F1, sensitivity/specificity

---

## 11. 接下來可直接做的事

1. 先在 A6000 主機安裝 CUDA 對應 PyTorch 與需求套件。
2. 放入 `idh_labels.csv` 後執行 manifest + 分類訓練。
3. 先跑 U-Net baseline，再切到 SAM3/YOLOv11 實驗分支比較指標。

如果你想直接串起 baseline 流程：

```bash
./scripts/run_baseline_pipeline.sh
```

如果你在 `datasets/` 新增了資料，建議改用自動偵測啟動腳本：

```bash
# BraTS / Ultralytics 自動判斷
./scripts/start_training.sh --dataset-path datasets --mode auto

# 明確指定 BraTS 路徑（可加 IDH 標籤）
./scripts/start_training.sh \
  --mode brats \
  --dataset-path datasets/BraTS-TCGA-LGG \
  --idh-labels artifacts/idh_labels.csv

# 明確指定 Ultralytics 資料集（需有 brain-tumor.yaml 或 dataset.yaml）
./scripts/start_training.sh \
  --mode ultralytics \
  --dataset-path datasets/Ultralytics \
  --yolo-task detect

# Kaggle / Kaggle_multimodal 類別資料夾格式（YOLO classify）
./scripts/start_training.sh \
  --mode kaggle \
  --dataset-path datasets/Kaggle
```

`--profile a6000` 會在 GPU 可用時自動套用較高吞吐參數（batch size / workers / prefetch / AMP / TF32）。

---

## 12. 參考文件

- SAM3 官方 Repo: `https://github.com/facebookresearch/sam3`
- SAM3 權重申請: `https://huggingface.co/facebook/sam3`
- YOLOv11 官方文件: `https://docs.ultralytics.com/models/yolo11/`

---

## 13. 版本與相容性建議

此專案在遷移到 A6000 主機後，請優先固定環境版本（Torch + CUDA + Ultralytics + SAM3），避免因套件升級造成行為漂移。

- 建議做法：每次實驗後更新 lock 檔（`uv lock`）並提交 `uv.lock`
- 在 README 指令中，先跑 baseline（U-Net + MobileNetV3），再擴充到 SAM3 / YOLOv11

依賴更新標準流程：

```bash
uv lock
uv sync --frozen
```
