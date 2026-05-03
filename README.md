# Detect and Segment IDH Mutation Status in Brain Gliomas

本專案提供一個可直接落地的 End-to-End 流程，整合：

- 腦瘤分割：2D `UNet` baseline、3D `SegResNet` baseline、MONAI Model Zoo `brats_mri_segmentation`
- IDH mutation 狀態分類：2D `MobileNetV3` baseline、3D `DenseNet121` ROI classifier
- End-to-End ROI/config calibration：共享 `artifacts/e2e_idh_config.json`
- 多來源 IDH manifest 建置：`TCGA-LGG`、`TCGA-GBM`、`UCSF-PDGM`、`EGD`
- YOLO 匯出與訓練介接（detect/segment/classify 任務）

---

## Molecular IDH (RNA-seq)

本 repo 現在另外提供一條「分子層」IDH 分類路徑，使用 TCGA-GBM + TCGA-LGG 的 RNA-seq (`log2(TPM+1)`) 與 public masked MAF 標籤，和影像 pipeline 並行。

Quickstart:

```bash
# 1) Prepare pooled molecular artifacts
uv run prepare-idh-molecular \
  --include-sources tcga_gbm tcga_lgg \
  --gbm-data-root datasets/TCGA-GBM/tcga_gbm_downloads/data \
  --lgg-data-root datasets/TCGA-LGG-Molecular/tcga_lgg_downloads/data \
  --output-dir artifacts/molecular \
  --skip-download

# 2) Train all molecular models
uv run train-idh-molecular \
  --input-dir artifacts/molecular \
  --model all \
  --top-k 2000 \
  --save-dir checkpoints/molecular_idh \
  --seed 42

# 3) Run B3 evaluation
uv run eval-idh-molecular \
  --input-dir artifacts/molecular \
  --checkpoint-dir checkpoints/molecular_idh \
  --mode all \
  --output-dir artifacts/molecular_idh_eval \
  --folds 5 \
  --seed 42
```

Example output (real run on current workspace):

```text
prepare: expression_matrix=(60616 genes x 809 patients), idh_labels=880, labeled_with_expression=759
train AUC: logistic=1.0000, lightgbm=1.0000, mlp=0.9972
pooled 5-fold CV AUC: logistic=0.9916±0.0098, lightgbm=0.9924±0.0089, mlp=0.9908±0.0083
source holdout AUC: LGG->GBM {logistic=0.9801, lightgbm=0.9650, mlp=0.9866}
source holdout AUC: GBM->LGG {logistic=0.9722, lightgbm=0.9555, mlp=0.9624}
GBM minority AUPRC: logistic=0.9326, lightgbm=0.9469, mlp=0.9368
```

Artifacts:

- `artifacts/molecular/{expression_matrix.parquet, idh_labels.parquet, cohort_manifest.json, feature_panel.json}`
- `checkpoints/molecular_idh/{logistic.joblib, lightgbm.txt, mlp.pt, training_report.json}`
- `artifacts/molecular_idh_eval/{pooled_cv_results.json, source_holdout_results.json, minority_metrics.json, figures/*.png}`

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

### 2.2 MRIBrainTumor（已停用）

`datasets/MRIBrainTumor/` 的 9 個 zip 是 2020 腦瘤分割挑戰賽資料，**單模態**（每個 case 只有一個 NIfTI 影像 + 一個 mask），不適合 4-channel U-Net。早期 pipeline 用 4 個 symlink 假裝多模態導致模型 Dice 低落，現已從訓練流程移除。Segmentation 與 IDH 分類統一使用 TCGA-LGG（真 4 模態）。如需重新接上單模態資料流，需另外新增 1-channel 模型分支。

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

## 4. 準備 IDH 標籤

`artifacts/idh_labels.csv` 已從 cBioPortal 公開臨床資料填好（study `lgggbm_tcga_pub` 的 `IDH_STATUS` attribute，64/65 cases 匹配）。格式：

```csv
case_id,idh_label
TCGA-CS-4942,1
TCGA-CS-4944,1
```

- `idh_label`: `0=IDH wildtype`, `1=IDH mutant`
- `case_id` 必須對應病例資料夾名稱

如要重新產生（例如新增 case 或更新標籤源），可用以下 Python 程式片段（需網路）：

```python
import json, csv, urllib.request
url = ("https://www.cbioportal.org/api/studies/lgggbm_tcga_pub/clinical-data"
       "?clinicalDataType=SAMPLE&attributeId=IDH_STATUS&pageSize=2000")
data = json.loads(urllib.request.urlopen(url).read())
patient_idh = {d['patientId']: d['value'] for d in data}
with open('artifacts/idh_labels.csv') as f:
    cases = [r['case_id'] for r in csv.DictReader(f)]
mapping = {'Mutant': 1, 'WT': 0}
with open('artifacts/idh_labels.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['case_id', 'idh_label'])
    for cid in cases:
        v = mapping.get(patient_idh.get(cid, ''), '')
        w.writerow([cid, v])
```

模板仍保留於 `artifacts/idh_labels.template.csv`。

---

## 5. 建立 manifest 與資料切分

如果要最短可直接使用流程，先跑一鍵腳本：

```bash
./scripts/prepare_idh_data.sh --dataset-path datasets/BraTS-TCGA-LGG
```

`artifacts/idh_labels.csv` 已預先填好（見 §4），執行該指令會直接 join 標籤並產出 `artifacts/manifest.json`。若 CSV 不存在，腳本會從 template 複製空白 CSV 並提示先填值。

```bash
uv run python -m idh_glioma.data.prepare_dataset \
  --brats-root datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations \
  --idh-labels artifacts/idh_labels.csv \
  --output artifacts/manifest.json
```

若暫時沒有 `idh_labels.csv`，也可先不帶 `--idh-labels`，先跑 segmentation。

也可讓程式自動從 `datasets/` 掃描 BraTS 根目錄：

```bash
uv run python -m idh_glioma.data.prepare_dataset \
  --dataset-root datasets \
  --output artifacts/manifest.json
```

### 5.1 多來源 IDH manifest v2

若要開始把 `TCGA-GBM`、`UCSF-PDGM`、`EGD` 納入同一套 IDH workflow，請改用新的多來源 manifest builder：

```bash
uv run python -m idh_glioma.data.prepare_idh_multisource \
  --dataset-root datasets \
  --idh-labels artifacts/idh_labels.csv \
  --output artifacts/manifest_v2.json
```

安裝後也可直接使用 console script：

```bash
uv run prepare-idh-multisource \
  --dataset-root datasets \
  --idh-labels artifacts/idh_labels.csv \
  --output artifacts/manifest_v2.json
```

`manifest_v2` 會保留目前 training code 相容欄位：

- `case_id`
- `date`
- `modalities`
- `mask_path`
- `idh_label`

並額外加入多來源資訊：

- `source_dataset`
- `source_subject_id`
- `cohort_id`
- `acquisition_stage`
- `mask_kind`
- `label_source`
- `inclusion_flags`
- `qc_flags`
- `provenance`

完整 contract 見：

- `configs/idh_manifest_v2_contract.yaml`

目前 repo 也已提交一份實際生成結果：

- `artifacts/manifest_v2.json`

在 2026-05-02 這次快照中，它反映的是「本地目前可掃描到的來源」：

- 只有 `brats_tcga_lgg`
- `cohort_id = tcga_lgg_only`
- `split = train 45 / val 10 / test 10`

也就是說，多來源 importer 已接上，但若你本地尚未放入 `TCGA-GBM`、`UCSF-PDGM`、`EGD`，輸出的 `manifest_v2` 仍會先退化成目前單來源 `TCGA-LGG` cohort。

### 5.2 本地資料夾規格

`prepare_idh_multisource.py` 目前支援下列來源與預設 root：

- `brats_tcga_lgg` -> `datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations`
- `tcga_gbm` -> `datasets/TCGA-GBM`
- `ucsf_pdgm` -> `datasets/UCSF-PDGM`
- `egd` -> `datasets/EGD`

#### TCGA-LGG

沿用目前單來源流程：

```text
datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/
  TCGA-DU-7015/
    TCGA-DU-7015_1989.06.18_flair.nii.gz
    TCGA-DU-7015_1989.06.18_t1.nii.gz
    TCGA-DU-7015_1989.06.18_t1Gd.nii.gz
    TCGA-DU-7015_1989.06.18_t2.nii.gz
    TCGA-DU-7015_1989.06.18_GlistrBoost_ManuallyCorrected.nii.gz
```

#### TCGA-GBM

目前 importer 假設一個 case 一個資料夾，檔名允許 `t1ce` 作為 `t1Gd` alias：

```text
datasets/TCGA-GBM/
  TCGA-06-0125/
    TCGA-06-0125_flair.nii.gz
    TCGA-06-0125_t1.nii.gz
    TCGA-06-0125_t1ce.nii.gz
    TCGA-06-0125_t2.nii.gz
```

若有 segmentation，可放入與現有 suffix 相容的 mask；若沒有也可，這類 case 仍可當作 classification-only record。

#### UCSF-PDGM

官方 collection 是 skull-stripped / registered NIfTI。Importer 已支援下列常見命名：

- `T2FLAIR` -> `flair`
- `T1` -> `t1`
- `T1c_bias` / `T1gad_bias` -> `t1Gd`
- `T2` -> `t2`
- `tumor_segmentation` -> segmentation mask

建議本地結構：

```text
datasets/UCSF-PDGM/
  UCSF-PDGM-0001/
    UCSF-PDGM-0001_T2FLAIR.nii.gz
    UCSF-PDGM-0001_T1.nii.gz
    UCSF-PDGM-0001_T1c_bias.nii.gz
    UCSF-PDGM-0001_T2.nii.gz
    UCSF-PDGM-0001_tumor_segmentation.nii.gz
```

#### EGD

EGD 論文描述的 structural MRI 檔名是固定的：

- `FLAIR.nii.gz`
- `T1.nii.gz`
- `T1GD.nii.gz`
- `T2.nii.gz`

建議本地結構：

```text
datasets/EGD/
  EGD-0001/
    FLAIR.nii.gz
    T1.nii.gz
    T1GD.nii.gz
    T2.nii.gz
```

### 5.3 Label join 規格

`prepare_idh_multisource.py` 目前支援下列 label metadata 形式。

#### 通用二元格式

```csv
case_id,idh_label
TCGA-DU-7015,1
TCGA-06-0125,0
```

#### TCGA-GBM join 表

目前已支援：

- `submitter_id`
- `paper_IDH_status`

範例：

```tsv
submitter_id	paper_IDH_status
TCGA-06-0125	WT
TCGA-06-0126	IDH1-mutant
TCGA-06-0127	IDH1 and/or IDH2-mutant
```

匯入時會自動映成：

- `WT -> 0`
- `IDH1-mutant -> 1`
- `IDH1 and/or IDH2-mutant -> 1`

#### UCSF-PDGM clinical metadata

目前已支援：

- `ID`
- `IDH status`

範例：

```csv
ID,IDH status
UCSF-PDGM-0001,Mutant
UCSF-PDGM-0002,Wildtype
```

#### EGD genetic labels

目前已支援：

- `subject`
- `IDH mutation status`

範例：

```csv
subject,IDH mutation status
EGD-0001,1
EGD-0002,0
EGD-0003,-1
```

其中：

- `1 = IDH mutant`
- `0 = IDH wildtype`
- `-1` 會被視為缺失值並跳過

### 5.4 常用命令

只做目前本地 `TCGA-LGG`：

```bash
uv run python -m idh_glioma.data.prepare_idh_multisource \
  --include-sources brats_tcga_lgg \
  --output artifacts/manifest_v2.json
```

做 `TCGA-LGG + TCGA-GBM` pooled manifest：

```bash
uv run python -m idh_glioma.data.prepare_idh_multisource \
  --include-sources brats_tcga_lgg tcga_gbm \
  --idh-labels artifacts/idh_labels_tcga_combined.tsv \
  --output artifacts/manifest_v2.json
```

把 `UCSF-PDGM` 留作 external cohort：

```bash
uv run python -m idh_glioma.data.prepare_idh_multisource \
  --include-sources brats_tcga_lgg tcga_gbm ucsf_pdgm \
  --split-mode source_holdout \
  --idh-labels artifacts/idh_labels_multisource.csv \
  --output artifacts/manifest_v2.json
```

加入 `EGD` 做更大的 pooled cohort：

```bash
uv run python -m idh_glioma.data.prepare_idh_multisource \
  --include-sources brats_tcga_lgg tcga_gbm ucsf_pdgm egd \
  --split-mode source_holdout \
  --idh-labels artifacts/idh_labels_multisource.csv \
  --output artifacts/manifest_v2.json
```

---

## 6. 訓練流程

### 6.1 U-Net 分割訓練（TCGA-LGG, 4 模態）

採用 Focal+Dice loss、cosine warmup、val Dice-based checkpoint。45 train cases 達到 test Dice ~0.76：

```bash
uv run train-seg \
  --manifest artifacts/manifest.json \
  --profile a6000 \
  --epochs 100 \
  --output checkpoints/unet2d_tcga_v1.pt
```

### 6.2 MobileNetV3 IDH 分類訓練（TCGA-LGG, 3 模態）

採用 ImageNet pretrained backbone、`pos_weight = sqrt(neg/pos)` 處理 5:1 imbalance、cosine warmup（3 epochs）、val AUC-based checkpoint：

```bash
uv run train-idh \
  --manifest artifacts/manifest.json \
  --profile a6000 \
  --epochs 25 \
  --output checkpoints/mobilenetv3_idh_v3.pt
```

預期 best val AUC 0.7–0.75，test slice AUC 0.6–0.65。若出現 `No IDH labels found`，代表 `idh_labels.csv` 為空或欄位不符（見 §4）。

### 6.3 評估

```bash
uv run eval-seg --ckpt checkpoints/unet2d_tcga_v1.pt    # Dice/IoU (2D U-Net)
uv run eval-idh --ckpt checkpoints/mobilenetv3_idh_v3.pt  # AUC + per-class (2D)
uv run eval-ct                                            # CT 分類器（預設 ckpt + manifest）
```

### 6.4 3D MONAI pipeline（生產推薦）

從 v0.2 開始加入 MONAI 框架的 3D 模型，在所有指標上都優於 2D baseline：

```bash
# 訓練 3D segmentation
uv run train-seg-monai --output checkpoints/segresnet_tcga.pt
# 訓練 3D IDH 分類（含 bbox jitter）
uv run train-idh-monai --output checkpoints/densenet3d_idh_jitter.pt \
  --jitter-expand-max 12 --jitter-shift-max 6
# 校準 IDH threshold
uv run python scripts/calibrate_idh_threshold.py \
  --ckpt checkpoints/densenet3d_idh_jitter.pt
# 校準 end-to-end macro-F1 threshold / ROI config
uv run python scripts/calibrate_idh_e2e.py \
  --split val \
  --output artifacts/e2e_idh_config.json
# 評估
uv run eval-seg-monai      # 3D Dice (TCGA-LGG): 0.910
uv run eval-seg-zoo        # MONAI Model Zoo zero-shot Dice: 0.926
uv run eval-idh-monai      # 3D IDH (GT mask): AUC 1.00
uv run eval-e2e-zoo --e2e-config artifacts/e2e_idh_config.json
                         # E2E (bundle + jitter cls): shared ROI + threshold config
```

下載 MONAI Model Zoo BraTS 預訓練 bundle（首次使用）：

```bash
uv run python -c "from monai.bundle import download; \
  download(name='brats_mri_segmentation', bundle_dir='checkpoints/monai_zoo')"
```

### 6.5 模型 performance 對照

| Pipeline | Test Dice | IDH AUC | IDH accuracy |
|----------|-----------|---------|--------------|
| 2D U-Net + 2D MobileNet | 0.760 | 0.875 (case) | 0.80 |
| 3D SegResNet + 3D DenseNet (GT mask) | 0.910 | **1.00** | 1.00 |
| **MONAI Model Zoo + jitter 3D DenseNet (E2E)** | **0.926** | 0.75 | 0.80 |

3D pipeline 訓練資料：TCGA-LGG（45 train / 10 val / 10 test cases）。MONAI bundle 額外從 BraTS 公開競賽 (~500 cases) 預訓練得來。

補充：最新 `artifacts/e2e_idh_config.json` 是重新校準後的 runtime decision rule，不是上表那個歷史 held-out test 結果。它目前記錄：

- `threshold = 0.13`
- `aggregation = mean`
- `view_margins = [0]`
- `keep_largest = true`
- `dilate_iters = 0`
- validation `macro_f1 = 1.0`
- validation `accuracy = 1.0`
- validation `auc = 1.0`

但這組數值只來自 `10` 個 validation cases，主要用途是固定 app / eval / infer 的共用 decision rule，不應過度解讀成穩定泛化結果。

---

## 7. 端到端推論

### 7.1 生產推薦（MONAI 3D + Model Zoo）

```bash
uv run infer-monai-zoo \
  --case-dir datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-CS-4942 \
  --e2e-config artifacts/e2e_idh_config.json \
  --output-mask outputs/TCGA-CS-4942_pred_mask_zoo.nii.gz
```

### 7.2 End-to-End calibration artifact

`artifacts/e2e_idh_config.json` 現在用來保存 predicted-mask pipeline 的共享設定，供以下路徑共用：

- `uv run eval-e2e-zoo --e2e-config ...`
- `uv run infer-monai-zoo --e2e-config ...`
- `src/idh_glioma/app_idh_monai.py`（預設讀取 `artifacts/e2e_idh_config.json`）

這份設定獨立於 classifier checkpoint 內的 GT-mask threshold metadata，目的在於：

- 用 validation split 的 `macro F1` 校準 end-to-end threshold
- 同步保存 ROI view 與 mask postprocess 參數
- 避免 app / eval / infer 之間使用不同 decision rule

### 7.3 Phase 2 retrain knobs

若 Phase 1 的 end-to-end calibration 無法穩定拉高 held-out macro F1，可對 3D IDH classifier 啟用更強的 noisy-ROI augmentation：

```bash
uv run train-idh-monai \
  --output checkpoints/densenet3d_idh_context.pt \
  --jitter-expand-max 12 \
  --jitter-shift-max 6 \
  --context-view-prob 0.35 \
  --context-extra-max 6
```

- 兩階段：MONAI bundle 切割（zero-shot）→ KeepLargestCC → 3D DenseNet 分類
- ckpt 預設：`checkpoints/monai_zoo/brats_mri_segmentation/models/model.pt` + `checkpoints/densenet3d_idh_jitter.pt`
- 自動讀取 ckpt metadata 中的 calibrated threshold
- Latency ~10–16 s/case（RTX PRO 5000）

### 7.2 2D legacy pipeline

```bash
uv run python -m idh_glioma.infer.pipeline \
  --case-dir datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-CS-4942 \
  --seg-ckpt checkpoints/unet2d_tcga_v1.pt \
  --cls-ckpt checkpoints/mobilenetv3_idh_v3.pt \
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
2. 確認 `idh_labels.csv` 已填好（見 §4），執行 `./scripts/prepare_idh_data.sh` 產生 manifest 後即可訓練。
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
