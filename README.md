# Brain Glioma — Segmentation + IDH Mutation Detection

> 一套可直接落地的 **End-to-End 腦膠質瘤分析平台**：從 MRI 影像做腫瘤**分割**、預測 **IDH 突變狀態**(影像 + 分子兩條路徑)，並提供互動式 Web UI。

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.12-blue">
<img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.5-ee4c2c">
<img alt="monai" src="https://img.shields.io/badge/MONAI-3D-00a3e0">
<img alt="status" src="https://img.shields.io/badge/pipeline-end--to--end-success">
<img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-green">
</p>

---

## 🎯 這個專案做什麼

| 任務 | 說明 | 最佳模型 |
|------|------|----------|
| 🔍 **CT/MRI 腫瘤偵測** | 單張影像 → 有/無腫瘤 | MobileNetV3-Large |
| 🧩 **腫瘤分割** | MRI 體積 → 腫瘤遮罩 (3D) | MONAI Model Zoo / SegResNet 3D |
| 🧬 **IDH 突變分類(影像)** | 4 模態 MRI → IDH mutant/wildtype | DenseNet121 3D (ROI) |
| 🔬 **IDH 突變分類(分子)** | RNA-seq / methylation → IDH 狀態 | LightGBM / MLP 多體學融合 |

> IDH 突變狀態是膠質瘤分級與治療決策的關鍵生物標記；本專案同時從**影像**與**分子**兩個層面預測它。

---

## 📊 成績一覽 (held-out test)

| 模型 | 指標 | 數值 |
|------|------|------|
| **CT/MRI 偵測** (MobileNetV3-Large) | Accuracy / AUC / ECE | **99.65% / 0.9997 / 0.0026** (TTA 下 AUC 1.0000;溫度校準後 ECE 0.026→0.0026) |
| **分割** (MONAI Zoo， zero-shot) | Dice | **0.926** |
| **分割** (SegResNet 3D) | Dice | 0.910 |
| **IDH 影像** (DenseNet121 3D， GT-mask ROI) | AUC | 1.00 (single-split) / CV **0.916 ± 0.073** |
| **IDH 影像** (E2E predicted-mask) | Accuracy / AUC | 0.80 / 0.75 |
| **IDH 分子** (LightGBM， RNA-seq) | Pooled 5-fold CV AUC | **0.992 ± 0.009** |
| **IDH 分子** (RNA-seq + methylation 多體學) | Pooled 5-fold CV AUC | **0.993 ± 0.007** |

<sub>影像模型訓練資料:TCGA-LGG(45 train / 10 val / 10 test cases)；MONAI bundle 另在 BraTS 公開競賽 (~500 cases) 預訓練。分子模型:TCGA-GBM + TCGA-LGG pooled cohort。數字為 in-dataset benchmark，非臨床泛化證據。</sub>

---

## 🚀 Quick Start

```bash
# 1) 安裝環境 (uv 為套件管理器)
uv sync --frozen

# 2) 啟動 Web UI (3 個分頁:CT/MRI 偵測、IDH 2D、IDH 3D MONAI)
uv run tumor-app                 # → http://localhost:7860

# 3) 生產推薦的端到端推論 (MONAI 3D + Model Zoo bundle)
uv run infer-monai-zoo \
  --case-dir datasets/BraTS-TCGA-LGG/Pre-operative_TCGA_LGG_NIfTI_and_Segmentations/TCGA-CS-4942 \
  --e2e-config artifacts/e2e_idh_config.json \
  --output-mask outputs/TCGA-CS-4942_pred_mask.nii.gz
```

> 首次使用分割 bundle 需先下載:
> ```bash
> uv run python -c "from monai.bundle import download; download(name='brats_mri_segmentation', bundle_dir='checkpoints/monai_zoo')"
> ```

---

## 🧩 三條 Pipeline

```
                 ┌─────────────────────────────────────────────┐
   MRI volume →  │  Segmentation  →  ROI crop  →  IDH classifier │  → IDH 機率 + 遮罩
   (4 modality)  │  (MONAI Zoo)      (KeepCC)     (DenseNet 3D)   │
                 └─────────────────────────────────────────────┘

   單張影像   →   MobileNetV3-Large  →  有/無腫瘤 + GradCAM

   RNA-seq /     →  特徵選擇 (top-K ∪ prior)  →  LightGBM / MLP  →  IDH 狀態
   methylation      (log2(TPM+1))                (pooled / multi-omics)
```

`configs/pipeline_contract.yaml` 定義統一資料契約(輸入模態、標註優先序、split 規則、輸出格式)，讓 U-Net / MobileNetV3 / SAM3 / YOLOv11 共用同一套資料語意。

---

## 🗂️ 主要指令

<details>
<summary><b>資料準備</b></summary>

```bash
uv run prepare-mri              # BraTS 掃描 → manifest.json
uv run prepare-idh-multisource  # 多來源 IDH manifest_v2 (TCGA-LGG/GBM/UCSF/EGD)
uv run prepare-ct               # Kaggle CT/MRI → ct_manifest.json
uv run prepare-idh-molecular    # 分子 IDH cohort artifacts
```
</details>

<details>
<summary><b>訓練</b></summary>

```bash
# 3D MONAI (生產推薦)
uv run train-seg-monai          # SegResNet 3D 分割
uv run train-idh-monai          # DenseNet121 3D + bbox jitter
uv run train-idh-molecular      # Logistic + LightGBM + MLP (RNA-seq)

# 2D legacy / 其他
uv run train-ct                 # CT/MRI 偵測 (MobileNetV3-Large)
uv run train-seg                # U-Net 2D 分割
uv run train-idh                # MobileNetV3 IDH (2D)
uv run train-yolo               # YOLOv8/v11 偵測
```
</details>

<details>
<summary><b>評估</b></summary>

```bash
uv run eval-ct [--tta]          # CT 偵測 Accuracy/AUC/F1
uv run eval-seg-zoo             # MONAI Zoo zero-shot Dice
uv run eval-seg-monai           # SegResNet 3D Dice
uv run eval-idh-monai           # 3D IDH (GT-mask ROI) AUC
uv run eval-e2e-zoo             # 完整 pipeline:bundle seg + jitter cls
uv run eval-idh-molecular       # B3 報告:pooled CV + source holdout + minority
```
</details>

<details>
<summary><b>推論 / Web UI</b></summary>

```bash
uv run infer-monai-zoo          # 3D 端到端 (Model Zoo bundle, 推薦)
uv run infer-monai              # 3D 端到端 (自訓 SegResNet)
uv run infer                    # 2D 端到端
uv run tumor-app                # Gradio Web UI (3 分頁)
```
</details>

---

## 📁 專案結構

```text
src/idh_glioma/
├── app.py / app_idh.py / app_idh_monai.py   # Gradio Web UI (3 分頁 + GradCAM)
├── data/          # manifest 建置 + Dataset (BraTS / CT / 多來源)
├── models/        # unet2d / mobilenetv3_classifier
├── train/         # 各任務訓練腳本 (CLI)
├── infer/         # 端到端推論 pipeline
├── molecular/     # RNA-seq + methylation 分子 IDH pipeline (與影像獨立)
├── integrations/  # SAM3 / YOLOv11 wrapper
└── eval/          # 指標計算 + 視覺化輸出
```

---

## 📦 資料集

| 資料集 | 用途 | 狀態 |
|--------|------|------|
| **BraTS-TCGA-LGG** (4 模態 flair/t1/t1Gd/t2 + mask) | 分割 + IDH 影像 | ✅ 主力 cohort |
| **TCGA-GBM / UCSF-PDGM / EGD** | 多來源 IDH 擴充 | 🔌 importer 已接，放入即用 |
| **TCGA-GBM + LGG 分子層** (RNA-seq / methylation) | 分子 IDH | ✅ |
| **Kaggle CT/MRI** | 二元腫瘤偵測 | ✅ (6，732 train / 1，443 test) |

> ⚠️ `datasets/MRIBrainTumor/`(2020 單模態挑戰賽)**已停用** —— 早期用 4 個 symlink 假裝多模態導致 Dice 低落，現已移除。分割與 IDH 分類統一使用真 4 模態的 TCGA-LGG。

<details>
<summary><b>IDH 標籤準備 (cBioPortal)</b></summary>

`artifacts/idh_labels.csv` 已從 cBioPortal 公開臨床資料填好(study `lgggbm_tcga_pub` 的 `IDH_STATUS`，64/65 cases 匹配)。格式:

```csv
case_id,idh_label     # 0 = IDH wildtype, 1 = IDH mutant
TCGA-CS-4942,1
```

重新產生(需網路):

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
        w.writerow([cid, mapping.get(patient_idh.get(cid, ''), '')])
```
</details>

<details>
<summary><b>多來源 manifest v2 (TCGA-GBM / UCSF-PDGM / EGD)</b></summary>

```bash
uv run prepare-idh-multisource \
  --include-sources brats_tcga_lgg tcga_gbm ucsf_pdgm egd \
  --split-mode source_holdout \
  --idh-labels artifacts/idh_labels_multisource.csv \
  --output artifacts/manifest_v2.json
```

`manifest_v2` 保留 training 相容欄位 (`case_id` / `modalities` / `mask_path` / `idh_label`)，並加上 `source_dataset` / `cohort_id` / `acquisition_stage` / `qc_flags` / `provenance` 等多來源資訊。完整 contract 見 `configs/idh_manifest_v2_contract.yaml`。

支援的來源與預設路徑、各資料集命名規則(T1c/t1ce → t1Gd alias 等)、各來源 label join 表格式，詳見 `prepare_idh_multisource.py` 的 docstring。
</details>

---

## 🔬 分子 IDH (RNA-seq / 多體學)

與影像 pipeline 並行的「分子層」路徑，使用 TCGA-GBM + LGG 的 RNA-seq(`log2(TPM+1)`)與 public masked MAF 標籤。

```bash
# 1) Pooled 分子 artifacts
uv run prepare-idh-molecular --include-sources tcga_gbm tcga_lgg --skip-download \
  --gbm-data-root datasets/TCGA-GBM/.../data --lgg-data-root datasets/TCGA-LGG-Molecular/.../data \
  --output-dir artifacts/molecular

# 2) 訓練三個模型
uv run train-idh-molecular --input-dir artifacts/molecular --model all --top-k 2000 --seed 42

# 3) B3 評估 (pooled CV + source holdout + GBM minority)
uv run eval-idh-molecular --input-dir artifacts/molecular --mode all --folds 5 --seed 42
```

<details>
<summary><b>多體學融合 (RNA-seq + methylation) 與 late-fusion 結果</b></summary>

```bash
uv run prepare-idh-molecular --modalities rnaseq methylation --skip-download ... \
  --output-dir artifacts/molecular_multimodal
uv run train-idh-molecular --modalities rnaseq methylation --input-dir artifacts/molecular_multimodal --top-k 2000 --seed 42
uv run eval-idh-molecular  --modalities rnaseq methylation --input-dir artifacts/molecular_multimodal --mode all --folds 5 --seed 42
```

最新結果:
- Pooled 5-fold CV best AUC **0.993 ± 0.007**(MLP， early concat)
- Source-holdout best AUC **0.985**(LGG→GBM， MLP)/ **0.977**(GBM→LGG， Logistic)
- GBM minority AUPRC best **0.9502**(per-modality LightGBM， late fusion via mean of probs)
  → 勝過 RNA-seq-only LightGBM(0.9469)與 early-concat LightGBM(0.9425)

重現 late-fusion:`uv run python scripts/exp_late_fusion_idh.py --input-dir artifacts/molecular_multimodal --output artifacts/molecular_idh_multimodal_eval/late_fusion_results.json`
</details>

---

## 🧪 訓練要點 (best practices)

- **分割**:Focal + Dice loss、mask resize 一律 `mode="nearest"`、cosine warmup、依 val Dice 存檔。
- **CT/MRI 偵測**:MobileNetV3-Large + ImageNet 預訓練、cosine warmup、label smoothing 0.05、EMA(0.999)、**依 val AUC 存檔**、可選 `--tta`(水平翻轉平均)、**溫度校準**(`calibrate_ct_temperature.py`，T≈0.51，ECE→0.0026，不影響準確率)。
- **IDH 影像**:3 模態 z-score(非 ImageNet 正規化)、`pos_weight = sqrt(neg/pos)` 處理 imbalance、bbox jitter 增強。
- **效能**:`channels_last`、AMP、TF32、`pin_memory` + `non_blocking`、NIfTI `lru_cache` per-worker 清快取。
- **常見坑**:不要對 MobileNetV3 用 `torch.jit.trace`(SE block 有資料相依控制流);root 分割區常 99% 滿，pip 安裝用 `TMPDIR=/mnt/8tb_hdd2/johnson/tmp`。

---

## 🐳 Docker (GPU)

```bash
docker compose -f docker-compose.gpu.yml build
docker compose -f docker-compose.gpu.yml run --rm trainer
# 或一鍵 baseline:
./scripts/run_baseline_docker.sh
```

Base image:`pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`

---

## 🔌 整合 (選用)

| 工具 | 指令入口 | 用途 |
|------|----------|------|
| **SAM3** (Meta) | `idh_glioma.integrations.sam3_runner` | prompt-based 分割(逐 slice 推論) |
| **YOLOv11** | `idh_glioma.integrations.yolov11_runner` | detect / segment / classify |
| **YOLO 匯出** | `idh_glioma.data.export_yolo` | manifest → YOLO 格式 |

參考：[SAM3](https://github.com/facebookresearch/sam3) ·  [SAM3 權重](https://huggingface.co/facebook/sam3) ·  [YOLOv11](https://docs.ultralytics.com/models/yolo11/)

---

## 🛠️ 開發

```bash
uv run pytest tests/ -v          # 跑測試
basedpyright src/                # 型別檢查 (lenient)
uv build                         # 打包 wheel
```

- Python 3.12(`.python-version` 釘住)
- 所有模組使用 `from __future__ import annotations`
- 每次實驗後 `uv lock` 並提交 `uv.lock`，避免套件漂移

---

## 🗺️ Roadmap

- [x] CT 偵測補 **ECE 校準指標** + **溫度校準** (`eval-ct` 輸出 ECE;`scripts/calibrate_ct_temperature.py` 將 ECE 降到 0.0026)
- [ ] 多分類腫瘤**類型** (glioma / meningioma / pituitary)
- [ ] CI/CD (GitHub Actions:pytest + type check)
- [ ] 放入真實 TCGA-GBM / UCSF-PDGM / EGD，跑 source-holdout 泛化

---

## 📄 License

[Apache License 2.0](LICENSE) © 2026 Johnson Wang. 原始碼、腳本與文件依此授權；資料集與模型權重保留各自的授權條款。
