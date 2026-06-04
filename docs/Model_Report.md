# 腦腫瘤偵測 — 資料集 · 模型 · 評估報告

> IDH 突變偵測與膠質瘤分割管線（Brain Tumor MRI / CT）
> 涵蓋 **Classification（分類）** 與 **Segmentation（分割）** 兩大任務，附推論結果圖（成功 + 失敗案例）。
> 產出日期：2026-06-04 ／ 分支：`develop`

---

## 目錄
1. [資料集 Datasets](#1-資料集-datasets)
2. [模型 Models](#2-模型-models)
3. [評估指標 Evaluation Metrics](#3-評估指標-evaluation-metrics)
4. [推論結果圖 — Classification](#4-推論結果圖--classification)
5. [推論結果圖 — Segmentation](#5-推論結果圖--segmentation)
6. [結論與已知限制](#6-結論與已知限制)

---

## 1. 資料集 Datasets

| 資料集 | 任務 | 模態 | 規模 | 標註 |
|--------|------|------|------|------|
| **Kaggle Brain Tumor (CT + MRI)** | 分類（腫瘤 / 健康） | CT scan、MRI（2D 影像） | CT 4,618（Tumor 2,318 / Healthy 2,300）＋ MRI 5,000（Tumor 3,000 / Healthy 2,000） | 影像級二元標籤 |
| **BraTS-TCGA-LGG** | 分割 + IDH 分類 | 4 模態 NIfTI（FLAIR / T1 / T1Gd / T2） | 65 cases（train 45 / val 10 / test 10） | 體素級腫瘤遮罩 + IDH 標籤 |
| **TCGA-GBM + TCGA-LGG（分子）** | IDH 分類（RNA-seq / 甲基化） | 基因表現 log2(TPM+1) + 甲基化 | pooled 880 患者（759 具表現資料；WT 442 / Mut 438） | 公開 MAF 之 IDH1/IDH2 missense |

**資料契約重點**
- 影像 manifest：`artifacts/manifest.json`，每筆含 `case_id`、`modalities`、`mask_path`、`idh_label`（0=wildtype、1=mutant）。
- 分割目標：whole tumor（GT 遮罩 BraTS 標籤 `{1,2,4}` 二值化為 >0）。
- IDH 影像分類輸入：3-channel slice（flair / t1Gd / t2），per-volume z-score（非 ImageNet 正規化）。
- 分子表現矩陣 index 為去版本號 base ENSG，值為 `log2(TPM+1)`；特徵選擇為 fold-aware 的 `top-K variance ∪ 先驗 panel`。

---

## 2. 模型 Models

| 任務 | 模型 | 架構重點 |
|------|------|----------|
| CT/MRI 腫瘤分類 | **MobileNetV3-Small** | ImageNet 預訓練、3-channel stem、GradCAM 可視化 |
| 膠質瘤分割（2D，legacy） | **U-Net 2D** | 4-channel 輸入 → 1-channel 遮罩，Focal + Dice loss |
| 膠質瘤分割（3D，自訓） | **MONAI SegResNet 3D** | sliding-window + DiceCE，較 2D +0.15 Dice 且 std 減半 |
| 膠質瘤分割（3D，**生產推薦**） | **MONAI Model Zoo `brats_mri_segmentation`** | 在完整 BraTS（~500 cases）預訓練、zero-shot 即勝過自訓模型 |
| IDH 分類（2D） | **MobileNetV3-Large** | ROI crop、校準閾值 0.876（Youden's J） |
| IDH 分類（3D，**生產推薦**） | **MONAI DenseNet121 3D** | GT-mask ROI crop + bbox jitter，閾值 0.0775 |
| 分子 IDH 分類 | **Logistic / LightGBM / MLP** | pooled-cohort；多體學以 per-modality late fusion |
| 腫瘤偵測 | **YOLOv8n / YOLO11n / YOLO11s** | 偵測框，資料量為主要瓶頸 |

**生產管線（推薦）**：`infer-monai-zoo` = Model Zoo bundle 分割（zero-shot Dice 0.926）→ 3D DenseNet121（jitter 訓練）IDH 分類。

---

## 3. 評估指標 Evaluation Metrics

| 模型 | 指標 | 數值 |
|------|------|------|
| CT/MRI 分類（MobileNetV3-Small） | Accuracy / AUC | **96.4%** / **0.993** |
| 分割 U-Net 2D（TCGA-LGG） | Dice | 0.760 ± 0.090（test）；val best 0.806 |
| 分割 SegResNet 3D（TCGA-LGG，自訓） | Dice | **0.910 ± 0.036**（test, 10 cases）；val best 0.918 |
| 分割 MONAI Model Zoo bundle（zero-shot） | Dice | **0.926 ± 0.031**（test, WT channel）— 生產推薦 |
| IDH 分類 MobileNetV3-Large 2D | AUC（case） | 0.875（single-split）；5-fold CV 0.764 ± 0.076 |
| IDH 分類 DenseNet121 3D（GT-mask ROI） | AUC（case） | 5-fold CV **0.916 ± 0.073**（+0.15 vs 2D） |
| IDH 端到端（預測遮罩，閾值 0.0775） | Acc / AUC / macroF1 | **0.80** / **0.75** / 0.69 |
| YOLO 偵測（best） | mAP50 | 0.497（yolo11s） |
| 分子 IDH（LightGBM, RNA-seq pooled） | AUC | pooled 5-fold CV **0.992 ± 0.009** |
| 分子 IDH（RNA-seq + methylation 多體學） | AUC | pooled 5-fold CV best **0.993 ± 0.007**（MLP）；GBM 少數類 AUPRC best **0.9502**（per-modality late fusion） |

評估圖（由 eval 腳本產生）：

| CT/MRI 分類 ROC | CT/MRI 分類 混淆矩陣 |
|---|---|
| ![CT ROC](report_assets/eval_ct_roc.png) | ![CT confusion](report_assets/eval_ct_confusion.png) |

| 分割 3D Dice 分布 | IDH 端到端 ROC（val） | IDH 端到端 混淆矩陣（val） |
|---|---|---|
| ![Seg Dice hist](report_assets/eval_seg_monai_dice_hist.png) | ![E2E ROC](report_assets/eval_e2e_zoo_roc.png) | ![E2E confusion](report_assets/eval_e2e_zoo_confusion.png) |

| 分子 IDH ROC（多體學） | 分子 IDH 校準曲線 |
|---|---|
| ![Mol ROC](report_assets/mol_roc_curves.png) | ![Mol calib](report_assets/mol_calibration_curve.png) |

---

## 4. 推論結果圖 — Classification

### CT/MRI 腫瘤分類（MobileNetV3-Small）— 成功 vs 失敗

![CT/MRI classification montage](report_assets/cls_ct_montage.png)

> 綠色標題 = 預測正確；紅色標題 = 預測錯誤。閾值 0.5。整體測試集 Accuracy 96.4%（1,443 張中 52 張錯誤）。

| 類型 | 案例 | 真實 | 預測 | P(tumor) | 說明 |
|------|------|------|------|----------|------|
| ✅ 成功 | `ct_tumor (917).jpg` | Tumor | Tumor | 1.000 | 高信心正確 |
| ✅ 成功 | `mri_healthy (10).jpg` | Healthy | Healthy | 0.02 | 健康腦正確排除 |
| ❌ 失敗（偽陽性 FP） | `ct_healthy (264).jpg` | Healthy | Tumor | 0.999 | 高信心**誤判**為腫瘤 |
| ❌ 失敗（偽陰性 FN） | `ct_tumor (291).jpg` | Tumor | Healthy | 0.006 | 高信心**漏掉**腫瘤 |

**失敗模式分析**：錯誤多集中於 CT（健康/腫瘤對比較低）與部分 MRI 腫瘤類型（pituitary / meningioma 訊號偏弱，prob 接近閾值）。偽陽性常出現在帶有明顯亮區或偽影的健康影像。

### IDH 突變分類（3D 端到端，DenseNet121 + 預測遮罩）

以 BraTS-TCGA-LGG test 10 cases 為例（閾值 0.0775）：

| 類型 | 案例 | 真實 | 預測 | P(mutant) | 說明 |
|------|------|------|------|-----------|------|
| ✅ 成功 | TCGA-CS-4942 | Mut(1) | Mut(1) | 1.000 | 高信心正確 |
| ✅ 成功 | TCGA-CS-6669 | WT(0) | WT(0) | 0.138 | jitter 訓練成功救回的難案例 |
| ❌ 失敗（FN） | TCGA-HT-8111 | Mut(1) | WT(0) | 0.202 | 突變漏判 |
| ❌ 失敗（FP） | TCGA-DU-8162 | WT(0) | Mut(1) | 0.216 | 野生型誤判為突變 |

> 端到端（含分割誤差傳遞）：Accuracy 0.80、AUC 0.75。多數錯誤的機率落在閾值附近（0.20–0.22），反映 ROI 邊界誤差對 IDH 判讀的敏感性。

---

## 5. 推論結果圖 — Segmentation

### ✅ 成功案例 — MONAI Model Zoo bundle（3D）

![Segmentation success](report_assets/seg_success.png)

> **TCGA-CS-6669 | Dice = 0.794**（此圖為端到端預測遮罩；該案例在純分割 MONAI 評估中 Dice 0.838）。
> 綠線 = Ground Truth，紅線 = 預測。兩者輪廓高度吻合，腫瘤主體與邊界皆精確捕捉。

### ❌ 失敗案例 — Legacy U-Net 2D

![Segmentation failure](report_assets/seg_failure.png)

> **TCGA-DU-7301 | Dice = 0.000 — 預測遮罩為空**。
> 影像中央有明顯腫瘤（綠線標出 GT），但舊版 2D U-Net 在此 slice 幾何下**完全未偵測到任何腫瘤體素**（無紅線）。這正是改用 3D 架構的主因：2D 逐切片推論缺乏體積上下文，在非典型切片上會整片崩潰。

### 分割模型對照（test, 10 cases）

| 模型 | Dice mean ± std | min | max |
|------|------|------|------|
| U-Net 2D（legacy） | 0.760 ± 0.090 | — | — |
| SegResNet 3D（自訓） | 0.910 ± 0.036 | 0.838 | 0.964 |
| **Model Zoo bundle（zero-shot，推薦）** | **0.926 ± 0.031** | — | — |

逐案例 Dice（SegResNet 3D）：最佳 TCGA-DU-7301 = 0.964，最難 TCGA-CS-6669 = 0.838。

---

## 6. 結論與已知限制

**結論**
- **分類**：CT/MRI 腫瘤分類達 96.4% / AUC 0.993，生產可用；分子 IDH（RNA-seq）AUC 0.99 為各路徑最強。
- **分割**：MONAI Model Zoo bundle（zero-shot Dice 0.926）勝過自訓模型，列為生產推薦。
- **IDH 影像分類**：3D（CV AUC 0.916）顯著優於 2D（0.764）；端到端受分割誤差影響降至 AUC 0.75。

**已知限制**
- BraTS-TCGA-LGG 僅 65 cases，影像 IDH 分類樣本量小，CV 變異偏大（±0.07）。
- 端到端 IDH 錯誤多落在閾值附近，對 ROI 邊界誤差敏感。
- YOLO 偵測在現有資料量（~1,100 張）下飽和於 mAP50 ≈ 0.50，需更多資料而非更大 backbone。
- Legacy 2D U-Net 在非典型切片幾何下可能輸出空遮罩，不應用於生產。

---

### 附錄：重現本報告圖片
```bash
# 分割疊圖 + CT 分類蒙太奇（即時由 GT/預測遮罩繪製）
python3 scripts/_make_report_overlays.py        # → docs/report_assets/

# 重新產生評估圖
uv run eval-ct                 # CT 分類 ROC / 混淆矩陣
uv run eval-seg-monai          # 3D 分割 Dice 直方圖
uv run eval-e2e-zoo            # IDH 端到端 ROC / 混淆矩陣
uv run eval-idh-molecular --modalities rnaseq methylation  # 分子 IDH 圖
```
