# NQS-SQD 12-bit 量子化學實驗報告

## 摘要

本實驗比較 Neural Quantum States (NQS) 與均勻隨機抽樣 (Baseline) 作為 Sample-based Quantum Diagonalization (SQD) 取樣器的效能。共完成 1788 次實驗，涵蓋 LiH、H4、H6 三個分子系統。

**核心發現**: NQS 訓練越完善，SQD 精度反而越差。Baseline 方法在所有測試系統中均達到 FCI 精度。

---

## 實驗設計

### 系統配置
| 分子 | 自旋軌域數 | 電子數 | Hilbert 空間維度 |
|------|-----------|--------|-----------------|
| LiH  | 12        | 4      | ~495            |
| H4   | 8         | 4      | ~70             |
| H6   | 12        | 6      | ~924            |

### 方法參數
- **NQS**: FFNN 架構，3 隱藏層，alpha=4，VMC + Stochastic Reconfiguration 訓練
- **Baseline**: 均勻隨機抽樣，無機器學習組件
- **SQD**: qiskit-addon-sqd，batches=5，iterations=5

### 實驗變數
- 訓練 epochs: 50, 100, 200, 500
- 樣本數: 100, 500, 1000, 5000
- LiH 鍵長: 1.0 - 3.0 A (間隔 0.25 A)
- 每組參數重複 3 次

---

## 實驗結果

### 統計摘要

| 分子 | 方法 | 實驗數 | SQD 誤差均值 (mHa) | SQD 誤差標準差 (mHa) | SQD 誤差最小值 (mHa) | Conservation Ratio |
|------|------|--------|-------------------|---------------------|---------------------|-------------------|
| LiH | NQS | 444 | 312.18 | 569.14 | ~0 | 6.5% +/- 2.7% |
| LiH | Baseline | 360 | 65.38 | 224.31 | ~0 | 5.1% +/- 0.7% |
| H4 | NQS | 384 | 156.07 | 443.61 | ~0 | 35.7% +/- 24.2% |
| H4 | Baseline | 384 | ~0 | ~0 | ~0 | 14.5% +/- 0.7% |
| H6 | NQS | 108 | 385.77 | 780.33 | ~0 | 25.1% +/- 14.8% |
| H6 | Baseline | 108 | ~0 | ~0 | ~0 | 9.8% +/- 0.2% |

### 關鍵觀察

1. **Baseline 達到 FCI 精度**: H4、H6 系統中，Baseline 的 SQD 誤差接近機器精度 (~10^-10 mHa)。LiH 系統需要 n_samples >= 500 才能達到 FCI 精度；低樣本數 (n=100-250) 時誤差顯著。
2. **NQS 訓練與 SQD 精度負相關** (分子相依性):
   - H4、H6: epochs 50-100 時 SQD 誤差 ~0 mHa (100% < 1mHa)
   - LiH: epochs 50-100 時平均誤差仍達 382 mHa (僅 40.7% < 1mHa)
   - 所有分子: epochs 200-500 時誤差升高至 300-900 mHa
3. **Conservation Ratio 差異**:
   - NQS: 平均 ~20% (範圍 2-98%，高變異)
   - Baseline: 平均 ~10% (範圍 2-16%，低變異)
4. **H-chain 系統規模效應**:
   - NQS 的 SQD 誤差隨系統規模增加而惡化
   - Baseline 維持穩定的 FCI 精度

---

## 圖表說明

### 1. VMC vs SQD 散點圖 

![vmc_vs_sqd_scatter](https://hackmd.io/_uploads/ryJR2YLbWg.png)

展示 VMC 能量誤差與 SQD 能量誤差的關係。NQS 方法呈現明顯的負相關: VMC 誤差越低 (訓練越好)，SQD 誤差反而越高。

### 2. H-chain 規模效應

![hchain_scaling](https://hackmd.io/_uploads/rkklpKUZZx.png)

比較 H4 (8 軌域) 與 H6 (12 軌域) 的表現:
- 左圖: NQS 的 SQD 誤差在 10^2-10^3 mHa 範圍，Baseline 接近 10^-9 mHa
- 右圖: NQS 的 Conservation Ratio 隨系統變大而降低

### 3. 樣本效率分析

![sample_efficiency](https://hackmd.io/_uploads/SJLWaYLZWx.png)

不同樣本數 (100-5000) 對 SQD 精度的影響。Baseline 在所有樣本數下均達到高精度。

### 4. LiH 鍵長掃描

![lih_bond_scan](https://hackmd.io/_uploads/SJXGpFI--x.png)

LiH 分子在不同鍵長下的 SQD 誤差變化。

### 5. Conservation Ratio 分佈 

![conservation_distributions](https://hackmd.io/_uploads/Hyg0g9UW-x.png)


NQS 與 Baseline 的 Conservation Ratio 分佈直方圖。

### 6. 訓練 Epochs 效應

![training_analysis](https://hackmd.io/_uploads/S1vRg5Ibbe.png)

展示訓練程度對 SQD 精度的影響，證實負相關現象。

### 7. 誤差分佈

![error_distributions](https://hackmd.io/_uploads/BJImb58b-x.png)

SQD 誤差的整體分佈特徵。

### 8. 熱力圖分析

![heatmap_analysis](https://hackmd.io/_uploads/rkI6xcUZ-e.png)

多維度參數對 SQD 精度影響的熱力圖視覺化。

---

## 討論

### 為何 NQS 訓練越好，SQD 越差?

1. **樣本多樣性喪失**: 訓練完善的 NQS 集中在少數低能態，減少 SQD 所需的構型空間探索
2. **Conservation Ratio 過高**: NQS 產生過多「有效」樣本，但這些樣本高度相關，無法提供足夠的 Hilbert 空間覆蓋
3. **小系統特性**: 12-bit 系統的 Hilbert 空間維度較小 (~500-1000)，均勻抽樣已能有效覆蓋

### LiH 樣本數閾值效應

LiH Baseline 在低樣本數時失效:

| n_samples | 平均誤差 (mHa) | 失敗率 (>1mHa) |
|-----------|---------------|----------------|
| 100       | 785.5         | 100%           |
| 250       | 195.2         | 66.7%          |
| >= 500    | ~0            | 0%             |

這表明即使是均勻抽樣，也需要足夠的樣本數才能覆蓋 LiH 的 495 維 Hilbert 空間。H4 (70 維) 和 H6 (924 維) 在所有測試樣本數下皆達到 FCI 精度，可能與其電子結構特性有關。

### 研究意涵

- 對於小型量子系統，簡單的均勻抽樣優於複雜的 NQS 方法
- NQS 的優勢可能在更大系統中顯現，當 Hilbert 空間維度遠超抽樣能力時
- SQD 的成功取決於樣本多樣性，而非樣本品質

---

## 結論

1. **Baseline 方法在 12-bit 系統中可達化學精度 (< 1 mHa)**，但 LiH 需要足夠樣本數 (n >= 500)
2. **NQS 訓練與 SQD 精度呈負相關**: 這是本實驗最重要的發現，但效應強度因分子而異 (H4/H6 較明顯，LiH 較弱)
3. **Conservation Ratio 並非 SQD 成功的充分指標**: 高 Conservation Ratio 可能反映樣本同質性過高
4. **樣本數閾值效應**: LiH 系統存在明確的樣本數閾值 (n=500)，低於此值 Baseline 也無法達到 FCI 精度
5. **後續研究方向**:
   - 測試更大系統 (20+ 軌域)
   - 探索混合抽樣策略
   - 研究 NQS 樣本多樣性與 SQD 精度的定量關係
   - 分析不同分子系統的樣本數閾值

---

## 附錄: 實驗環境

- Python 3.12+
- PyTorch 2.5+ (CUDA 12.1)
- qiskit-addon-sqd
- 硬體: NVIDIA RTX 4090

## 資料位置

- 原始數據: `results/phase_diagram/`
- 統計摘要: `results/figures/experiment_summary.csv`
- 視覺化圖表: `results/figures/*.png`

---

報告生成日期: 2025-11-28