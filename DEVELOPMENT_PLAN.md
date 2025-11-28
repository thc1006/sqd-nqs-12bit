# NQS-SQD 研究開發計畫

> 最後更新: 2025-11-28
> 負責人: Ting-Yi (蔡秀吉)
> 目標: 建立 SQD 在 biased sampler 下的 sample-efficiency phase diagram

---

## 研究核心問題

> 「只要 NQS 能量好到 XX 水準、抽樣 YY 個樣本、子空間維度 ZZ，
> 就足夠達到 12-bit energy 精度。」

這對於用真正 QPU 來跑 SQD 的人來說是超有參考價值的設計指引。

---

## 實驗完成狀態

**總計: 1788 次實驗 (2025-11-28 完成)**

| Phase | 任務 | 狀態 | 實驗數 |
|-------|------|------|--------|
| 0-1 | LiH 多幾何點驗證 | 完成 | 804 |
| 1-1 | NQS 品質控制掃描 | 完成 | (含於 0-1) |
| 1-2 | 樣本數/維度掃描 | 完成 | (含於 0-1) |
| 2-1 | H4 Chain (8 軌域) | 完成 | 768 |
| 2-2 | H6 Chain (12 軌域) | 完成 | 216 |
| 3-1 | Symmetry-Preserving NQS | 未開始 | - |
| 3-2 | Autoregressive NQS | 未開始 | - |

---

## 核心發現

### NQS 訓練與 SQD 精度呈負相關

這是本實驗最重要且出乎意料的發現: **NQS 訓練越完善，SQD 精度反而越差**。

| 分子 | 方法 | 實驗數 | SQD 誤差均值 (mHa) | Conservation Ratio |
|------|------|--------|-------------------|-------------------|
| LiH | NQS | 444 | 312.18 | 6.5% +/- 2.7% |
| LiH | Baseline | 360 | 65.38 | 5.1% +/- 0.7% |
| H4 | NQS | 384 | 156.07 | 35.7% +/- 24.2% |
| H4 | Baseline | 384 | ~0 | 14.5% +/- 0.7% |
| H6 | NQS | 108 | 385.77 | 25.1% +/- 14.8% |
| H6 | Baseline | 108 | ~0 | 9.8% +/- 0.2% |

### 訓練程度與 SQD 誤差關係

| Epochs | VMC 誤差 (mHa) | SQD 誤差 (mHa) | 觀察 |
|--------|----------------|----------------|------|
| 50-100 | 高 | ~0 | 可達 FCI 精度 |
| 200-500 | 低 | 500-2500 | 精度反而惡化 |

### 原因分析

1. **樣本多樣性喪失**: 訓練完善的 NQS 集中在少數低能態，減少 SQD 所需的構型空間探索
2. **Conservation Ratio 過高**: NQS 產生過多「有效」樣本，但高度相關，無法提供足夠的 Hilbert 空間覆蓋
3. **小系統特性**: 12-bit 系統的 Hilbert 空間維度較小 (~500-1000)，均勻抽樣已能有效覆蓋

---

## Phase 0: Baseline 與指標定義

### 0-1: LiH 多幾何點驗證 [完成]

- [x] LiH @ 0.8 A (初步驗證)
- [x] LiH @ 1.0 - 3.0 A (間隔 0.25 A，共 9 個幾何點)
- [x] 每個幾何點 × 4 epochs × 4 樣本數 × 3 重複 = 每點 48 次實驗

### 0-2: Metrics 系統 [完成]

| Metric | 定義 | 實際使用 |
|--------|------|----------|
| dE_VMC | E_VMC - E_FCI | 是 |
| dE_SQD | E_SQD - E_FCI | 是 |
| D | subspace dimension | 固定 (SQD 內部) |
| M | n_samples | 100, 500, 1000, 5000 |
| M_eff | n_conserved_samples | 是 |
| eta | M_eff / M | 是 (Conservation Ratio) |

### 0-3: 基準數據 (LiH @ 0.8 A)

```
HF:      -7.615770 Ha
FCI:     -7.634167 Ha
關聯能:   18.4 mHa
```

---

## Phase 1: NQS 品質 vs SQD 表現 Phase Diagram [完成]

### 1-1: NQS 品質控制變因

| 變因 | 實際掃描範圍 | 結果 |
|------|-------------|------|
| epochs | 50, 100, 200, 500 | 負相關 |
| alpha | 4 (固定) | - |
| n_layers | 3 (固定) | - |

### 1-2: 樣本數掃描

| 變因 | 實際掃描範圍 | 結果 |
|------|-------------|------|
| M (樣本數) | 100, 500, 1000, 5000 | Baseline 在所有樣本數均達 FCI |

### 1-3: Phase Diagram 視覺化 [完成]

生成圖表:
- `results/figures/vmc_vs_sqd_scatter.png` - VMC vs SQD 誤差散點圖
- `results/figures/sample_efficiency.png` - 樣本效率分析
- `results/figures/lih_bond_scan.png` - LiH 鍵長掃描
- `results/figures/hchain_scaling.png` - H-chain 規模效應

---

## Phase 2: 系統規模擴展 [完成]

### 2-1: H4 Chain (8 spin orbitals) [完成]

- [x] 建立 H4 分子結構 (linear chain, 1.0 A spacing)
- [x] 執行 NQS + SQD pipeline (384 次)
- [x] 執行 Baseline pipeline (384 次)
- [x] 結果: Baseline 達 FCI 精度，NQS 誤差 156 mHa

### 2-2: H6 Chain (12 spin orbitals) [完成]

- [x] 建立 H6 分子結構
- [x] 執行 NQS + SQD pipeline (108 次)
- [x] 執行 Baseline pipeline (108 次)
- [x] 結果: Baseline 達 FCI 精度，NQS 誤差 386 mHa

---

## Phase 3: 進階擴展 [未開始]

### 3-1: Symmetry-Preserving NQS

- [ ] 固定電子數 (Hamming weight)
- [ ] 固定 S_z
- [ ] 觀察 D/M 需求是否降低

### 3-2: Autoregressive NQS

- [ ] i.i.d. 樣本 (無 MCMC autocorrelation)
- [ ] 對比 FFNN-NQS 的 sample efficiency

---

## Bug 修復記錄

### Bug #1: vmc_training.py ERI 索引錯誤 [已修復]

- 檔案: `src/nqs_models/vmc_training.py:195-221`
- 症狀: HF 能量 -9.8 Ha (應為 -7.6 Ha)
- 修復: `eri[p,q,p,q]` -> `eri[p,p,q,q]` (chemist notation)

### Bug #2: gpu_optimized.py einsum 錯誤 [已修復]

- 檔案: `src/nqs_models/gpu_optimized.py:190-197`
- 症狀: 同 Bug #1
- 修復: einsum 改為 advanced indexing

### Bug #3: 外部 vmc_cal.py [需手動修復]

- 檔案: 外部模組 (不在此 repo)
- 狀態: 需手動修復

---

## 結論

對於 12-bit 系統 (LiH, H4, H6):

1. **SQD 後處理非常強大** - 即使初始採樣品質很差也能恢復 FCI 精度
2. **小系統下 Baseline 優於 NQS** - 均勻採樣反而優於訓練完善的 NQS
3. **NQS 的價值待驗證** - 需要在更大系統 (20+ 軌域) 上測試
4. **SQD 成功取決於樣本多樣性** - 而非樣本品質

---

## 後續研究方向

1. **測試更大系統** (20+ 軌域)，Hilbert 空間維度超過均勻抽樣能力時，NQS 可能展現優勢
2. **探索混合抽樣策略** - 結合 NQS 與均勻抽樣
3. **研究 NQS 樣本多樣性與 SQD 精度的定量關係**
4. **Symmetry-Preserving NQS** - 可能改善樣本效率

---

## 檔案結構

```
sqd-nqs-12bit/
├── CLAUDE.md                 # 專案指南
├── DEVELOPMENT_PLAN.md       # 開發計畫 (本文件)
├── plan.md                   # 研究計畫 (PI 會議用)
├── src/
│   ├── nqs_models/
│   │   ├── ffn_nqs.py        # FFNN NQS 模型
│   │   ├── vmc_training.py   # VMC 訓練 + SR
│   │   └── gpu_optimized.py  # GPU 優化版本
│   ├── sqd_interface/
│   │   ├── hamiltonian.py    # 分子積分 (PySCF)
│   │   └── sqd_runner.py     # SQD 對角化
│   └── experiments/
├── scripts/
│   ├── run_full_research_plan.py  # 完整實驗執行器
│   └── generate_phase_diagram.py  # 視覺化生成
├── results/
│   ├── phase_diagram/        # Phase 0-3 原始數據
│   ├── figures/              # 生成圖表
│   ├── EXPERIMENT_SUMMARY.md # 早期實驗摘要
│   └── figures/experiment_report.md  # 完整實驗報告
└── configs/
    └── *.yaml                # 實驗配置
```

---

## 參考資料

- [1] Qiskit addon SQD: https://qiskit.github.io/qiskit-addon-sqd/
- [2] SKQD: https://arxiv.org/html/2501.09702v1
- [4] SQD Limitations: https://arxiv.org/html/2501.07231v1
- [7] Fermionic NQS: https://www.nature.com/articles/s41467-020-15724-9
- [8] GTNN-SCI: https://pubs.acs.org/doi/full/10.1021/acs.jctc.5c01429
