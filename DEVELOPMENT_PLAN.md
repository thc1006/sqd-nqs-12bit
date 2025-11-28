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

## Phase 0: Baseline 與指標定義

### 0-1: LiH 多幾何點驗證
- [x] LiH @ 0.8 A (已完成)
- [ ] LiH @ 1.0 A
- [ ] LiH @ 1.5 A (接近解離)
- [ ] LiH @ 2.0 A (強關聯區)

### 0-2: Metrics 系統
| Metric | 定義 | 用途 |
|--------|------|------|
| ΔE_VMC | E_VMC - E_FCI | NQS 變分能量誤差 |
| ΔE_SQD | E_SQD - E_FCI | SQD 最終能量誤差 |
| D | subspace dimension | 子空間維度 |
| M | n_samples | 總樣本數 |
| M_eff | n_conserved_samples | 有效樣本數 (通過守恆篩選) |
| η | M_eff / M | 守恆比率 |

### 0-3: 已完成的基準數據 (LiH @ 0.8 A)
```
HF:      -7.615770 Ha
FCI:     -7.634167 Ha
關聯能:   18.4 mHa

Epoch 掃描結果:
| Epochs | ΔE_VMC (mHa) | ΔE_SQD (mHa) | η (%) |
|--------|--------------|--------------|-------|
| 2      | 4565         | ~0           | 5.7   |
| 5      | 4250         | ~0           | 5.3   |
| 10     | 3950         | ~0           | 5.2   |
| 20     | 3154         | ~0           | 4.9   |
| 40     | 2216         | ~0           | 4.6   |
| 100    | 762          | ~0           | 7.4   |
```

---

## Phase 1: NQS 品質 vs SQD 表現 Phase Diagram

### 1-1: NQS 品質控制變因
| 變因 | 掃描範圍 | 目的 |
|------|---------|------|
| epochs | 2, 5, 10, 20, 40, 100, 200 | 訓練程度 |
| alpha (hidden/visible) | 2, 4, 8 | 網路容量 |
| n_layers | 1, 2, 3 | 網路深度 |

### 1-2: 樣本數與子空間維度掃描
| 變因 | 掃描範圍 |
|------|---------|
| M (樣本數) | 500, 1000, 2000, 5000, 10000 |
| max_dim | 50, 100, 200, 500 (如果支援) |

### 1-3: Phase Diagram 視覺化
目標圖表:
- 3D/contour: x=ΔE_VMC, y=D, color=ΔE_SQD
- 標出 12-bit (1 mHa) 和 chemical accuracy (1.6 mHa) contour
- 標出當前 LiH 案例位置

---

## Phase 2: 系統規模擴展

### 2-1: H4 Chain (16 spin orbitals)
- [ ] 建立 H4 分子結構
- [ ] 執行 NQS + SQD pipeline
- [ ] 對比 NQS vs Baseline
- [ ] 分析 sample efficiency

### 2-2: H6 Chain (24 spin orbitals)
- [ ] 建立 H6 分子結構
- [ ] 執行實驗 (預計 baseline 開始失效)
- [ ] 展示 NQS 優勢

---

## Phase 3: 進階擴展 (資源允許時)

### 3-1: Symmetry-Preserving NQS
- 固定電子數 (Hamming weight)
- 固定 S_z
- 觀察 D/M 需求是否降低

### 3-2: Autoregressive NQS
- i.i.d. 樣本 (無 MCMC autocorrelation)
- 對比 FFNN-NQS 的 sample efficiency

---

## 進度追蹤

| Phase | 任務 | 狀態 | 完成日期 |
|-------|------|------|----------|
| 0-1 | LiH @ 0.8 A | 完成 | 2025-11-28 |
| 0-1 | LiH bond length 掃描 | 待開始 | |
| 0-2 | Metrics 系統 | 待開始 | |
| 1-1 | NQS 品質控制掃描 | 待開始 | |
| 1-2 | 樣本數/維度掃描 | 待開始 | |
| 1-3 | Phase Diagram | 待開始 | |
| 2-1 | H4 Chain | 待開始 | |
| 2-2 | H6 Chain | 待開始 | |
| 3-1 | Symmetry NQS | 待開始 | |
| 3-2 | Autoregressive NQS | 待開始 | |

---

## Bug 修復記錄

### Bug #1: vmc_training.py ERI 索引錯誤
- 檔案: `src/nqs_models/vmc_training.py:195-221`
- 症狀: HF 能量 -9.8 Ha (應為 -7.6 Ha)
- 修復: `eri[p,q,p,q]` → `eri[p,p,q,q]` (chemist notation)
- 狀態: ✅ 已修復

### Bug #2: gpu_optimized.py einsum 錯誤
- 檔案: `src/nqs_models/gpu_optimized.py:190-197`
- 症狀: 同 Bug #1
- 修復: einsum 改為 advanced indexing
- 狀態: ✅ 已修復

### Bug #3: 外部 vmc_cal.py
- 檔案: 外部模組 (不在此 repo)
- 狀態: ⚠️ 需手動修復

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
│       ├── h2_12bit_small_sample.py
│       ├── ablation_nqs_vs_baseline.py
│       └── h_chain_scaling.py
├── scripts/
│   ├── run_phase_diagram.py  # Phase 1 主實驗
│   └── plot_results.py       # 視覺化
├── results/
│   ├── raw/                  # 原始 JSON 結果
│   ├── ablation/             # 消融實驗
│   ├── epoch_sweep/          # Epoch 掃描
│   ├── phase_diagram/        # Phase Diagram 數據
│   ├── figures/              # 圖表
│   └── EXPERIMENT_SUMMARY.md # 實驗摘要
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
